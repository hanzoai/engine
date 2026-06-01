//! BitDelta wire codec — opt-in compressed delta path.
//!
//! Uses the **same envelope** as the canonical bf16 codec (u64 LE hdr_len + JSON
//! header + body), but per-tensor:
//!
//! * `dtype` = `"BITDELTA"` (informational; the authoritative dispatch field is
//!   `codec`)
//! * `codec` = `"bitdelta"`
//! * Body slice `[offsets[0]..offsets[1]]` is
//!   `[scale_f32_le: 4 bytes][sign_bytes: ceil(numel/8) bytes]`
//!
//! Reconstruction (per [`hanzo_quant::BitDelta`]): element `i` decodes to
//! `(sign_bit ? +1 : -1) * scale`. Compression vs raw bf16 is ~16x
//! (2 bytes/elt → 1 bit/elt + 4 bytes scale).
//!
//! Back-compat: the canonical bf16 path stays untouched in [`crate::codec`].
//! Workers without the `compression` feature simply never emit `codec` and the
//! decoder treats the body as bf16 — so this module is purely additive.

use anyhow::{anyhow, Context, Result};
use bytes::BytesMut;
use serde_json::{json, Map, Value};

use crate::codec::TensorMeta;

/// Body prefix bytes (the f32 scale, little-endian).
const SCALE_BYTES: usize = 4;

/// Encode `(name, base, weight)`-style deltas using BitDelta and pack into
/// the canonical envelope. The caller passes paired `(name, current_f32)` and
/// `(name, base_f32)` slices; we compute `delta = current - base` per element,
/// then run [`hanzo_quant::BitDelta::encode_delta`]-equivalent math.
///
/// Both slices are matched by name; missing names in either side produce an
/// error rather than silently dropping tensors. The header order follows
/// `params`' iteration order.
pub fn encode_bitdelta(
    params: &[(String, Vec<f32>)],
    base: &[(String, Vec<f32>)],
) -> Result<Vec<u8>> {
    use std::collections::HashMap;
    let base_map: HashMap<&str, &Vec<f32>> = base.iter().map(|(n, v)| (n.as_str(), v)).collect();

    let mut header = Map::new();
    let mut body = BytesMut::new();
    let mut offset: u64 = 0;
    for (name, curr) in params {
        let b = base_map
            .get(name.as_str())
            .ok_or_else(|| anyhow!("base missing tensor {name:?}"))?;
        if curr.len() != b.len() {
            return Err(anyhow!(
                "tensor {name:?} length mismatch: curr={} base={}",
                curr.len(),
                b.len()
            ));
        }
        if curr.is_empty() {
            return Err(anyhow!("tensor {name:?} has 0 elements"));
        }

        // delta = curr - base; scale = mean(|delta|), clamped > 0; signs packed LE.
        let numel = curr.len();
        let nbytes_signs = numel.div_ceil(8);
        let mut sign_bits = vec![0u8; nbytes_signs];
        let mut abs_sum = 0.0f32;
        for i in 0..numel {
            let d = curr[i] - b[i];
            abs_sum += d.abs();
            if d >= 0.0 {
                sign_bits[i / 8] |= 1u8 << (i % 8);
            }
        }
        let scale = (abs_sum / numel as f32).max(1e-8);

        // Body slice: [scale_f32_le | sign_bytes]
        let start = offset;
        body.extend_from_slice(&scale.to_le_bytes());
        body.extend_from_slice(&sign_bits);
        offset += (SCALE_BYTES + nbytes_signs) as u64;

        header.insert(
            name.clone(),
            json!({
                "dtype": "BITDELTA",
                "shape": [numel as u64],
                "offsets": [start, offset],
                "codec": "bitdelta",
            }),
        );
    }

    let hdr_bytes = serde_json::to_vec(&Value::Object(header))
        .expect("BitDelta header is JSON-serializable");
    let mut out = Vec::with_capacity(8 + hdr_bytes.len() + body.len());
    out.extend_from_slice(&(hdr_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&hdr_bytes);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode a full BitDelta blob to `(name, f32_delta)` pairs. Returns f32
/// because the immediate consumer (the coordinator's trim-mean) needs f32 to
/// aggregate; the caller can re-bf16 the result if it wants to retransmit.
pub fn decode_bitdelta(blob: &[u8]) -> Result<Vec<(String, Vec<f32>)>> {
    let triples = crate::codec::decode_delta(blob)?;
    let mut out = Vec::with_capacity(triples.len());
    for (name, meta, body) in triples {
        let codec = meta.codec.as_deref().unwrap_or("bf16");
        if codec != "bitdelta" {
            return Err(anyhow!(
                "decode_bitdelta: tensor {name:?} has codec={codec:?}, expected \"bitdelta\""
            ));
        }
        let values = decode_bitdelta_tensor(&meta, &body)
            .with_context(|| format!("decoding bitdelta tensor {name:?}"))?;
        out.push((name, values));
    }
    Ok(out)
}

/// Decode a single bitdelta-coded body slice using its TensorMeta. Returns
/// the reconstructed f32 delta values. Cross-codec dispatch (in
/// [`crate::codec::decode_delta_to_f32`]) routes here for `codec="bitdelta"`.
pub fn decode_bitdelta_tensor(meta: &TensorMeta, body: &[u8]) -> Result<Vec<f32>> {
    if body.len() < SCALE_BYTES {
        return Err(anyhow!("bitdelta body too small for scale ({} bytes)", body.len()));
    }
    let scale = f32::from_le_bytes(body[..SCALE_BYTES].try_into().unwrap());
    let sign_bytes = &body[SCALE_BYTES..];
    let numel: usize = meta.element_count() as usize;
    let want_sign_bytes = numel.div_ceil(8);
    if sign_bytes.len() < want_sign_bytes {
        return Err(anyhow!(
            "bitdelta body truncated: want {} sign bytes, got {}",
            want_sign_bytes,
            sign_bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        let bit = (sign_bytes[i / 8] >> (i % 8)) & 1;
        out.push(if bit == 1 { scale } else { -scale });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_recovers_signs_and_scale() {
        // curr - base produces alternating ±0.2 with mean(|.|) = 0.2
        let base = vec![("w".to_string(), vec![0.0f32; 16])];
        let curr_vals: Vec<f32> =
            (0..16).map(|i| if i % 2 == 0 { 0.2 } else { -0.2 }).collect();
        let params = vec![("w".to_string(), curr_vals.clone())];
        let blob = encode_bitdelta(&params, &base).unwrap();
        let decoded = decode_bitdelta(&blob).unwrap();
        assert_eq!(decoded.len(), 1);
        let (name, dvals) = &decoded[0];
        assert_eq!(name, "w");
        assert_eq!(dvals.len(), 16);
        for (got, want_sign) in dvals.iter().zip(curr_vals.iter()) {
            assert!(got.signum() == want_sign.signum());
            assert!((got.abs() - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn header_marks_codec_bitdelta() {
        let base = vec![("w".to_string(), vec![0.0f32; 4])];
        let params = vec![("w".to_string(), vec![1.0f32, -1.0, 1.0, -1.0])];
        let blob = encode_bitdelta(&params, &base).unwrap();
        let hdr_len = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
        let hdr_str = std::str::from_utf8(&blob[8..8 + hdr_len]).unwrap();
        assert!(hdr_str.contains("\"codec\":\"bitdelta\""));
        assert!(hdr_str.contains("\"dtype\":\"BITDELTA\""));
    }
}
