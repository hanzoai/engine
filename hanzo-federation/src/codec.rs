//! Canonical BF16 LoRA-delta wire format.
//!
//! Layout — must be byte-identical to `coordinator.py::_encode/_decode`:
//!
//! ```text
//! u64 little-endian header_length
//! json header: {name: {dtype: "BF16", shape: [...], offsets: [start, end]}}
//! concatenated bf16 little-endian body (one u16 per element)
//! ```
//!
//! The body is opaque bytes from the caller's perspective — we never look
//! at numeric values, only at offsets and shapes. Trim-mean aggregation
//! does interpret bf16 as f32 in [`crate::coordinator`].

use anyhow::{anyhow, Context, Result};
use bytes::{BufMut, BytesMut};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// Header entry for one tensor in the canonical blob.
///
/// `codec` is optional for back-compat: absent or `"bf16"` => the body slice
/// at `offsets` is raw little-endian bf16. `"bitdelta"` (only when the
/// `compression` feature is built) => the body slice is
/// `[scale_f32_le:4][sign_bytes:ceil(N/8)]`. Decoding is via
/// [`crate::codec_bitdelta::decode_bitdelta_tensor`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<u64>,
    /// `[start, end]` bytes within the body region (i.e. after the header).
    pub offsets: [u64; 2],
    /// Per-tensor codec hint. `None` or `Some("bf16")` => raw bf16 body.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codec: Option<String>,
}

impl TensorMeta {
    pub fn nbytes(&self) -> u64 {
        self.offsets[1] - self.offsets[0]
    }

    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// Encode `(name, raw_bf16_le_bytes)` pairs into a canonical blob.
///
/// The caller is responsible for the bytes already being in bf16 LE form —
/// this function does no numeric conversion. Order of names in the output
/// header preserves insertion order (matching Python's dict ordering).
pub fn encode_delta<S: AsRef<str>, B: AsRef<[u8]>>(items: &[(S, B)]) -> Vec<u8> {
    // Build header preserving insertion order (Map serializes in insert order).
    let mut header = Map::new();
    let mut body = BytesMut::new();
    let mut offset: u64 = 0;
    for (name, raw) in items {
        let raw = raw.as_ref();
        let entry = json!({
            "dtype": "BF16",
            "shape": [(raw.len() / 2) as u64],
            "offsets": [offset, offset + raw.len() as u64],
        });
        header.insert(name.as_ref().to_string(), entry);
        body.put_slice(raw);
        offset += raw.len() as u64;
    }
    encode_with_header(Value::Object(header), &body)
}

/// Encode with caller-provided per-tensor metadata (so shape isn't flattened).
pub fn encode_delta_with_meta<S: AsRef<str>, B: AsRef<[u8]>>(
    items: &[(S, B, Vec<u64>)],
) -> Vec<u8> {
    let mut header = Map::new();
    let mut body = BytesMut::new();
    let mut offset: u64 = 0;
    for (name, raw, shape) in items {
        let raw = raw.as_ref();
        let entry = json!({
            "dtype": "BF16",
            "shape": shape,
            "offsets": [offset, offset + raw.len() as u64],
        });
        header.insert(name.as_ref().to_string(), entry);
        body.put_slice(raw);
        offset += raw.len() as u64;
    }
    encode_with_header(Value::Object(header), &body)
}

fn encode_with_header(header: Value, body: &[u8]) -> Vec<u8> {
    // `serde_json::to_vec` matches Python's `json.dumps(..., separators=(",", ":"))`
    // when there are no `to_string`-ambiguous values — i.e. for our schema.
    let hdr_json = serde_json::to_vec(&header).expect("header is JSON-serializable");
    let mut out = Vec::with_capacity(8 + hdr_json.len() + body.len());
    out.extend_from_slice(&(hdr_json.len() as u64).to_le_bytes());
    out.extend_from_slice(&hdr_json);
    out.extend_from_slice(body);
    out
}

/// Decode a canonical blob into `(name, meta, body_bytes)` triples.
///
/// `body_bytes` is a copy of the raw bf16 bytes (matching the Python `.copy()`
/// — important because the caller may later free the blob).
pub fn decode_delta(blob: &[u8]) -> Result<Vec<(String, TensorMeta, Vec<u8>)>> {
    if blob.len() < 8 {
        return Err(anyhow!("blob too short for header length"));
    }
    let hdr_len = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
    if blob.len() < 8 + hdr_len {
        return Err(anyhow!("blob too short for declared header"));
    }
    let hdr_bytes = &blob[8..8 + hdr_len];
    let hdr: serde_json::Map<String, Value> =
        serde_json::from_slice(hdr_bytes).context("parse delta header")?;
    let base = 8 + hdr_len;
    let body = &blob[base..];

    let mut out = Vec::with_capacity(hdr.len());
    for (name, raw_meta) in hdr {
        let meta: TensorMeta = serde_json::from_value(raw_meta)
            .with_context(|| format!("parse meta for tensor {name:?}"))?;
        let s = meta.offsets[0] as usize;
        let e = meta.offsets[1] as usize;
        if e > body.len() {
            return Err(anyhow!(
                "tensor {name:?} offsets [{s},{e}] exceed body len {}",
                body.len()
            ));
        }
        out.push((name, meta, body[s..e].to_vec()));
    }
    Ok(out)
}

/// View-cast bf16 LE bytes to f32. Same algorithm as
/// `_bf16_to_f32` in coordinator.py.
pub fn bf16_to_f32(raw: &[u8]) -> Vec<f32> {
    assert!(raw.len() % 2 == 0, "bf16 buffer must be even length");
    raw.chunks_exact(2)
        .map(|c| {
            let u = u16::from_le_bytes([c[0], c[1]]);
            // bf16 → f32 by left-shifting 16 bits and reinterpreting.
            f32::from_bits((u as u32) << 16)
        })
        .collect()
}

/// Round-trip the other way: f32 → bf16 LE bytes by truncation, matching
/// `_f32_to_bf16` in coordinator.py.
pub fn f32_to_bf16(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &f in values {
        let bits = f.to_bits();
        let trunc = (bits >> 16) as u16;
        out.extend_from_slice(&trunc.to_le_bytes());
    }
    out
}

/// Decode any blob (bf16 or bitdelta per-tensor) into `(name, shape, f32_values)`
/// triples. Dispatches on `TensorMeta.codec`:
///
/// * `None` or `Some("bf16")` → raw bf16 LE body, decoded via [`bf16_to_f32`].
/// * `Some("bitdelta")` → requires the `compression` feature; routes to
///   [`crate::codec_bitdelta::decode_bitdelta_tensor`]. If the feature is off,
///   returns an error so the caller never silently corrupts data.
///
/// This is the single dispatch point the coordinator uses — keeps codec choice
/// out of the aggregation math.
pub fn decode_delta_to_f32(blob: &[u8]) -> Result<Vec<(String, Vec<u64>, Vec<f32>)>> {
    let triples = decode_delta(blob)?;
    let mut out = Vec::with_capacity(triples.len());
    for (name, meta, body) in triples {
        let codec = meta.codec.as_deref().unwrap_or("bf16");
        let values = match codec {
            "bf16" => bf16_to_f32(&body),
            "bitdelta" => {
                #[cfg(feature = "compression")]
                {
                    crate::codec_bitdelta::decode_bitdelta_tensor(&meta, &body)
                        .with_context(|| format!("bitdelta decode of {name:?}"))?
                }
                #[cfg(not(feature = "compression"))]
                {
                    return Err(anyhow!(
                        "tensor {name:?} uses codec=\"bitdelta\" but hanzo-federation was \
                         built without the `compression` feature; rebuild with \
                         `--features compression` to decode"
                    ));
                }
            }
            other => {
                return Err(anyhow!(
                    "tensor {name:?} uses unknown codec {other:?}; expected \"bf16\" or \"bitdelta\""
                ));
            }
        };
        out.push((name, meta.shape, values));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_trip_matches_python() {
        // Python: (f32.view(uint32) >> 16).astype(uint16)
        // For f=1.0, bits = 0x3F800000, trunc = 0x3F80 → bf16(1.0) → f32(1.0)
        let xs: Vec<f32> = vec![1.0, -0.5, 0.0, 3.14159];
        let bf = f32_to_bf16(&xs);
        let back = bf16_to_f32(&bf);
        assert_eq!(back.len(), xs.len());
        // bf16 has 7-bit mantissa → rel error ~1e-2 max for 3.14159
        for (a, b) in xs.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.05, "got {b} for {a}");
        }
    }

    #[test]
    fn encode_decode_round_trip() {
        let raw_a = f32_to_bf16(&[1.0, 2.0, 3.0, 4.0]);
        let raw_b = f32_to_bf16(&[-1.0, -2.0]);
        let items: Vec<(String, &[u8], Vec<u64>)> = vec![
            ("A".to_string(), raw_a.as_slice(), vec![2, 2]),
            ("B".to_string(), raw_b.as_slice(), vec![2]),
        ];
        let blob = encode_delta_with_meta(&items);

        // Manually parse header length.
        let hdr_len = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
        let hdr_str = std::str::from_utf8(&blob[8..8 + hdr_len]).unwrap();
        assert!(hdr_str.contains("\"BF16\""));
        assert!(hdr_str.contains("\"A\""));
        assert!(hdr_str.contains("\"B\""));

        let decoded = decode_delta(&blob).unwrap();
        assert_eq!(decoded.len(), 2);
        let map: std::collections::HashMap<_, _> =
            decoded.into_iter().map(|(n, m, b)| (n, (m, b))).collect();
        assert_eq!(map["A"].0.shape, vec![2, 2]);
        assert_eq!(map["A"].1, raw_a);
        assert_eq!(map["B"].0.shape, vec![2]);
        assert_eq!(map["B"].1, raw_b);
    }

    #[test]
    fn header_is_compact_json() {
        // Python uses separators=(",", ":") — no spaces. serde_json default
        // also has no spaces in compact form (which is the default).
        let raw = f32_to_bf16(&[1.0]);
        let items: Vec<(&str, &[u8])> = vec![("x", raw.as_slice())];
        let blob = encode_delta(&items);
        let hdr_len = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
        let hdr_str = std::str::from_utf8(&blob[8..8 + hdr_len]).unwrap();
        assert!(!hdr_str.contains(", "), "header had ', ' — not compact");
        assert!(!hdr_str.contains(": "), "header had ': ' — not compact");
    }
}
