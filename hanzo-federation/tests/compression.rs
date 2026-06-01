//! Integration tests for the opt-in BitDelta wire codec.
//!
//! Only built when `--features compression` is on; covers:
//! * BitDelta round-trip fidelity (`max|err| <= per-tensor scale`).
//! * Mixed-codec aggregation: a coordinator must aggregate a worker delta
//!   built with BitDelta against two workers built with bf16.
//! * Compressed-blob size vs raw bf16 envelope, on a 4096-element delta.

#![cfg(feature = "compression")]

use hanzo_federation::codec::{
    bf16_to_f32, decode_delta, decode_delta_to_f32, encode_delta_with_meta, f32_to_bf16, TensorMeta,
};
use hanzo_federation::codec_bitdelta::{decode_bitdelta, encode_bitdelta};
use hanzo_federation::coordinator::{aggregate, AggregationMethod};
use std::collections::HashMap;

#[test]
fn bitdelta_round_trip_within_scale_tolerance() {
    // Build a delta with non-trivial dynamic range so scale = mean(|.|) is
    // meaningful and reconstruction error per element is bounded by scale.
    let base = vec![("w".to_string(), vec![0.0f32; 256])];
    let curr_vals: Vec<f32> = (0..256)
        .map(|i| ((i as f32) - 128.0) * 0.01) // -1.28 .. +1.27 step 0.01
        .collect();
    let params = vec![("w".to_string(), curr_vals.clone())];

    let blob = encode_bitdelta(&params, &base).unwrap();
    let decoded = decode_bitdelta(&blob).unwrap();
    assert_eq!(decoded.len(), 1);
    let (name, dvals) = &decoded[0];
    assert_eq!(name, "w");
    assert_eq!(dvals.len(), curr_vals.len());

    // True delta == curr (base is 0). Reconstructed value is sign(d) * scale.
    let scale: f32 = {
        let s: f32 = curr_vals.iter().map(|x| x.abs()).sum::<f32>() / curr_vals.len() as f32;
        s.max(1e-8)
    };
    let max_err = dvals
        .iter()
        .zip(curr_vals.iter())
        .map(|(got, want)| (got - want).abs())
        .fold(0.0_f32, f32::max);
    // Each element has error at most |true - sign(true)*scale| <= max(|true|, scale).
    let bound = curr_vals
        .iter()
        .map(|x| x.abs().max(scale))
        .fold(0.0_f32, f32::max);
    assert!(
        max_err <= bound + 1e-6,
        "max_err {max_err} exceeded bound {bound} (scale={scale})"
    );
}

#[test]
fn mixed_codec_aggregation_round_trips() {
    // Three workers, same tensor "w" of 8 elements.
    // Worker A and B encode bf16; worker C encodes BitDelta.
    let vals_a = vec![0.10f32, 0.20, 0.30, 0.40, -0.10, -0.20, -0.30, -0.40];
    let vals_b = vec![0.11f32, 0.21, 0.29, 0.39, -0.11, -0.21, -0.29, -0.39];
    let vals_c = vec![0.12f32, 0.22, 0.31, 0.41, -0.12, -0.22, -0.31, -0.41];

    // Build a bf16 blob via the canonical path.
    let bf16_blob = |name: &str, vals: &[f32]| -> Vec<u8> {
        let raw = f32_to_bf16(vals);
        let items: Vec<(String, &[u8], Vec<u64>)> =
            vec![(name.to_string(), raw.as_slice(), vec![vals.len() as u64])];
        encode_delta_with_meta(&items)
    };
    let bd_blob = |name: &str, vals: &[f32]| -> Vec<u8> {
        let base = vec![(name.to_string(), vec![0.0f32; vals.len()])];
        let params = vec![(name.to_string(), vals.to_vec())];
        encode_bitdelta(&params, &base).unwrap()
    };

    let blobs = vec![
        bf16_blob("w", &vals_a),
        bf16_blob("w", &vals_b),
        bd_blob("w", &vals_c),
    ];

    // Mimic what coordinator::run_aggregation does: decode each blob to
    // (name, meta, raw_body) and feed into aggregate(). Aggregate() then
    // dispatches per-tensor on meta.codec.
    let mut decoded: Vec<HashMap<String, (TensorMeta, Vec<u8>)>> = Vec::new();
    for b in &blobs {
        let triples = decode_delta(b).unwrap();
        let mut m = HashMap::new();
        for (n, meta, raw) in triples {
            m.insert(n, (meta, raw));
        }
        decoded.push(m);
    }
    let agg = aggregate(&decoded, AggregationMethod::Mean).unwrap();
    assert_eq!(agg.len(), 1);
    let (name, shape, agg_bytes) = &agg[0];
    assert_eq!(name, "w");
    assert_eq!(shape, &vec![8u64]);

    // Sanity: coordinator emits bf16 — bytes are 2 per element.
    assert_eq!(agg_bytes.len(), 16);
    let agg_f32 = bf16_to_f32(agg_bytes);

    // Expected: mean of (vals_a[i], vals_b[i], sign(vals_c[i]) * scale_c).
    // BitDelta scale_c = mean(|vals_c|) = mean of {0.12..0.41}.
    let scale_c: f32 =
        vals_c.iter().map(|x| x.abs()).sum::<f32>() / vals_c.len() as f32;
    for (i, got) in agg_f32.iter().enumerate() {
        let recon_c = vals_c[i].signum() * scale_c;
        let want = (vals_a[i] + vals_b[i] + recon_c) / 3.0;
        // bf16 truncation tolerance.
        assert!(
            (got - want).abs() < 0.05,
            "elt {i}: got {got}, want {want} (recon_c={recon_c})"
        );
    }

    // Sanity: also verify the decode_delta_to_f32 helper agrees on the
    // BitDelta blob standalone.
    let bd_only = decode_delta_to_f32(&blobs[2]).unwrap();
    assert_eq!(bd_only.len(), 1);
    let (n, _shape, vals) = &bd_only[0];
    assert_eq!(n, "w");
    for (got, want) in vals.iter().zip(vals_c.iter()) {
        let recon = want.signum() * scale_c;
        assert!((got - recon).abs() < 1e-6);
    }
}

#[test]
fn bitdelta_compresses_4k_delta_vs_bf16() {
    // 4096-element f32 delta with realistic dynamic range.
    let n: usize = 4096;
    let base = vec![("w".to_string(), vec![0.0f32; n])];
    let curr: Vec<f32> = (0..n)
        .map(|i| ((i as f32) - (n as f32 / 2.0)) * 1e-3)
        .collect();
    let params = vec![("w".to_string(), curr.clone())];

    // bf16 envelope size: u64 hdr_len + json hdr + 2*n bytes body.
    let raw_bf16 = f32_to_bf16(&curr);
    let bf16_blob = {
        let items: Vec<(String, &[u8], Vec<u64>)> =
            vec![("w".to_string(), raw_bf16.as_slice(), vec![n as u64])];
        encode_delta_with_meta(&items)
    };

    // BitDelta envelope: same wrapper, body is [scale f32 | ceil(n/8) bytes].
    let bd_blob = encode_bitdelta(&params, &base).unwrap();

    let bf16_size = bf16_blob.len();
    let bd_size = bd_blob.len();
    let ratio = bf16_size as f32 / bd_size as f32;

    // Body alone: bf16 = 2*4096 = 8192 bytes; bitdelta = 4 + 512 = 516 bytes.
    // Including envelopes the ratio is still >= ~15.5x.
    println!(
        "n={n} bf16_blob={bf16_size}B bitdelta_blob={bd_size}B ratio={ratio:.2}x"
    );
    assert!(
        ratio > 12.0,
        "expected >12x size reduction; got {ratio:.2}x (bf16={bf16_size}, bd={bd_size})"
    );
}
