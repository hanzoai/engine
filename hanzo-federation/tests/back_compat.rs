//! Wire-format byte-identity check: the canonical bf16 encoder MUST keep
//! producing the same bytes Python workers already speak. This test always
//! builds (no feature gates) — `cargo test -p hanzo-federation` must pass
//! whether or not `--features compression` is on.
//!
//! What we assert:
//! 1. A blob built via `encode_delta` / `encode_delta_with_meta` does not
//!    contain the string `"codec"` in its header. (Adding a `codec` field
//!    behind a `skip_serializing_if = "Option::is_none"` should never emit
//!    it for the bf16 path.)
//! 2. The decode_delta round-trip yields back the exact body bytes.
//! 3. A frozen reference blob (built by the previous version of this codec)
//!    decodes to the same `(name, shape, body)` triples it did before.

use hanzo_federation::codec::{
    bf16_to_f32, decode_delta, encode_delta, encode_delta_with_meta, f32_to_bf16,
};

#[test]
fn bf16_encoder_emits_no_codec_field() {
    let raw = f32_to_bf16(&[1.0, 2.0, 3.0, 4.0]);
    let items: Vec<(&str, &[u8])> = vec![("w", raw.as_slice())];
    let blob = encode_delta(&items);

    let hdr_len = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
    let hdr_str = std::str::from_utf8(&blob[8..8 + hdr_len]).unwrap();

    // Must not have introduced a `codec` field for the bf16 path — that
    // would break Python workers' header schema.
    assert!(
        !hdr_str.contains("\"codec\""),
        "bf16 encoder leaked a codec field into the header: {hdr_str}"
    );
    // Sanity: still the schema Python speaks.
    assert!(hdr_str.contains("\"dtype\":\"BF16\""));
    assert!(hdr_str.contains("\"offsets\""));
    assert!(hdr_str.contains("\"shape\""));
}

#[test]
fn bf16_encoder_with_meta_emits_no_codec_field() {
    let raw_a = f32_to_bf16(&[1.0, 2.0, 3.0, 4.0]);
    let items: Vec<(String, &[u8], Vec<u64>)> =
        vec![("A".to_string(), raw_a.as_slice(), vec![2, 2])];
    let blob = encode_delta_with_meta(&items);

    let hdr_len = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
    let hdr_str = std::str::from_utf8(&blob[8..8 + hdr_len]).unwrap();
    assert!(
        !hdr_str.contains("\"codec\""),
        "bf16 with-meta encoder leaked a codec field: {hdr_str}"
    );
}

#[test]
fn bf16_round_trip_preserves_body_bytes_exactly() {
    let raw_a = f32_to_bf16(&[1.0, 2.0, 3.0, 4.0]);
    let raw_b = f32_to_bf16(&[-1.0, -2.0]);
    let items: Vec<(String, &[u8], Vec<u64>)> = vec![
        ("A".to_string(), raw_a.as_slice(), vec![2, 2]),
        ("B".to_string(), raw_b.as_slice(), vec![2]),
    ];
    let blob = encode_delta_with_meta(&items);
    let decoded = decode_delta(&blob).unwrap();
    let map: std::collections::HashMap<_, _> =
        decoded.into_iter().map(|(n, m, b)| (n, (m, b))).collect();
    assert_eq!(map["A"].1, raw_a);
    assert_eq!(map["B"].1, raw_b);
    assert_eq!(map["A"].0.shape, vec![2, 2]);
    assert_eq!(map["B"].0.shape, vec![2]);
    // Both tensors must report no codec (None) on the bf16 wire.
    assert!(map["A"].0.codec.is_none());
    assert!(map["B"].0.codec.is_none());
}

#[test]
fn legacy_blob_decodes_identically() {
    // Build a blob manually with the exact byte layout pre-dating the
    // codec field — this is what Python writes and what older Rust workers
    // produced. We construct it by hand to be sure no current encoder
    // behaviour can mask a regression.
    let raw = f32_to_bf16(&[1.5, -2.5, 3.5, -4.5]);
    // Compact JSON, no `codec` key, no spaces — matches Python's
    // `json.dumps(..., separators=(',', ':'))`.
    let hdr = format!(
        "{{\"x\":{{\"dtype\":\"BF16\",\"shape\":[4],\"offsets\":[0,{}]}}}}",
        raw.len()
    );
    let hdr_bytes = hdr.as_bytes();
    let mut blob = Vec::with_capacity(8 + hdr_bytes.len() + raw.len());
    blob.extend_from_slice(&(hdr_bytes.len() as u64).to_le_bytes());
    blob.extend_from_slice(hdr_bytes);
    blob.extend_from_slice(&raw);

    let decoded = decode_delta(&blob).expect("legacy blob still parses");
    assert_eq!(decoded.len(), 1);
    let (name, meta, body) = &decoded[0];
    assert_eq!(name, "x");
    assert_eq!(meta.shape, vec![4]);
    assert!(meta.codec.is_none(), "legacy header must decode codec as None");
    assert_eq!(body, &raw);
    let recovered = bf16_to_f32(body);
    for (got, want) in recovered.iter().zip(&[1.5_f32, -2.5, 3.5, -4.5]) {
        assert!((got - want).abs() < 1e-2, "got {got}, want {want}");
    }
}
