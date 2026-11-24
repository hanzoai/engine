use std::sync::{Arc, Barrier};
use std::time::Instant;

use half::f16;
use hanzo_paged_attn::kivi::*;

const OUTLIER_CHANNELS: [(usize, f32); 4] = [(0, 68.0), (16, -54.0), (42, 92.0), (99, -76.0)];

#[derive(Clone, Copy)]
struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.state >> 40) as u32;
        (bits as f32 / (1u32 << 23) as f32) - 1.0
    }
}

fn is_outlier_channel(ch: usize) -> Option<f32> {
    for &(idx, bias) in &OUTLIER_CHANNELS {
        if idx == ch {
            return Some(bias);
        }
    }
    None
}

struct NeedleCase {
    position: usize,
    query: Vec<f32>,
    target_value: Vec<f32>,
}

fn build_outlier_workload(
    seq_len: usize,
    num_kv_heads: usize,
    head_size: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>, Vec<NeedleCase>) {
    let mut rng = Lcg64::new(seed);
    let stride = num_kv_heads * head_size;
    let mut keys = vec![0.0f32; seq_len * stride];
    let mut vals = vec![0.0f32; seq_len * stride];

    for t in 0..seq_len {
        let base = t * stride;
        for h in 0..num_kv_heads {
            let off = base + h * head_size;
            for c in 0..head_size {
                keys[off + c] = match is_outlier_channel(c) {
                    Some(bias) => bias + rng.next_f32() * 1.4,
                    None => rng.next_f32() * 1.1,
                };
                vals[off + c] = rng.next_f32() * 1.5;
            }
        }
    }

    let positions = [
        seq_len / 10,
        seq_len / 2,
        (seq_len * 17) / 20,
        seq_len.saturating_sub(12),
    ];
    let mut needles = Vec::with_capacity(positions.len());

    for (n_idx, &pos) in positions.iter().enumerate() {
        let mut pattern = vec![0.0f32; head_size];
        let mut target_val = vec![0.0f32; head_size];
        for c in 0..head_size {
            if is_outlier_channel(c).is_none() {
                pattern[c] = if ((c >> n_idx) & 1) == 0 { 1.0 } else { -1.0 };
            }
            let shifted = (c + n_idx * 17) % head_size;
            let ramp = -2.5 + 5.0 * (shifted as f32 / (head_size - 1) as f32);
            target_val[c] = if n_idx.is_multiple_of(2) { ramp } else { -ramp };
        }

        for h in 0..num_kv_heads {
            let off = pos * stride + h * head_size;
            for c in 0..head_size {
                if is_outlier_channel(c).is_none() {
                    keys[off + c] = 2.8 * pattern[c];
                }
                vals[off + c] = target_val[c];
            }
        }

        let query: Vec<f32> = pattern.iter().map(|&p| 3.2 * p).collect();
        needles.push(NeedleCase {
            position: pos,
            query,
            target_value: target_val,
        });
    }

    (keys, vals, needles)
}

fn fp16_reference_gqa(
    keys: &[f32],
    vals: &[f32],
    seq_len: usize,
    num_kv_heads: usize,
    queries: &[f32],
    num_q_heads: usize,
    head_size: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_size as f32).sqrt();
    let stride = num_kv_heads * head_size;
    let q_per_kv = num_q_heads / num_kv_heads;
    let mut out = vec![0.0f32; num_q_heads * head_size];

    for (q_idx, (q_slice, out_slice)) in queries
        .chunks_exact(head_size)
        .zip(out.chunks_exact_mut(head_size))
        .enumerate()
    {
        let kv_h = q_idx / q_per_kv;
        let mut max_s = f32::NEG_INFINITY;
        let mut logits = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let k_off = t * stride + kv_h * head_size;
            let mut dot = 0.0f32;
            for c in 0..head_size {
                let k_fp16 = f16::from_f32(keys[k_off + c]).to_f32();
                dot += q_slice[c] * k_fp16;
            }
            let s = dot * scale;
            max_s = max_s.max(s);
            logits.push(s);
        }
        let mut sum_w = 0.0f32;
        for s in &mut logits {
            *s = (*s - max_s).exp();
            sum_w += *s;
        }
        let inv = 1.0 / sum_w;
        for (t, &w) in logits.iter().enumerate() {
            let prob = w * inv;
            let v_off = t * stride + kv_h * head_size;
            for c in 0..head_size {
                let v_fp16 = f16::from_f32(vals[v_off + c]).to_f32();
                out_slice[c] += prob * v_fp16;
            }
        }
    }
    out
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let xf = f64::from(x);
        let yf = f64::from(y);
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 {
        1.0
    } else {
        (dot / denom) as f32
    }
}

fn mean_squared_error(a: &[f32], b: &[f32]) -> f32 {
    let sum_sq: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let diff = f64::from(x) - f64::from(y);
            diff * diff
        })
        .sum();
    (sum_sq / a.len() as f64) as f32
}

#[test]
fn kivi_bit_packing_and_validation_guards() -> Result<(), KiviError> {
    let (word, s_back, z_back) = pack_affine_params_f16(0.375, -14.5);
    let (s_unpacked, z_unpacked) = unpack_affine_params_f16(word);
    assert_eq!(s_back, s_unpacked);
    assert_eq!(z_back, z_unpacked);
    assert!((s_unpacked - 0.375).abs() < 1e-4);
    assert!((z_unpacked - (-14.5)).abs() < 1e-3);

    // Non-finite values must be sanitized without poisoning scales or outputs.
    let mut dirty = vec![0.25f32; 64];
    dirty[3] = f32::NAN;
    dirty[19] = f32::INFINITY;
    dirty[41] = f32::NEG_INFINITY;
    let mut packed = vec![0u32; KiviBits::Int4.packed_words(64)];
    let (scale, zero) = quantize_affine_f32(KiviBits::Int4, &dirty, &mut packed);
    assert!(scale.is_finite() && zero.is_finite());
    let mut rec = vec![0.0f32; 64];
    dequantize_affine_f32(KiviBits::Int4, &packed, scale, zero, &mut rec);
    assert!(rec.iter().all(|v| v.is_finite()));

    assert!(matches!(
        KiviKvCache::new(2, 60, KiviConfig::int4_default()),
        Err(KiviError::InvalidHeadSize { head_size: 60 })
    ));
    assert!(matches!(
        KiviKvCache::new(2, 128, KiviConfig::new(KiviBits::Int4, 24, 64)),
        Err(KiviError::InvalidGroupSize { group_size: 24 })
    ));
    let bad_v_cfg = KiviConfig {
        value_group_size: Some(48),
        ..KiviConfig::int4_default()
    };
    assert!(matches!(
        KiviKvCache::new(2, 128, bad_v_cfg),
        Err(KiviError::InvalidValueGroupSize {
            value_group_size: 48,
            head_size: 128,
        })
    ));

    let cache = KiviKvCache::new(2, 64, KiviConfig::int4_default())?;
    let mut out = vec![0.0f32; 3 * 64];
    let queries = vec![0.0f32; 3 * 64];
    assert!(matches!(
        cache.fused_gqa_decode(&queries, 3, 0.125, &mut out),
        Err(KiviError::InvalidHeadTopology {
            num_q_heads: 3,
            num_kv_heads: 2,
        })
    ));
    Ok(())
}

#[test]
fn kivi_sliding_residual_window_flushes_atomically() -> Result<(), KiviError> {
    let mut rng = Lcg64::new(0x1020_3040_5060_7080);
    let num_kv_heads = 2;
    let head_size = 64;
    let stride = num_kv_heads * head_size;
    let cfg = KiviConfig {
        bits: KiviBits::Int2,
        key_group_size: 32,
        value_group_size: Some(32),
        residual_length: 64,
        residual_precision: ResidualPrecision::Fp32,
    };
    let mut cache = KiviKvCache::new(num_kv_heads, head_size, cfg)?;

    for step in 1..=165usize {
        let k_tok: Vec<f32> = (0..stride).map(|_| rng.next_f32() * 4.0).collect();
        let v_tok: Vec<f32> = (0..stride).map(|_| rng.next_f32() * 2.0).collect();
        let flushed = cache.append_token(&k_tok, &v_tok)?;

        assert_eq!(cache.seq_len(), step);
        assert!(cache.quantized_len().is_multiple_of(32));
        if step <= 64 {
            assert_eq!(flushed, 0);
            assert_eq!(cache.residual_len(), step);
        } else {
            assert!(
                (33..=64).contains(&cache.residual_len()),
                "step={step} residual_len={} outside 33..=64",
                cache.residual_len()
            );
            if (step - 65).is_multiple_of(32) {
                assert_eq!(flushed, 1);
            } else {
                assert_eq!(flushed, 0);
            }
        }
    }
    Ok(())
}

#[test]
fn kivi_beats_uniform_symmetric_on_outlier_long_context() -> Result<(), KiviError> {
    let num_kv_heads = 2;
    let num_q_heads = 8;
    let head_size = 128;
    let scale = 1.0 / (head_size as f32).sqrt();

    let mut q_rng = Lcg64::new(0x7777_8888_9999_aaaa);
    let mut eval_queries = vec![0.0f32; num_q_heads * head_size];
    for q_row in eval_queries.chunks_exact_mut(head_size) {
        for (c, slot) in q_row.iter_mut().enumerate() {
            let weight = if is_outlier_channel(c).is_some() {
                0.05
            } else {
                2.4
            };
            *slot = q_rng.next_f32() * weight;
        }
    }

    for &(ctx_label, seq_len) in &[("5K", 5_120usize), ("30K", 30_720usize)] {
        let (keys, vals, needles) =
            build_outlier_workload(seq_len, num_kv_heads, head_size, 0x4242 + seq_len as u64);
        let ref_out = fp16_reference_gqa(
            &keys,
            &vals,
            seq_len,
            num_kv_heads,
            &eval_queries,
            num_q_heads,
            head_size,
        );

        let mut sym_int4 = UniformSymmetricKvCache::new(KiviBits::Int4, num_kv_heads, head_size)?;
        sym_int4.append_batch(&keys, &vals, seq_len)?;
        let mut sym4_out = vec![0.0f32; num_q_heads * head_size];
        sym_int4.decode_gqa(&eval_queries, num_q_heads, scale, &mut sym4_out)?;

        let mut sym_int2 = UniformSymmetricKvCache::new(KiviBits::Int2, num_kv_heads, head_size)?;
        sym_int2.append_batch(&keys, &vals, seq_len)?;
        let mut sym2_out = vec![0.0f32; num_q_heads * head_size];
        sym_int2.decode_gqa(&eval_queries, num_q_heads, scale, &mut sym2_out)?;

        let mut kivi_int4 = KiviKvCache::new(num_kv_heads, head_size, KiviConfig::int4_default())?;
        let t0 = Instant::now();
        kivi_int4.append_batch(&keys, &vals, seq_len)?;
        let kivi4_prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut kivi4_out = vec![0.0f32; num_q_heads * head_size];
        let t_dec4 = Instant::now();
        let dec_iters = 16usize;
        for _ in 0..dec_iters {
            kivi_int4.fused_gqa_decode(&eval_queries, num_q_heads, scale, &mut kivi4_out)?;
        }
        let kivi4_decode_us = (t_dec4.elapsed().as_secs_f64() * 1e6) / dec_iters as f64;

        let kivi2_cfg = KiviConfig::new(KiviBits::Int2, 128, 32);
        let mut kivi_int2 = KiviKvCache::new(num_kv_heads, head_size, kivi2_cfg)?;
        let t1 = Instant::now();
        kivi_int2.append_batch(&keys, &vals, seq_len)?;
        let kivi2_prefill_ms = t1.elapsed().as_secs_f64() * 1e3;
        let mut kivi2_out = vec![0.0f32; num_q_heads * head_size];
        let t_dec2 = Instant::now();
        for _ in 0..dec_iters {
            kivi_int2.fused_gqa_decode(&eval_queries, num_q_heads, scale, &mut kivi2_out)?;
        }
        let kivi2_decode_us = (t_dec2.elapsed().as_secs_f64() * 1e6) / dec_iters as f64;

        let sym4_cos = cosine_similarity(&ref_out, &sym4_out);
        let sym2_cos = cosine_similarity(&ref_out, &sym2_out);
        let kivi4_cos = cosine_similarity(&ref_out, &kivi4_out);
        let kivi2_cos = cosine_similarity(&ref_out, &kivi2_out);

        let sym4_mse = mean_squared_error(&ref_out, &sym4_out);
        let sym2_mse = mean_squared_error(&ref_out, &sym2_out);
        let kivi4_mse = mean_squared_error(&ref_out, &kivi4_out);
        let kivi2_mse = mean_squared_error(&ref_out, &kivi2_out);

        let fp4 = kivi_int4.memory_footprint();
        let fp2 = kivi_int2.memory_footprint();

        let mut kivi4_needle_hits = 0usize;
        let mut kivi2_needle_hits = 0usize;
        let mut needle_out = vec![0.0f32; head_size];
        for needle in &needles {
            kivi_int4.fused_decode_head(0, &needle.query, scale, &mut needle_out)?;
            if cosine_similarity(&needle_out, &needle.target_value) > 0.99 {
                kivi4_needle_hits += 1;
            }
            kivi_int2.fused_decode_head(0, &needle.query, scale, &mut needle_out)?;
            if cosine_similarity(&needle_out, &needle.target_value) > 0.92 {
                kivi2_needle_hits += 1;
            }
            let _ = needle.position;
        }

        let r4 = fp4.compression_ratio_vs_fp16();
        let r2 = fp2.compression_ratio_vs_fp16();
        let n_cnt = needles.len();
        println!(
            "[scorecard ctx={ctx_label}] \
             KIVI-INT4: ratio={r4:.2}x cos={kivi4_cos:.6} mse={kivi4_mse:.2e} \
             needles={kivi4_needle_hits}/{n_cnt} prefill_ms={kivi4_prefill_ms:.2} \
             decode_us={kivi4_decode_us:.1} | \
             KIVI-INT2: ratio={r2:.2}x cos={kivi2_cos:.6} mse={kivi2_mse:.2e} \
             needles={kivi2_needle_hits}/{n_cnt} prefill_ms={kivi2_prefill_ms:.2} \
             decode_us={kivi2_decode_us:.1} | \
             Sym4 cos={sym4_cos:.4} mse={sym4_mse:.2e} | \
             Sym2 cos={sym2_cos:.4} mse={sym2_mse:.2e}"
        );

        assert!(fp4.compression_ratio_vs_fp16() >= 3.50);
        assert!(fp2.compression_ratio_vs_fp16() >= 7.00);
        assert!(kivi4_cos > 0.995, "kivi4_cos={kivi4_cos}");
        assert!(kivi2_cos > 0.935, "kivi2_cos={kivi2_cos}");
        assert!(kivi4_cos > sym4_cos, "KIVI INT4 must beat Symmetric INT4");
        assert!(kivi2_cos > sym2_cos, "KIVI INT2 must beat Symmetric INT2");
        assert!(kivi4_mse < sym4_mse);
        assert!(kivi2_mse < sym2_mse);
        assert_eq!(kivi4_needle_hits, needles.len());
        assert_eq!(kivi2_needle_hits, needles.len());
    }
    Ok(())
}

#[test]
fn kivi_128k_footprint_and_hybrid_4b_9b_27b_matrix() -> Result<(), KiviError> {
    let seq_len = 131_072usize;
    let head_size = 128;
    let (keys, vals, needles) = build_outlier_workload(seq_len, 1, head_size, 0x128_000);

    let cfg_int4 = KiviConfig::new(KiviBits::Int4, 64, 64);
    let cfg_int2 = KiviConfig::new(KiviBits::Int2, 128, 32);

    let mut cache_int4 = KiviKvCache::new(1, head_size, cfg_int4)?;
    cache_int4.append_batch(&keys, &vals, seq_len)?;
    let mem4 = cache_int4.memory_footprint();

    let mut cache_int2 = KiviKvCache::new(1, head_size, cfg_int2)?;
    cache_int2.append_batch(&keys, &vals, seq_len)?;
    let mem2 = cache_int2.memory_footprint();

    assert!(
        mem4.compression_ratio_vs_fp16() >= 3.65,
        "128K INT4 ratio={}",
        mem4.compression_ratio_vs_fp16()
    );
    assert!(
        mem2.compression_ratio_vs_fp16() >= 7.10,
        "128K INT2 ratio={}",
        mem2.compression_ratio_vs_fp16()
    );

    let scale = 1.0 / (head_size as f32).sqrt();
    let mut needle_out = vec![0.0f32; head_size];
    for needle in &needles {
        cache_int2.fused_decode_head(0, &needle.query, scale, &mut needle_out)?;
        let cos = cosine_similarity(&needle_out, &needle.target_value);
        assert!(cos > 0.92, "128K INT2 needle cos={cos}");
    }

    // Shared 4B / 9B / 27B hybrid attention + unquantized GDN state matrix.
    let models = [
        HybridModelSpec {
            label: "4B-Hybrid",
            num_attention_layers: 9,
            num_gdn_layers: 27,
            num_kv_heads: 2,
            head_size: 128,
            gdn_heads: 16,
            gdn_state_elems_per_head: 128 * 128,
        },
        HybridModelSpec {
            label: "9B-Hybrid",
            num_attention_layers: 10,
            num_gdn_layers: 30,
            num_kv_heads: 4,
            head_size: 128,
            gdn_heads: 24,
            gdn_state_elems_per_head: 128 * 128,
        },
        HybridModelSpec {
            label: "27B-Hybrid",
            num_attention_layers: 16,
            num_gdn_layers: 48,
            num_kv_heads: 4,
            head_size: 256,
            gdn_heads: 32,
            gdn_state_elems_per_head: 128 * 256,
        },
    ];

    for spec in &models {
        let p4 = project_hybrid_model_memory(spec, seq_len, cfg_int4)?;
        let p2 = project_hybrid_model_memory(spec, seq_len, cfg_int2)?;
        assert_eq!(p4.unquantized_gdn_bytes, p2.unquantized_gdn_bytes);
        assert!(p4.unquantized_gdn_bytes > 0);
        assert!(p4.attention_kv_ratio >= 3.65);
        assert!(p2.attention_kv_ratio >= 7.10);
        println!(
            "[hybrid-128K model={}] baseline_mib={:.1} int4_mib={:.1} ({:.2}x) \
             int2_mib={:.1} ({:.2}x) unquant_gdn_mib={:.1}",
            spec.label,
            p4.baseline_total_bytes as f64 / (1024.0 * 1024.0),
            p4.compressed_total_bytes as f64 / (1024.0 * 1024.0),
            p4.total_state_ratio,
            p2.compressed_total_bytes as f64 / (1024.0 * 1024.0),
            p2.total_state_ratio,
            p4.unquantized_gdn_bytes as f64 / (1024.0 * 1024.0),
        );
    }
    Ok(())
}

#[test]
fn recurrent_gdn_state_negative_control_proves_exemption_necessity() {
    let cfg = KiviConfig::int2_default();
    assert_eq!(
        resolve_layer_plan(LayerStateKind::RecurrentGdn, cfg),
        LayerCompressionPlan::UnquantizedRecurrentGdn
    );

    // Negative control: 2-bit per-row state quantization inside a recurrent DeltaNet update
    // compounds rounding error multiplicatively across steps.
    let d = 64usize;
    let mut rng = Lcg64::new(0x2026_1118);
    let mut state_exact = vec![0.0f32; d * d];
    let mut state_quant = vec![0.0f32; d * d];
    let mut packed = vec![0u32; KiviBits::Int2.packed_words(d)];

    for _ in 0..384 {
        let mut k: Vec<f32> = (0..d).map(|_| rng.next_f32() * 0.2).collect();
        let norm = k.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-6;
        for x in &mut k {
            *x /= norm;
        }
        let v: Vec<f32> = (0..d).map(|_| rng.next_f32() * 0.3).collect();
        let decay = 0.985f32;
        let beta = 0.25f32;

        for r in 0..d {
            let row_ex = &mut state_exact[r * d..(r + 1) * d];
            let pred_ex: f32 = row_ex.iter().zip(&k).map(|(&s, &ki)| s * ki).sum();
            let delta_ex = beta * (v[r] - pred_ex);
            for (s, &ki) in row_ex.iter_mut().zip(&k) {
                *s = decay * (*s) + delta_ex * ki;
            }

            let row_q = &mut state_quant[r * d..(r + 1) * d];
            let pred_q: f32 = row_q.iter().zip(&k).map(|(&s, &ki)| s * ki).sum();
            let delta_q = beta * (v[r] - pred_q);
            for (s, &ki) in row_q.iter_mut().zip(&k) {
                *s = decay * (*s) + delta_q * ki;
            }
            let (sc, zp) = quantize_affine_f32(KiviBits::Int2, row_q, &mut packed);
            dequantize_affine_f32(KiviBits::Int2, &packed, sc, zp, row_q);
        }
    }

    let err_sq: f64 = state_exact
        .iter()
        .zip(&state_quant)
        .map(|(&a, &b)| {
            let diff = f64::from(a) - f64::from(b);
            diff * diff
        })
        .sum();
    let ref_sq: f64 = state_exact.iter().map(|&a| f64::from(a) * f64::from(a)).sum();
    let rel_l2 = (err_sq / ref_sq).sqrt();
    println!("[gdn-negative-control] recurrent_state_rel_l2_drift={rel_l2:.4}");
    assert!(
        rel_l2 > 0.20,
        "expected substantial drift on quantized recurrent state, got {rel_l2}"
    );
}

#[test]
fn twenty_turn_coding_agent_prefix_reuse_loop() -> Result<(), KiviError> {
    let num_kv_heads = 2;
    let num_q_heads = 8;
    let head_size = 128;
    let stride = num_kv_heads * head_size;
    let scale = 1.0 / (head_size as f32).sqrt();

    let mut cache_kivi4 = KiviKvCache::new(num_kv_heads, head_size, KiviConfig::int4_default())?;
    let mut cache_kivi2 = KiviKvCache::new(num_kv_heads, head_size, KiviConfig::int2_default())?;

    let mut all_keys = Vec::new();
    let mut all_vals = Vec::new();
    let mut rng = Lcg64::new(0xc0de_a9e7_2026);

    let loop_start = Instant::now();
    for turn in 0..20usize {
        // Each turn appends a tool/file observation chunk (180 tokens) + 32 decoded tokens.
        let prefill_tokens = 180usize;
        let decode_tokens = 32usize;

        let mut turn_k = vec![0.0f32; prefill_tokens * stride];
        let mut turn_v = vec![0.0f32; prefill_tokens * stride];
        for t in 0..prefill_tokens {
            for h in 0..num_kv_heads {
                let off = (t * num_kv_heads + h) * head_size;
                for c in 0..head_size {
                    turn_k[off + c] = match is_outlier_channel(c) {
                        None => rng.next_f32(),
                        Some(bias) => bias + rng.next_f32(),
                    };
                    turn_v[off + c] = rng.next_f32() * 1.2;
                }
            }
        }
        cache_kivi4.append_batch(&turn_k, &turn_v, prefill_tokens)?;
        cache_kivi2.append_batch(&turn_k, &turn_v, prefill_tokens)?;
        all_keys.extend_from_slice(&turn_k);
        all_vals.extend_from_slice(&turn_v);

        let mut q_buf = vec![0.0f32; num_q_heads * head_size];
        let mut out4 = vec![0.0f32; num_q_heads * head_size];
        let mut out2 = vec![0.0f32; num_q_heads * head_size];

        for _ in 0..decode_tokens {
            let mut step_k = vec![0.0f32; stride];
            let mut step_v = vec![0.0f32; stride];
            for h in 0..num_kv_heads {
                let off = h * head_size;
                for c in 0..head_size {
                    step_k[off + c] = match is_outlier_channel(c) {
                        None => rng.next_f32(),
                        Some(bias) => bias + rng.next_f32(),
                    };
                    step_v[off + c] = rng.next_f32() * 1.2;
                }
            }
            cache_kivi4.append_token(&step_k, &step_v)?;
            cache_kivi2.append_token(&step_k, &step_v)?;
            all_keys.extend_from_slice(&step_k);
            all_vals.extend_from_slice(&step_v);

            for (c_idx, slot) in q_buf.iter_mut().enumerate() {
                let ch = c_idx % head_size;
                let w = if is_outlier_channel(ch).is_some() {
                    0.05
                } else {
                    2.4
                };
                *slot = rng.next_f32() * w;
            }
            cache_kivi4.fused_gqa_decode(&q_buf, num_q_heads, scale, &mut out4)?;
            cache_kivi2.fused_gqa_decode(&q_buf, num_q_heads, scale, &mut out2)?;

            if turn == 19 {
                let ref_out = fp16_reference_gqa(
                    &all_keys,
                    &all_vals,
                    cache_kivi4.seq_len(),
                    num_kv_heads,
                    &q_buf,
                    num_q_heads,
                    head_size,
                );
                assert!(cosine_similarity(&ref_out, &out4) > 0.995);
                assert!(cosine_similarity(&ref_out, &out2) > 0.915);
            }
        }
    }
    let wall_ms = loop_start.elapsed().as_secs_f64() * 1e3;
    assert_eq!(cache_kivi4.seq_len(), 20 * 212);
    println!(
        "[20-turn-agent-loop] final_tokens={} wall_ms={wall_ms:.2} \
         int4_ratio={:.2}x int2_ratio={:.2}x",
        cache_kivi4.seq_len(),
        cache_kivi4.memory_footprint().compression_ratio_vs_fp16(),
        cache_kivi2.memory_footprint().compression_ratio_vs_fp16(),
    );
    Ok(())
}

#[test]
fn multithreaded_concurrent_gqa_decode_is_deterministic() -> Result<(), KiviError> {
    let num_kv_heads = 2;
    let num_q_heads = 8;
    let head_size = 128;
    let seq_len = 1_050usize;
    let scale = 1.0 / (head_size as f32).sqrt();

    let (keys, vals, _) = build_outlier_workload(seq_len, num_kv_heads, head_size, 0x9988_7766);
    let mut base_cache = KiviKvCache::new(num_kv_heads, head_size, KiviConfig::int2_default())?;
    base_cache.append_batch(&keys, &vals, seq_len)?;
    let shared_cache = Arc::new(base_cache);

    let mut q_rng = Lcg64::new(0x1357_2468);
    let q_elems = num_q_heads * head_size;
    let q_vec: Vec<f32> = (0..q_elems).map(|_| q_rng.next_f32()).collect();
    let queries = Arc::new(q_vec);
    let mut expected = vec![0.0f32; num_q_heads * head_size];
    shared_cache.fused_gqa_decode(&queries, num_q_heads, scale, &mut expected)?;

    let num_threads = 16usize;
    let barrier = Arc::new(Barrier::new(num_threads));

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(num_threads);
        for tid in 0..num_threads {
            let cache_ref = Arc::clone(&shared_cache);
            let q_ref = Arc::clone(&queries);
            let bar = Arc::clone(&barrier);
            let exp_clone = expected.clone();
            let keys_ref = &keys;
            let vals_ref = &vals;

            handles.push(scope.spawn(move || -> Result<(), KiviError> {
                bar.wait();
                if tid < 12 {
                    let mut local_out = vec![0.0f32; num_q_heads * head_size];
                    for _ in 0..12 {
                        cache_ref.fused_gqa_decode(&q_ref, num_q_heads, scale, &mut local_out)?;
                        assert_eq!(local_out, exp_clone);
                    }
                } else {
                    let mut own =
                        KiviKvCache::new(num_kv_heads, head_size, KiviConfig::int2_default())?;
                    own.append_batch(keys_ref, vals_ref, seq_len)?;
                    let mut own_out = vec![0.0f32; num_q_heads * head_size];
                    own.fused_gqa_decode(&q_ref, num_q_heads, scale, &mut own_out)?;
                    assert_eq!(own_out, exp_clone);
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle.join().expect("worker thread panicked")?;
        }
        Ok::<(), KiviError>(())
    })?;

    Ok(())
}
