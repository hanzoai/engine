//! Reproducible CPU benchmark for `hanzo_engine::resonance`.
//!
//! Measures pairwise throughput/latency, shortlist reranking latency (p50/p95/p99),
//! 10K full-scan cost, memory footprint, and 500-query contrastive polarity metrics
//! vs dense cosine and structured polarity filtering.
//!
//! Run with:
//!   cargo run --release -p hanzo-engine --example resonance_bench
use hanzo_engine::resonance::{rerank_candidates, Candidate, PhaseOperator, WavePattern};
use std::hint::black_box;
use std::time::Instant;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn unit_vec(&mut self, dim: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(dim);
        let mut sum = 0.0f64;
        for _ in 0..dim {
            let x = (self.next_f64() * 2.0 - 1.0) as f32;
            sum += f64::from(x) * f64::from(x);
            v.push(x);
        }
        let inv = (1.0 / sum.sqrt()) as f32;
        for x in &mut v {
            *x *= inv;
        }
        v
    }
}

fn cosine_baseline(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let fx = f64::from(x);
        let fy = f64::from(y);
        dot += fx * fy;
        na += fx * fx;
        nb += fy * fy;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na * nb).sqrt()).clamp(-1.0, 1.0)
}

fn blend_unit_vec(base: &[f32], noise: &[f32], target_cos: f64) -> Vec<f32> {
    let proj: f64 = base
        .iter()
        .zip(noise)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum();
    let mut orth = Vec::with_capacity(base.len());
    let mut norm2 = 0.0f64;
    for (&a, &b) in base.iter().zip(noise) {
        let o = f64::from(b) - proj * f64::from(a);
        norm2 += o * o;
        orth.push(o);
    }
    let inv_orth = 1.0 / norm2.sqrt();
    let sin_t = (1.0 - target_cos * target_cos).max(0.0).sqrt();
    let mut out = Vec::with_capacity(base.len());
    let mut out_norm2 = 0.0f64;
    for (&a, &o) in base.iter().zip(&orth) {
        let val = (target_cos * f64::from(a) + sin_t * o * inv_orth) as f32;
        out_norm2 += f64::from(val) * f64::from(val);
        out.push(val);
    }
    let inv_out = (1.0 / out_norm2.sqrt()) as f32;
    for x in &mut out {
        *x *= inv_out;
    }
    out
}

fn percentiles(samples_us: &mut [f64]) -> (f64, f64, f64) {
    samples_us.sort_by(f64::total_cmp);
    let n = samples_us.len();
    (
        samples_us[n / 2],
        samples_us[(n * 95) / 100],
        samples_us[(n * 99) / 100],
    )
}

fn bench_pairwise(dim: usize, iters: usize) {
    let mut rng = SplitMix64(1000 + dim as u64);
    let a_raw = rng.unit_vec(dim);
    let b_raw = rng.unit_vec(dim);
    let a_wave = WavePattern::from_embedding(&a_raw).unwrap();
    let b_wave = WavePattern::from_embedding(&b_raw).unwrap();

    // Warmup
    for _ in 0..10_000 {
        black_box(cosine_baseline(&a_raw, &b_raw));
        black_box(a_wave.resonance(&b_wave).unwrap());
    }

    let t0 = Instant::now();
    let mut acc = 0.0f64;
    for _ in 0..iters {
        acc += cosine_baseline(black_box(&a_raw), black_box(&b_raw));
    }
    black_box(acc);
    let cos_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    let t1 = Instant::now();
    let mut acc2 = 0.0f64;
    for _ in 0..iters {
        acc2 += black_box(&a_wave).resonance(black_box(&b_wave)).unwrap();
    }
    black_box(acc2);
    let res_ns = t1.elapsed().as_nanos() as f64 / iters as f64;

    let real_bytes = dim * std::mem::size_of::<f32>();
    let wave_bytes = std::mem::size_of::<WavePattern>() + 2 * dim * std::mem::size_of::<f32>();
    println!(
        "pairwise dim={dim:4} | cosine={cos_ns:7.1} ns ({:6.2}M/s) | resonance={res_ns:7.1} ns ({:6.2}M/s) | ratio={:.2}x | mem_real={real_bytes} B mem_wave={wave_bytes} B",
        1e3 / cos_ns,
        1e3 / res_ns,
        res_ns / cos_ns
    );
}

fn bench_rerank(dim: usize, n: usize, rounds: usize) {
    let mut rng = SplitMix64(42 + (dim as u64) * 1000 + n as u64);
    let q_raw = rng.unit_vec(dim);
    let q_wave = WavePattern::from_embedding(&q_raw).unwrap();
    let mut ids = Vec::with_capacity(n);
    let mut patterns = Vec::with_capacity(n);
    let mut cosines = Vec::with_capacity(n);
    for i in 0..n {
        let v = rng.unit_vec(dim);
        let c = cosine_baseline(&q_raw, &v);
        let mut w = WavePattern::from_embedding(&v).unwrap();
        if i % 5 == 0 {
            w = w.apply(PhaseOperator::Negate).unwrap();
        }
        ids.push(format!("doc-{i:06}"));
        patterns.push(w);
        cosines.push(c);
    }
    let cands: Vec<Candidate<'_>> = (0..n)
        .map(|i| Candidate {
            id: ids[i].as_str(),
            pattern: &patterns[i],
            dense_cosine: cosines[i],
        })
        .collect();

    for _ in 0..5 {
        black_box(rerank_candidates(&q_wave, cands.iter().copied(), 0.75, 10).unwrap());
    }
    let mut samples_us = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let t0 = Instant::now();
        let hits = rerank_candidates(&q_wave, cands.iter().copied(), 0.75, 10).unwrap();
        black_box(hits);
        samples_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    let (p50, p95, p99) = percentiles(&mut samples_us);
    let total_mb =
        (n * (std::mem::size_of::<WavePattern>() + 2 * dim * 4)) as f64 / (1024.0 * 1024.0);
    println!(
        "rerank   dim={dim:4} N={n:5} | p50={p50:8.2} us | p95={p95:8.2} us | p99={p99:8.2} us | qps={:9.1} | wave_mem={total_mb:.2} MiB",
        1e6 / p50
    );
}

fn eval_contrastive() {
    const NUM_QUERIES: usize = 500;
    const SHORTLIST: usize = 50;
    const DIM: usize = 384;
    let mut rng = SplitMix64(20261125);

    #[derive(Default)]
    struct Metrics {
        r1: f64,
        mrr: f64,
        ndcg10: f64,
        contra_fp: f64,
        total_us: f64,
    }

    let score = |m: &mut Metrics, ids: &[&str]| {
        if ids.first() == Some(&"target") {
            m.r1 += 1.0;
        }
        if ids.first() == Some(&"negated") {
            m.contra_fp += 1.0;
        }
        for (i, &id) in ids.iter().enumerate() {
            if id == "target" {
                m.mrr += 1.0 / (i + 1) as f64;
                if i < 10 {
                    m.ndcg10 += 1.0 / ((i + 2) as f64).log2();
                }
                break;
            }
        }
    };

    let mut m_cos = Metrics::default();
    let mut m_nophase = Metrics::default();
    let mut m_res50 = Metrics::default();
    let mut m_res75 = Metrics::default();
    let mut m_filter = Metrics::default();

    for _ in 0..NUM_QUERIES {
        let q_vec = rng.unit_vec(DIM);
        let q_wave = WavePattern::from_embedding(&q_vec).unwrap();
        let target_cos = 0.86 + 0.05 * rng.next_f64();
        let neg_cos = 0.93 + 0.05 * rng.next_f64();
        let n1 = rng.unit_vec(DIM);
        let n2 = rng.unit_vec(DIM);
        let target_vec = blend_unit_vec(&q_vec, &n1, target_cos);
        let neg_vec = blend_unit_vec(&q_vec, &n2, neg_cos);

        let mut names = vec!["target".to_string(), "negated".to_string()];
        let mut vecs = vec![target_vec, neg_vec];
        let mut pols = vec![1i8, -1i8];
        for d in 2..SHORTLIST {
            let c = 0.25 + 0.55 * rng.next_f64();
            let noise = rng.unit_vec(DIM);
            names.push(format!("dist-{d:02}"));
            vecs.push(blend_unit_vec(&q_vec, &noise, c));
            pols.push(0);
        }
        let cosines: Vec<f64> = vecs.iter().map(|v| cosine_baseline(&q_vec, v)).collect();
        let waves_nophase: Vec<WavePattern> = vecs
            .iter()
            .map(|v| WavePattern::from_embedding(v).unwrap())
            .collect();
        let waves_phased: Vec<WavePattern> = waves_nophase
            .iter()
            .enumerate()
            .map(|(i, w)| {
                if pols[i] == -1 {
                    w.apply(PhaseOperator::Negate).unwrap()
                } else {
                    w.clone()
                }
            })
            .collect();

        // A: Dense cosine baseline
        let t0 = Instant::now();
        let mut idx: Vec<usize> = (0..SHORTLIST).collect();
        idx.sort_by(|&a, &b| {
            cosines[b]
                .total_cmp(&cosines[a])
                .then_with(|| names[a].cmp(&names[b]))
        });
        m_cos.total_us += t0.elapsed().as_secs_f64() * 1e6;
        let ids_cos: Vec<&str> = idx.iter().take(10).map(|&i| names[i].as_str()).collect();
        score(&mut m_cos, &ids_cos);

        // B: Resonance without phase
        let cands_np: Vec<Candidate<'_>> = (0..SHORTLIST)
            .map(|i| Candidate {
                id: names[i].as_str(),
                pattern: &waves_nophase[i],
                dense_cosine: cosines[i],
            })
            .collect();
        let t0 = Instant::now();
        let h_np = rerank_candidates(&q_wave, cands_np, 0.75, 10).unwrap();
        m_nophase.total_us += t0.elapsed().as_secs_f64() * 1e6;
        let ids_np: Vec<&str> = h_np.iter().map(|h| h.id).collect();
        assert_eq!(
            ids_np[0], ids_cos[0],
            "no-phase resonance diverged from cosine"
        );
        score(&mut m_nophase, &ids_np);

        // C1/C2: Resonance with grounded phase
        let cands_p: Vec<Candidate<'_>> = (0..SHORTLIST)
            .map(|i| Candidate {
                id: names[i].as_str(),
                pattern: &waves_phased[i],
                dense_cosine: cosines[i],
            })
            .collect();
        let t0 = Instant::now();
        let h_50 = rerank_candidates(&q_wave, cands_p.iter().copied(), 0.50, 10).unwrap();
        m_res50.total_us += t0.elapsed().as_secs_f64() * 1e6;
        let ids_50: Vec<&str> = h_50.iter().map(|h| h.id).collect();
        score(&mut m_res50, &ids_50);

        let t0 = Instant::now();
        let h_75 = rerank_candidates(&q_wave, cands_p, 0.75, 10).unwrap();
        m_res75.total_us += t0.elapsed().as_secs_f64() * 1e6;
        let ids_75: Vec<&str> = h_75.iter().map(|h| h.id).collect();
        score(&mut m_res75, &ids_75);

        // D: Structured polarity filter
        let t0 = Instant::now();
        let mut fidx: Vec<usize> = (0..SHORTLIST).filter(|&i| pols[i] != -1).collect();
        fidx.sort_by(|&a, &b| {
            cosines[b]
                .total_cmp(&cosines[a])
                .then_with(|| names[a].cmp(&names[b]))
        });
        m_filter.total_us += t0.elapsed().as_secs_f64() * 1e6;
        let ids_f: Vec<&str> = fidx.iter().take(10).map(|&i| names[i].as_str()).collect();
        score(&mut m_filter, &ids_f);
    }

    let n = NUM_QUERIES as f64;
    for (name, m) in [
        ("A. Dense cosine baseline", &m_cos),
        ("B. Resonance (no phase, w=0.75)", &m_nophase),
        ("C1. Resonance (grounded phase, 0.50)", &m_res50),
        ("C2. Resonance (grounded phase, 0.75)", &m_res75),
        ("D. Structured polarity filter", &m_filter),
    ] {
        println!(
            "{name:<36} | R@1={:.4} | MRR={:.4} | nDCG@10={:.4} | ContraFP={:.4} | avg={:.2} us/query",
            m.r1 / n,
            m.mrr / n,
            m.ndcg10 / n,
            m.contra_fp / n,
            m.total_us / n
        );
    }
}

fn main() {
    println!("=== hanzo_engine::resonance pairwise throughput & footprint ===");
    for dim in [384usize, 768, 1024] {
        bench_pairwise(dim, 200_000);
    }
    println!("\n=== hanzo_engine::resonance shortlist & scan latency ===");
    for &(dim, n, rounds) in &[
        (384, 50, 1000),
        (384, 200, 500),
        (768, 50, 1000),
        (768, 200, 500),
        (1024, 50, 1000),
        (1024, 200, 500),
        (1024, 10_000, 40),
    ] {
        bench_rerank(dim, n, rounds);
    }
    println!("\n=== 500-query contrastive polarity benchmark (dim=384, N=50) ===");
    eval_contrastive();
}
