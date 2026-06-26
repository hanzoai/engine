// Numeric unit test for the FUSED ROCm MoE router (hanzo_ml::quantized::moe_route ->
// RocmStorage::moe_route -> quant.hip `moe_route`). The fused kernel replaces the per-token
// softmax->sort->narrow->sum->div routing chain with one launch; this gates it bit-faithfully
// against a CPU reference of the SAME algorithm (full-softmax weight, topk by descending logit,
// optional norm_topk_prob renorm). A wrong router selects the wrong experts -> garbage output.
// Lives in hanzo-cli so it links the HIP runtime via hanzo-cli's build.rs.
#![cfg(feature = "rocm")]

use hanzo_ml::{Device, Tensor};

fn val(i: usize, salt: usize) -> f32 {
    let x = ((i.wrapping_mul(2654435761).wrapping_add(salt.wrapping_mul(40503)) >> 8) & 0xffff)
        as f32
        / 65535.0; // [0,1)
    (x - 0.5) * 8.0 // [-4,4]
}

// CPU reference: matches the moe_route kernel exactly.
fn route_ref(logits: &[f32], n_experts: usize, topk: usize, norm: bool) -> (Vec<u32>, Vec<f32>) {
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let se: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
    let z: f32 = se.iter().sum();
    let mut used = vec![false; n_experts];
    let mut ids = Vec::with_capacity(topk);
    let mut w = Vec::with_capacity(topk);
    let mut wsum = 0f32;
    for _ in 0..topk {
        let mut best = usize::MAX;
        let mut bestv = f32::NEG_INFINITY;
        for e in 0..n_experts {
            if !used[e] && logits[e] > bestv {
                bestv = logits[e];
                best = e;
            }
        }
        used[best] = true;
        let p = se[best] / z;
        ids.push(best as u32);
        w.push(p);
        wsum += p;
    }
    if norm {
        for x in w.iter_mut() {
            *x /= wsum;
        }
    }
    (ids, w)
}

fn check(dev: &Device, log: &mut String, ntok: usize, n_experts: usize, topk: usize, norm: bool) {
    let salt = n_experts * 31 + topk * 7 + usize::from(norm);
    let logits: Vec<f32> = (0..ntok * n_experts)
        .map(|i| val(i, salt) + (i % n_experts) as f32 * 1e-3)
        .collect();
    let lt = Tensor::from_vec(logits.clone(), (ntok, n_experts), dev).expect("logits");

    let (ids_t, w_t) = hanzo_ml::quantized::moe_route(&lt, topk, norm).expect("moe_route");
    assert_eq!(ids_t.dims(), &[ntok, topk], "ids shape");
    assert_eq!(w_t.dims(), &[ntok, topk], "w shape");
    let ids = ids_t.flatten_all().unwrap().to_vec1::<u32>().expect("ids vec");
    let w = w_t.flatten_all().unwrap().to_vec1::<f32>().expect("w vec");

    let tol = 1e-5f32;
    let mut nbad = 0usize;
    let mut max_w_err = 0f32;
    for t in 0..ntok {
        let (rid, rw) = route_ref(&logits[t * n_experts..(t + 1) * n_experts], n_experts, topk, norm);
        for r in 0..topk {
            let k = t * topk + r;
            if ids[k] != rid[r] {
                nbad += 1;
            }
            let e = (w[k] - rw[r]).abs();
            max_w_err = max_w_err.max(e);
            if e > tol {
                nbad += 1;
            }
        }
    }
    log.push_str(&format!(
        "ntok={ntok} n_experts={n_experts} topk={topk} norm={norm} nbad={nbad}/{} max_w_err={max_w_err:.8}\n",
        ntok * topk
    ));
    assert_eq!(nbad, 0, "moe_route ntok={ntok} E={n_experts} k={topk} norm={norm}: nbad={nbad} max_w_err={max_w_err}");
}

#[test]
fn moe_route_numeric() {
    let mut log = String::new();
    let dev = Device::new_rocm(0).expect("rocm device");

    // Qwen3-30B-A3B: 128 experts, topk=8, norm_topk_prob=true. Decode (ntok=1) + prefill + tails.
    check(&dev, &mut log, 1, 128, 8, true);
    check(&dev, &mut log, 1, 128, 8, false);
    check(&dev, &mut log, 128, 128, 8, true);
    check(&dev, &mut log, 1024, 128, 8, true);
    check(&dev, &mut log, 7, 64, 6, true); // non-pow2 experts/topk
    check(&dev, &mut log, 33, 60, 4, false);
    check(&dev, &mut log, 1, 256, 8, true); // max experts

    eprintln!("{log}");
}
