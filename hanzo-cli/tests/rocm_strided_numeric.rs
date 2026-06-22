// Correctness gate for the inline-DimsStrides ROCm strided-op dispatch (ml commit 6a09dcaf:
// dims/strides passed by-value in the kernel param block instead of a per-op clone_htod). That
// change touches EVERY strided op (Map1/Map2/Affine/Powf/Elu/FastReduce/to_dtype/copy_strided/
// const_set/where_cond/index_select) + 8 HIP kernels, so a stride-offset/rank/dtype bug could
// silently corrupt non-contiguous tensors, non-last-axis reductions, broadcasting, casts, etc.
//
// Method: run the SAME op on CPU (the oracle) and on rocm[0] from identical inputs, pull both
// back to host, assert agreement. The non-contiguous cases (transpose/narrow/broadcast/permute)
// are the load-bearing ones -- they force the strided gather loop that reads dims/strides.
// Lives in hanzo-cli so it links the HIP runtime via hanzo-cli's build.rs.
#![cfg(feature = "rocm")]

use hanzo_ml::{DType, Device, Tensor, D};
use std::io::Write;

fn val(i: usize) -> f32 {
    let x = ((i.wrapping_mul(2654435761) >> 8) & 0xffff) as f32 / 65535.0;
    (x - 0.5) * 6.0
}

fn to_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .expect("to f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec1")
}

fn max_abs_err(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len(), "length mismatch cpu={} rocm={}", a.len(), b.len());
    let mut m = 0f32;
    let mut idx = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let e = (x - y).abs();
        if e > m {
            m = e;
            idx = i;
        }
    }
    (m, idx)
}

// Build the input on CPU and on rocm from identical f32 data + dtype, run `op` on both, compare.
fn cmp<F>(log: &mut String, name: &str, dims: &[usize], dtype: DType, tol: f32, dev: &Device, op: F)
where
    F: Fn(&Tensor) -> Tensor,
{
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(val).collect();

    let cpu_in = Tensor::from_vec(data.clone(), dims, &Device::Cpu)
        .expect("cpu in")
        .to_dtype(dtype)
        .expect("cpu dtype");
    let roc_in = Tensor::from_vec(data, dims, dev)
        .expect("rocm in")
        .to_dtype(dtype)
        .expect("rocm dtype");

    let cpu_out = op(&cpu_in);
    let roc_out = op(&roc_in);

    assert_eq!(
        cpu_out.dims(),
        roc_out.dims(),
        "{name}: shape divergence cpu={:?} rocm={:?}",
        cpu_out.dims(),
        roc_out.dims()
    );
    assert_eq!(
        cpu_out.dtype(),
        roc_out.dtype(),
        "{name}: dtype divergence cpu={:?} rocm={:?}",
        cpu_out.dtype(),
        roc_out.dtype()
    );

    let (err, idx) = max_abs_err(&to_f32(&cpu_out), &to_f32(&roc_out));
    log.push_str(&format!(
        "{name:<40} dims={dims:?} dtype={dtype:?} max_abs_err={err:.6} @ {idx}\n"
    ));
    assert!(
        err <= tol,
        "{name}: ROCm vs CPU mismatch dims={dims:?} dtype={dtype:?} max_abs_err={err} > tol {tol} @ idx {idx}"
    );
}

// f16/bf16 round-trip both sides, so tolerances are dtype-aware; f32/int compare exactly.
fn tol_for(dtype: DType) -> f32 {
    match dtype {
        DType::F32 | DType::F64 => 1e-4,
        DType::BF16 => 5e-2,
        DType::F16 => 1e-2,
        _ => 0.0, // integer / exact
    }
}

const FLOATS: [DType; 3] = [DType::F32, DType::F16, DType::BF16];

#[test]
fn rocm_strided_numeric() {
    let mut log = String::new();
    let dev = Device::new_rocm(0).expect("rocm device");

    // ---- copy_strided_src: contiguous, transposed, narrowed, permuted, ranks 1..8 ----
    for &dt in &FLOATS {
        let t = tol_for(dt);
        // contiguous .copy()
        cmp(&mut log, "copy.contig", &[4, 7], dt, t, &dev, |x| x.copy().unwrap());
        // transpose forces a non-contiguous gather (strides reordered)
        cmp(&mut log, "copy.transpose.contiguous", &[5, 6], dt, t, &dev, |x| {
            x.transpose(0, 1).unwrap().contiguous().unwrap()
        });
        // narrow (slice) -> non-zero start_offset; .contiguous() copies it
        cmp(&mut log, "copy.narrow.contiguous", &[8, 8], dt, t, &dev, |x| {
            x.narrow(0, 2, 4).unwrap().narrow(1, 1, 5).unwrap().contiguous().unwrap()
        });
        // permute a 4D tensor (multi-axis stride reorder)
        cmp(&mut log, "copy.permute4d.contiguous", &[2, 3, 4, 5], dt, t, &dev, |x| {
            x.permute((2, 0, 3, 1)).unwrap().contiguous().unwrap()
        });
        // rank 8 (the ROCM_DS_MAX ceiling) contiguous + transposed-tail
        cmp(&mut log, "copy.rank8.transpose67", &[2, 1, 2, 1, 2, 1, 2, 3], dt, t, &dev, |x| {
            x.transpose(6, 7).unwrap().contiguous().unwrap()
        });
    }

    // ---- Map1 / unary on non-contiguous input (gelu, silu, neg, sqr) ----
    for &dt in &FLOATS {
        let t = tol_for(dt).max(2e-2);
        cmp(&mut log, "unary.gelu.transpose", &[6, 5], dt, t, &dev, |x| {
            x.transpose(0, 1).unwrap().gelu().unwrap()
        });
        cmp(&mut log, "unary.silu.narrow", &[8, 8], dt, t, &dev, |x| {
            x.narrow(1, 2, 4).unwrap().silu().unwrap()
        });
        cmp(&mut log, "unary.neg.permute", &[2, 3, 4], dt, t, &dev, |x| {
            x.permute((2, 1, 0)).unwrap().neg().unwrap()
        });
    }

    // ---- Powf / Elu (unary-scalar) on non-contiguous input ----
    for &dt in &FLOATS {
        let t = tol_for(dt).max(2e-2);
        cmp(&mut log, "powf.transpose", &[5, 6], dt, t, &dev, |x| {
            // abs first so fractional power is real-valued
            x.transpose(0, 1).unwrap().abs().unwrap().powf(1.5).unwrap()
        });
        cmp(&mut log, "elu.narrow", &[8, 6], dt, t, &dev, |x| {
            x.narrow(0, 1, 5).unwrap().elu(1.0).unwrap()
        });
    }

    // ---- Affine on non-contiguous input ----
    for &dt in &FLOATS {
        let t = tol_for(dt).max(2e-2);
        cmp(&mut log, "affine.transpose", &[7, 4], dt, t, &dev, |x| {
            x.transpose(0, 1).unwrap().affine(0.5, -1.25).unwrap()
        });
        cmp(&mut log, "affine.permute3d", &[2, 3, 4], dt, t, &dev, |x| {
            x.permute((1, 2, 0)).unwrap().affine(2.0, 0.5).unwrap()
        });
    }

    // ---- Map2 / broadcast binary (the 2-stride-set path): add/mul/sub with broadcasting ----
    // Broadcasting forces stride-0 dims on one operand -> exercises lhs/rhs stride sets independently.
    for &dt in &FLOATS {
        let t = tol_for(dt).max(2e-2);
        // [3,4] op broadcast([4] -> [3,4]); rhs has a stride-0 leading dim
        cmp(&mut log, "binary.add.broadcast_row", &[3, 4], dt, t, &dev, |x| {
            let row = x.narrow(0, 0, 1).unwrap().broadcast_as((3, 4)).unwrap();
            x.broadcast_add(&row).unwrap()
        });
        cmp(&mut log, "binary.mul.broadcast_col", &[3, 4], dt, t, &dev, |x| {
            let col = x.narrow(1, 0, 1).unwrap().broadcast_as((3, 4)).unwrap();
            x.broadcast_mul(&col).unwrap()
        });
        // both operands non-contiguous (transposed) elementwise
        cmp(&mut log, "binary.sub.both_transposed", &[5, 6], dt, t, &dev, |x| {
            let a = x.transpose(0, 1).unwrap();
            let b = x.affine(1.0, 1.0).unwrap().transpose(0, 1).unwrap();
            a.sub(&b).unwrap()
        });
        // 3D broadcast: [2,3,4] * broadcast([2,1,4])
        cmp(&mut log, "binary.mul.broadcast_mid3d", &[2, 3, 4], dt, t, &dev, |x| {
            let mid = x.narrow(1, 0, 1).unwrap().broadcast_as((2, 3, 4)).unwrap();
            x.broadcast_mul(&mid).unwrap()
        });
    }

    // ---- FastReduce: sum/max/min over LAST and NON-LAST axes, multi-axis, contiguous + non-contig ----
    for &dt in &FLOATS {
        let t = tol_for(dt).max(3e-2); // sum over many elems accumulates f16 error
        // sum over last axis (contiguous)
        cmp(&mut log, "reduce.sum.lastaxis", &[6, 32], dt, t, &dev, |x| x.sum(D::Minus1).unwrap());
        // sum over FIRST axis (non-last -> layout reordered so axis 0 is gathered last)
        cmp(&mut log, "reduce.sum.axis0", &[32, 6], dt, t, &dev, |x| x.sum(0).unwrap());
        // sum over a MIDDLE axis of a 3D tensor
        cmp(&mut log, "reduce.sum.mid3d", &[4, 8, 5], dt, t, &dev, |x| x.sum(1).unwrap());
        // multi-axis sum (axes 0 and 2)
        cmp(&mut log, "reduce.sum.multiaxis", &[3, 5, 4], dt, t, &dev, |x| x.sum((0, 2)).unwrap());
        // max / min over the first axis (non-last). max/min are exact (no accumulation), tight tol.
        let te = tol_for(dt);
        cmp(&mut log, "reduce.max.axis0", &[16, 5], dt, te, &dev, |x| x.max(0).unwrap());
        cmp(&mut log, "reduce.min.axis0", &[16, 5], dt, te, &dev, |x| x.min(0).unwrap());
        cmp(&mut log, "reduce.max.lastaxis", &[5, 16], dt, te, &dev, |x| x.max(D::Minus1).unwrap());
        // reduce over a NON-CONTIGUOUS (transposed) input
        cmp(&mut log, "reduce.sum.transposed", &[8, 6], dt, t, &dev, |x| {
            x.transpose(0, 1).unwrap().sum(D::Minus1).unwrap()
        });
        // sum_keepdim over a middle axis (used by softmax/layernorm)
        cmp(&mut log, "reduce.sum_keepdim.mid", &[3, 7, 4], dt, t, &dev, |x| {
            x.sum_keepdim(1).unwrap()
        });
    }

    // ---- to_dtype / cast: every cross-dtype path, contiguous + non-contiguous ----
    // f32 -> f16/bf16/u32/i64 and back; verify the cast kernel reads strides correctly.
    let cast_cases: &[(DType, DType, f32)] = &[
        (DType::F32, DType::F16, 1e-2),
        (DType::F32, DType::BF16, 5e-2),
        (DType::F16, DType::F32, 1e-2),
        (DType::BF16, DType::F32, 5e-2),
        (DType::F32, DType::U32, 0.0),
        (DType::F32, DType::I64, 0.0),
        (DType::U32, DType::F32, 0.0),
        (DType::I64, DType::F32, 0.0),
        (DType::U32, DType::I64, 0.0),
    ];
    for &(from, to, t) in cast_cases {
        // contiguous cast
        cmp(&mut log, &format!("cast.{from:?}_{to:?}.contig"), &[4, 8], from, t, &dev, |x| {
            // for int targets, take abs+floor-ish via affine to keep values in range and integral
            let x = if to == DType::U32 || to == DType::I64 {
                x.abs().unwrap()
            } else {
                x.clone()
            };
            x.to_dtype(to).unwrap()
        });
        // non-contiguous cast (transposed) -- the strided cast path
        cmp(&mut log, &format!("cast.{from:?}_{to:?}.transpose"), &[5, 6], from, t, &dev, |x| {
            let x = if to == DType::U32 || to == DType::I64 {
                x.abs().unwrap()
            } else {
                x.clone()
            };
            x.transpose(0, 1).unwrap().to_dtype(to).unwrap()
        });
    }

    // ---- where_cond / ternary (the 3-stride-set path): contiguous + non-contiguous operands ----
    for &dt in &FLOATS {
        let t = tol_for(dt);
        // cond from a comparison, t/f from affine variants. All contiguous.
        cmp(&mut log, "where.contig", &[6, 5], dt, t, &dev, |x| {
            let cond = x.gt(0f64).unwrap(); // u8 mask
            let on_t = x.affine(2.0, 0.0).unwrap();
            let on_f = x.affine(-1.0, 0.0).unwrap();
            cond.where_cond(&on_t, &on_f).unwrap()
        });
        // all three operands NON-CONTIGUOUS (transposed) -> cond/t/f stride sets all exercised
        cmp(&mut log, "where.all_transposed", &[5, 6], dt, t, &dev, |x| {
            let xt = x.transpose(0, 1).unwrap();
            let cond = xt.gt(0f64).unwrap(); // u8 mask, non-contiguous (shares xt strides)
            let on_t = x.affine(2.0, 1.0).unwrap().transpose(0, 1).unwrap();
            let on_f = x.affine(-1.0, 1.0).unwrap().transpose(0, 1).unwrap();
            cond.where_cond(&on_t, &on_f).unwrap()
        });
    }

    // ---- index_select: contiguous-ids fast path + the strided source path, ranks/dtypes ----
    for &dt in &FLOATS {
        let t = tol_for(dt);
        let ids = Tensor::from_vec(vec![3u32, 0u32, 2u32, 0u32, 1u32], (5,), &dev).unwrap();
        let ids_cpu = Tensor::from_vec(vec![3u32, 0u32, 2u32, 0u32, 1u32], (5,), &Device::Cpu).unwrap();
        // select rows of a [4,7] matrix (dim 0)
        cmp(&mut log, "index_select.dim0", &[4, 7], dt, t, &dev, |x| {
            let i = if x.device().is_cpu() { &ids_cpu } else { &ids };
            x.index_select(i, 0).unwrap()
        });
        // select along last dim of a [3,4] matrix
        cmp(&mut log, "index_select.dim1", &[3, 4], dt, t, &dev, |x| {
            // ids must be < dim size 4; reuse the <=3 ids above (all valid for size 4)
            let i = if x.device().is_cpu() { &ids_cpu } else { &ids };
            x.index_select(i, 1).unwrap()
        });
    }
    // index_select on u32/i64 payloads (non-float dtype path of indexing.hip)
    {
        let ids = Tensor::from_vec(vec![2u32, 0u32, 3u32], (3,), &dev).unwrap();
        let ids_cpu = Tensor::from_vec(vec![2u32, 0u32, 3u32], (3,), &Device::Cpu).unwrap();
        for &dt in &[DType::U32, DType::I64] {
            cmp(&mut log, &format!("index_select.{dt:?}.dim0"), &[4, 5], dt, 0.0, &dev, |x| {
                let x = x.abs().unwrap();
                let i = if x.device().is_cpu() { &ids_cpu } else { &ids };
                x.index_select(i, 0).unwrap()
            });
        }
    }

    // ---- const_set / fill: write a constant into a (possibly sliced) tensor ----
    // Tensor::ones/zeros/full go through const_set; verify against CPU.
    for &dt in &FLOATS {
        let t = tol_for(dt);
        cmp(&mut log, "fill.ones_like", &[4, 6], dt, t, &dev, |x| x.ones_like().unwrap());
        cmp(&mut log, "fill.zeros_like", &[3, 5], dt, t, &dev, |x| x.zeros_like().unwrap());
    }

    // ---- integer-dtype Map1/binary (u32/i64) to cover non-float kernel template instantiations ----
    {
        for &dt in &[DType::U32, DType::I64] {
            cmp(&mut log, &format!("binary.add.{dt:?}.transpose"), &[5, 4], dt, 0.0, &dev, |x| {
                let x = x.abs().unwrap();
                let a = x.transpose(0, 1).unwrap();
                a.add(&a).unwrap()
            });
            cmp(&mut log, &format!("copy.{dt:?}.narrow", ), &[6, 6], dt, 0.0, &dev, |x| {
                x.abs().unwrap().narrow(0, 1, 4).unwrap().contiguous().unwrap()
            });
        }
    }

    let _ = std::fs::File::create("/home/z/rocm-strided-test.txt")
        .and_then(|mut f| f.write_all(log.as_bytes()));
    eprintln!("{log}");
    eprintln!("rocm_strided_numeric: ALL PASS");
}
