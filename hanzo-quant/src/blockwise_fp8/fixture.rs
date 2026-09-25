//! vLLM's block-FP8 GEMM outputs on synthetic operands, and the fp64 model
//! every output is judged against.
//!
//! The fixture is written by `scripts/blockwise_fp8_golden.py`, which runs
//! vLLM 0.29.0's `cutlass_scaled_mm` (the op `CutlassFp8BlockScaledMMKernel`
//! reaches) on seeded operands. Layouts are hanzo's: activation scales
//! row-major `[M, K/128]`, weight scales `[N/128, K/128]`.

use std::collections::HashMap;

use float8::F8E4M3;
use hanzo_ml::{DType, Device, Result, Tensor};

pub(crate) const BLOCK: usize = 128;

/// One weight shape with vLLM's output for the largest M it was run at.
/// vLLM's rows do not depend on M (the generator asserts it), so row `i` of
/// `out` is the answer for every M > i.
pub(crate) struct Case {
    pub name: String,
    pub n: usize,
    pub k: usize,
    pub m: Vec<usize>,
    /// e4m3 `[Mmax, K]`.
    pub qa: Tensor,
    /// f32 `[Mmax, K/128]`.
    pub sa: Tensor,
    /// e4m3 `[N, K]`.
    pub qw: Tensor,
    /// f32 `[N/128, K/128]`.
    pub sw: Tensor,
    /// bf16 `[Mmax, N]`.
    pub out: Tensor,
}

pub(crate) struct Fixture {
    pub cases: Vec<Case>,
    pub metadata: HashMap<String, String>,
}

pub(crate) fn path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/blockwise_fp8.safetensors")
}

pub(crate) fn load() -> Result<Fixture> {
    let bytes = std::fs::read(path())?;
    let (_, header) =
        safetensors::SafeTensors::read_metadata(&bytes).map_err(hanzo_ml::Error::wrap)?;
    let metadata = header.metadata().clone().unwrap_or_default();
    let mut tensors = hanzo_ml::safetensors::load_buffer(&bytes, &Device::Cpu)?;
    let cases_json = metadata
        .get("cases")
        .ok_or_else(|| hanzo_ml::Error::Msg("fixture metadata has no cases".into()))?;
    let described: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(cases_json).map_err(hanzo_ml::Error::wrap)?;

    let take = |key: &str| {
        tensors
            .get(key)
            .cloned()
            .ok_or_else(|| hanzo_ml::Error::Msg(format!("fixture has no tensor {key}")))
    };
    let mut cases = Vec::new();
    for (name, d) in &described {
        let n = d["n"].as_u64().unwrap() as usize;
        let k = d["k"].as_u64().unwrap() as usize;
        let act = d["act"].as_str().unwrap();
        let m = d["m"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        cases.push(Case {
            name: name.clone(),
            n,
            k,
            m,
            qa: take(&format!("{act}.qa"))?,
            sa: take(&format!("{act}.sa"))?,
            qw: take(&format!("{name}.qw"))?,
            sw: take(&format!("{name}.sw"))?,
            out: take(&format!("{name}.out"))?,
        });
    }
    tensors.clear();
    Ok(Fixture { cases, metadata })
}

/// OCP E4M3 (fn): no infinities, 0x7F/0xFF are NaN.
pub(crate) fn e4m3(bits: u8) -> f64 {
    let sign = if bits & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 3) & 0xF) as i32;
    let man = (bits & 7) as f64;
    if exp == 0xF && man == 7.0 {
        return f64::NAN;
    }
    let mag = if exp == 0 {
        man / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + man / 8.0) * 2f64.powi(exp - 7)
    };
    sign * mag
}

fn codes(t: &Tensor) -> Result<Vec<f64>> {
    Ok(t.flatten_all()?
        .to_vec1::<F8E4M3>()?
        .into_iter()
        .map(|v| e4m3(v.to_bits()))
        .collect())
}

/// Exact block-FP8 product in fp64 for the first `m` rows, and the matching
/// magnitude sum C = sum_kb |s_a s_w| sum_k |q_a q_w| the tolerance scales with.
/// Per-block dot products of e4m3 values are exact in fp64.
pub(crate) struct Reference {
    pub y: Vec<f64>,
    pub c: Vec<f64>,
}

pub(crate) fn reference(qa: &Tensor, sa: &Tensor, qw: &Tensor, sw: &Tensor, m: usize) -> Result<Reference> {
    let (_, k) = qa.dims2()?;
    let (n, _) = qw.dims2()?;
    let kb = k / BLOCK;
    let a = codes(&qa.narrow(0, 0, m)?)?;
    let w = codes(qw)?;
    let sa: Vec<f64> = sa.narrow(0, 0, m)?.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
    let sw: Vec<f64> = sw.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
    let mut y = vec![0.0; m * n];
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let (mut acc, mut mag) = (0.0f64, 0.0f64);
            for b in 0..kb {
                let (mut dot, mut abs) = (0.0f64, 0.0f64);
                for t in b * BLOCK..(b + 1) * BLOCK {
                    let p = a[i * k + t] * w[j * k + t];
                    dot += p;
                    abs += p.abs();
                }
                let s = sa[i * kb + b] * sw[(j / BLOCK) * kb + b];
                acc += s * dot;
                mag += s.abs() * abs;
            }
            y[i * n + j] = acc;
            c[i * n + j] = mag;
        }
    }
    Ok(Reference { y, c })
}

/// |y - y64| <= 2^-8 |y64| + 2^-12 C: bf16 rounding plus FP32 accumulation,
/// far below the >= C/(K/128) a wrong scale block or a dropped K block costs.
pub(crate) fn within_bound(y: f64, y64: f64, c: f64) -> bool {
    (y - y64).abs() <= 2f64.powi(-8) * y64.abs() + 2f64.powi(-12) * c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_decodes_the_codebook_edges() {
        assert_eq!(e4m3(0x00), 0.0);
        assert_eq!(e4m3(0x80), 0.0);
        assert_eq!(e4m3(0x7E), 448.0);
        assert_eq!(e4m3(0xFE), -448.0);
        assert_eq!(e4m3(0x01), 2f64.powi(-9));
        assert_eq!(e4m3(0x38), 1.0);
        assert!(e4m3(0x7F).is_nan());
    }

    #[test]
    fn golden_within_fp64_bound() -> Result<()> {
        let fixture = load()?;
        let meta = &fixture.metadata;
        assert_eq!(meta.get("synthetic").map(String::as_str), Some("true"));
        assert_eq!(meta.get("vllm_version").map(String::as_str), Some("0.29.0"));
        assert_eq!(meta.get("cc").map(String::as_str), Some("12.1"));
        println!(
            "fixture: vllm {} ({}), torch {}, cuda {}, {}",
            meta["vllm_version"], meta["vllm_commit"], meta["torch"], meta["cuda"], meta["device"]
        );
        assert_eq!(fixture.cases.len(), 5);

        for case in &fixture.cases {
            let m_max = *case.m.iter().max().unwrap();
            assert_eq!(case.qa.dims2()?, (m_max, case.k));
            assert_eq!(case.sa.dims2()?, (m_max, case.k / BLOCK));
            assert_eq!(case.qw.dims2()?, (case.n, case.k));
            assert_eq!(case.sw.dims2()?, (case.n / BLOCK, case.k / BLOCK));
            assert_eq!(case.out.dims2()?, (m_max, case.n));

            let reference = reference(&case.qa, &case.sa, &case.qw, &case.sw, m_max)?;
            let got: Vec<f64> = case.out.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
            let mut worst = 0.0f64;
            let mut bad = 0usize;
            for (idx, g) in got.iter().enumerate() {
                let (y64, c) = (reference.y[idx], reference.c[idx]);
                let slack = 2f64.powi(-8) * y64.abs() + 2f64.powi(-12) * c;
                if slack > 0.0 {
                    worst = worst.max((g - y64).abs() / slack);
                }
                if !within_bound(*g, y64, c) {
                    if bad < 5 {
                        println!(
                            "{} [{}, {}]: vllm {g}, fp64 {y64}, C {c}",
                            case.name,
                            idx / case.n,
                            idx % case.n
                        );
                    }
                    bad += 1;
                }
            }
            println!(
                "{}: N={} K={} Mmax={m_max}: {bad} of {} outside the bound, worst {worst:.3} of the slack",
                case.name,
                case.n,
                case.k,
                got.len()
            );
            assert_eq!(bad, 0, "{}: vLLM output outside the fp64 bound", case.name);
        }
        Ok(())
    }
}
