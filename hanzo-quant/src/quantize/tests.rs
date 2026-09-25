use hanzo_ml::{DType, Device, Result, Tensor};

use super::*;

/// E2M1 midpoints go to the even code, zero keeps its sign, and magnitudes past 6 saturate.
fn check_rne(encode: impl Fn(&[f32]) -> Result<Vec<u8>>) -> Result<()> {
    let cases: [(f32, u8); 24] = [
        (0.25, 0),
        (0.75, 2),
        (1.25, 2),
        (1.75, 4),
        (2.5, 4),
        (3.5, 6),
        (5.0, 6),
        (-0.25, 8),
        (-0.75, 10),
        (-1.25, 10),
        (-1.75, 12),
        (-2.5, 12),
        (-3.5, 14),
        (-5.0, 14),
        (0.0, 0),
        (-0.0, 8),
        (-0.1, 8),
        (6.0, 7),
        (7.0, 7),
        (1e9, 7),
        (f32::INFINITY, 7),
        (-1e9, 15),
        (0.2500001, 1),
        (4.9999, 6),
    ];
    let v: Vec<f32> = cases.iter().map(|c| c.0).collect();
    let got = encode(&v)?;
    for ((x, want), got) in cases.iter().zip(got) {
        assert_eq!(got, *want, "e2m1({x}) = {got:#06b}, want {want:#06b}");
    }
    Ok(())
}

#[test]
fn e2m1_encode_is_rne() -> Result<()> {
    check_rne(|v| Ok(v.iter().map(|&x| e2m1_encode(x)).collect()))?;
    for c in 0..16u8 {
        assert_eq!(e2m1_encode(e2m1_decode(c)), c, "code {c} round trip");
    }
    #[cfg(feature = "cuda")]
    if let Ok(dev) = Device::new_cuda(0) {
        check_rne(|v| {
            let t = Tensor::from_slice(v, v.len(), &dev)?;
            e2m1_encode_cuda(&t)?.to_vec1::<u8>()
        })?;
    }
    Ok(())
}

#[test]
fn cpu_quantizers_round_trip() -> Result<()> {
    // A block holding exactly the codebook times a power of two decodes back exactly.
    let block: Vec<f32> = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        .iter()
        .flat_map(|v| [*v, -*v])
        .collect();
    let x = Tensor::from_vec(block.clone(), (1, 16), &Device::Cpu)?.to_dtype(DType::BF16)?;
    // gs such that SF = 1 exactly: SF = gs * 6 / 6.
    let (codes, scales) = nvfp4(&x, 1.0)?;
    assert_eq!(scales.to_dtype(DType::F32)?.to_vec2::<f32>()?, [[1.0f32]]);
    let bytes = codes.to_vec2::<u8>()?;
    let decoded: Vec<f32> = bytes[0]
        .iter()
        .flat_map(|b| [e2m1_decode(b & 15), e2m1_decode(b >> 4)])
        .collect();
    assert_eq!(decoded, block);

    // FP8: the group maximum lands on 448 and survives.
    let mut v = vec![0.001f32; 256];
    v[3] = 3.5;
    v[200] = -7.0;
    let x = Tensor::from_vec(v, (1, 256), &Device::Cpu)?;
    for mode in [Fp8Mode::Linear, Fp8Mode::Eager, Fp8Mode::Gather] {
        let (q, s) = fp8(&x, mode)?;
        let q = q.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let s = s.to_vec2::<f32>()?;
        assert!((q[0][3] * s[0][0] - 3.5).abs() <= 3.5 / 16.0, "{mode:?}");
        assert!((q[0][200] * s[0][1] + 7.0).abs() <= 7.0 / 16.0, "{mode:?}");
    }
    Ok(())
}

#[cfg(feature = "cuda")]
mod served {
    use std::collections::HashMap;

    use super::*;

    fn vectors(dev: &Device) -> Result<HashMap<String, Tensor>> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/vllm_vectors.safetensors"
        );
        hanzo_ml::safetensors::load(path, dev)
    }

    fn bits(t: &Tensor) -> Result<Vec<u8>> {
        match t.dtype() {
            DType::U8 => t.flatten_all()?.to_vec1::<u8>(),
            DType::F8E4M3 => Ok(t
                .flatten_all()?
                .to_vec1::<float8::F8E4M3>()?
                .into_iter()
                .map(|v| v.to_bits())
                .collect()),
            d => panic!("no bytes view of {d:?}"),
        }
    }

    fn f32s(t: &Tensor) -> Result<Vec<u32>> {
        Ok(t.to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    /// Bytes equal, reporting the first few that differ.
    fn assert_bytes(what: &str, got: &[u8], want: &[u8]) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let bad: Vec<(usize, u8, u8)> = got
            .iter()
            .zip(want)
            .enumerate()
            .filter(|(_, (g, w))| g != w)
            .map(|(i, (g, w))| (i, *g, *w))
            .collect();
        assert!(
            bad.is_empty(),
            "{what}: {} of {} differ, first {:?}",
            bad.len(),
            got.len(),
            &bad[..bad.len().min(8)]
        );
    }

    fn gpu() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    fn check_fp8(mode: Fp8Mode, prefix: &str, inputs: &[(&str, &str)]) -> Result<()> {
        let Some(dev) = gpu() else { return Ok(()) };
        let v = vectors(&dev)?;
        for (tag, input) in inputs {
            let x = &v[*input];
            let (q, s) = fp8(x, mode)?;
            assert_bytes(
                &format!("{prefix}.{tag}.q"),
                &bits(&q)?,
                &bits(&v[&format!("{prefix}.{tag}.q")])?,
            );
            assert_eq!(
                f32s(&s)?,
                f32s(&v[&format!("{prefix}.{tag}.s")])?,
                "{prefix}.{tag}.s"
            );
        }
        Ok(())
    }

    #[test]
    fn fp8_linear_matches_served() -> Result<()> {
        check_fp8(
            Fp8Mode::Linear,
            "fp8.linear",
            &[
                ("syn", "syn.x"),
                ("l0u", "x.l0u"),
                ("l3u", "x.l3u"),
                ("l3mu", "x.l3mu"),
                ("l0n", "fp8.linear.l0n.x"),
            ],
        )
    }

    #[test]
    fn fp8_eager_matches_served() -> Result<()> {
        check_fp8(
            Fp8Mode::Eager,
            "fp8.eager",
            &[("syn", "syn.x"), ("l3mu", "x.l3mu"), ("l3act", "fp8.eager.l3act.x")],
        )
    }

    #[test]
    fn fp8_gather_matches_served() -> Result<()> {
        check_fp8(Fp8Mode::Gather, "fp8.group", &[("syn", "syn.x")])
    }

    /// The o_proj quantizer reads attn * sigmoid(gate) in f32: the product handed over unrounded
    /// quantizes to the served codes.
    #[test]
    fn fp8_gated_product_matches_served() -> Result<()> {
        let Some(dev) = gpu() else { return Ok(()) };
        let v = vectors(&dev)?;
        let core = v["fp8.gated.core"].to_dtype(DType::F32)?;
        let gate = v["fp8.gated.gate"].to_dtype(DType::F32)?;
        let sig = (gate.neg()?.exp()? + 1.0)?.recip()?;
        let (q, s) = fp8(&(core * sig)?, Fp8Mode::Linear)?;
        let (gq, wq) = (bits(&q)?, bits(&v["fp8.gated.q"])?);
        let differ = gq.iter().zip(&wq).filter(|(a, b)| a != b).count();
        // exp and the reciprocal are Triton's tl.sigmoid on the server: allow a last-bit step
        assert!(
            differ as f64 <= 1e-4 * gq.len() as f64,
            "{differ} of {} codes differ",
            gq.len()
        );
        let _ = s;
        Ok(())
    }

    fn check_nvfp4(prefix: &str, gs: f32, inputs: &[(&str, &str)]) -> Result<()> {
        let Some(dev) = gpu() else { return Ok(()) };
        let v = vectors(&dev)?;
        for (tag, input) in inputs {
            let x = &v[*input];
            let (c, s) = nvfp4(x, gs)?;
            assert_bytes(
                &format!("{prefix}.{tag}.q"),
                &bits(&c)?,
                &bits(&v[&format!("{prefix}.{tag}.q")])?,
            );
            assert_bytes(
                &format!("{prefix}.{tag}.s"),
                &bits(&s)?,
                &bits(&v[&format!("{prefix}.{tag}.s")])?,
            );
        }
        Ok(())
    }

    fn gs(i: usize) -> Result<f32> {
        let dev = Device::Cpu;
        Ok(vectors(&dev)?["fp4.gs"].to_vec1::<f32>()?[i])
    }

    #[test]
    fn nvfp4_matches_served() -> Result<()> {
        let Some(dev) = gpu() else { return Ok(()) };
        let syn640 = vectors(&dev)?["syn.x"].narrow(1, 0, 640)?.contiguous()?;
        check_nvfp4("fp4.input", gs(0)?, &[("syn", "syn.x"), ("l3mu", "x.l3mu")])?;
        check_nvfp4("fp4.inter", gs(1)?, &[("l3", "fp4.inter.l3.x")])?;
        // the synthetic slice for the FlashInfer quantizer
        let v = vectors(&dev)?;
        let (c, s) = nvfp4(&syn640, gs(1)?)?;
        assert_bytes("fp4.inter.syn.q", &bits(&c)?, &bits(&v["fp4.inter.syn.q"])?);
        assert_bytes("fp4.inter.syn.s", &bits(&s)?, &bits(&v["fp4.inter.syn.s"])?);
        Ok(())
    }

    /// CPU (IEEE) and CUDA (served approximations) agree except for a code step on at most 1e-4 of
    /// elements, each at a rounding boundary.
    #[test]
    fn quantizers_cpu_match_cuda() -> Result<()> {
        let Some(dev) = gpu() else { return Ok(()) };
        let v = vectors(&dev)?;
        for input in ["syn.x", "x.l0u", "x.l3u", "x.l3mu"] {
            let x = &v[input];
            for mode in [Fp8Mode::Linear, Fp8Mode::Eager, Fp8Mode::Gather] {
                let (qg, _) = fp8(x, mode)?;
                let (qc, _) = fp8(&x.to_device(&Device::Cpu)?, mode)?;
                let (a, b) = (bits(&qg)?, bits(&qc)?);
                let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
                assert!(
                    differ as f64 <= 1e-4 * a.len() as f64,
                    "{input} {mode:?}: {differ} of {} differ",
                    a.len()
                );
            }
            let (cg, sg) = nvfp4(x, gs(0)?)?;
            let (cc, sc) = nvfp4(&x.to_device(&Device::Cpu)?, gs(0)?)?;
            let (a, b) = (bits(&cg)?, bits(&cc)?);
            let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            assert!(
                differ as f64 <= 2e-4 * a.len() as f64,
                "{input} nvfp4: {differ} of {} bytes differ",
                a.len()
            );
            let (a, b) = (bits(&sg)?, bits(&sc)?);
            let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            assert!(
                differ as f64 <= 1e-4 * a.len() as f64,
                "{input} nvfp4 scales: {differ} differ"
            );
        }
        Ok(())
    }

    /// The CUTLASS path's quantizer, unswizzled, writes the same bytes as `nvfp4`.
    #[cfg(has_nvfp4_cutlass_kernels)]
    #[test]
    fn cutlass_quantizer_equals_nvfp4() -> Result<()> {
        let Some(dev) = gpu() else { return Ok(()) };
        let v = vectors(&dev)?;
        let gs = gs(0)?;
        for input in ["syn.x", "x.l3mu"] {
            let x = &v[input];
            // The CUTLASS path takes the checkpoint's input_scale and forms 1 / input_scale.
            let input_scale = 1.0 / gs;
            let (codes, scales) = crate::nvfp4::cutlass::quantize_linear(x, input_scale)?;
            let (c, s) = nvfp4(x, 1.0 / input_scale)?;
            assert_bytes(&format!("{input} codes"), &bits(&codes)?, &bits(&c)?);
            assert_bytes(&format!("{input} scales"), &scales, &bits(&s)?);
        }
        Ok(())
    }
}
