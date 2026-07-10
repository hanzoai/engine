//! FASHN VTON v1.5: pixel-space maskless virtual try-on (fashn-ai/fashn-vton-1.5, Apache-2.0).
//!
//! A 972M-parameter MMDiT operating directly on RGB pixels (12x12 patch embedding, no VAE, no text
//! encoder). Conditioning is person + garment images, single-channel pose maps, and a garment
//! category. Sampling is rectified-flow Euler with classifier-free guidance.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub mod config;
pub mod model;
pub mod pipeline;
pub mod sampling;

pub use config::VtonConfig;
pub use model::TryOnModel;
pub use pipeline::{load_model, vton_generate, Category, TryOnInputs, DEFAULT_MODEL_ID};
pub use sampling::SampleParams;

use std::collections::BTreeSet;

/// A registered buffer kept in the checkpoint for training compatibility; unused at inference.
pub const UNUSED_BUFFER_KEY: &str = "patch_mixer_token";

/// Every tensor key the model consumes from the checkpoint. Used to verify a loaded checkpoint has
/// no missing or unexpected tensors (the sole extra key is [`UNUSED_BUFFER_KEY`]).
pub fn expected_tensor_keys(cfg: &VtonConfig) -> BTreeSet<String> {
    fn wb(keys: &mut BTreeSet<String>, p: &str) {
        keys.insert(format!("{p}.weight"));
        keys.insert(format!("{p}.bias"));
    }
    fn qk(keys: &mut BTreeSet<String>, p: &str) {
        keys.insert(format!("{p}.norm.query_norm.scale"));
        keys.insert(format!("{p}.norm.key_norm.scale"));
    }
    fn single(keys: &mut BTreeSet<String>, p: &str) {
        wb(keys, &format!("{p}.linear1"));
        wb(keys, &format!("{p}.linear2"));
        wb(keys, &format!("{p}.modulation.lin"));
        qk(keys, p);
    }
    fn double(keys: &mut BTreeSet<String>, p: &str) {
        for side in ["img", "txt"] {
            wb(keys, &format!("{p}.{side}_mod.lin"));
            wb(keys, &format!("{p}.{side}_attn.qkv"));
            wb(keys, &format!("{p}.{side}_attn.proj"));
            qk(keys, &format!("{p}.{side}_attn"));
            wb(keys, &format!("{p}.{side}_mlp.0"));
            wb(keys, &format!("{p}.{side}_mlp.2"));
        }
    }

    let mut keys = BTreeSet::new();
    wb(&mut keys, "t_embedder.mlp.in_layer");
    wb(&mut keys, "t_embedder.mlp.out_layer");
    keys.insert("y_embedder.weight".into());
    wb(&mut keys, "x_embedder.proj");
    wb(&mut keys, "garment_embedder.proj");
    for i in 0..cfg.patch_mixer_depth {
        single(&mut keys, &format!("x_patch_mixer.{i}"));
    }
    for i in 0..cfg.double_blocks_depth {
        double(&mut keys, &format!("double_blocks.{i}"));
    }
    for i in 0..cfg.single_blocks_depth {
        single(&mut keys, &format!("single_blocks.{i}"));
    }
    wb(&mut keys, "final_layer.linear");
    wb(&mut keys, "final_layer.adaLN_modulation.1");
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_key_count() {
        // 4 (t_embedder) + 1 (y) + 2 (x_embedder) + 2 (garment) + 4*8 + 8*24 + 16*8 + 4 = 365.
        let keys = expected_tensor_keys(&VtonConfig::default());
        assert_eq!(keys.len(), 365);
    }

    // --- Reference-parity tests (ignored: need the real checkpoint + the Python oracle dump) ---
    //
    // Run with:
    //   VTON_WEIGHTS=/path/to/model.safetensors VTON_ORACLE=/path/to/vton_parity.safetensors \
    //   cargo test -p hanzo-engine vton::tests -- --ignored --nocapture --test-threads=1

    use hanzo_ml::{Device, Tensor};
    use std::collections::BTreeSet;

    fn weights_path() -> String {
        std::env::var("VTON_WEIGHTS").unwrap_or_else(|_| {
            "/home/z/.cache/huggingface/hub/models--fashn-ai--fashn-vton-1.5/snapshots/\
             7720683168567eb5a2a4c67f15116c6e29c83ded/model.safetensors"
                .into()
        })
    }
    fn oracle_path() -> String {
        std::env::var("VTON_ORACLE")
            .unwrap_or_else(|_| "/home/z/work/vton-ref/vton_parity.safetensors".into())
    }

    fn stats(name: &str, got: &Tensor, want: &Tensor) -> (f64, f64, f64) {
        let a = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len(), "{name}: length mismatch");
        let (mut se, mut dot, mut na, mut nb, mut maxabs) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(&b) {
            let (x, y) = (*x as f64, *y as f64);
            se += (x - y).powi(2);
            dot += x * y;
            na += x * x;
            nb += y * y;
            maxabs = maxabs.max((x - y).abs());
        }
        let mse = se / a.len() as f64;
        let cos = dot / (na.sqrt() * nb.sqrt());
        println!("  {name:12} MSE={mse:.3e}  cos={cos:.8}  max|Δ|={maxabs:.3e}");
        (mse, cos, maxabs)
    }

    #[test]
    #[ignore = "needs real checkpoint + oracle"]
    fn loader_completeness() {
        // Gate 1: every tensor the model reads is present, and the only unread key is the buffer.
        let file_keys: BTreeSet<String> = {
            let dev = Device::Cpu;
            hanzo_ml::safetensors::load(weights_path(), &dev)
                .unwrap()
                .into_keys()
                .collect()
        };
        let mut expected = expected_tensor_keys(&VtonConfig::default());
        expected.insert(UNUSED_BUFFER_KEY.to_string());
        let missing: Vec<_> = expected.difference(&file_keys).cloned().collect();
        let unexpected: Vec<_> = file_keys.difference(&expected).cloned().collect();
        assert!(missing.is_empty(), "missing tensors: {missing:?}");
        assert!(unexpected.is_empty(), "unexpected tensors: {unexpected:?}");
        // And the model actually constructs from the real checkpoint.
        load_model(&weights_path(), &Device::Cpu).unwrap();
        println!(
            "  loader OK: 365 consumed + 1 unused buffer = {} keys",
            file_keys.len()
        );
    }

    #[test]
    #[ignore = "needs real checkpoint + oracle"]
    fn parity_vs_reference() {
        let dev = Device::Cpu;
        let model = load_model(&weights_path(), &dev).unwrap();
        let o = hanzo_ml::safetensors::load(oracle_path(), &dev).unwrap();
        let g = |k: &str| o[k].clone();

        let (x, times) = (g("x"), g("times"));
        let (ca, garment) = (g("ca_images"), g("garment_images"));
        let (pp, gp) = (g("person_poses"), g("garment_poses"));
        let cats = Tensor::from_vec(vec![1u32], 1, &dev).unwrap();

        // Gate 2a: single forward (full conditioning) == reference single_out.
        println!("single forward:");
        let out = model
            .forward(&x, &times, &ca, &garment, &pp, &gp, &cats)
            .unwrap();
        let (mse, cos, _) = stats("single_out", &out, &g("single_out"));
        assert!(cos > 0.99999 && mse < 1e-4, "single forward parity failed");

        // Gate 2b: CFG unconditional branch (all conditioning zeroed, null category) == cfg_vu.
        println!("cfg unconditional:");
        let z = |t: &Tensor| t.zeros_like().unwrap();
        let null_cat = Tensor::from_vec(vec![0u32], 1, &dev).unwrap();
        let vu = model
            .forward(
                &x,
                &times,
                &z(&ca),
                &z(&garment),
                &z(&pp),
                &z(&gp),
                &null_cat,
            )
            .unwrap();
        let (mse, cos, _) = stats("cfg_vu", &vu, &g("cfg_vu"));
        assert!(cos > 0.99999 && mse < 1e-4, "cfg uncond parity failed");

        // Gate 2c: full 4-step rectified-flow + CFG generation == reference gen_out.
        println!("generation (4-step euler + cfg):");
        let cond = sampling::Conditioning {
            ca_images: ca,
            garment_images: garment,
            person_poses: pp,
            garment_poses: gp,
            categories: cats,
        };
        let params = sampling::SampleParams {
            num_timesteps: 4,
            time_shift_mu: 1.5,
            guidance_scale: 1.5,
            skip_cfg_last_n_steps: 1,
        };
        let gen = sampling::denoise(&model, &cond, &g("gen_init"), &params).unwrap();
        let (mse, cos, _) = stats("gen_out", &gen, &g("gen_out"));
        assert!(cos > 0.99999 && mse < 1e-4, "generation parity failed");
        println!("PARITY OK");
    }

    // Gate 3: a real person+garment pair through the engine produces a coherent try-on image.
    // Steps are reduced (VTON_E2E_STEPS, default 8) for CPU time; person pose is blank (DWPose
    // is a follow-up). Output saved to VTON_E2E_OUT.
    #[test]
    #[ignore = "needs real checkpoint + example images; slow on CPU"]
    fn e2e_generate() {
        let dir = std::env::var("VTON_REF_DIR").unwrap_or_else(|_| "/home/z/work/vton-ref".into());
        let steps: usize = std::env::var("VTON_E2E_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let out = std::env::var("VTON_E2E_OUT")
            .unwrap_or_else(|_| format!("{dir}/vton_engine_output.png"));

        let person = image::open(format!("{dir}/model.png")).unwrap();
        let garment = image::open(format!("{dir}/garment.png")).unwrap();
        let inputs = TryOnInputs {
            model_id: weights_path(),
            person,
            garment,
            person_pose: None,
            garment_pose: None,
            category: Category::Tops,
            num_samples: 1,
            seed: 42,
            sample: SampleParams {
                num_timesteps: steps,
                ..SampleParams::default()
            },
        };
        let images = vton_generate(&inputs, &Device::Cpu).unwrap();
        assert_eq!(images.len(), 1);
        let img = images.into_iter().next().unwrap();
        let (w, h) = (img.width(), img.height());
        assert!(w > 64 && h > 64, "degenerate output {w}x{h}");
        img.save(&out).unwrap();
        println!("E2E OK: {steps}-step try-on -> {out} ({w}x{h})");
    }
}
