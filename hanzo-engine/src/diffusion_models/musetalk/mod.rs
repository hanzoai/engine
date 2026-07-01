pub mod animator;
pub mod config;
pub mod pipeline;
pub mod unet;
pub mod vae;

pub use animator::{AnimatorOptions, MuseTalkAnimator, MuseTalkGenerator, S3fdLocator};
pub use config::{MuseTalkConfig, UNetConfig, VaeConfig};
pub use pipeline::MuseTalk;
pub use unet::UNet2DConditionModel;
pub use vae::AutoencoderKl;

#[cfg(test)]
mod tests {
    use super::config::{MuseTalkConfig, UNetConfig, VaeConfig};
    use super::pipeline::MuseTalk;
    use hanzo_ml::{DType, Device, Result, Shape, Tensor};
    use hanzo_nn::var_builder::SimpleBackend;
    use hanzo_nn::Init;
    use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};

    const WEIGHT_STD: f64 = 0.02;

    struct RandnBackend;

    impl SimpleBackend for RandnBackend {
        fn get(
            &self,
            s: Shape,
            name: &str,
            _h: Init,
            dtype: DType,
            dev: &Device,
        ) -> Result<Tensor> {
            if name.ends_with("bias") {
                Tensor::zeros(s, dtype, dev)
            } else if name.ends_with("weight") && s.rank() == 1 {
                Tensor::ones(s, dtype, dev)
            } else {
                Tensor::randn(0f64, WEIGHT_STD, s, dev)?.to_dtype(dtype)
            }
        }

        fn get_unchecked(&self, _name: &str, _dtype: DType, _dev: &Device) -> Result<Tensor> {
            hanzo_ml::bail!("RandnBackend requires an explicit shape")
        }

        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    fn random_vb(dtype: DType, dev: &Device) -> ShardedVarBuilder {
        ShardedSafeTensors::wrap(Box::new(RandnBackend), dtype, dev.clone())
    }

    fn tiny_config() -> MuseTalkConfig {
        MuseTalkConfig {
            unet: UNetConfig {
                sample_size: 4,
                in_channels: 8,
                out_channels: 4,
                layers_per_block: 1,
                block_out_channels: vec![32, 64],
                down_block_types: vec![
                    "CrossAttnDownBlock2D".to_string(),
                    "DownBlock2D".to_string(),
                ],
                up_block_types: vec!["UpBlock2D".to_string(), "CrossAttnUpBlock2D".to_string()],
                cross_attention_dim: 384,
                attention_head_dim: 8,
                norm_num_groups: 32,
                norm_eps: 1e-5,
                flip_sin_to_cos: true,
                freq_shift: 0.0,
            },
            vae: VaeConfig {
                in_channels: 3,
                out_channels: 3,
                block_out_channels: vec![32, 64],
                layers_per_block: 1,
                latent_channels: 4,
                norm_num_groups: 32,
                scaling_factor: 0.18215,
                sample_size: 16,
            },
            resized_img: 16,
        }
    }

    fn assert_finite(t: &Tensor) -> Result<()> {
        let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(!v.is_empty());
        let bad = v.iter().filter(|x| !x.is_finite()).count();
        assert_eq!(bad, 0, "found {bad} non-finite values in tensor");
        let max = v.iter().cloned().fold(f32::MIN, f32::max);
        let min = v.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            max != min,
            "output is constant (degenerate), min==max=={min}"
        );
        Ok(())
    }

    #[test]
    fn unet_single_step_forward_no_nan() -> Result<()> {
        let dev = Device::Cpu;
        let dtype = DType::F32;
        let cfg = tiny_config();
        let model = MuseTalk::new(
            cfg.clone(),
            random_vb(dtype, &dev),
            random_vb(dtype, &dev),
            &dev,
            dtype,
        )?;

        let n = cfg.unet.sample_size;
        let latent_input = Tensor::randn(0f64, 1.0, (1, cfg.unet.in_channels, n, n), &dev)?;
        let audio = Tensor::randn(0f64, 1.0, (1, 50, cfg.unet.cross_attention_dim), &dev)?;
        let ts = Tensor::zeros(1, DType::F32, &dev)?;

        let out = model_unet_forward(&model, &latent_input, &ts, &audio)?;
        assert_eq!(out.dims(), &[1, cfg.unet.out_channels, n, n]);
        assert_finite(&out)?;
        Ok(())
    }

    #[test]
    fn full_pipeline_inpaint_forward_no_nan() -> Result<()> {
        let dev = Device::Cpu;
        let dtype = DType::F32;
        let cfg = tiny_config();
        let model = MuseTalk::new(
            cfg.clone(),
            random_vb(dtype, &dev),
            random_vb(dtype, &dev),
            &dev,
            dtype,
        )?;

        let sz = cfg.resized_img;
        let face = Tensor::rand(0f32, 1f32, (1, 3, sz, sz), &dev)?;
        let audio = Tensor::randn(0f64, 1.0, (1, 50, cfg.unet.cross_attention_dim), &dev)?;

        let latents = model.latents_for_unet(&face)?;
        assert_eq!(latents.dim(1)?, 2 * cfg.vae.latent_channels);
        assert_finite(&latents)?;

        let image = model.forward(&face, &audio)?;
        assert_eq!(image.dims(), &[1, 3, sz, sz]);
        assert_finite(&image)?;

        let blended = model.blend(&face, &image)?;
        assert_eq!(blended.dims(), &[1, 3, sz, sz]);
        assert_finite(&blended)?;
        Ok(())
    }

    fn model_unet_forward(
        model: &MuseTalk,
        latent: &Tensor,
        ts: &Tensor,
        audio: &Tensor,
    ) -> Result<Tensor> {
        model.unet_forward(latent, ts, audio)
    }

    #[test]
    fn default_sd_v1_4_config_constructs() -> Result<()> {
        let dev = Device::Cpu;
        let dtype = DType::F32;
        let cfg = MuseTalkConfig::default();
        let model = MuseTalk::new(
            cfg.clone(),
            random_vb(dtype, &dev),
            random_vb(dtype, &dev),
            &dev,
            dtype,
        )?;
        assert_eq!(model.cross_attention_dim(), 384);
        assert_eq!(model.latent_size(), 32);
        assert_eq!(model.resized_img(), 256);
        Ok(())
    }
}
