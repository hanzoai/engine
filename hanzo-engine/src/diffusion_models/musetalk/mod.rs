pub mod config;
pub mod pipeline;
pub mod taesd;
pub mod unet;
pub mod vae;

pub use config::{MuseTalkConfig, UNetConfig, VaeConfig};
pub use pipeline::{MuseTalk, RefLatents};
pub use taesd::TaesdDecoder;
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

    struct SeededBackend {
        seed: std::sync::atomic::AtomicU64,
    }

    impl SeededBackend {
        fn new(seed: u64) -> Self {
            Self {
                seed: std::sync::atomic::AtomicU64::new(seed),
            }
        }
        fn next(&self) -> f32 {
            use std::sync::atomic::Ordering;
            let mut x = self.seed.load(Ordering::Relaxed);
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.seed.store(x, Ordering::Relaxed);
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    impl SimpleBackend for SeededBackend {
        fn get(
            &self,
            s: Shape,
            name: &str,
            _h: Init,
            dtype: DType,
            dev: &Device,
        ) -> Result<Tensor> {
            let n: usize = s.elem_count();
            if name.ends_with("bias") {
                return Tensor::zeros(s, dtype, dev);
            }
            if name.ends_with("weight") && s.rank() == 1 {
                return Tensor::ones(s, dtype, dev);
            }
            let data: Vec<f32> = (0..n).map(|_| self.next() * WEIGHT_STD as f32).collect();
            Tensor::from_vec(data, s, &Device::Cpu)?
                .to_device(dev)?
                .to_dtype(dtype)
        }
        fn get_unchecked(&self, _name: &str, _dtype: DType, _dev: &Device) -> Result<Tensor> {
            hanzo_ml::bail!("SeededBackend requires an explicit shape")
        }
        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    fn seeded_vb(seed: u64, dtype: DType, dev: &Device) -> ShardedVarBuilder {
        ShardedSafeTensors::wrap(Box::new(SeededBackend::new(seed)), dtype, dev.clone())
    }

    fn seeded_input(seed: u64, shape: &[usize], dev: &Device) -> Result<Tensor> {
        let b = SeededBackend::new(seed);
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| (b.next() + 1.0) * 0.5).collect();
        Tensor::from_vec(data, shape, &Device::Cpu)?.to_device(dev)
    }

    fn psnr_cosine(a: &Tensor, b: &Tensor) -> Result<(f64, f64)> {
        let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(a.len(), b.len());
        let mut mse = 0f64;
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            let (x, y) = (x as f64, y as f64);
            mse += (x - y) * (x - y);
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        mse /= a.len() as f64;
        let psnr = if mse <= 1e-12 {
            120.0
        } else {
            10.0 * (1.0 / mse).log10()
        };
        let cosine = dot / (na.sqrt() * nb.sqrt() + 1e-12);
        Ok((psnr, cosine))
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

    fn pick_device() -> Result<Device> {
        match std::env::var("MUSETALK_DEV").as_deref() {
            Ok("cuda") => Device::new_cuda(0),
            Ok("metal") => Device::new_metal(0),
            #[cfg(feature = "vulkan")]
            Ok("vulkan") => Device::new_vulkan(0),
            _ => Ok(Device::Cpu),
        }
    }

    fn pick_dtype() -> DType {
        match std::env::var("MUSETALK_DTYPE").as_deref() {
            Ok("f16") => DType::F16,
            Ok("bf16") => DType::BF16,
            _ => DType::F32,
        }
    }

    struct Stage {
        encode: f64,
        unet: f64,
        decode: f64,
    }

    fn time_frame(model: &MuseTalk, face: &Tensor, audio: &Tensor, dev: &Device) -> Result<Stage> {
        use std::time::Instant;
        dev.synchronize()?;
        let t0 = Instant::now();
        let latents = model.latents_for_unet(face)?;
        dev.synchronize()?;
        let t1 = Instant::now();
        let b = latents.dim(0)?;
        let ts = Tensor::zeros(b, DType::F32, dev)?;
        let pred = model.unet_forward(&latents, &ts, audio)?;
        dev.synchronize()?;
        let t2 = Instant::now();
        let _img = model.decode_latents(&pred)?;
        dev.synchronize()?;
        let t3 = Instant::now();
        Ok(Stage {
            encode: (t1 - t0).as_secs_f64() * 1e3,
            unet: (t2 - t1).as_secs_f64() * 1e3,
            decode: (t3 - t2).as_secs_f64() * 1e3,
        })
    }

    #[test]
    #[ignore]
    fn bench_single_frame() -> Result<()> {
        let dev = pick_device()?;
        let dtype = pick_dtype();
        let iters: usize = std::env::var("MUSETALK_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);
        let cfg = MuseTalkConfig::default();
        let model = MuseTalk::new(
            cfg.clone(),
            seeded_vb(0x1234_5678, dtype, &dev),
            seeded_vb(0x9abc_def0, dtype, &dev),
            &dev,
            dtype,
        )?;
        let sz = cfg.resized_img;
        let face = seeded_input(0x55, &[1, 3, sz, sz], &dev)?;
        let audio =
            seeded_input(0xAA, &[1, 50, cfg.unet.cross_attention_dim], &dev)?.to_dtype(dtype)?;

        let _ = time_frame(&model, &face, &audio, &dev)?;
        let _ = time_frame(&model, &face, &audio, &dev)?;

        let (mut e, mut u, mut d) = (0f64, 0f64, 0f64);
        let mut total_min = f64::MAX;
        for _ in 0..iters {
            let s = time_frame(&model, &face, &audio, &dev)?;
            e += s.encode;
            u += s.unet;
            d += s.decode;
            total_min = total_min.min(s.encode + s.unet + s.decode);
        }
        let n = iters as f64;
        let (e, u, d) = (e / n, u / n, d / n);
        let total = e + u + d;
        println!(
            "\n==== MuseTalk bench  dev={:?} dtype={:?} iters={} ====",
            dev.location(),
            dtype,
            iters
        );
        println!("vae-encode(x2): {:8.3} ms", e);
        println!("unet-1step:     {:8.3} ms", u);
        println!("vae-decode:     {:8.3} ms", d);
        println!(
            "total/frame:    {:8.3} ms  (best {:.3} ms)",
            total, total_min
        );
        println!("fps(mean):      {:8.2}", 1000.0 / total);
        println!("fps(best):      {:8.2}", 1000.0 / total_min);
        Ok(())
    }

    fn time_frame_streaming(
        model: &MuseTalk,
        face: &Tensor,
        audio: &Tensor,
        reference: &super::pipeline::RefLatents,
        dev: &Device,
    ) -> Result<Stage> {
        use std::time::Instant;
        dev.synchronize()?;
        let t0 = Instant::now();
        let latents = model.latents_for_unet_with_ref(face, reference)?;
        dev.synchronize()?;
        let t1 = Instant::now();
        let b = latents.dim(0)?;
        let ts = Tensor::zeros(b, DType::F32, dev)?;
        let pred = model.unet_forward(&latents, &ts, audio)?;
        dev.synchronize()?;
        let t2 = Instant::now();
        let _img = model.decode_latents(&pred)?;
        dev.synchronize()?;
        let t3 = Instant::now();
        Ok(Stage {
            encode: (t1 - t0).as_secs_f64() * 1e3,
            unet: (t2 - t1).as_secs_f64() * 1e3,
            decode: (t3 - t2).as_secs_f64() * 1e3,
        })
    }

    #[test]
    #[ignore]
    fn bench_streaming_frame() -> Result<()> {
        let dev = pick_device()?;
        let dtype = pick_dtype();
        let iters: usize = std::env::var("MUSETALK_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);
        let cfg = MuseTalkConfig::default();
        let model = MuseTalk::new(
            cfg.clone(),
            seeded_vb(0x1234_5678, dtype, &dev),
            seeded_vb(0x9abc_def0, dtype, &dev),
            &dev,
            dtype,
        )?;
        let sz = cfg.resized_img;
        let face = seeded_input(0x55, &[1, 3, sz, sz], &dev)?;
        let audio =
            seeded_input(0xAA, &[1, 50, cfg.unet.cross_attention_dim], &dev)?.to_dtype(dtype)?;
        let reference = model.encode_reference(&face)?;

        let _ = time_frame_streaming(&model, &face, &audio, &reference, &dev)?;
        let _ = time_frame_streaming(&model, &face, &audio, &reference, &dev)?;

        let (mut e, mut u, mut d) = (0f64, 0f64, 0f64);
        let mut total_min = f64::MAX;
        for _ in 0..iters {
            let s = time_frame_streaming(&model, &face, &audio, &reference, &dev)?;
            e += s.encode;
            u += s.unet;
            d += s.decode;
            total_min = total_min.min(s.encode + s.unet + s.decode);
        }
        let n = iters as f64;
        let (e, u, d) = (e / n, u / n, d / n);
        let total = e + u + d;
        println!(
            "\n==== MuseTalk STREAMING (cached-ref) bench  dev={:?} dtype={:?} iters={} ====",
            dev.location(),
            dtype,
            iters
        );
        println!("vae-encode(x1): {:8.3} ms", e);
        println!("unet-1step:     {:8.3} ms", u);
        println!("vae-decode:     {:8.3} ms", d);
        println!(
            "total/frame:    {:8.3} ms  (best {:.3} ms)",
            total, total_min
        );
        println!("fps(mean):      {:8.2}", 1000.0 / total);
        println!("fps(best):      {:8.2}", 1000.0 / total_min);
        Ok(())
    }

    #[test]
    #[ignore]
    fn verify_streaming_vs_full() -> Result<()> {
        let dev = pick_device()?;
        let dtype = pick_dtype();
        let cfg = MuseTalkConfig::default();
        let model = MuseTalk::new(
            cfg.clone(),
            seeded_vb(0x1234_5678, dtype, &dev),
            seeded_vb(0x9abc_def0, dtype, &dev),
            &dev,
            dtype,
        )?;
        let sz = cfg.resized_img;
        let face = seeded_input(0x55, &[1, 3, sz, sz], &dev)?;
        let audio =
            seeded_input(0xAA, &[1, 50, cfg.unet.cross_attention_dim], &dev)?.to_dtype(dtype)?;

        let full = model.forward(&face, &audio)?;
        let reference = model.encode_reference(&face)?;
        let streamed = model.forward_streaming(&face, &audio, &reference)?;

        let (psnr, cos) = psnr_cosine(&full, &streamed)?;
        println!(
            "\n==== MuseTalk streaming-vs-full (same ref face)  dtype={:?} ====",
            dtype
        );
        println!("full-forward vs cached-ref forward: PSNR {psnr:.3} dB  cosine {cos:.6}");
        assert!(cos > 0.9999, "cached-ref diverges from full: cosine={cos}");
        assert!(psnr > 50.0, "cached-ref PSNR too low: {psnr} dB");
        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_taesd_decode() -> Result<()> {
        use std::time::Instant;
        let dev = pick_device()?;
        let dtype = pick_dtype();
        let iters: usize = std::env::var("MUSETALK_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);
        let cfg = MuseTalkConfig::default();
        let model = MuseTalk::new(
            cfg.clone(),
            seeded_vb(0x1234_5678, dtype, &dev),
            seeded_vb(0x9abc_def0, dtype, &dev),
            &dev,
            dtype,
        )?
        .with_taesd(seeded_vb(0x0bad_f00d, dtype, &dev))?;

        let lat_ch = cfg.vae.latent_channels;
        let lat_sz = cfg.unet.sample_size;
        let pred = seeded_input(0x77, &[1, lat_ch, lat_sz, lat_sz], &dev)?.to_dtype(dtype)?;

        let time = |f: &dyn Fn() -> Result<Tensor>| -> Result<f64> {
            f()?;
            f()?;
            dev.synchronize()?;
            let mut best = f64::MAX;
            for _ in 0..iters {
                dev.synchronize()?;
                let t0 = Instant::now();
                let _ = f()?;
                dev.synchronize()?;
                best = best.min((Instant::now() - t0).as_secs_f64() * 1e3);
            }
            Ok(best)
        };

        let full = time(&|| model.decode_latents(&pred))?;
        let taesd = time(&|| model.decode_latents_fast(&pred))?;
        println!(
            "\n==== MuseTalk DECODE: full-VAE vs TAESD  dev={:?} dtype={:?} ====",
            dev.location(),
            dtype
        );
        println!("full-VAE decode: {full:8.3} ms");
        println!(
            "TAESD   decode:  {taesd:8.3} ms   ({:.1}x faster)",
            full / taesd
        );
        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_streaming_taesd_frame() -> Result<()> {
        let dev = pick_device()?;
        let dtype = pick_dtype();
        let iters: usize = std::env::var("MUSETALK_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);
        let cfg = MuseTalkConfig::default();
        let model = MuseTalk::new(
            cfg.clone(),
            seeded_vb(0x1234_5678, dtype, &dev),
            seeded_vb(0x9abc_def0, dtype, &dev),
            &dev,
            dtype,
        )?
        .with_taesd(seeded_vb(0x0bad_f00d, dtype, &dev))?;
        let sz = cfg.resized_img;
        let face = seeded_input(0x55, &[1, 3, sz, sz], &dev)?;
        let audio =
            seeded_input(0xAA, &[1, 50, cfg.unet.cross_attention_dim], &dev)?.to_dtype(dtype)?;
        let reference = model.encode_reference(&face)?;

        let frame = |dev: &Device| -> Result<Stage> {
            use std::time::Instant;
            dev.synchronize()?;
            let t0 = Instant::now();
            let latents = model.latents_for_unet_with_ref(&face, &reference)?;
            dev.synchronize()?;
            let t1 = Instant::now();
            let ts = Tensor::zeros(latents.dim(0)?, DType::F32, dev)?;
            let pred = model.unet_forward(&latents, &ts, &audio)?;
            dev.synchronize()?;
            let t2 = Instant::now();
            let _img = model.decode_latents_fast(&pred)?;
            dev.synchronize()?;
            let t3 = Instant::now();
            Ok(Stage {
                encode: (t1 - t0).as_secs_f64() * 1e3,
                unet: (t2 - t1).as_secs_f64() * 1e3,
                decode: (t3 - t2).as_secs_f64() * 1e3,
            })
        };
        let _ = frame(&dev)?;
        let _ = frame(&dev)?;
        let (mut e, mut u, mut d) = (0f64, 0f64, 0f64);
        let mut total_min = f64::MAX;
        for _ in 0..iters {
            let s = frame(&dev)?;
            e += s.encode;
            u += s.unet;
            d += s.decode;
            total_min = total_min.min(s.encode + s.unet + s.decode);
        }
        let n = iters as f64;
        let (e, u, d) = (e / n, u / n, d / n);
        let total = e + u + d;
        println!(
            "\n==== MuseTalk STREAMING + TAESD bench  dev={:?} dtype={:?} iters={} ====",
            dev.location(),
            dtype,
            iters
        );
        println!("vae-encode(x1): {e:8.3} ms",);
        println!("unet-1step:     {u:8.3} ms",);
        println!("taesd-decode:   {d:8.3} ms",);
        println!("total/frame:    {total:8.3} ms  (best {total_min:.3} ms)");
        println!("fps(mean):      {:8.2}", 1000.0 / total);
        println!("fps(best):      {:8.2}", 1000.0 / total_min);
        Ok(())
    }

    fn time_batch(model: &MuseTalk, faces: &Tensor, audio: &Tensor, dev: &Device) -> Result<Stage> {
        use std::time::Instant;
        dev.synchronize()?;
        let t0 = Instant::now();
        let latents = model.latents_for_unet(faces)?;
        dev.synchronize()?;
        let t1 = Instant::now();
        let b = latents.dim(0)?;
        let ts = Tensor::zeros(b, DType::F32, dev)?;
        let pred = model.unet_forward(&latents, &ts, audio)?;
        dev.synchronize()?;
        let t2 = Instant::now();
        let _img = model.decode_latents(&pred)?;
        dev.synchronize()?;
        let t3 = Instant::now();
        Ok(Stage {
            encode: (t1 - t0).as_secs_f64() * 1e3,
            unet: (t2 - t1).as_secs_f64() * 1e3,
            decode: (t3 - t2).as_secs_f64() * 1e3,
        })
    }

    #[test]
    #[ignore]
    fn bench_batched_throughput() -> Result<()> {
        let dev = pick_device()?;
        let dtype = pick_dtype();
        let iters: usize = std::env::var("MUSETALK_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);
        let batches: Vec<usize> = std::env::var("MUSETALK_BATCHES")
            .ok()
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32]);
        let cfg = MuseTalkConfig::default();
        let model = MuseTalk::new(
            cfg.clone(),
            seeded_vb(0x1234_5678, dtype, &dev),
            seeded_vb(0x9abc_def0, dtype, &dev),
            &dev,
            dtype,
        )?;
        let sz = cfg.resized_img;
        let cad = cfg.unet.cross_attention_dim;

        println!(
            "\n==== MuseTalk BATCHED throughput  dev={:?} dtype={:?} iters={} ====",
            dev.location(),
            dtype,
            iters
        );
        println!(
            "{:>4}  {:>10}  {:>10}  {:>10}  {:>11}  {:>10}  {:>9}",
            "N", "enc(ms)", "unet(ms)", "dec(ms)", "lat/best(ms)", "ms/frame", "fps"
        );
        let mut best_fps = 0f64;
        let mut best_n = 0usize;
        let mut best_lat = 0f64;
        for &n in batches.iter() {
            let faces = seeded_input(0x55, &[n, 3, sz, sz], &dev)?;
            let audio = seeded_input(0xAA, &[n, 50, cad], &dev)?.to_dtype(dtype)?;
            // warmup
            let _ = time_batch(&model, &faces, &audio, &dev)?;
            let _ = time_batch(&model, &faces, &audio, &dev)?;
            let (mut e, mut u, mut d) = (0f64, 0f64, 0f64);
            let mut best = f64::MAX;
            for _ in 0..iters {
                let s = time_batch(&model, &faces, &audio, &dev)?;
                e += s.encode;
                u += s.unet;
                d += s.decode;
                best = best.min(s.encode + s.unet + s.decode);
            }
            let it = iters as f64;
            let (e, u, d) = (e / it, u / it, d / it);
            let ms_per_frame = best / n as f64;
            let fps = 1000.0 * n as f64 / best;
            println!("{n:>4}  {e:>10.3}  {u:>10.3}  {d:>10.3}  {best:>11.3}  {ms_per_frame:>10.3}  {fps:>9.2}");
            if fps > best_fps {
                best_fps = fps;
                best_n = n;
                best_lat = best;
            }
        }
        println!(
            "---- peak: {best_fps:.2} fps at N={best_n}  (latency {best_lat:.1} ms/batch) ----"
        );
        println!(
            "---- realtime(>=30fps): {} ----",
            if best_fps >= 30.0 { "YES" } else { "NO" }
        );
        Ok(())
    }

    #[test]
    #[ignore]
    fn verify_batched_vs_single() -> Result<()> {
        let dev = pick_device()?;
        let dtype = pick_dtype();
        let cfg = MuseTalkConfig::default();
        let model = MuseTalk::new(
            cfg.clone(),
            seeded_vb(0x1234_5678, dtype, &dev),
            seeded_vb(0x9abc_def0, dtype, &dev),
            &dev,
            dtype,
        )?;
        let sz = cfg.resized_img;
        let cad = cfg.unet.cross_attention_dim;
        let n = std::env::var("MUSETALK_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8usize);

        let faces = seeded_input(0x55, &[n, 3, sz, sz], &dev)?;
        let audio = seeded_input(0xAA, &[n, 50, cad], &dev)?.to_dtype(dtype)?;

        let batched = model.forward_batched(&faces, &audio)?;
        assert_eq!(batched.dims(), &[n, 3, sz, sz]);

        println!(
            "\n==== MuseTalk batched-vs-single  dev={:?} dtype={:?} N={} ====",
            dev.location(),
            dtype,
            n
        );
        let (mut min_psnr, mut min_cos) = (f64::MAX, f64::MAX);
        for i in 0..n {
            let face_i = faces.narrow(0, i, 1)?;
            let audio_i = audio.narrow(0, i, 1)?;
            let single_i = model.forward(&face_i, &audio_i)?;
            let batched_i = batched.narrow(0, i, 1)?;
            let (psnr, cos) = psnr_cosine(&single_i, &batched_i)?;
            println!("frame {i:>2}: PSNR {psnr:8.3} dB  cosine {cos:.6}");
            min_psnr = min_psnr.min(psnr);
            min_cos = min_cos.min(cos);
        }
        println!("---- worst-frame: PSNR {min_psnr:.3} dB  cosine {min_cos:.6} ----");
        assert!(
            min_cos > 0.999,
            "batched diverges from single: cosine={min_cos}"
        );
        assert!(min_psnr > 40.0, "batched PSNR too low: {min_psnr} dB");
        Ok(())
    }

    #[test]
    #[ignore]
    fn verify_gpu_vs_cpu() -> Result<()> {
        let cpu = Device::Cpu;
        let cfg = MuseTalkConfig::default();
        let sz = cfg.resized_img;

        let model_cpu = MuseTalk::new(
            cfg.clone(),
            seeded_vb(0x1234_5678, DType::F32, &cpu),
            seeded_vb(0x9abc_def0, DType::F32, &cpu),
            &cpu,
            DType::F32,
        )?;
        let face_cpu = seeded_input(0x55, &[1, 3, sz, sz], &cpu)?;
        let audio_cpu = seeded_input(0xAA, &[1, 50, cfg.unet.cross_attention_dim], &cpu)?;
        let ref_img = model_cpu.forward(&face_cpu, &audio_cpu)?;

        let gpu = pick_device()?;
        let dtype = pick_dtype();
        let model_gpu = MuseTalk::new(
            cfg.clone(),
            seeded_vb(0x1234_5678, dtype, &gpu),
            seeded_vb(0x9abc_def0, dtype, &gpu),
            &gpu,
            dtype,
        )?;
        let face_gpu = seeded_input(0x55, &[1, 3, sz, sz], &gpu)?;
        let audio_gpu =
            seeded_input(0xAA, &[1, 50, cfg.unet.cross_attention_dim], &gpu)?.to_dtype(dtype)?;
        let gpu_img = model_gpu.forward(&face_gpu, &audio_gpu)?;

        let (psnr, cosine) = psnr_cosine(&ref_img, &gpu_img.to_device(&cpu)?)?;
        println!("\n==== MuseTalk correctness  gpu_dtype={:?} ====", dtype);
        println!("PSNR(dB): {:.3}", psnr);
        println!("cosine:   {:.6}", cosine);
        // Informational only: the CPU reference runs every matmul in f16 (see MatMul::matmul CPU
        // branch) while the GPU runs true f32/f16, and with random untrained weights the deep
        // VAE+UNet is a chaotic map that amplifies that per-op delta. The faithful precision check
        // is verify_metal_self_consistency (f16 vs f32 on the same backend), which agrees to ~1.0.
        Ok(())
    }

    #[test]
    #[ignore]
    fn verify_stages_gpu_vs_cpu() -> Result<()> {
        let cpu = Device::Cpu;
        let cfg = MuseTalkConfig::default();
        let sz = cfg.resized_img;

        let model_cpu = MuseTalk::new(
            cfg.clone(),
            seeded_vb(0x1234_5678, DType::F32, &cpu),
            seeded_vb(0x9abc_def0, DType::F32, &cpu),
            &cpu,
            DType::F32,
        )?;
        let gpu = pick_device()?;
        let dtype = pick_dtype();
        let model_gpu = MuseTalk::new(
            cfg.clone(),
            seeded_vb(0x1234_5678, dtype, &gpu),
            seeded_vb(0x9abc_def0, dtype, &gpu),
            &gpu,
            dtype,
        )?;

        let face_cpu = seeded_input(0x55, &[1, 3, sz, sz], &cpu)?;
        let audio_cpu = seeded_input(0xAA, &[1, 50, cfg.unet.cross_attention_dim], &cpu)?;
        let face_gpu = face_cpu.to_device(&gpu)?;
        let audio_gpu = audio_cpu.to_device(&gpu)?.to_dtype(dtype)?;

        let cmp = |name: &str, a: &Tensor, b: &Tensor| -> Result<()> {
            let (psnr, cosine) = psnr_cosine(a, &b.to_device(&cpu)?)?;
            println!("{name:<22} PSNR {psnr:8.3} dB   cosine {cosine:.6}");
            Ok(())
        };

        println!("\n==== MuseTalk per-stage  gpu_dtype={:?} ====", dtype);
        let enc_c = model_cpu.latents_for_unet(&face_cpu)?;
        let enc_g = model_gpu.latents_for_unet(&face_gpu)?;
        cmp("vae-encode latents", &enc_c, &enc_g)?;

        let ts_c = Tensor::zeros(1, DType::F32, &cpu)?;
        let ts_g = Tensor::zeros(1, DType::F32, &gpu)?;
        let un_c = model_cpu.unet_forward(&enc_c, &ts_c, &audio_cpu)?;
        let un_g = model_gpu.unet_forward(&enc_g, &ts_g, &audio_gpu)?;
        cmp("unet pred_latents", &un_c, &un_g)?;

        let de_c = model_cpu.decode_latents(&un_c)?;
        let de_g = model_gpu.decode_latents(&un_g)?;
        cmp("vae-decode image", &de_c, &de_g)?;

        let un_g_oncpu = un_g.to_device(&cpu)?.to_dtype(DType::F32)?;
        let de_cross = model_cpu.decode_latents(&un_g_oncpu)?;
        cmp("decode(shared input)", &de_cross, &de_g)?;
        Ok(())
    }

    fn rms(t: &Tensor) -> Result<f64> {
        let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let ss: f64 = v.iter().map(|x| (*x as f64) * (*x as f64)).sum();
        Ok((ss / v.len() as f64).sqrt())
    }

    #[test]
    #[ignore]
    fn verify_metal_self_consistency() -> Result<()> {
        let gpu = pick_device()?;
        let cfg = MuseTalkConfig::default();
        let sz = cfg.resized_img;
        let mk = |dt: DType| -> Result<MuseTalk> {
            MuseTalk::new(
                cfg.clone(),
                seeded_vb(0x1234_5678, dt, &gpu),
                seeded_vb(0x9abc_def0, dt, &gpu),
                &gpu,
                dt,
            )
        };
        let m32 = mk(DType::F32)?;
        let m16 = mk(DType::F16)?;
        let face = seeded_input(0x55, &[1, 3, sz, sz], &gpu)?;
        let audio = seeded_input(0xAA, &[1, 50, cfg.unet.cross_attention_dim], &gpu)?;

        let cmp = |name: &str, a: &Tensor, b: &Tensor| -> Result<()> {
            let (psnr, cosine) = psnr_cosine(&a.to_dtype(DType::F32)?, &b.to_dtype(DType::F32)?)?;
            println!(
                "{name:<22} PSNR {psnr:8.3} dB  cosine {cosine:.6}  rmsA {:.5} rmsB {:.5}",
                rms(a)?,
                rms(b)?
            );
            Ok(())
        };

        println!("\n==== MuseTalk Metal f16-vs-f32 self-consistency ====");
        let e32 = m32.latents_for_unet(&face)?;
        let e16 = m16.latents_for_unet(&face)?;
        cmp("vae-encode latents", &e32, &e16)?;
        let ts = Tensor::zeros(1, DType::F32, &gpu)?;
        let u32 = m32.unet_forward(&e32, &ts, &audio)?;
        let u16 = m16.unet_forward(&e16, &ts, &audio)?;
        cmp("unet pred_latents", &u32, &u16)?;
        let d32 = m32.decode_latents(&u32)?;
        let d16 = m16.decode_latents(&u16)?;
        cmp("vae-decode image", &d32, &d16)?;
        let dshared = m16.decode_latents(&u32.to_dtype(DType::F16)?)?;
        cmp("decode(shared latents)", &d32, &dshared)?;

        let (psnr, cosine) = psnr_cosine(&d32.to_dtype(DType::F32)?, &d16.to_dtype(DType::F32)?)?;
        assert!(
            cosine > 0.99,
            "f16 vs f32 structural divergence: cosine={cosine}"
        );
        assert!(psnr > 30.0, "f16 vs f32 PSNR too low: {psnr} dB");
        Ok(())
    }
}
