//! Oasis world-model rollout: prompt image + Minecraft action stream -> generated frames as PNGs.
//!
//! Run (CPU):
//!   cargo run -p hanzo-engine --release --example oasis_generate -- \
//!       --frames 32 --steps 10 --out /data/worldmodel-tests/run1
//!
//! Assemble a video from the PNGs afterward (frames are `frame_%03d.png`):
//!   ffmpeg -y -framerate 20 -i out/frame_%03d.png -pix_fmt yuv420p out/rollout.mp4
//!   ffmpeg -y -framerate 20 -i out/frame_%03d.png out/rollout.gif

use std::path::PathBuf;

use anyhow::{Context, Result};
use hanzo_engine::diffusion_models::oasis::{
    frames_to_images, image_to_tensor, sampling::SampleParams, WorldModel,
};
use hanzo_ml::{DType, Device};

const WDIR: &str = "/home/z/work/hanzo/oasis-weights";
const REF: &str = "/home/z/work/hanzo/open-oasis-ref";

struct Args {
    vae: PathBuf,
    dit: PathBuf,
    image: PathBuf,
    actions: PathBuf,
    out: PathBuf,
    frames: usize,
    steps: usize,
    seed: u64,
    device: String,
}

fn parse() -> Args {
    let mut a = Args {
        vae: PathBuf::from(format!("{WDIR}/vit-l-20.safetensors")),
        dit: PathBuf::from(format!("{WDIR}/oasis500m.safetensors")),
        image: PathBuf::from(format!("{REF}/sample_data/sample_image_0.png")),
        actions: PathBuf::from(format!("{WDIR}/actions.safetensors")),
        out: PathBuf::from(format!("{WDIR}/rollout")),
        frames: 32,
        steps: 10,
        seed: 0,
        device: "cpu".to_string(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().expect("missing value");
        match k.as_str() {
            "--vae" => a.vae = v().into(),
            "--dit" => a.dit = v().into(),
            "--image" => a.image = v().into(),
            "--actions" => a.actions = v().into(),
            "--out" => a.out = v().into(),
            "--frames" => a.frames = v().parse().unwrap(),
            "--steps" => a.steps = v().parse().unwrap(),
            "--seed" => a.seed = v().parse().unwrap(),
            "--device" => a.device = v(),
            other => panic!("unknown arg {other}"),
        }
    }
    a
}

fn device(spec: &str) -> Result<Device> {
    if spec == "cpu" {
        return Ok(Device::Cpu);
    }
    let ord = spec
        .strip_prefix("cuda:")
        .or_else(|| spec.strip_prefix("cuda"))
        .unwrap_or("0");
    let ord: usize = if ord.is_empty() { 0 } else { ord.parse()? };
    Ok(Device::new_cuda(ord)?)
}

fn main() -> Result<()> {
    let args = parse();
    let dev = device(&args.device)?;
    let dtype = if matches!(dev, Device::Cpu) {
        DType::F32
    } else {
        DType::F16
    };
    println!("loading world model ({:?}, {dtype:?})", args.device);
    let wm = WorldModel::load(args.vae.clone(), args.dit.clone(), dtype, &dev)?;

    let img = image::open(&args.image).with_context(|| format!("open {}", args.image.display()))?;
    let prompt = image_to_tensor(&img, &dev)?; // [1,3,360,640]

    let acts_all = hanzo_ml::safetensors::load(&args.actions, &dev)?;
    let acts = acts_all
        .get("actions")
        .context("actions.safetensors missing 'actions'")?
        .narrow(1, 0, args.frames)?
        .to_dtype(DType::F32)?;

    let params = SampleParams {
        total_frames: args.frames,
        ddim_steps: args.steps,
        seed: args.seed,
    };
    println!("rollout: {} frames, {} ddim steps", args.frames, args.steps);
    let t0 = std::time::Instant::now();
    let out = wm.generate(&prompt, &acts, &params)?;
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "generated {} frames in {dt:.1}s ({:.2} fps)",
        args.frames,
        args.frames as f64 / dt
    );

    std::fs::create_dir_all(&args.out)?;
    for (i, frame) in frames_to_images(&out)?.into_iter().enumerate() {
        frame.save(args.out.join(format!("frame_{i:03}.png")))?;
    }
    println!("saved {} frames to {}", args.frames, args.out.display());
    Ok(())
}
