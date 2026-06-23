// Micro-timing: isolate the qmatvec_q4k kernel cost from the model pipeline. Times many back-to-back
// matvec_q4k launches on production decode shapes and compares against matvec_q8_0 (the proven-fast
// path) on the equivalent logical matrix, plus computes the effective GB/s each sustains. This tells
// us whether a slow end-to-end Q4_K decode is the KERNEL or the surrounding per-call Rust work.
#![cfg(feature = "rocm")]

use half::{bf16, f16};
use hanzo_ml::backend::BackendDevice;
use hanzo_ml::{RocmDevice, RocmStorage};
use std::io::Write;
use std::time::Instant;

fn val(i: usize) -> f32 {
    let x = ((i.wrapping_mul(2654435761) >> 8) & 0xffff) as f32 / 65535.0;
    (x - 0.5) * 6.0
}

fn time_q4k(dev: &RocmDevice, n: usize, k: usize, iters: usize) -> (f64, f64) {
    let nblk = k / 256;
    let xb: Vec<bf16> = (0..k).map(|i| bf16::from_f32(val(i))).collect();
    let xst: RocmStorage = dev.storage_from_slice(&xb).expect("x");
    let mut wq: Vec<u8> = vec![0u8; n * nblk * 144];
    for (i, b) in wq.iter_mut().enumerate() {
        *b = ((i * 131 + 7) % 256) as u8;
    }
    let wst = dev.storage_from_slice(&wq).expect("w");
    // warmup (also forces kernel compile/module-load out of the timed region)
    for _ in 0..5 {
        let _ = dev.matvec_q4k(&wst, &xst, n, k).expect("q4k");
    }
    dev.synchronize().expect("sync");
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = dev.matvec_q4k(&wst, &xst, n, k).expect("q4k");
    }
    dev.synchronize().expect("sync");
    let secs = t0.elapsed().as_secs_f64();
    let bytes_per = (n * nblk * 144) as f64; // weight traffic per launch
    let gbps = (bytes_per * iters as f64) / secs / 1e9;
    (secs / iters as f64 * 1e6, gbps) // (us/launch, GB/s)
}

fn time_q8(dev: &RocmDevice, n: usize, k: usize, iters: usize) -> (f64, f64) {
    let nblk = k / 32;
    let xb: Vec<bf16> = (0..k).map(|i| bf16::from_f32(val(i))).collect();
    let xst: RocmStorage = dev.storage_from_slice(&xb).expect("x");
    let wq: Vec<u8> = vec![1u8; n * nblk * 34];
    let wst = dev.storage_from_slice(&wq).expect("w");
    for _ in 0..5 {
        let _ = dev.matvec_q8_0(&wst, &xst, n, k).expect("q8");
    }
    dev.synchronize().expect("sync");
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = dev.matvec_q8_0(&wst, &xst, n, k).expect("q8");
    }
    dev.synchronize().expect("sync");
    let secs = t0.elapsed().as_secs_f64();
    let bytes_per = (n * nblk * 34) as f64;
    let gbps = (bytes_per * iters as f64) / secs / 1e9;
    (secs / iters as f64 * 1e6, gbps)
}

#[test]
fn qmatvec_q4k_perf() {
    let _ = f16::from_f32(0.0);
    let mut log = String::new();
    let dev = RocmDevice::new(0).expect("rocm device");
    let iters = 300;
    for &(n, k) in &[
        (5120usize, 5120usize),
        (17408, 5120),
        (1024, 5120),
        (151936, 5120),
    ] {
        let (us4, gb4) = time_q4k(&dev, n, k, iters);
        let (us8, gb8) = time_q8(&dev, n, k, iters);
        log.push_str(&format!(
            "n={n} k={k}: q4k {us4:.1} us/launch {gb4:.1} GB/s | q8_0 {us8:.1} us/launch {gb8:.1} GB/s | q4k bytes/launch={:.2} MiB q8 bytes/launch={:.2} MiB\n",
            (n * (k / 256) * 144) as f64 / 1048576.0,
            (n * (k / 32) * 34) as f64 / 1048576.0,
        ));
    }
    let _ = std::fs::File::create("C:\\qmatvec-q4k-perf.txt")
        .and_then(|mut f| f.write_all(log.as_bytes()));
    eprintln!("{log}");
}
