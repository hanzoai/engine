//! DeltaSoup Byzantine-resistance demo.
//!
//! Generates 7 contributor deltas (5 honest, 2 Byzantine) and verifies that
//! TrimmedMean drops the outliers while plain Mean is corrupted by them.
//!
//! ```sh
//! cargo run --release --example deltasoup_byzantine
//! ```

use candle_core::{Device, Tensor};
use hanzo_quant::{aggregate, Method};

fn t(v: &[f32]) -> Tensor {
    Tensor::from_vec(v.to_vec(), v.len(), &Device::Cpu).unwrap()
}

fn main() -> anyhow::Result<()> {
    // Honest contributors: small positive deltas near 0.1.
    // Byzantine: huge values designed to drag any naive aggregator.
    let deltas = vec![
        t(&[0.10, 0.10, 0.10, 0.10]),    // honest
        t(&[0.09, 0.11, 0.10, 0.10]),    // honest
        t(&[0.10, 0.10, 0.09, 0.11]),    // honest
        t(&[0.11, 0.10, 0.10, 0.10]),    // honest
        t(&[0.10, 0.10, 0.11, 0.09]),    // honest
        t(&[100.0, 100.0, 100.0, 100.0]), // Byzantine
        t(&[-100.0, -100.0, -100.0, -100.0]), // Byzantine
    ];

    let mean: Vec<f32> = aggregate(Method::Mean, &deltas)?.to_vec1()?;
    let trimmed: Vec<f32> =
        aggregate(Method::TrimmedMean { trim: 1.0 / 7.0 }, &deltas)?.to_vec1()?;
    let median: Vec<f32> = aggregate(Method::Median, &deltas)?.to_vec1()?;
    let krum: Vec<f32> = aggregate(Method::Krum { f: 2 }, &deltas)?.to_vec1()?;

    println!("Byzantine resistance demo — 5 honest, 2 outliers");
    println!("  honest target ~ [0.10, 0.10, 0.10, 0.10]");
    println!("  Mean         : {:?}", mean);
    println!("  TrimmedMean  : {:?}", trimmed);
    println!("  Median       : {:?}", median);
    println!("  Krum  (f=2)  : {:?}", krum);

    // Plain mean is wrecked: 2 Byzantines balance to ~0 contribution per coord,
    // but the average is (sum_honest + 0) / 7 -> still pulled away from 0.1.
    // TrimmedMean drops the +100 and -100 outliers and lands close to 0.1.
    let trim_err: f32 = trimmed.iter().map(|x| (x - 0.10).abs()).sum::<f32>() / 4.0;
    let median_err: f32 = median.iter().map(|x| (x - 0.10).abs()).sum::<f32>() / 4.0;
    assert!(trim_err < 0.05, "TrimmedMean should be near 0.10, got mean err {}", trim_err);
    assert!(median_err < 0.05, "Median should be near 0.10, got mean err {}", median_err);

    println!("OK: trimmed-mean and median both reject the Byzantines.");
    Ok(())
}
