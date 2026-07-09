// Standalone ring-collective numerics check. Run one process per rank with RING_CONFIG set.
// Rank r contributes (r+1) to an all-reduce (global sum must be sum_{r}(r+1)) and [r] to an
// all-gather (result must be [0,1,...,world-1]). Validates any world_size >= 2.
use std::sync::Arc;

use hanzo_ml::{Device, Result, Tensor};
use hanzo_quant::distributed::AllGather;
use hanzo_quant::{Comm, Id, RingConfig, SumAllReduce};

fn main() -> Result<()> {
    let cfg = RingConfig::load();
    let (rank, world) = (cfg.rank, cfg.world_size);
    let dev = Device::Cpu;
    let comm = Arc::new(Comm::from_device(Id::new(), &dev, rank, world).unwrap());

    let n = 8usize;
    let local = vec![(rank + 1) as f32; n];
    let xs = Tensor::from_slice(&local, (n,), &dev)?;
    let reduced = SumAllReduce::new(&comm)
        .sum_all_reduce(&xs)?
        .to_vec1::<f32>()?;
    let expected = (1..=world).sum::<usize>() as f32;
    assert!(
        reduced.iter().all(|&v| (v - expected).abs() < 1e-3),
        "rank {rank}: all_reduce got {reduced:?}, expected all {expected}"
    );
    println!("[rank {rank}/{world}] sum_all_reduce OK -> {expected}");

    let g = Tensor::from_slice(&[rank as f32], (1,), &dev)?;
    let gathered = AllGather::new(&comm, 0).all_gather(&g)?.to_vec1::<f32>()?;
    let want: Vec<f32> = (0..world).map(|r| r as f32).collect();
    assert_eq!(gathered, want, "rank {rank}: all_gather mismatch");
    println!("[rank {rank}/{world}] all_gather OK -> {gathered:?}");
    Ok(())
}
