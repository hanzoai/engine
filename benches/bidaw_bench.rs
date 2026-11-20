use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hanzo::scheduler::{BidawScheduler, KVCacheLocation, RequestMetadata};
use std::time::{Duration, Instant};

fn bench_bidaw_scheduling(c: &mut Criterion) {
    let scheduler = BidawScheduler::new(1.0, 5.0);
    let now = Instant::now();

    let mut requests = Vec::new();
    for i in 0..100 {
        let loc = if i % 3 == 0 {
            KVCacheLocation::GPU
        } else if i % 3 == 1 {
            KVCacheLocation::Host
        } else {
            KVCacheLocation::Remote
        };

        requests.push(RequestMetadata {
            id: format!("req_{}", i),
            arrival_time: now - Duration::from_millis(i * 10),
            session_id: format!("sess_{}", i),
            tokens_in_cache: 512,
            location: loc,
        });
    }

    c.bench_function("bidaw_reorder_100_reqs", |b| {
        b.iter(|| {
            let mut q = requests.clone();
            scheduler.schedule(black_box(&mut q));
        })
    });
}

criterion_group!(benches, bench_bidaw_scheduling);
criterion_main!(benches);
