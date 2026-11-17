use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hanzo::io::prewarm::ModelPrewarmer;
use std::fs::File;

fn bench_prewarm(c: &mut Criterion) {
    let file = File::open("calibration_data/calibration_datav3.txt").unwrap();
    let metadata = file.metadata().unwrap();
    let size = metadata.len() as usize;

    c.bench_function("prewarm_io_latency", |b| {
        b.iter(|| {
            ModelPrewarmer::prewarm_file(black_box(&file), black_box(size)).unwrap();
        })
    });
}

criterion_group!(benches, bench_prewarm);
criterion_main!(benches);
