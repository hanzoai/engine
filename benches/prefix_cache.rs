use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hanzo::cache::PrefixCacheManager;

fn bench_lookup(c: &mut Criterion) {
    let mut manager = PrefixCacheManager::new();
    let key = "model_4b:tenant_1:hash_abc".to_string();
    
    c.bench_function("prefix_cache_lookup", |b| {
        b.iter(|| manager.lookup(black_box(&key)))
    });
}

criterion_group!(benches, bench_lookup);
criterion_main!(benches);
