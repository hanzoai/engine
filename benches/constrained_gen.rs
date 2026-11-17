use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::sync::Arc;
use hanzo::logits::xgrammar::XGrammarLogitProcessor;

fn bench_xgrammar_overhead(c: &mut Criterion) {
    let vocab = Arc::new((0..32000).map(|i| format!("token_{}", i)).collect::<Vec<_>>());
    let schema = r#"\{"type": "object", "properties": \{"name": \{"type": "string"\}\}\}"#;
    
    let mut processor = XGrammarLogitProcessor::new(schema, vocab.clone());
    let mut logits = vec![0.0f32; 32000];

    c.benchmark_group("Constrained Generation")
        .bench_function("XGrammar-2 Apply (Cached)", |b| {
            // Pre-warm cache
            processor.apply(&mut logits);
            b.iter(|| {
                processor.apply(black_box(&mut logits));
            })
        });
}

criterion_group!(benches, bench_xgrammar_overhead);
criterion_main!(benches);
