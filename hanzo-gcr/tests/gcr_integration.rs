// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Integration, concurrency, fault-injection, and crash-safety tests for `hanzo-gcr`.

#![forbid(unsafe_code)]

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use hanzo_gcr::bench::{run_full_benchmark, BenchOptions};
use hanzo_gcr::store::StoredChunkInfo;
use hanzo_gcr::{
    AtomicDirtyBitmap, CheckpointConfig, CheckpointEngine, CheckpointMode, CheckpointStore, Digest,
    FsCheckpointStore, GcrError, GcrResult, GpuMemorySpace, IoStats, MemoryCheckpointStore,
    Namespace, PageRegion, PageSize, SharedWeightBase,
};

fn build_weights(page_size: PageSize, count: usize, seed: u8) -> SharedWeightBase {
    let ps = page_size.bytes().unwrap();
    let mut pages = Vec::with_capacity(count);
    for i in 0..count {
        let mut p = vec![0u8; ps];
        for (j, b) in p.iter_mut().enumerate() {
            *b = seed
                .wrapping_add((i * 29) as u8)
                .wrapping_add((j * 13) as u8);
        }
        pages.push(p);
    }
    SharedWeightBase::new(page_size, pages).unwrap()
}

#[test]
fn supports_4kib_64kib_and_2mib_logical_page_sizes() {
    let store = MemoryCheckpointStore::new();
    for (idx, page_size) in [PageSize::KiB4, PageSize::KiB64, PageSize::MiB2]
        .iter()
        .enumerate()
    {
        let weights = build_weights(*page_size, 2, (idx as u8) + 1);
        let ns = Namespace::new("hanzo-sizes", format!("job-{idx}")).unwrap();
        let mut vram = GpuMemorySpace::new(
            ns,
            weights,
            vec![PageRegion::KvCache, PageRegion::LoRaAdapter],
        )
        .unwrap();

        let ps = page_size.bytes().unwrap();
        vram.write_page_range(2, 0, b"kv-header").unwrap();
        vram.write_page_range(3, ps - 8, b"lora-end").unwrap();

        let engine = CheckpointEngine::new(&store, CheckpointConfig::default());
        let r0 = engine.checkpoint(&vram, "p0", None).unwrap();
        assert_eq!(r0.manifest.page_size as usize, ps);

        vram.write_page_range(3, 128, b"lora-delta").unwrap();
        let r1 = engine.checkpoint(&vram, "p1", Some("p0")).unwrap();
        assert_eq!(r1.dirty_pages_drained, 1);
        assert_eq!(r1.manifest.entries.len(), 1);

        let mut cold = vram.cold_boot_clone_layout().unwrap();
        let restored = engine.restore(&mut cold, "p1").unwrap();
        assert!(restored.bit_exact_verified);
        assert_eq!(cold.state_digest(), vram.state_digest());
    }
}

#[test]
fn atomic_dirty_bitmap_concurrent_writers_never_lose_bits() {
    let bitmap = Arc::new(AtomicDirtyBitmap::new(1024));
    let drained_total = Arc::new(std::sync::Mutex::new(HashSet::new()));

    std::thread::scope(|s| {
        for t in 0..8usize {
            let bm = Arc::clone(&bitmap);
            s.spawn(move || {
                for p in (t * 128)..((t + 1) * 128) {
                    bm.mark_dirty(p).unwrap();
                }
            });
        }
        let bm_drain = Arc::clone(&bitmap);
        let acc = Arc::clone(&drained_total);
        s.spawn(move || {
            for _ in 0..32 {
                let batch = bm_drain.take_and_clear_all();
                let mut guard = acc.lock().unwrap();
                guard.extend(batch);
                std::thread::yield_now();
            }
        });
    });

    // Final drain after all writers complete must capture every remaining bit.
    let remaining = bitmap.take_and_clear_all();
    let mut guard = drained_total.lock().unwrap();
    guard.extend(remaining);
    assert_eq!(guard.len(), 1024);
    assert_eq!(bitmap.dirty_count(), 0);
}

#[test]
fn concurrent_multi_tenant_fs_store_has_zero_cross_tenant_leak() {
    let dir = std::env::temp_dir().join(format!(
        "hanzo_gcr_mt_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let store = Arc::new(FsCheckpointStore::new(&dir, false).unwrap());
    let weights = build_weights(PageSize::KiB4, 4, 0x77);

    std::thread::scope(|s| {
        for t in 0..6usize {
            let store = Arc::clone(&store);
            let weights = weights.clone();
            s.spawn(move || {
                let ns = Namespace::new(format!("tenant-{t}"), "agent-job").unwrap();
                let mut vram =
                    GpuMemorySpace::new(ns, weights, vec![PageRegion::KvCache; 8]).unwrap();
                let engine = CheckpointEngine::new(
                    store.as_ref(),
                    CheckpointConfig {
                        mode: CheckpointMode::GcrIncremental,
                        max_chain_depth: 2,
                        restore_workers: 2,
                        ..CheckpointConfig::default()
                    },
                );

                let mut prev: Option<String> = None;
                for step in 0..5usize {
                    let page_id = 4 + (step % 8);
                    let marker = format!("tenant-{t}-step-{step}");
                    vram.write_page_range(page_id, 0, marker.as_bytes())
                        .unwrap();
                    let cid = format!("ckpt-{step}");
                    engine.checkpoint(&vram, &cid, prev.as_deref()).unwrap();
                    prev = Some(cid);
                }

                // Own restore succeeds bit-exact.
                let mut cold = vram.cold_boot_clone_layout().unwrap();
                let restored = engine.restore(&mut cold, "ckpt-4").unwrap();
                assert!(restored.bit_exact_verified);
                assert_eq!(cold.state_digest(), vram.state_digest());

                // Attempt to restore into another tenant's namespace MUST fail.
                let foreign_ns =
                    Namespace::new(format!("tenant-{}", (t + 1) % 6), "agent-job").unwrap();
                let mut foreign_vram = GpuMemorySpace::new(
                    foreign_ns,
                    vram.shared_weights().clone(),
                    vec![PageRegion::KvCache; 8],
                )
                .unwrap();
                // Even if both tenants have a checkpoint named "ckpt-4", foreign_vram gets its OWN
                // data (or not found yet), NEVER tenant `t`'s marker!
                if engine.restore(&mut foreign_vram, "ckpt-4").is_ok() {
                    let p4 = foreign_vram.read_page(4).unwrap();
                    let expected_foreign_prefix = format!("tenant-{}", (t + 1) % 6);
                    assert!(p4.starts_with(expected_foreign_prefix.as_bytes()));
                }
            });
        }
    });

    let _ = std::fs::remove_dir_all(&dir);
}

/// Fault-injecting wrapper store that fails after `fail_after_writes` chunk writes to verify
/// dirty-bitmap rollback on mid-checkpoint crashes/errors.
struct FailingStore {
    inner: MemoryCheckpointStore,
    writes_seen: AtomicUsize,
    fail_after_writes: usize,
}

impl CheckpointStore for FailingStore {
    fn put_chunk(
        &self,
        namespace: &Namespace,
        raw_page: &[u8],
        allow_zero_rle: bool,
        dedup: bool,
    ) -> GcrResult<StoredChunkInfo> {
        let n = self.writes_seen.fetch_add(1, Ordering::SeqCst);
        if n >= self.fail_after_writes {
            return Err(GcrError::Io(
                "simulated mid-checkpoint NVMe write fault".to_string(),
            ));
        }
        self.inner
            .put_chunk(namespace, raw_page, allow_zero_rle, dedup)
    }

    fn get_chunk(
        &self,
        namespace: &Namespace,
        digest: &Digest,
        expected_page_size: usize,
    ) -> GcrResult<Vec<u8>> {
        self.inner.get_chunk(namespace, digest, expected_page_size)
    }

    fn put_manifest(
        &self,
        namespace: &Namespace,
        checkpoint_id: &str,
        envelope: &[u8],
    ) -> GcrResult<()> {
        self.inner.put_manifest(namespace, checkpoint_id, envelope)
    }

    fn get_manifest(&self, namespace: &Namespace, checkpoint_id: &str) -> GcrResult<Vec<u8>> {
        self.inner.get_manifest(namespace, checkpoint_id)
    }

    fn gc_unreferenced_chunks(
        &self,
        namespace: &Namespace,
        live_digests: &HashSet<Digest>,
    ) -> GcrResult<(u64, u64)> {
        self.inner.gc_unreferenced_chunks(namespace, live_digests)
    }

    fn namespace_storage_bytes(&self, namespace: &Namespace) -> GcrResult<u64> {
        self.inner.namespace_storage_bytes(namespace)
    }

    fn io_stats(&self) -> IoStats {
        self.inner.io_stats()
    }
}

#[test]
fn failed_checkpoint_restores_dirty_bitmap_for_safe_retry() {
    let store = FailingStore {
        inner: MemoryCheckpointStore::new(),
        writes_seen: AtomicUsize::new(0),
        fail_after_writes: 5, // Base has 4 mutable pages; delta will fail on its 2nd dirty page!
    };
    let weights = build_weights(PageSize::Custom(512), 2, 0x11);
    let ns = Namespace::new("acme", "crash-recovery").unwrap();
    let mut vram = GpuMemorySpace::new(ns, weights, vec![PageRegion::KvCache; 4]).unwrap();
    let engine = CheckpointEngine::new(&store, CheckpointConfig::default());

    engine.checkpoint(&vram, "base", None).unwrap();

    // Dirty 3 pages: 2, 3, 5.
    vram.overwrite_page(2, &vec![0xAA; 512]).unwrap();
    vram.overwrite_page(3, &vec![0xBB; 512]).unwrap();
    vram.overwrite_page(5, &vec![0xCC; 512]).unwrap();
    assert_eq!(vram.dirty_bitmap().dirty_indices(), vec![2, 3, 5]);

    // Checkpoint fails mid-way after 1st dirty chunk write -> dirty bitmap MUST be fully restored!
    assert!(matches!(
        engine.checkpoint(&vram, "delta-fail", Some("base")),
        Err(GcrError::Io(_))
    ));
    assert_eq!(vram.dirty_bitmap().dirty_indices(), vec![2, 3, 5]);
}

#[test]
fn compaction_by_delta_ratio_and_gc_reclaim_superseded_chunks() {
    let store = MemoryCheckpointStore::new();
    let weights = build_weights(PageSize::Custom(512), 2, 0x33);
    let ns = Namespace::new("acme", "gc-job").unwrap();
    let mut vram = GpuMemorySpace::new(ns.clone(), weights, vec![PageRegion::KvCache; 4]).unwrap();

    let engine = CheckpointEngine::new(
        &store,
        CheckpointConfig {
            mode: CheckpointMode::GcrIncremental,
            max_chain_depth: 100, // Force compaction via max_chain_delta_ratio instead!
            max_chain_delta_ratio: 1.5,
            gc_on_compaction: true,
            ..CheckpointConfig::default()
        },
    );

    let r0 = engine.checkpoint(&vram, "c0", None).unwrap();
    assert!(!r0.compacted);

    // Pause 1: dirty 2 of 4 pages (cumulative = 6 / 4 = 1.5 -> not > 1.5 yet).
    vram.overwrite_page(2, &vec![1; 512]).unwrap();
    vram.overwrite_page(3, &vec![2; 512]).unwrap();
    let r1 = engine.checkpoint(&vram, "c1", Some("c0")).unwrap();
    assert!(!r1.compacted);
    let storage_before_gc = store.namespace_storage_bytes(&ns).unwrap();

    // Pause 2: dirty 2 of 4 pages again (cumulative = 8 / 4 = 2.0 > 1.5 -> compacts & GCs!).
    vram.overwrite_page(2, &vec![3; 512]).unwrap();
    vram.overwrite_page(3, &vec![4; 512]).unwrap();
    let r2 = engine.checkpoint(&vram, "c2", Some("c1")).unwrap();
    assert!(r2.compacted);
    assert_eq!(r2.manifest.chain_depth, 0);
    assert!(r2.gc_chunks_deleted >= 2);
    let storage_after_gc = store.namespace_storage_bytes(&ns).unwrap();
    assert!(storage_after_gc < storage_before_gc + 1024);

    let mut cold = vram.cold_boot_clone_layout().unwrap();
    let restored = engine.restore(&mut cold, "c2").unwrap();
    assert!(restored.bit_exact_verified);
    assert_eq!(restored.manifests_traversed, 1);
    assert_eq!(cold.state_digest(), vram.state_digest());
}

#[test]
fn layout_and_model_base_mismatch_are_rejected() {
    let store = MemoryCheckpointStore::new();
    let weights_v1 = build_weights(PageSize::Custom(512), 4, 0x01);
    let weights_v2 = build_weights(PageSize::Custom(512), 4, 0x02);

    let ns = Namespace::new("acme", "model-check").unwrap();
    let mut vram =
        GpuMemorySpace::new(ns.clone(), weights_v1, vec![PageRegion::KvCache; 4]).unwrap();
    vram.overwrite_page(4, &vec![0x99; 512]).unwrap();

    let engine = CheckpointEngine::new(&store, CheckpointConfig::default());
    engine.checkpoint(&vram, "c0", None).unwrap();

    // Different model weights MUST be rejected.
    let mut wrong_weights =
        GpuMemorySpace::new(ns.clone(), weights_v2, vec![PageRegion::KvCache; 4]).unwrap();
    assert!(matches!(
        engine.restore(&mut wrong_weights, "c0"),
        Err(GcrError::LayoutMismatch { .. })
    ));

    // Different mutable page count MUST be rejected.
    let mut wrong_count = GpuMemorySpace::new(
        ns,
        vram.shared_weights().clone(),
        vec![PageRegion::KvCache; 6],
    )
    .unwrap();
    assert!(matches!(
        engine.restore(&mut wrong_count, "c0"),
        Err(GcrError::LayoutMismatch { .. })
    ));
}

#[test]
fn benchmark_harness_runs_cleanly_and_passes_all_invariants() {
    let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let mut opts = BenchOptions::with_repo_root(repo_root);
    opts.iterations = 1;
    opts.page_size_override = Some(PageSize::KiB4);
    let report = run_full_benchmark(&opts).unwrap();

    assert_eq!(report.workloads.len(), 4);
    assert!((report.security.bit_exact_post_resume_pct - 100.0).abs() < f64::EPSILON);
    assert!((report.security.cross_tenant_leak_pct - 0.0).abs() < f64::EPSILON);
    assert!((report.security.cross_tenant_rejection_pct - 100.0).abs() < f64::EPSILON);
    assert!((report.security.corrupted_chunk_rejection_pct - 100.0).abs() < f64::EPSILON);
    assert!((report.security.truncated_chunk_rejection_pct - 100.0).abs() < f64::EPSILON);
    assert!((report.security.cyclic_chain_rejection_pct - 100.0).abs() < f64::EPSILON);
    assert!((report.security.missing_parent_rejection_pct - 100.0).abs() < f64::EPSILON);
    assert!(report.security.crash_safe_fsync_atomic_rename_ok);

    // Negative regime must honestly show >= 1.0x overhead vs full_mutable_only.
    let neg = report
        .workloads
        .iter()
        .find(|w| w.is_negative_regime)
        .unwrap();
    let neg_gcr = neg
        .strategies
        .iter()
        .find(|s| s.strategy == "gcr_incremental_delta")
        .unwrap();
    assert!(neg_gcr.overhead_vs_mutable_only >= 1.0);
}
