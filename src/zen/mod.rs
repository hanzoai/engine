pub mod kv_offload;

pub use kv_offload::{KVBlockManager, PredictiveFetcher, CacheSalt};

/// Zen Generation Engine Integration
pub struct ZenEngine {
    kv_manager: std::sync::Arc<KVBlockManager>,
    fetcher: PredictiveFetcher,
}

impl ZenEngine {
    pub fn new(max_ram_gb: usize, offload_path: std::path::PathBuf) -> Self {
        // Assume 2MB blocks
        let block_size = 2 * 1024 * 1024;
        let max_blocks = (max_ram_gb * 1024 * 1024 * 1024) / block_size;
        
        let kv_manager = std::sync::Arc::new(KVBlockManager::new(
            max_blocks,
            block_size,
            offload_path
        ));
        
        let fetcher = PredictiveFetcher::new(kv_manager.clone());
        
        Self {
            kv_manager,
            fetcher,
        }
    }

    /// Placeholder for the main generation loop integration
    pub async fn step_generation(&self, current_block_id: u64, next_expected_blocks: Vec<u64>, salt: CacheSalt) {
        // 1. Trigger predictive fetch for future blocks
        self.fetcher.prefetch(next_expected_blocks, salt).await;
        
        // 2. Fetch current required block
        let _data = self.kv_manager.fetch_block(current_block_id, salt);
        
        // ... perform attention computation ...
    }
}
