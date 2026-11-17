use crate::lora::paging::LoraPager;
use std::sync::Arc;

pub struct AdapterManager {
    pager: Arc<LoraPager>,
    max_rank: usize,
}

impl AdapterManager {
    pub fn new(pager: Arc<LoraPager>, max_rank: usize) -> Self {
        Self { pager, max_rank }
    }

    /// Registers a new tenant adapter and reserves paged memory.
    pub fn register_adapter(&self, tenant_id: &str, rank: usize) -> Result<(), String> {
        if rank > self.max_rank {
            return Err(format!("Rank {} exceeds maximum supported rank {}", rank, self.max_rank));
        }
        
        // Estimate size for A and B matrices (simplified)
        // size = (hidden_dim * rank + rank * hidden_dim) * sizeof(float16)
        let estimated_size = 4096 * rank * 2 * 2; 
        
        self.pager.allocate_for_adapter(tenant_id, estimated_size)
    }

    pub fn evict_tenant(&self, tenant_id: &str) {
        self.pager.deallocate_adapter(tenant_id);
    }
}
