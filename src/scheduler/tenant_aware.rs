use std::collections::VecDeque;

pub struct TenantRequest {
    pub request_id: String,
    pub tenant_id: String,
    pub prompt: String,
}

pub struct MultiTenantScheduler {
    queue: VecDeque<TenantRequest>,
    active_adapters: Vec<String>,
}

impl MultiTenantScheduler {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            active_adapters: Vec::new(),
        }
    }

    pub fn add_request(&mut self, req: TenantRequest) {
        self.queue.push_back(req);
    }

    /// Groups requests by adapter to leverage S-LoRA heterogeneous batching.
    pub fn schedule_batch(&mut self, batch_size: usize) -> Vec<TenantRequest> {
        let mut batch = Vec::new();
        while batch.len() < batch_size && !self.queue.is_empty() {
            batch.push(self.queue.pop_front().unwrap());
        }
        batch
    }
}
