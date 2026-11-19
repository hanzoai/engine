pub mod bidaw;

pub use bidaw::{BidawScheduler, KVCacheLocation, RequestMetadata};

/// Interface for the Hanzo Engine to interact with the scheduler.
pub struct SchedulerService {
    inner: BidawScheduler,
}

impl SchedulerService {
    pub fn default() -> Self {
        // Default weights: alpha=1.0 (wait time), beta=5.0 (residency priority)
        Self {
            inner: BidawScheduler::new(1.0, 5.0),
        }
    }

    pub fn optimize_queue(&self, requests: &mut Vec<RequestMetadata>) {
        self.inner.schedule(requests);
    }
}
