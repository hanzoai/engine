use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Represents the location of a KV Cache block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVCacheLocation {
    GPU,
    Host,
    Remote, // S3 / VFS
}

impl KVCacheLocation {
    /// Estimated latency penalty in milliseconds for loading from this location.
    /// Values based on typical PCIe Gen4 and NVLink speeds.
    pub fn latency_penalty_ms(&self) -> f64 {
        match self {
            KVCacheLocation::GPU => 0.0,
            KVCacheLocation::Host => 15.0,   // Host to GPU transfer
            KVCacheLocation::Remote => 150.0, // S3/Disk to GPU transfer
        }
    }
}

#[derive(Debug, Clone)]
pub struct RequestMetadata {
    pub id: String,
    pub arrival_time: Instant,
    pub session_id: String,
    pub tokens_in_cache: usize,
    pub location: KVCacheLocation,
}

/// Bidaw Scheduler: KV-residency-aware request ordering.
pub struct BidawScheduler {
    alpha: f64, // Weight for wait time (starvation prevention)
    beta: f64,  // Weight for residency benefit
}

impl BidawScheduler {
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }

    /// Calculates the priority score. Higher is better.
    /// Score = alpha * (WaitTime) - beta * (EstimatedLoadingLatency)
    pub fn calculate_score(&self, req: &RequestMetadata, now: Instant) -> f64 {
        let wait_time = now.duration_since(req.arrival_time).as_secs_f64();
        let latency_penalty = req.location.latency_penalty_ms();
        
        (self.alpha * wait_time) - (self.beta * (latency_penalty / 1000.0))
    }

    /// Reorders the queue based on Bidaw logic.
    pub fn schedule(&self, queue: &mut Vec<RequestMetadata>) {
        let now = Instant::now();
        queue.sort_by(|a, b| {
            let score_a = self.calculate_score(a, now);
            let score_b = self.calculate_score(b, now);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bidaw_reordering() {
        let scheduler = BidawScheduler::new(1.0, 10.0);
        let now = Instant::now();
        
        let req_cold = RequestMetadata {
            id: "cold".to_string(),
            arrival_time: now - Duration::from_secs(1),
            session_id: "s1".to_string(),
            tokens_in_cache: 100,
            location: KVCacheLocation::Remote,
        };

        let req_warm = RequestMetadata {
            id: "warm".to_string(),
            arrival_time: now,
            session_id: "s2".to_string(),
            tokens_in_cache: 100,
            location: KVCacheLocation::GPU,
        };

        let mut queue = vec![req_cold.clone(), req_warm.clone()];
        scheduler.schedule(&mut queue);

        // Even though req_warm arrived later, it should be first because it's resident in GPU
        assert_eq!(queue[0].id, "warm");
    }
}
