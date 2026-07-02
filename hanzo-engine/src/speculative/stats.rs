//! Process-wide speculative-decoding acceptance counters.
//!
//! A tiny diagnostic accumulator so examples/benchmarks can report the mean accepted draft
//! length per verify round without threading a channel through the whole engine. The driver
//! bumps it once per verify step (see [`record_verify`]); a benchmark [`reset`]s before a run
//! and [`snapshot`]s after. Cheap relaxed atomics — off the hot numeric path.

use std::sync::atomic::{AtomicU64, Ordering};

static VERIFY_ROUNDS: AtomicU64 = AtomicU64::new(0);
static ACCEPTED_SUM: AtomicU64 = AtomicU64::new(0);
static PROPOSED_SUM: AtomicU64 = AtomicU64::new(0);

/// One verify round: `accepted` drafts confirmed out of `proposed` staged.
pub fn record_verify(accepted: usize, proposed: usize) {
    VERIFY_ROUNDS.fetch_add(1, Ordering::Relaxed);
    ACCEPTED_SUM.fetch_add(accepted as u64, Ordering::Relaxed);
    PROPOSED_SUM.fetch_add(proposed as u64, Ordering::Relaxed);
}

/// Zero the counters before a measured run.
pub fn reset() {
    VERIFY_ROUNDS.store(0, Ordering::Relaxed);
    ACCEPTED_SUM.store(0, Ordering::Relaxed);
    PROPOSED_SUM.store(0, Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SpeculativeStats {
    pub verify_rounds: u64,
    pub accepted_sum: u64,
    pub proposed_sum: u64,
}

impl SpeculativeStats {
    /// Mean accepted drafts per verify round (excludes the always-emitted verified token).
    pub fn mean_accepted(&self) -> f64 {
        if self.verify_rounds == 0 {
            0.0
        } else {
            self.accepted_sum as f64 / self.verify_rounds as f64
        }
    }

    /// Mean staged draft length per verify round.
    pub fn mean_proposed(&self) -> f64 {
        if self.verify_rounds == 0 {
            0.0
        } else {
            self.proposed_sum as f64 / self.verify_rounds as f64
        }
    }
}

/// Read the accumulated counters.
pub fn snapshot() -> SpeculativeStats {
    SpeculativeStats {
        verify_rounds: VERIFY_ROUNDS.load(Ordering::Relaxed),
        accepted_sum: ACCEPTED_SUM.load(Ordering::Relaxed),
        proposed_sum: PROPOSED_SUM.load(Ordering::Relaxed),
    }
}
