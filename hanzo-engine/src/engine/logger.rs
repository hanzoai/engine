#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use tracing::info;

pub struct IntervalLogger {
    enable_logging: Arc<AtomicBool>,
    prefix_cache_hits: Arc<AtomicUsize>,
    tokens_processed: Arc<AtomicUsize>,
    total_new_seqs: Arc<AtomicUsize>,
    num_running: Arc<AtomicUsize>,
    num_waiting: Arc<AtomicUsize>,
    encoder_cache_hits: Option<Arc<AtomicUsize>>,
    encoder_cache_misses: Option<Arc<AtomicUsize>>,
    /// The engine loop's phase (top two bits) and when it entered it, in ms since `origin`.
    beat: AtomicU64,
    origin: Instant,
    /// Creation time (ms since the Unix epoch) of the oldest running sequence; 0 when none runs.
    oldest: AtomicU64,
    /// Paged KV pool: token positions held by live sequences, and the pool's size (0: no pool).
    kv_used: AtomicUsize,
    kv_positions: AtomicUsize,
    prefix: PrefixCounters,
}

/// What the engine loop is doing, for the liveness probe behind `/health` (Halogen spec §10.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Waiting for a request; it answers the moment one arrives.
    Idle,
    /// Between model steps: taking requests, scheduling, reaping.
    Loop,
    /// Inside a prompt step.
    Prefill,
    /// Inside a decode step.
    Decode,
}

impl Phase {
    /// The longest the loop may stay in this phase before it counts as unresponsive. Between steps
    /// that is Halogen's PING budget, 30 s (spec §10.1). Inside a step the loop cannot answer until
    /// the step returns, so the limits are Halogen's engine-silence limits: 1800 s for a prefill,
    /// 300 s for a decode step (spec §3.4).
    pub fn budget(self) -> Option<Duration> {
        match self {
            Self::Idle => None,
            Self::Loop => Some(Duration::from_secs(30)),
            Self::Prefill => Some(Duration::from_secs(1800)),
            Self::Decode => Some(Duration::from_secs(300)),
        }
    }

    fn bits(self) -> u64 {
        match self {
            Self::Idle => 0,
            Self::Loop => 1,
            Self::Prefill => 2,
            Self::Decode => 3,
        }
    }

    fn from_bits(bits: u64) -> Self {
        match bits {
            1 => Self::Loop,
            2 => Self::Prefill,
            3 => Self::Decode,
            _ => Self::Idle,
        }
    }
}

const PHASE_SHIFT: u32 = 62;
const SINCE_MASK: u64 = (1 << PHASE_SHIFT) - 1;

/// The engine loop's last heartbeat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Beat {
    pub phase: Phase,
    /// Time spent in `phase` so far.
    pub elapsed: Duration,
}

impl Beat {
    /// Whether the loop has overstayed its phase's budget.
    pub fn stalled(&self) -> bool {
        self.phase
            .budget()
            .is_some_and(|budget| self.elapsed > budget)
    }
}

/// How the engine reuses prompt prefixes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixMode {
    Off,
    /// Whole finished sequences are kept and the longest shared prefix is reused.
    Sequence,
    /// Full KV blocks of this many tokens are hashed and reused (paged attention).
    Blocks(usize),
}

/// Prefix-cache counters. Lookups are counted where the engine looks a prompt up; entries, stores
/// and evictions are copied from the cache after every engine step.
#[derive(Default)]
struct PrefixCounters {
    /// 0 off, 1 sequence mode, otherwise block mode with a block size of `mode - 1`.
    mode: AtomicUsize,
    misses: AtomicU64,
    tokens_saved: AtomicU64,
    entries: AtomicUsize,
    stores: AtomicU64,
    evicted: AtomicU64,
}

/// A read of the prefix cache's counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixStats {
    pub mode: PrefixMode,
    /// Prompts that reused a cached prefix.
    pub hits: u64,
    /// Prompts that found none.
    pub misses: u64,
    /// Prompt tokens served from the cache instead of being prefilled.
    pub tokens_saved: u64,
    /// Entries held now: sequences, or KV blocks in block mode.
    pub entries: usize,
    /// Entries stored since start.
    pub stores: u64,
    /// Entries evicted since start.
    pub evicted: u64,
}

/// A read of the engine's load: what runs, what waits, and the paged KV pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Load {
    /// Sequences in the scheduler's running set.
    pub running: usize,
    /// Sequences waiting for a slot or for KV room.
    pub waiting: usize,
    /// Creation time (ms since the Unix epoch) of the oldest running sequence.
    pub oldest: Option<u64>,
    /// Token positions held by live sequences, and the pool's size, when the pool is paged.
    pub kv: Option<(usize, usize)>,
}

impl IntervalLogger {
    /// Starts an interval logger. Call `begin_logging` to begin the logging process.
    pub fn new(
        interval: Duration,
        encoder_cache_counters: Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)>,
    ) -> Self {
        let prefix_cache_hits = Arc::new(AtomicUsize::new(0));
        let tokens_processed = Arc::new(AtomicUsize::new(0));
        let total_new_seqs = Arc::new(AtomicUsize::new(0));
        let enable_logging = Arc::new(AtomicBool::new(false));
        let num_running = Arc::new(AtomicUsize::new(0));
        let num_waiting = Arc::new(AtomicUsize::new(0));

        let t_prefix_cache_hits = prefix_cache_hits.clone();
        let t_tokens_processed = tokens_processed.clone();
        let t_total_new_seqs = total_new_seqs.clone();
        let t_enable_logging = enable_logging.clone();
        let t_num_running = num_running.clone();
        let t_num_waiting = num_waiting.clone();
        let (encoder_cache_hits, encoder_cache_misses) = match encoder_cache_counters {
            Some((h, m)) => (Some(h), Some(m)),
            None => (None, None),
        };
        let t_enc_hits = encoder_cache_hits.clone();
        let t_enc_misses = encoder_cache_misses.clone();
        thread::spawn(move || {
            // Speculative counters are cumulative; the line reports each window's share, so
            // the previous reading is kept and advanced every interval, logged or not.
            let mut drafts = crate::speculative::stats::snapshot();
            // Start the actual logging
            loop {
                thread::sleep(interval);
                let drafts_now = crate::speculative::stats::snapshot();
                let draft_info = draft_window(drafts, drafts_now);
                drafts = drafts_now;
                if !t_enable_logging.load(Ordering::Relaxed) {
                    continue;
                }

                let total_new_seqs = t_total_new_seqs.load(Ordering::Relaxed);
                let prefix_cache_hits = t_prefix_cache_hits.load(Ordering::Relaxed);
                let tokens_processed = t_tokens_processed.swap(0, Ordering::Relaxed);
                let num_running = t_num_running.load(Ordering::Relaxed);
                let num_waiting = t_num_waiting.load(Ordering::Relaxed);

                if total_new_seqs != 0 && tokens_processed != 0 {
                    let enc_cache_info =
                        if let (Some(ref hits), Some(ref misses)) = (&t_enc_hits, &t_enc_misses) {
                            let h = hits.load(Ordering::Relaxed);
                            let m = misses.load(Ordering::Relaxed);
                            let total = h + m;
                            if total > 0 {
                                format!(
                                    ", Encoder cache hitrate {:.2}%",
                                    100. * h as f64 / total as f64
                                )
                            } else {
                                String::new()
                            }
                        } else {
                            String::new()
                        };

                    // Throughput = tokens processed during this interval / interval duration.
                    // Combines both prefill and decode tokens. The counter is atomically
                    // swapped to 0 each interval, so the metric reflects only the current
                    // window and is not cumulative.
                    info!(
                        "Throughput (T/s) {:.2}, Prefix cache hitrate {:.2}%{enc_cache_info}{draft_info}, {num_running} running, {num_waiting} waiting",
                        tokens_processed as f64 / interval.as_secs_f64(),
                        100. * prefix_cache_hits as f64 / total_new_seqs as f64,
                    );
                }
            }
        });

        Self {
            prefix_cache_hits,
            tokens_processed,
            total_new_seqs,
            enable_logging,
            num_running,
            num_waiting,
            encoder_cache_hits,
            encoder_cache_misses,
            beat: AtomicU64::new(0),
            origin: Instant::now(),
            oldest: AtomicU64::new(0),
            kv_used: AtomicUsize::new(0),
            kv_positions: AtomicUsize::new(0),
            prefix: PrefixCounters::default(),
        }
    }

    pub fn enable_logging(&self) {
        self.enable_logging.store(true, Ordering::Relaxed);
    }

    /// Reset all counters to zero. Call after warmup/dummy runs to get clean stats.
    pub fn reset(&self) {
        self.prefix_cache_hits.store(0, Ordering::Relaxed);
        self.prefix.misses.store(0, Ordering::Relaxed);
        self.prefix.tokens_saved.store(0, Ordering::Relaxed);
        self.tokens_processed.store(0, Ordering::Relaxed);
        self.total_new_seqs.store(0, Ordering::Relaxed);
        self.num_running.store(0, Ordering::Relaxed);
        self.num_waiting.store(0, Ordering::Relaxed);
        if let Some(ref hits) = self.encoder_cache_hits {
            hits.store(0, Ordering::Relaxed);
        }
        if let Some(ref misses) = self.encoder_cache_misses {
            misses.store(0, Ordering::Relaxed);
        }
    }

    pub fn add_tokens_processed(&self, num_tokens: usize) {
        self.tokens_processed
            .fetch_add(num_tokens, Ordering::Relaxed);
    }

    pub fn add_new_sequence(&self) {
        self.total_new_seqs.fetch_add(1, Ordering::Relaxed);
    }

    /// One prompt looked up in the prefix cache: `cached` of its tokens were found there.
    pub fn add_prefix_lookup(&self, cached: usize) {
        if cached > 0 {
            self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
            self.prefix
                .tokens_saved
                .fetch_add(cached as u64, Ordering::Relaxed);
        } else {
            self.prefix.misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// How the engine reuses prompt prefixes; set once, when the engine is built.
    pub fn set_prefix_mode(&self, mode: PrefixMode) {
        let bits = match mode {
            PrefixMode::Off => 0,
            PrefixMode::Sequence => 1,
            PrefixMode::Blocks(block_size) => block_size + 1,
        };
        self.prefix.mode.store(bits, Ordering::Relaxed);
    }

    /// The prefix cache's current entries and its cumulative stores and evictions.
    pub fn set_prefix_counts(&self, entries: usize, stores: u64, evicted: u64) {
        self.prefix.entries.store(entries, Ordering::Relaxed);
        self.prefix.stores.store(stores, Ordering::Relaxed);
        self.prefix.evicted.store(evicted, Ordering::Relaxed);
    }

    pub fn prefix_stats(&self) -> PrefixStats {
        let mode = match self.prefix.mode.load(Ordering::Relaxed) {
            0 => PrefixMode::Off,
            1 => PrefixMode::Sequence,
            bits => PrefixMode::Blocks(bits - 1),
        };
        PrefixStats {
            mode,
            hits: self.prefix_cache_hits.load(Ordering::Relaxed) as u64,
            misses: self.prefix.misses.load(Ordering::Relaxed),
            tokens_saved: self.prefix.tokens_saved.load(Ordering::Relaxed),
            entries: self.prefix.entries.load(Ordering::Relaxed),
            stores: self.prefix.stores.load(Ordering::Relaxed),
            evicted: self.prefix.evicted.load(Ordering::Relaxed),
        }
    }

    /// The engine loop entered `phase` now.
    pub fn beat(&self, phase: Phase) {
        let since = u64::try_from(self.origin.elapsed().as_millis()).unwrap_or(SINCE_MASK);
        self.beat.store(
            (phase.bits() << PHASE_SHIFT) | since.min(SINCE_MASK),
            Ordering::Relaxed,
        );
    }

    /// The engine loop's phase and how long it has been in it.
    pub fn last_beat(&self) -> Beat {
        let beat = self.beat.load(Ordering::Relaxed);
        let now = u64::try_from(self.origin.elapsed().as_millis()).unwrap_or(SINCE_MASK);
        Beat {
            phase: Phase::from_bits(beat >> PHASE_SHIFT),
            elapsed: Duration::from_millis(now.saturating_sub(beat & SINCE_MASK)),
        }
    }

    /// Publish the engine's load. The engine calls this after every step and before it idles, so
    /// a reader sees the live state without taking any engine lock.
    pub fn set_load(&self, load: Load) {
        self.num_running.store(load.running, Ordering::Relaxed);
        self.num_waiting.store(load.waiting, Ordering::Relaxed);
        self.oldest
            .store(load.oldest.unwrap_or(0), Ordering::Relaxed);
        let (used, positions) = load.kv.unwrap_or((0, 0));
        self.kv_used.store(used, Ordering::Relaxed);
        self.kv_positions.store(positions, Ordering::Relaxed);
    }

    pub fn load(&self) -> Load {
        let oldest = self.oldest.load(Ordering::Relaxed);
        let positions = self.kv_positions.load(Ordering::Relaxed);
        Load {
            running: self.num_running.load(Ordering::Relaxed),
            waiting: self.num_waiting.load(Ordering::Relaxed),
            oldest: (oldest > 0).then_some(oldest),
            kv: (positions > 0).then(|| (self.kv_used.load(Ordering::Relaxed), positions)),
        }
    }

    pub fn set_num_running(&self, running: usize) {
        self.num_running.store(running, Ordering::Relaxed);
    }

    pub fn set_num_waiting(&self, waiting: usize) {
        self.num_waiting.store(waiting, Ordering::Relaxed);
    }

    /// Return cumulative prefix cache (hits, total_sequences).
    pub fn prefix_cache_stats(&self) -> (usize, usize) {
        (
            self.prefix_cache_hits.load(Ordering::Relaxed),
            self.total_new_seqs.load(Ordering::Relaxed),
        )
    }

    /// Return cumulative encoder cache (hits, misses), or `None` if no encoder cache exists.
    pub fn encoder_cache_stats(&self) -> Option<(usize, usize)> {
        match (&self.encoder_cache_hits, &self.encoder_cache_misses) {
            (Some(h), Some(m)) => Some((h.load(Ordering::Relaxed), m.load(Ordering::Relaxed))),
            _ => None,
        }
    }
}

/// What speculative decoding did over one logging window, for the throughput line.
///
/// Empty when nothing was verified in the window, so a server without a draft logs
/// exactly what it logged before. "tokens per verify" adds the one token every verify
/// step emits beyond its accepted drafts -- the target's correction, or the bonus
/// token when every draft held -- which is the figure sglang reports as its accept
/// length, so two engines serving one draft compare directly. (The step a sequence
/// ends on can emit fewer; over a window that is noise.)
fn draft_window(
    before: crate::speculative::stats::SpeculativeStats,
    now: crate::speculative::stats::SpeculativeStats,
) -> String {
    // A benchmark may reset the counters mid-run; the window is then all since the reset.
    let window = if now.verify_rounds < before.verify_rounds {
        now
    } else {
        crate::speculative::stats::SpeculativeStats {
            verify_rounds: now.verify_rounds - before.verify_rounds,
            accepted_sum: now.accepted_sum.saturating_sub(before.accepted_sum),
            proposed_sum: now.proposed_sum.saturating_sub(before.proposed_sum),
        }
    };
    if window.verify_rounds == 0 {
        return String::new();
    }
    let held = if window.proposed_sum == 0 {
        0.
    } else {
        100. * window.accepted_sum as f64 / window.proposed_sum as f64
    };
    format!(
        ", Drafts {:.2} of {:.2} accepted per verify ({held:.1}%), {:.2} tokens per verify",
        window.mean_accepted(),
        window.mean_proposed(),
        window.mean_accepted() + 1.,
    )
}

#[cfg(test)]
mod tests {
    use super::{draft_window, Beat, IntervalLogger, Load, Phase, PrefixMode};
    use crate::speculative::stats::SpeculativeStats;
    use std::time::Duration;

    fn logger() -> IntervalLogger {
        IntervalLogger::new(Duration::from_secs(3600), None)
    }

    /// Idle never stalls; each other phase stalls only past its own budget.
    #[test]
    fn a_phase_stalls_past_its_budget() {
        let at = |phase, secs| Beat {
            phase,
            elapsed: Duration::from_secs(secs),
        };
        assert!(!at(Phase::Idle, 1_000_000).stalled());
        assert!(!at(Phase::Loop, 30).stalled());
        assert!(at(Phase::Loop, 31).stalled());
        assert!(!at(Phase::Decode, 300).stalled());
        assert!(at(Phase::Decode, 301).stalled());
        assert!(!at(Phase::Prefill, 1800).stalled());
        assert!(at(Phase::Prefill, 1801).stalled());
    }

    /// A beat reads back as the phase it set, just entered.
    #[test]
    fn a_beat_reads_back() {
        let logger = logger();
        assert_eq!(logger.last_beat().phase, Phase::Idle);
        for phase in [Phase::Loop, Phase::Prefill, Phase::Decode, Phase::Idle] {
            logger.beat(phase);
            let beat = logger.last_beat();
            assert_eq!(beat.phase, phase);
            assert!(beat.elapsed < Duration::from_secs(5));
        }
    }

    /// Published load reads back; no pool and no running sequence read as absent.
    #[test]
    fn load_reads_back() {
        let logger = logger();
        assert_eq!(logger.load(), Load::default());
        let load = Load {
            running: 2,
            waiting: 3,
            oldest: Some(1_700_000_000_000),
            kv: Some((4096, 655_360)),
        };
        logger.set_load(load);
        assert_eq!(logger.load(), load);
        logger.set_load(Load::default());
        assert_eq!(logger.load(), Load::default());
    }

    /// A lookup that found cached tokens is a hit and saves them; one that found none is a miss.
    #[test]
    fn prefix_lookups_count_hits_misses_and_tokens() {
        let logger = logger();
        logger.set_prefix_mode(PrefixMode::Blocks(16));
        logger.add_prefix_lookup(0);
        logger.add_prefix_lookup(512);
        logger.add_prefix_lookup(48);
        logger.set_prefix_counts(7, 9, 2);
        let stats = logger.prefix_stats();
        assert_eq!(stats.mode, PrefixMode::Blocks(16));
        assert_eq!((stats.hits, stats.misses, stats.tokens_saved), (2, 1, 560));
        assert_eq!((stats.entries, stats.stores, stats.evicted), (7, 9, 2));
        for mode in [PrefixMode::Off, PrefixMode::Sequence, PrefixMode::Blocks(1)] {
            logger.set_prefix_mode(mode);
            assert_eq!(logger.prefix_stats().mode, mode);
        }
        logger.reset();
        let stats = logger.prefix_stats();
        assert_eq!((stats.hits, stats.misses, stats.tokens_saved), (0, 0, 0));
    }

    fn at(verify_rounds: u64, accepted_sum: u64, proposed_sum: u64) -> SpeculativeStats {
        SpeculativeStats {
            verify_rounds,
            accepted_sum,
            proposed_sum,
        }
    }

    /// No draft, or a window with no verify step: the line is exactly what it was.
    #[test]
    fn nothing_verified_adds_nothing() {
        assert_eq!(draft_window(at(0, 0, 0), at(0, 0, 0)), "");
        assert_eq!(draft_window(at(7, 20, 56), at(7, 20, 56)), "");
    }

    /// The window is the difference, not the running total: 4 rounds, 12 of 28 held.
    #[test]
    fn a_window_reports_its_own_share() {
        assert_eq!(
            draft_window(at(10, 20, 70), at(14, 32, 98)),
            ", Drafts 3.00 of 7.00 accepted per verify (42.9%), 4.00 tokens per verify"
        );
    }

    /// Counters reset between two readings: the window is everything since the reset,
    /// never a wrapped subtraction.
    #[test]
    fn a_reset_mid_window_is_not_a_wraparound() {
        assert_eq!(
            draft_window(at(100, 300, 800), at(2, 5, 16)),
            ", Drafts 2.50 of 8.00 accepted per verify (31.2%), 3.50 tokens per verify"
        );
    }
}
