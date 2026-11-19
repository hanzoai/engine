//! KV isolation: the real scheduler and block pool, driven against a simulated KV store.
//!
//! Each slot of the store remembers the context its KV was computed from (the tokens up to and
//! including its position) and the cache salt of the request that wrote it. A forward reads every
//! earlier position of its sequence through the block table the scheduler gave it, so a read of a
//! slot no forward wrote, of KV computed from a different context, or of another tenant's KV is
//! caught at the read that would have decoded against it.

use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, Mutex},
    time::Duration,
};

use tokio::sync::mpsc::{channel, Receiver, Sender};

use super::{
    block_hash::{compute_block_hashes, salt_keys},
    scheduler::{PagedAttentionScheduler, PagedAttentionSchedulerConfig},
    CacheConfig, PagedCacheType,
};
use crate::{
    engine::IntervalLogger,
    get_mut_arcmutex,
    response::Response,
    sampler::{Logprobs, Sampler},
    scheduler::Scheduler,
    sequence::{
        SeqStepType, Sequence, SequenceGroup, SequenceRecognizer, SequenceState, StopReason,
    },
};

const BLOCK: usize = 16;
const MAX_STEPS: usize = 500;

#[derive(Clone)]
struct Slot {
    salt: Option<String>,
    ctx: u64,
}

/// What a test asks of one request: its prompt, the tokens it generates, and when it is canceled.
struct Script {
    answer: Vec<u32>,
    cancel_at_len: Option<usize>,
}

struct Sim {
    sched: PagedAttentionScheduler,
    store: HashMap<usize, Vec<Option<Slot>>>,
    scripts: HashMap<usize, Script>,
    seqs: HashMap<usize, Arc<Mutex<Sequence>>>,
    /// Prompt tokens served from the prefix cache, from each request's latest admission.
    hits: HashMap<usize, usize>,
    violations: Vec<String>,
    logger: IntervalLogger,
    tx: Sender<Response>,
    _rx: Receiver<Response>,
    next_id: usize,
}

fn ctx(tokens: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    tokens.hash(&mut h);
    h.finish()
}

fn toks(seed: u32, n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| seed * 10_000 + i).collect()
}

fn cat(parts: &[&[u32]]) -> Vec<u32> {
    parts.concat()
}

impl Sim {
    /// `blocks` usable blocks; the pool also holds the null block.
    fn new(blocks: usize, max_num_seqs: usize) -> Self {
        let mut sched = PagedAttentionScheduler::new(
            PagedAttentionSchedulerConfig { max_num_seqs },
            CacheConfig {
                block_size: BLOCK,
                num_gpu_blocks: blocks + 1,
                cache_type: PagedCacheType::default(),
            },
        );
        sched.set_prefix_caching_enabled_sync(true);
        let (tx, rx) = channel(1024);
        Self {
            sched,
            store: HashMap::new(),
            scripts: HashMap::new(),
            seqs: HashMap::new(),
            hits: HashMap::new(),
            violations: Vec::new(),
            logger: IntervalLogger::new(Duration::from_secs(3600), None),
            tx,
            _rx: rx,
            next_id: 0,
        }
    }

    fn submit(&mut self, salt: Option<&str>, prompt: Vec<u32>, answer: Vec<u32>) -> usize {
        self.submit_script(
            salt,
            prompt,
            Script {
                answer,
                cancel_at_len: None,
            },
        )
    }

    fn submit_script(&mut self, salt: Option<&str>, prompt: Vec<u32>, script: Script) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        let sampler = Sampler::new(None, 0, None, None, None, None, None, 32, 1.0, 0.0, vec![])
            .expect("greedy sampler");
        let group = Arc::new(tokio::sync::Mutex::new(SequenceGroup::new(
            1, false, true, None,
        )));
        let mut seq = Sequence::new_waiting(
            prompt,
            String::new(),
            id,
            id as u128,
            1,
            self.tx.clone(),
            sampler,
            vec![],
            vec![],
            None,
            false,
            false,
            group,
            0,
            0,
            SequenceRecognizer::None,
            None,
            None,
            None,
            None,
            None,
            Some(BLOCK),
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            vec![],
        );
        seq.set_cache_salt(salt.map(str::to_owned));
        self.scripts.insert(id, script);
        self.sched.add_seq(seq);
        id
    }

    /// One engine iteration: schedule, forward every scheduled sequence, sample, reap.
    fn step(&mut self) {
        let out = self.sched.schedule(&self.logger);
        for (i, seq) in out.scheduled.iter().enumerate() {
            let id = *get_mut_arcmutex!(seq).id();
            self.seqs.insert(id, seq.clone());
            if get_mut_arcmutex!(seq).is_prompt() {
                self.hits.insert(id, out.num_cached_tokens[i]);
            }
            self.forward(seq);
        }
        self.sched.free_finished_sequence_groups();
    }

    fn forward(&mut self, seq: &Arc<Mutex<Sequence>>) {
        let mut seq = get_mut_arcmutex!(seq);
        let id = *seq.id();
        let tokens = seq.get_toks().to_vec();
        let salt = seq.cache_salt().map(str::to_owned);
        let start = if seq.is_prompt() {
            seq.prefix_cache_len()
        } else {
            tokens.len() - 1
        };
        let blocks = get_mut_arcmutex!(self.sched.kv_cache_manager)
            .get_block_ids(id)
            .expect("a scheduled sequence holds blocks")
            .to_vec();

        for p in 0..start {
            let slot = self
                .store
                .get(&blocks[p / BLOCK])
                .and_then(|b| b[p % BLOCK].clone());
            let problem = match slot {
                None => Some("a slot no forward wrote".to_string()),
                Some(slot) if slot.salt != salt => {
                    Some(format!("KV written by tenant {:?}", slot.salt))
                }
                Some(slot) if slot.ctx != ctx(&tokens[..=p]) => {
                    Some("KV computed from a different context".to_string())
                }
                Some(_) => None,
            };
            if let Some(problem) = problem {
                self.violations.push(format!(
                    "seq {id} (salt {salt:?}) read position {p}: {problem}"
                ));
                break;
            }
        }
        for p in start..tokens.len() {
            let block = self
                .store
                .entry(blocks[p / BLOCK])
                .or_insert_with(|| vec![None; BLOCK]);
            block[p % BLOCK] = Some(Slot {
                salt: salt.clone(),
                ctx: ctx(&tokens[..=p]),
            });
        }

        let script = &self.scripts[&id];
        let generated = tokens.len() - seq.prompt_tokens();
        let token = script.answer[generated];
        let done = (generated + 1 == script.answer.len()).then_some(StopReason::Length(0));
        seq.add_token(
            Logprobs {
                token,
                logprob: 0.,
                bytes: None,
                top_logprobs: None,
            },
            vec![],
            &done,
        );
        if let Some(reason) = done {
            seq.set_state(SequenceState::Done(reason));
        } else if script.cancel_at_len == Some(seq.get_toks().len()) {
            seq.set_state(SequenceState::Done(StopReason::Canceled));
        }
    }

    fn run(&mut self) {
        for _ in 0..MAX_STEPS {
            if self.sched.waiting_len() == 0 && self.sched.running_len() == 0 {
                return;
            }
            self.step();
        }
        panic!("requests still unfinished after {MAX_STEPS} steps");
    }

    fn tokens(&self, id: usize) -> Vec<u32> {
        get_mut_arcmutex!(self.seqs[&id]).get_toks().to_vec()
    }

    fn assert_clean(&self) {
        assert!(
            self.violations.is_empty(),
            "a forward read KV it must not see:\n{}",
            self.violations.join("\n")
        );
    }
}

/// A salt changes the hash of every block, so two partitions never name the same block.
#[test]
fn salt_partitions_every_block() {
    let tokens = toks(1, 5 * BLOCK);
    let hashes = |salt| compute_block_hashes(&tokens, BLOCK, &[], &salt_keys(salt));
    let (none, acme, evil) = (hashes(None), hashes(Some("acme")), hashes(Some("evil")));
    assert_eq!(acme, hashes(Some("acme")));
    for i in 0..none.len() {
        assert_ne!(
            none[i], acme[i],
            "block {i} shared between unsalted and acme"
        );
        assert_ne!(acme[i], evil[i], "block {i} shared between acme and evil");
    }
}

/// A tenant's cached prompt serves that tenant's next request and nobody else's.
#[test]
fn tenants_never_share_cached_prompts() {
    let mut sim = Sim::new(64, 8);
    let system = toks(1, 2 * BLOCK);

    sim.submit(Some("acme"), cat(&[&system, &toks(2, 8)]), toks(3, 4));
    sim.run();

    let acme = sim.submit(Some("acme"), cat(&[&system, &toks(4, 8)]), toks(5, 4));
    let evil = sim.submit(Some("evil"), cat(&[&system, &toks(6, 8)]), toks(7, 4));
    let open = sim.submit(None, cat(&[&system, &toks(8, 8)]), toks(9, 4));
    sim.run();

    sim.assert_clean();
    assert_eq!(
        sim.hits[&acme],
        2 * BLOCK,
        "the owning tenant reuses its prompt"
    );
    assert_eq!(
        sim.hits[&evil], 0,
        "another tenant hit acme's cached prompt"
    );
    assert_eq!(
        sim.hits[&open], 0,
        "an unsalted request hit acme's cached prompt"
    );
}

/// Two tenants send the same prompt at once; each follow-up reuses only its own tenant's blocks.
#[test]
fn concurrent_tenants_with_identical_prompts_stay_apart() {
    let mut sim = Sim::new(64, 8);
    let prompt = toks(1, 3 * BLOCK + 4);
    sim.submit(Some("acme"), prompt.clone(), toks(2, 6));
    sim.submit(Some("evil"), prompt.clone(), toks(3, 6));
    sim.run();

    let acme = sim.submit(Some("acme"), cat(&[&prompt, &toks(4, 4)]), toks(5, 2));
    let evil = sim.submit(Some("evil"), cat(&[&prompt, &toks(6, 4)]), toks(7, 2));
    sim.run();

    sim.assert_clean();
    assert_eq!(sim.hits[&acme], 3 * BLOCK);
    assert_eq!(sim.hits[&evil], 3 * BLOCK);
}

/// Two prompts share a prefix and contradict each other after it. Admitted in one pass with
/// different lengths, the longer is preempted before its first forward. Its fresh blocks hold no
/// KV of its own, only what they held before (nothing, or an earlier request that answered the
/// other way), so caching them would hand that stale KV to its own retry.
#[test]
fn identical_prefixes_with_contradictory_suffixes() {
    let mut sim = Sim::new(5, 8);
    let shared = toks(1, 2 * BLOCK);
    let no = toks(2, 2 * BLOCK);
    let yes = toks(3, 2 * BLOCK);

    sim.submit(Some("acme"), cat(&[&shared, &no]), toks(4, 1));
    sim.run();

    let long = sim.submit(Some("acme"), cat(&[&shared, &yes]), toks(5, 3));
    sim.submit(Some("acme"), cat(&[&shared, &toks(6, BLOCK)]), toks(7, 3));
    sim.step();
    assert!(
        sim.sched.waiting_len() == 1 && !sim.hits.contains_key(&long),
        "the longer prompt must be preempted before its first forward for this test to bite"
    );
    sim.run();

    sim.assert_clean();
}

/// A request canceled mid-generation has sampled a token whose KV no forward wrote. Its blocks
/// stay reusable up to the last written position and no further.
#[test]
fn cancellation_mid_generation_caches_only_written_kv() {
    let mut sim = Sim::new(32, 8);
    let canceled = sim.submit_script(
        Some("acme"),
        toks(1, 2 * BLOCK - 2),
        Script {
            answer: toks(2, 16),
            cancel_at_len: Some(2 * BLOCK),
        },
    );
    sim.run();
    let history = sim.tokens(canceled);
    assert_eq!(
        history.len(),
        2 * BLOCK,
        "canceled with its second block full"
    );

    let retry = sim.submit(Some("acme"), cat(&[&history, &toks(3, 8)]), toks(4, 2));
    sim.run();

    sim.assert_clean();
    assert_eq!(
        sim.hits[&retry], BLOCK,
        "the fully written block is still reused"
    );
}

/// Blocks evicted under pressure and rewritten by another tenant are never served under the hash
/// they were cached with.
#[test]
fn reuse_after_eviction() {
    let mut sim = Sim::new(8, 4);
    let prompt = toks(1, 3 * BLOCK);
    sim.submit(Some("acme"), prompt.clone(), toks(2, 2));
    sim.run();

    for seed in 10..14 {
        sim.submit(Some("evil"), toks(seed, 6 * BLOCK), toks(seed + 100, 3));
        sim.run();
    }

    let again = sim.submit(Some("acme"), cat(&[&prompt, &toks(3, 4)]), toks(4, 2));
    sim.run();

    sim.assert_clean();
    assert_eq!(
        sim.hits[&again], 0,
        "acme's blocks were evicted and rewritten"
    );
}
