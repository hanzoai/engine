//! XGrammar Context-Independent / Context-Dependent Token Mask Partitioning,
//! Dense/Sparse Adaptive [`BitMask`], Post-Pop Suffix Grouping, Compiled Grammar,
//! Runtime Matcher, and Matched Naive Baseline (`NaivePerStepConstrainedDecoder`).
//!
//! ## Mathematical Exactness Guarantee
//!
//! For every terminal leaf position `p` in a finalized [`Grammar`] and every
//! non-empty token `w` in [`Vocabulary`], compile-time simulation from `[p]`
//! (unknown caller stack `C`) partitions `w` into disjoint classes:
//!
//! 1. **Context-Independent Accepted (`accepted`)**: a branch survives all bytes
//!    of `w` strictly within depth 1 (tail-call optimized) without popping `p`.
//!    Valid for *every* caller stack `C`, even at `MAX_RECURSION_DEPTH`.
//! 2. **Caller-Live Accepted (`accepted_if_caller_live`)**: a branch pops `p` on
//!    the exact final byte of `w` (zero remaining bytes to match in `C`). Valid
//!    in O(1) whenever `expand(C)` is non-empty (checked once per active stack).
//! 3. **Context-Independent Rejected**: no branch survives and no branch pops `p`.
//!    Invalid for *every* caller stack `C` — zero runtime work!
//! 4. **Post-Pop Suffix Groups (`suffix_groups`)**: tokens whose ONLY exit from
//!    `p` is a clean pop at offset `k` leaving remaining suffix `s = w[k..]`
//!    (e.g., thousands of BPE tokens `foo",`, `bar",`, `baz",` all share suffix
//!    `","`). At runtime, `expand(C)` is tested **once per unique suffix**,
//!    collapsing thousands of per-token PDA checks into ~10–25 suffix checks!
//! 5. **General Context-Dependent (`uncertain`)**: remaining context-dependent
//!    tokens (e.g., ambiguous multi-pop or non-tail sub-frame pushes) evaluated
//!    directly against `[C, p]`.
//!
//! This partition guarantees **100% bit-identical token masks** compared to
//! [`NaivePerStepConstrainedDecoder`] on every decode step.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::grammar::{
    expand_caller_stack, leaves_can_consume_bytes, stack_can_consume_bytes, step_sim_byte, Grammar,
    LeafId, PdaCheckpoint, PdaConfig, PdaScratch, PdaState, RulePos, SimScratch, SimStack,
    SmallStack, Stack,
};
use crate::vocab::{VocabTrie, Vocabulary};

/// Dense bitmask (`Vec<u64>`, `ceil(V / 64)` words) over a vocabulary of size `V`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitMask {
    words: Vec<u64>,
    vocab_size: usize,
}

impl BitMask {
    pub fn new_empty(vocab_size: usize) -> Self {
        let num_words = vocab_size.div_ceil(64);
        Self {
            words: vec![0u64; num_words],
            vocab_size,
        }
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    #[inline]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    #[inline]
    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    #[inline]
    pub fn set(&mut self, token_id: u32) {
        let idx = token_id as usize;
        debug_assert!(idx < self.vocab_size);
        self.words[idx >> 6] |= 1u64 << (idx & 63);
    }

    #[inline]
    pub fn get(&self, token_id: u32) -> bool {
        let idx = token_id as usize;
        if idx >= self.vocab_size {
            return false;
        }
        ((self.words[idx >> 6] >> (idx & 63)) & 1) != 0
    }

    #[inline]
    pub fn or_with(&mut self, other: &BitMask) {
        debug_assert_eq!(self.words.len(), other.words.len());
        for (dst, &src) in self.words.iter_mut().zip(other.words.iter()) {
            *dst |= src;
        }
    }

    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Collect all allowed token IDs (useful for diagnostics and tests).
    pub fn allowed_tokens(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.count_ones());
        for (word_idx, &word) in self.words.iter().enumerate() {
            let mut w = word;
            let base = (word_idx << 6) as u32;
            while w != 0 {
                let tz = w.trailing_zeros();
                let tid = base + tz;
                if (tid as usize) < self.vocab_size {
                    out.push(tid);
                }
                w &= w - 1;
            }
        }
        out
    }

    /// Mask disallowed tokens in-place to `-inf` on a model logits vector.
    ///
    /// Optimized at the 64-bit word level: skips all-allowed words (`u64::MAX`)
    /// and bulk-fills all-disallowed words (`0`) directly.
    pub fn apply_to_logits(&self, logits: &mut [f32]) {
        let n = logits.len().min(self.vocab_size);
        let full_words = n >> 6;
        for (w_idx, &word) in self.words.iter().enumerate().take(full_words) {
            if word == u64::MAX {
                continue;
            }
            let base = w_idx << 6;
            if word == 0 {
                logits[base..base + 64].fill(f32::NEG_INFINITY);
                continue;
            }
            let mut disallowed = !word;
            while disallowed != 0 {
                let bit = disallowed.trailing_zeros() as usize;
                logits[base + bit] = f32::NEG_INFINITY;
                disallowed &= disallowed - 1;
            }
        }
        let rem_start = full_words << 6;
        if rem_start < n {
            let word = self.words[full_words];
            for (i, slot) in logits.iter_mut().enumerate().take(n).skip(rem_start) {
                if ((word >> (i & 63)) & 1) == 0 {
                    *slot = f32::NEG_INFINITY;
                }
            }
        }
    }

    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.words.capacity() * std::mem::size_of::<u64>()
    }
}

/// Adaptive sparse-vs-dense token set (XGrammar Section 4.1 storage optimization).
#[derive(Debug, Clone)]
pub enum CompactTokenSet {
    Empty,
    Sparse(Vec<u32>),
    Dense(BitMask),
}

impl CompactTokenSet {
    fn from_sorted_ids(ids: Vec<u32>, vocab_size: usize) -> Self {
        if ids.is_empty() {
            return CompactTokenSet::Empty;
        }
        if ids.len() * 32 > vocab_size {
            let mut bm = BitMask::new_empty(vocab_size);
            for tid in ids {
                bm.set(tid);
            }
            CompactTokenSet::Dense(bm)
        } else {
            CompactTokenSet::Sparse(ids)
        }
    }

    #[inline]
    pub fn or_into(&self, dst: &mut BitMask) {
        match self {
            CompactTokenSet::Empty => {}
            CompactTokenSet::Sparse(ids) => {
                for &tid in ids {
                    dst.set(tid);
                }
            }
            CompactTokenSet::Dense(bm) => dst.or_with(bm),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CompactTokenSet::Empty => 0,
            CompactTokenSet::Sparse(ids) => ids.len(),
            CompactTokenSet::Dense(bm) => bm.count_ones(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn memory_bytes(&self) -> usize {
        match self {
            CompactTokenSet::Empty => 0,
            CompactTokenSet::Sparse(v) => v.capacity() * std::mem::size_of::<u32>(),
            CompactTokenSet::Dense(bm) => bm.memory_bytes(),
        }
    }
}

/// Group of context-dependent tokens that all exit the leaf cleanly and share
/// the exact same post-pop remaining suffix (e.g., `","`, `":"`, `"}"`).
#[derive(Debug, Clone)]
pub struct SuffixGroup {
    pub suffix: Vec<u8>,
    pub tokens: CompactTokenSet,
}

impl SuffixGroup {
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.suffix.capacity() + self.tokens.memory_bytes()
    }
}

/// Precomputed XGrammar token partition for a single grammar leaf position.
#[derive(Debug, Clone)]
pub struct PositionMask {
    /// Unconditionally accepted regardless of caller stack `C`.
    pub accepted: CompactTokenSet,
    /// Accepted in O(1) whenever `expand(C)` is non-empty (exact end-of-token pop).
    pub accepted_if_caller_live: CompactTokenSet,
    /// Context-dependent tokens grouped by identical post-pop suffix (checked once per suffix!).
    pub suffix_groups: Vec<SuffixGroup>,
    /// General context-dependent tokens requiring per-token runtime PDA check against `[C, p]`.
    pub uncertain: Vec<u32>,
    /// Rare fallback tokens checked only if `expand(C)` is dead (at `MAX_RECURSION_DEPTH`).
    pub uncertain_if_caller_dead: Vec<u32>,
    /// Total context-dependent tokens (`sum(suffix_groups) + uncertain.len()`).
    pub total_context_dependent_tokens: usize,
}

impl PositionMask {
    pub fn memory_bytes(&self) -> usize {
        let sg_bytes: usize = self
            .suffix_groups
            .iter()
            .map(SuffixGroup::memory_bytes)
            .sum();
        std::mem::size_of::<Self>()
            + self.accepted.memory_bytes()
            + self.accepted_if_caller_live.memory_bytes()
            + self.suffix_groups.capacity() * std::mem::size_of::<SuffixGroup>()
            + sg_bytes
            + self.uncertain.capacity() * std::mem::size_of::<u32>()
            + self.uncertain_if_caller_dead.capacity() * std::mem::size_of::<u32>()
    }
}

/// Compile-time diagnostic statistics for a [`CompiledGrammar`].
#[derive(Debug, Clone)]
pub struct CompileStats {
    pub vocab_size: usize,
    pub num_rules: usize,
    pub num_leaf_positions: usize,
    pub compile_duration: Duration,
    pub memory_bytes: usize,
    /// Average fraction of vocabulary that is context-independent (accepted + rejected) across leaves.
    pub mean_context_independent_fraction: f64,
    /// Average fraction of vocabulary that is context-dependent across leaves.
    pub mean_context_dependent_fraction: f64,
    /// Maximum context-dependent token count on any single leaf position.
    pub max_uncertain_tokens: usize,
}

/// Compiled grammar with precomputed per-leaf XGrammar token masks.
#[derive(Debug, Clone)]
pub struct CompiledGrammar {
    grammar: Arc<Grammar>,
    vocab: Arc<Vocabulary>,
    config: PdaConfig,
    position_masks: Vec<PositionMask>,
    stats: CompileStats,
}

impl CompiledGrammar {
    /// Compile a [`Grammar`] against a [`Vocabulary`] using the compact [`VocabTrie`]
    /// to precompute context-independent masks, suffix groups, and uncertain token lists.
    pub fn compile(grammar: Arc<Grammar>, vocab: Arc<Vocabulary>, config: PdaConfig) -> Self {
        let start = Instant::now();
        let num_leaves = grammar.num_leaf_positions();
        let vocab_size = vocab.len();
        let mut position_masks = Vec::with_capacity(num_leaves);
        let mut sim_scratch = SimScratch::default();

        let mut total_cd = 0usize;
        let mut max_cd = 0usize;

        for leaf_id in 0..num_leaves {
            let pos = grammar.leaf_pos(leaf_id as LeafId);
            let pm = compile_leaf_mask_via_trie(&grammar, &vocab, pos, config, &mut sim_scratch);
            total_cd += pm.total_context_dependent_tokens;
            max_cd = max_cd.max(pm.total_context_dependent_tokens);
            position_masks.push(pm);
        }

        let compile_duration = start.elapsed();
        let mut memory_bytes = std::mem::size_of::<Self>() + grammar.memory_bytes();
        memory_bytes += position_masks.capacity() * std::mem::size_of::<PositionMask>();
        for pm in &position_masks {
            memory_bytes += pm.memory_bytes();
        }

        let denom = (num_leaves.max(1) * vocab_size.max(1)) as f64;
        let mean_cd = total_cd as f64 / denom;
        let stats = CompileStats {
            vocab_size,
            num_rules: grammar.num_rules(),
            num_leaf_positions: num_leaves,
            compile_duration,
            memory_bytes,
            mean_context_independent_fraction: (1.0 - mean_cd).clamp(0.0, 1.0),
            mean_context_dependent_fraction: mean_cd.clamp(0.0, 1.0),
            max_uncertain_tokens: max_cd,
        };

        Self {
            grammar,
            vocab,
            config,
            position_masks,
            stats,
        }
    }

    pub fn grammar(&self) -> &Arc<Grammar> {
        &self.grammar
    }

    pub fn vocab(&self) -> &Arc<Vocabulary> {
        &self.vocab
    }

    pub fn config(&self) -> PdaConfig {
        self.config
    }

    pub fn stats(&self) -> &CompileStats {
        &self.stats
    }

    pub fn memory_bytes(&self) -> usize {
        self.stats.memory_bytes
    }

    #[inline]
    pub fn position_mask(&self, leaf_id: LeafId) -> &PositionMask {
        &self.position_masks[leaf_id as usize]
    }
}

/// Compile a single leaf position's partition by walking [`VocabTrie`] iteratively.
fn compile_leaf_mask_via_trie(
    grammar: &Grammar,
    vocab: &Vocabulary,
    pos: RulePos,
    config: PdaConfig,
    sim_scratch: &mut SimScratch,
) -> PositionMask {
    let vocab_size = vocab.len();
    let trie = vocab.trie();
    let Some(first_bs) = grammar.byte_set_at(pos) else {
        return PositionMask {
            accepted: CompactTokenSet::Empty,
            accepted_if_caller_live: CompactTokenSet::Empty,
            suffix_groups: Vec::new(),
            uncertain: Vec::new(),
            uncertain_if_caller_dead: Vec::new(),
            total_context_dependent_tokens: 0,
        };
    };

    let mut accepted_ids: Vec<u32> = Vec::new();
    let mut caller_live_ids: Vec<u32> = Vec::new();
    let mut suffix_map: HashMap<Vec<u8>, Vec<u32>> = HashMap::new();
    let mut uncertain_ids: Vec<u32> = Vec::new();
    let mut caller_dead_uncertain_ids: Vec<u32> = Vec::new();

    let init_active = vec![SimStack {
        frames: SmallStack::from_single(pos),
        max_len: 1,
    }];

    // Iterative DFS worklist: (node_idx, active_sim_stacks, popped_mid_before_node)
    let mut stack: Vec<(u32, Vec<SimStack>, bool)> = Vec::new();

    // First-byte filter at trie root (skips ~250/256 root edges on structural states!).
    let root = trie.node(0);
    for &(byte, child_idx) in &root.children {
        if !first_bs.contains(byte) {
            continue;
        }
        let (next_active, popped_on_byte) =
            step_sim_byte(grammar, &init_active, byte, config, sim_scratch);
        handle_trie_transition(
            trie,
            child_idx,
            next_active,
            false,
            popped_on_byte,
            &mut accepted_ids,
            &mut caller_live_ids,
            &mut suffix_map,
            &mut uncertain_ids,
            &mut caller_dead_uncertain_ids,
            &mut stack,
        );
    }

    while let Some((node_idx, cur_active, cur_popped_mid)) = stack.pop() {
        let node = trie.node(node_idx);
        for &(byte, child_idx) in &node.children {
            let (next_active, popped_on_byte) =
                step_sim_byte(grammar, &cur_active, byte, config, sim_scratch);
            handle_trie_transition(
                trie,
                child_idx,
                next_active,
                cur_popped_mid,
                popped_on_byte,
                &mut accepted_ids,
                &mut caller_live_ids,
                &mut suffix_map,
                &mut uncertain_ids,
                &mut caller_dead_uncertain_ids,
                &mut stack,
            );
        }
    }

    let mut suffix_groups_count = 0usize;
    let mut suffix_groups: Vec<SuffixGroup> = Vec::new();
    let mut pairs: Vec<(Vec<u8>, Vec<u32>)> = suffix_map.into_iter().collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    for (suffix, mut tids) in pairs {
        tids.sort_unstable();
        tids.dedup();
        if tids.len() >= 4 {
            suffix_groups_count += tids.len();
            suffix_groups.push(SuffixGroup {
                suffix,
                tokens: CompactTokenSet::from_sorted_ids(tids, vocab_size),
            });
        } else {
            uncertain_ids.extend_from_slice(&tids);
        }
    }

    accepted_ids.sort_unstable();
    caller_live_ids.sort_unstable();
    uncertain_ids.sort_unstable();
    uncertain_ids.dedup();
    caller_dead_uncertain_ids.sort_unstable();

    let total_context_dependent_tokens = suffix_groups_count + uncertain_ids.len();

    PositionMask {
        accepted: CompactTokenSet::from_sorted_ids(accepted_ids, vocab_size),
        accepted_if_caller_live: CompactTokenSet::from_sorted_ids(caller_live_ids, vocab_size),
        suffix_groups,
        uncertain: uncertain_ids,
        uncertain_if_caller_dead: caller_dead_uncertain_ids,
        total_context_dependent_tokens,
    }
}

/// Collect all strict descendants of `parent_idx` in `trie` grouped by their relative
/// byte suffix after `parent_idx` (exact post-pop remaining bytes for caller stack `C`).
fn collect_post_pop_suffixes(
    trie: &VocabTrie,
    parent_idx: u32,
    suffix_map: &mut HashMap<Vec<u8>, Vec<u32>>,
) {
    let parent = trie.node(parent_idx);
    let mut dfs: Vec<(u32, Vec<u8>)> = Vec::new();
    for &(b, gc_idx) in &parent.children {
        dfs.push((gc_idx, vec![b]));
    }
    while let Some((u, path)) = dfs.pop() {
        let node = trie.node(u);
        if !node.tokens.is_empty() {
            suffix_map
                .entry(path.clone())
                .or_default()
                .extend_from_slice(&node.tokens);
        }
        for &(b, child_idx) in &node.children {
            let mut next_path = path.clone();
            next_path.push(b);
            dfs.push((child_idx, next_path));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_trie_transition(
    trie: &VocabTrie,
    child_idx: u32,
    next_active: Vec<SimStack>,
    cur_popped_mid: bool,
    popped_on_byte: bool,
    accepted_ids: &mut Vec<u32>,
    caller_live_ids: &mut Vec<u32>,
    suffix_map: &mut HashMap<Vec<u8>, Vec<u32>>,
    uncertain_ids: &mut Vec<u32>,
    caller_dead_uncertain_ids: &mut Vec<u32>,
    stack: &mut Vec<(u32, Vec<SimStack>, bool)>,
) {
    let child = trie.node(child_idx);
    if next_active.is_empty() {
        if !cur_popped_mid && !popped_on_byte {
            // Entire subtree is Context-Independent REJECTED -> O(1) prune!
            return;
        }
        if !cur_popped_mid && popped_on_byte {
            // Clean single-point exit from `p` right at `child`!
            // 1. Tokens ending right at `child` have 0 remaining bytes -> O(1) caller-live mask.
            for &tid in &child.tokens {
                caller_live_ids.push(tid);
            }
            // 2. Strictly deeper descendants have exact remaining suffix relative to `child`!
            if !child.children.is_empty() {
                collect_post_pop_suffixes(trie, child_idx, suffix_map);
            }
        } else if popped_on_byte {
            // Both earlier mid-pop and pop at `child`:
            for &tid in &child.tokens {
                caller_live_ids.push(tid);
                caller_dead_uncertain_ids.push(tid);
            }
            for &(_, grand_child_idx) in &child.children {
                let gc = trie.node(grand_child_idx);
                uncertain_ids.extend_from_slice(trie.subtree_tokens(gc));
            }
        } else {
            // `cur_popped_mid == true` and no survivors -> entire subtree is uncertain.
            uncertain_ids.extend_from_slice(trie.subtree_tokens(child));
        }
        return;
    }

    // At least one branch survived inside the rule!
    let has_depth1 = next_active.iter().any(|s| s.max_len == 1);
    for &tid in &child.tokens {
        if has_depth1 {
            accepted_ids.push(tid);
        } else if popped_on_byte {
            caller_live_ids.push(tid);
            caller_dead_uncertain_ids.push(tid);
        } else {
            uncertain_ids.push(tid);
        }
    }

    if !child.children.is_empty() {
        stack.push((child_idx, next_active, cur_popped_mid || popped_on_byte));
    }
}

/// Per-step telemetry returned by [`XGrammarMatcher::compute_mask`].
#[derive(Debug, Clone, Copy, Default)]
pub struct StepMaskStats {
    pub active_stacks: usize,
    /// Number of runtime PDA evaluations performed (unique suffix checks + per-token uncertain checks).
    pub runtime_pda_checks: usize,
    /// Total context-dependent tokens covered by active leaf positions (for reporting).
    pub context_dependent_tokens: usize,
    pub allowed_tokens: usize,
}

/// Fast runtime constrained-decoding matcher backed by a [`CompiledGrammar`].
pub struct XGrammarMatcher {
    compiled: Arc<CompiledGrammar>,
    state: PdaState,
    scratch: PdaScratch,
    caller_work: Vec<Stack>,
    caller_leaves_buf: Vec<Vec<Stack>>,
}

impl XGrammarMatcher {
    pub fn new(compiled: Arc<CompiledGrammar>) -> Self {
        let state = PdaState::new(compiled.grammar(), compiled.config());
        Self {
            compiled,
            state,
            scratch: PdaScratch::default(),
            caller_work: Vec::new(),
            caller_leaves_buf: Vec::new(),
        }
    }

    pub fn compiled(&self) -> &Arc<CompiledGrammar> {
        &self.compiled
    }

    pub fn state(&self) -> &PdaState {
        &self.state
    }

    pub fn can_accept(&self) -> bool {
        self.state.can_accept()
    }

    pub fn checkpoint(&self) -> PdaCheckpoint {
        self.state.checkpoint()
    }

    pub fn rollback(&mut self, cp: PdaCheckpoint) {
        self.state.rollback(cp);
    }

    /// Compute the exact allowed-token [`BitMask`] for the current decode step.
    pub fn compute_mask(&mut self, out: &mut BitMask) -> StepMaskStats {
        out.clear();
        let grammar = self.compiled.grammar();
        let vocab = self.compiled.vocab();
        let config = self.compiled.config();

        if self.state.can_accept() {
            out.set(vocab.eos_token_id());
        }

        let leaves = self.state.leaves();
        if self.caller_leaves_buf.len() < leaves.len() {
            self.caller_leaves_buf.resize_with(leaves.len(), Vec::new);
        }

        let mut runtime_pda_checks = 0usize;
        let mut context_dependent_tokens = 0usize;

        // Pass 1: O(1) union of precomputed context-independent masks & expand caller stack once.
        for (idx, &stack) in leaves.iter().enumerate() {
            let caller_buf = &mut self.caller_leaves_buf[idx];
            caller_buf.clear();

            let Some(top) = stack.last() else {
                continue;
            };
            let Some(leaf_id) = grammar.leaf_id_at(top) else {
                continue;
            };
            let pm = self.compiled.position_mask(leaf_id);
            pm.accepted.or_into(out);
            context_dependent_tokens += pm.total_context_dependent_tokens;

            let need_caller = !pm.accepted_if_caller_live.is_empty()
                || !pm.suffix_groups.is_empty()
                || !pm.uncertain_if_caller_dead.is_empty();
            if need_caller {
                let live =
                    expand_caller_stack(grammar, stack, config, &mut self.caller_work, caller_buf);
                if live {
                    pm.accepted_if_caller_live.or_into(out);
                }
            }
        }

        // Pass 2: Evaluate unique post-pop suffix groups (O(unique_suffixes) instead of O(tokens)!)
        // and any remaining general context-dependent (`uncertain`) tokens.
        for (idx, &stack) in leaves.iter().enumerate() {
            let Some(top) = stack.last() else {
                continue;
            };
            let Some(leaf_id) = grammar.leaf_id_at(top) else {
                continue;
            };
            let pm = self.compiled.position_mask(leaf_id);
            let caller_leaves = &self.caller_leaves_buf[idx];

            if !caller_leaves.is_empty() {
                for sg in &pm.suffix_groups {
                    runtime_pda_checks += 1;
                    if leaves_can_consume_bytes(
                        grammar,
                        caller_leaves,
                        &sg.suffix,
                        config,
                        &mut self.scratch,
                    ) {
                        sg.tokens.or_into(out);
                    }
                }
            }

            for &tid in &pm.uncertain {
                if out.get(tid) {
                    continue;
                }
                runtime_pda_checks += 1;
                if stack_can_consume_bytes(
                    grammar,
                    stack,
                    vocab.token_bytes(tid),
                    config,
                    &mut self.scratch,
                ) {
                    out.set(tid);
                }
            }

            if caller_leaves.is_empty() && !pm.uncertain_if_caller_dead.is_empty() {
                for &tid in &pm.uncertain_if_caller_dead {
                    if out.get(tid) {
                        continue;
                    }
                    runtime_pda_checks += 1;
                    if stack_can_consume_bytes(
                        grammar,
                        stack,
                        vocab.token_bytes(tid),
                        config,
                        &mut self.scratch,
                    ) {
                        out.set(tid);
                    }
                }
            }
        }

        StepMaskStats {
            active_stacks: leaves.len(),
            runtime_pda_checks,
            context_dependent_tokens,
            allowed_tokens: out.count_ones(),
        }
    }

    /// Consume a sampled token ID and advance the PDA state.
    pub fn consume_token(&mut self, token_id: u32) -> bool {
        let vocab = self.compiled.vocab();
        if token_id == vocab.eos_token_id() {
            return self.state.can_accept();
        }
        let bytes = vocab.token_bytes(token_id);
        self.state.advance_bytes(self.compiled.grammar(), bytes)
    }

    /// Compute bitmask and apply `-inf` directly to `logits`.
    pub fn mask_logits(&mut self, logits: &mut [f32], mask_buf: &mut BitMask) -> StepMaskStats {
        let stats = self.compute_mask(mask_buf);
        mask_buf.apply_to_logits(logits);
        stats
    }
}

/// Matched uncompiled baseline: evaluates the EXACT same PDA [`Grammar`] across all
/// `V` vocabulary tokens on every decode step (simulating standard per-step constrained
/// decoding without XGrammar's precomputed context-independent masks).
pub struct NaivePerStepConstrainedDecoder {
    grammar: Arc<Grammar>,
    vocab: Arc<Vocabulary>,
    state: PdaState,
    scratch: PdaScratch,
}

impl NaivePerStepConstrainedDecoder {
    pub fn new(grammar: Arc<Grammar>, vocab: Arc<Vocabulary>, config: PdaConfig) -> Self {
        let state = PdaState::new(&grammar, config);
        Self {
            grammar,
            vocab,
            state,
            scratch: PdaScratch::default(),
        }
    }

    pub fn can_accept(&self) -> bool {
        self.state.can_accept()
    }

    pub fn checkpoint(&self) -> PdaCheckpoint {
        self.state.checkpoint()
    }

    pub fn rollback(&mut self, cp: PdaCheckpoint) {
        self.state.rollback(cp);
    }

    /// Compute the exact allowed-token [`BitMask`] by checking all `V` tokens against
    /// the current PDA state.
    pub fn compute_mask(&mut self, out: &mut BitMask) {
        out.clear();
        let eos = self.vocab.eos_token_id();
        if self.state.can_accept() {
            out.set(eos);
        }
        for tid in 0..(self.vocab.len() as u32) {
            if tid == eos {
                continue;
            }
            let bytes = self.vocab.token_bytes(tid);
            if bytes.is_empty() {
                continue;
            }
            if self
                .state
                .can_consume_bytes(&self.grammar, bytes, &mut self.scratch)
            {
                out.set(tid);
            }
        }
    }

    pub fn consume_token(&mut self, token_id: u32) -> bool {
        if token_id == self.vocab.eos_token_id() {
            return self.state.can_accept();
        }
        let bytes = self.vocab.token_bytes(token_id);
        self.state.advance_bytes(&self.grammar, bytes)
    }

    pub fn mask_logits(&mut self, logits: &mut [f32], mask_buf: &mut BitMask) {
        self.compute_mask(mask_buf);
        mask_buf.apply_to_logits(logits);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{
        build_tool_call_schema, JsonSchema, SchemaDocument, ToolEnvelopeFormat, ToolSpec,
    };

    fn assert_exact_mask_parity_on_sequence(
        grammar: Arc<Grammar>,
        vocab: Arc<Vocabulary>,
        config: PdaConfig,
        target_text: &str,
    ) {
        let compiled = Arc::new(CompiledGrammar::compile(
            Arc::clone(&grammar),
            Arc::clone(&vocab),
            config,
        ));
        let mut xgm = XGrammarMatcher::new(compiled);
        let mut naive = NaivePerStepConstrainedDecoder::new(grammar, Arc::clone(&vocab), config);
        let mut xm = BitMask::new_empty(vocab.len());
        let mut nm = BitMask::new_empty(vocab.len());

        let bytes = target_text.as_bytes();
        let mut pos = 0usize;
        while pos < bytes.len() {
            xgm.compute_mask(&mut xm);
            naive.compute_mask(&mut nm);
            assert_eq!(xm, nm, "bitmask mismatch before offset {pos}");

            let mut best: Option<(u32, usize)> = None;
            for tid in 1..(vocab.len() as u32) {
                let tb = vocab.token_bytes(tid);
                if !tb.is_empty()
                    && bytes[pos..].starts_with(tb)
                    && best.is_none_or(|(_, blen)| tb.len() > blen)
                {
                    best = Some((tid, tb.len()));
                }
            }
            let (tid, tlen) = best.expect("byte-complete vocab always matches");
            assert!(
                xm.get(tid),
                "target token {tid} ({:?}) must be allowed",
                vocab.token_bytes(tid)
            );
            assert!(xgm.consume_token(tid));
            assert!(naive.consume_token(tid));
            pos += tlen;
        }

        xgm.compute_mask(&mut xm);
        naive.compute_mask(&mut nm);
        assert_eq!(xm, nm, "bitmask mismatch at final EOS step");
        assert!(xm.get(vocab.eos_token_id()));
    }

    #[test]
    fn bit_identical_mask_parity_json_object_utf8_and_escapes() {
        let vocab = Arc::new(Vocabulary::realistic_bpe(4096));
        let schema = SchemaDocument::from_json_str(
            r#"{
                "type": "object",
                "properties": {
                    "status": {"enum": ["ok", "retry"]},
                    "msg": {"type": "string"},
                    "nums": {"type": "array", "items": {"type": "number"}},
                    "flag": {"type": "boolean"}
                },
                "required": ["status", "msg", "nums", "flag"],
                "additionalProperties": false
            }"#,
        )
        .unwrap();
        let grammar = Arc::new(schema.compile_grammar().unwrap());
        let sample = r#"{"status": "ok", "msg": "hi\n\"日本語🚀\u0041", "nums": [0, -12, 3.14e-2], "flag": true}"#;
        assert_exact_mask_parity_on_sequence(grammar, vocab, PdaConfig::default(), sample);
    }

    #[test]
    fn bit_identical_mask_parity_recursive_ast_at_depth_limit() {
        let vocab = Arc::new(Vocabulary::realistic_bpe(2048));
        let schema = SchemaDocument::from_json_str(
            r##"{
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {
                            "k": {"enum": ["a", "b"]},
                            "c": {"type": "array", "items": {"$ref": "#/$defs/Node"}}
                        },
                        "required": ["k", "c"],
                        "additionalProperties": false
                    }
                },
                "$ref": "#/$defs/Node"
            }"##,
        )
        .unwrap();
        let grammar = Arc::new(schema.compile_grammar().unwrap());
        let cfg = PdaConfig {
            max_recursion_depth: 10,
            max_active_stacks: 128,
        };
        let sample = r#"{"k":"a","c":[{"k":"b","c":[]}]}"#;
        assert_exact_mask_parity_on_sequence(grammar, vocab, cfg, sample);
    }

    #[test]
    fn bit_identical_mask_parity_multi_tool_mcp_envelope() {
        let vocab = Arc::new(Vocabulary::realistic_bpe(4096));
        let doc = build_tool_call_schema(
            &[
                ToolSpec::new(
                    "search_code",
                    JsonSchema::Object {
                        properties: vec![
                            ("query".into(), JsonSchema::String),
                            ("limit".into(), JsonSchema::Integer),
                        ],
                        required: vec!["query".into(), "limit".into()],
                        additional_properties: false,
                    },
                ),
                ToolSpec::new(
                    "run_tests",
                    JsonSchema::Object {
                        properties: vec![("package".into(), JsonSchema::String)],
                        required: vec!["package".into()],
                        additional_properties: false,
                    },
                ),
            ],
            ToolEnvelopeFormat::ToolAndArguments,
        )
        .unwrap();
        let grammar = Arc::new(doc.compile_grammar().unwrap());
        let sample = r#"{"tool": "search_code", "arguments": {"query": "xgrammar", "limit": 25}}"#;
        assert_exact_mask_parity_on_sequence(grammar, vocab, PdaConfig::default(), sample);
    }

    #[test]
    fn apply_to_logits_masks_disallowed_to_neg_infinity() {
        let mut bm = BitMask::new_empty(130);
        bm.set(0);
        bm.set(63);
        bm.set(64);
        bm.set(129);
        let mut logits: Vec<f32> = (0..130).map(|i| i as f32).collect();
        bm.apply_to_logits(&mut logits);
        for (i, &v) in logits.iter().enumerate() {
            if matches!(i, 0 | 63 | 64 | 129) {
                assert_eq!(v, i as f32);
            } else {
                assert!(v.is_infinite() && v.is_sign_negative(), "idx {i} not -inf");
            }
        }
    }

    #[test]
    fn partial_multibyte_utf8_tokens_and_rollback_parity() {
        let vocab = Arc::new(Vocabulary::realistic_bpe(2048));
        let schema = SchemaDocument::from_json_str(r#"{"type":"string"}"#).unwrap();
        let grammar = Arc::new(schema.compile_grammar().unwrap());
        let compiled = Arc::new(CompiledGrammar::compile(
            Arc::clone(&grammar),
            Arc::clone(&vocab),
            PdaConfig::default(),
        ));
        let mut xgm = XGrammarMatcher::new(compiled);
        let mut naive =
            NaivePerStepConstrainedDecoder::new(grammar, Arc::clone(&vocab), PdaConfig::default());
        let mut xm = BitMask::new_empty(vocab.len());
        let mut nm = BitMask::new_empty(vocab.len());

        let find_tok = |bytes: &[u8]| -> u32 {
            (1..vocab.len() as u32)
                .find(|&t| vocab.token_bytes(t) == bytes)
                .expect("token present")
        };
        let quote = find_tok(b"\"");
        let split_hi = find_tok(&[0xE4, 0xB8]);
        let split_lo = find_tok(&[0xAD]);
        let ascii_space = find_tok(b" ");

        assert!(xgm.consume_token(quote));
        assert!(naive.consume_token(quote));

        xgm.compute_mask(&mut xm);
        naive.compute_mask(&mut nm);
        assert_eq!(xm, nm);
        assert!(xm.get(split_hi));

        let cp_x = xgm.checkpoint();
        let cp_n = naive.checkpoint();
        assert!(xgm.consume_token(split_hi));
        assert!(naive.consume_token(split_hi));

        xgm.compute_mask(&mut xm);
        naive.compute_mask(&mut nm);
        assert_eq!(xm, nm);
        assert!(xm.get(split_lo));
        assert!(!xm.get(ascii_space));
        assert!(!xm.get(quote));

        xgm.rollback(cp_x);
        naive.rollback(cp_n);
        xgm.compute_mask(&mut xm);
        naive.compute_mask(&mut nm);
        assert_eq!(xm, nm);
        assert!(xm.get(quote));

        assert!(xgm.consume_token(split_hi));
        assert!(xgm.consume_token(split_lo));
        assert!(xgm.consume_token(quote));
        assert!(xgm.can_accept());
    }

    #[test]
    fn bit_identical_mask_parity_optional_and_additional_properties() {
        let vocab = Arc::new(Vocabulary::realistic_bpe(2048));
        let schema = SchemaDocument::from_json_str(
            r#"{
                "type": "object",
                "properties": {
                    "req": {"type": "integer"},
                    "opt1": {"type": "string"},
                    "opt2": {"type": "boolean"}
                },
                "required": ["req"],
                "additionalProperties": true
            }"#,
        )
        .unwrap();
        let grammar = Arc::new(schema.compile_grammar().unwrap());
        assert_exact_mask_parity_on_sequence(
            grammar,
            vocab,
            PdaConfig::default(),
            r#"{"req": 7, "opt2": false, "extra_k": [1, "x", null]}"#,
        );
    }

    #[test]
    fn bit_identical_mask_parity_ambiguous_context_heavy_cfg() {
        use crate::grammar::{ByteSet, Element, GrammarBuilder};
        let mut b = GrammarBuilder::new();
        let expr = b.declare_rule("expr").unwrap();
        let atom = b.declare_rule("atom").unwrap();
        let ident = ByteSet::from_range(b'a', b'z');
        let mut op = ByteSet::empty();
        op.insert(b'+');
        op.insert(b' ');
        b.add_alt(atom, vec![Element::Byte(ident)]).unwrap();
        b.add_alt(
            atom,
            vec![
                Element::Byte(ByteSet::from_byte(b'(')),
                Element::RuleRef(expr),
                Element::Byte(ByteSet::from_byte(b')')),
            ],
        )
        .unwrap();
        b.add_alt(expr, vec![Element::RuleRef(atom)]).unwrap();
        b.add_alt(
            expr,
            vec![
                Element::RuleRef(atom),
                Element::Byte(op),
                Element::RuleRef(expr),
            ],
        )
        .unwrap();
        b.add_alt(expr, vec![Element::RuleRef(atom), Element::RuleRef(expr)])
            .unwrap();
        let grammar = Arc::new(b.finalize(expr).unwrap());
        let vocab = Arc::new(Vocabulary::realistic_bpe(2048));
        assert_exact_mask_parity_on_sequence(grammar, vocab, PdaConfig::default(), "(ab+(cd+ef))");
    }
}
