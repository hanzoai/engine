//! Byte-level Pushdown Automaton (PDA) Grammar IR and Execution Engine.
//!
//! Clean-room design inspired by XGrammar's persistent-stack PDA bytecode:
//! - Rules consist of alternatives, each a sequence of [`Element::Byte`]
//!   (256-bit byte class) or [`Element::RuleRef`] (non-terminal invocation).
//! - Compact 4-byte [`RulePos`] and inline `Copy` [`SmallStack`] eliminate
//!   heap allocations in the hot decoding and compilation loops.
//! - Precomputed `FIRST` sets (`alt_first_bytes`) and nullability (`alt_nullable`)
//!   enable next-byte lookahead pruning during multi-byte token evaluation.
//! - Tail-Call Optimization (TCO): when a `RuleRef` is the last element of an
//!   alternative, the frame is replaced in-place rather than pushing a return
//!   continuation, keeping stack depth O(1) for strings, numbers, whitespace,
//!   and flat arrays while preserving exact nesting depth for recursive rules.
//! - Compile-time left-recursion detection guarantees epsilon-expansion always
//!   terminates in bounded steps.
//! - Deterministic checkpoint/rollback and hard DoS bounds
//!   (`MAX_RECURSION_DEPTH`, `MAX_ACTIVE_STACKS`, `MAX_GRAMMAR_RULES`,
//!   `MAX_GRAMMAR_ELEMENTS`).

use std::fmt;

/// Default maximum non-tail PDA call stack depth.
pub const DEFAULT_MAX_RECURSION_DEPTH: usize = 64;
/// Hard compile-time inline capacity of [`SmallStack`].
pub const MAX_STACK_CAPACITY: usize = 64;
/// Default maximum concurrent nondeterministic PDA stacks per decode step.
pub const DEFAULT_MAX_ACTIVE_STACKS: usize = 256;
/// Hard cap on rules per grammar to prevent schema-bomb DoS.
pub const MAX_GRAMMAR_RULES: usize = 4096;
/// Hard cap on total bytecode elements per grammar to prevent schema-bomb DoS.
pub const MAX_GRAMMAR_ELEMENTS: usize = 65_536;

/// Dense 256-bit byte character class (`[u64; 4]`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ByteSet(pub [u64; 4]);

impl ByteSet {
    pub const fn empty() -> Self {
        Self([0; 4])
    }

    pub const fn all() -> Self {
        Self([u64::MAX; 4])
    }

    pub const fn from_byte(b: u8) -> Self {
        let mut words = [0u64; 4];
        words[(b >> 6) as usize] = 1u64 << (b & 63);
        Self(words)
    }

    pub const fn from_range(lo: u8, hi: u8) -> Self {
        let mut words = [0u64; 4];
        let mut b = lo as u16;
        let end = hi as u16;
        while b <= end {
            let idx = (b >> 6) as usize;
            words[idx] |= 1u64 << (b & 63);
            b += 1;
        }
        Self(words)
    }

    pub fn insert(&mut self, b: u8) {
        self.0[(b >> 6) as usize] |= 1u64 << (b & 63);
    }

    pub fn remove(&mut self, b: u8) {
        self.0[(b >> 6) as usize] &= !(1u64 << (b & 63));
    }

    pub const fn union(self, other: Self) -> Self {
        Self([
            self.0[0] | other.0[0],
            self.0[1] | other.0[1],
            self.0[2] | other.0[2],
            self.0[3] | other.0[3],
        ])
    }

    #[inline(always)]
    pub const fn contains(&self, b: u8) -> bool {
        ((self.0[(b >> 6) as usize] >> (b & 63)) & 1) != 0
    }

    pub const fn is_empty(&self) -> bool {
        (self.0[0] | self.0[1] | self.0[2] | self.0[3]) == 0
    }

    pub fn count(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }
}

impl fmt::Debug for ByteSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ByteSet(count={})", self.count())
    }
}

/// Identifier of a grammar rule (`0..rules.len()`).
pub type RuleId = u16;

/// Dense identifier of a terminal (`Element::Byte`) leaf position in a finalized grammar.
pub type LeafId = u32;

/// Single bytecode instruction inside a rule alternative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    /// Consume a single byte belonging to the 256-bit class.
    Byte(ByteSet),
    /// Invoke another rule (tail-call optimized if at end of alternative).
    RuleRef(RuleId),
}

/// A named grammar rule with one or more alternative sequences.
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub alts: Vec<Vec<Element>>,
    /// Parallel table mapping `[alt][elem]` -> `Some(LeafId)` for `Element::Byte`, else `None`.
    pub(crate) leaf_ids: Vec<Vec<Option<LeafId>>>,
    /// Precomputed FIRST byte set for each alternative (used for lookahead pruning).
    pub(crate) alt_first: Vec<ByteSet>,
    /// Precomputed nullability for each alternative.
    pub(crate) alt_nullable: Vec<bool>,
}

/// Compact cursor position `(rule, alt, elem)` inside grammar bytecode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct RulePos {
    pub rule: RuleId,
    pub alt: u16,
    pub elem: u16,
}

/// Inline, zero-heap-allocation PDA call stack (`Copy`, bounded by [`MAX_STACK_CAPACITY`]).
///
/// An empty stack (`len == 0`) represents the `Accept` (EOF/EOS) state.
#[derive(Clone, Copy)]
pub struct SmallStack {
    len: u8,
    frames: [RulePos; MAX_STACK_CAPACITY],
}

impl Default for SmallStack {
    fn default() -> Self {
        Self::new()
    }
}

impl SmallStack {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            len: 0,
            frames: [RulePos {
                rule: 0,
                alt: 0,
                elem: 0,
            }; MAX_STACK_CAPACITY],
        }
    }

    #[inline(always)]
    pub fn from_single(pos: RulePos) -> Self {
        let mut s = Self::new();
        s.frames[0] = pos;
        s.len = 1;
        s
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[RulePos] {
        &self.frames[..self.len as usize]
    }

    #[inline(always)]
    pub fn last(&self) -> Option<RulePos> {
        if self.len == 0 {
            None
        } else {
            Some(self.frames[self.len as usize - 1])
        }
    }

    #[inline(always)]
    pub fn last_mut(&mut self) -> Option<&mut RulePos> {
        if self.len == 0 {
            None
        } else {
            Some(&mut self.frames[self.len as usize - 1])
        }
    }

    #[inline(always)]
    pub fn push(&mut self, pos: RulePos) {
        debug_assert!((self.len as usize) < MAX_STACK_CAPACITY);
        self.frames[self.len as usize] = pos;
        self.len += 1;
    }

    #[inline(always)]
    pub fn pop(&mut self) -> Option<RulePos> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            Some(self.frames[self.len as usize])
        }
    }

    #[inline(always)]
    pub fn without_top(&self) -> Self {
        let mut s = *self;
        if s.len > 0 {
            s.len -= 1;
        }
        s
    }
}

impl PartialEq for SmallStack {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.as_slice() == other.as_slice()
    }
}

impl Eq for SmallStack {}

impl PartialOrd for SmallStack {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SmallStack {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl fmt::Debug for SmallStack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

/// Alias for PDA execution stacks.
pub type Stack = SmallStack;

/// Runtime safety limits for the PDA engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PdaConfig {
    pub max_recursion_depth: usize,
    pub max_active_stacks: usize,
}

impl Default for PdaConfig {
    fn default() -> Self {
        Self {
            max_recursion_depth: DEFAULT_MAX_RECURSION_DEPTH,
            max_active_stacks: DEFAULT_MAX_ACTIVE_STACKS,
        }
    }
}

impl PdaConfig {
    #[inline(always)]
    fn effective_depth(&self) -> usize {
        self.max_recursion_depth.min(MAX_STACK_CAPACITY)
    }
}

/// Error returned during grammar construction or validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarError {
    TooManyRules { count: usize, max: usize },
    TooManyElements { count: usize, max: usize },
    EmptyRule { rule: String },
    InvalidRuleRef { from: String, target: RuleId },
    LeftRecursionDetected { rule: String },
    InvalidRootRule { root: RuleId },
}

impl fmt::Display for GrammarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GrammarError::TooManyRules { count, max } => {
                write!(f, "grammar exceeds max rules ({count} > {max})")
            }
            GrammarError::TooManyElements { count, max } => {
                write!(f, "grammar exceeds max elements ({count} > {max})")
            }
            GrammarError::EmptyRule { rule } => {
                write!(f, "grammar rule '{rule}' has no alternatives")
            }
            GrammarError::InvalidRuleRef { from, target } => {
                write!(f, "rule '{from}' references unknown rule id {target}")
            }
            GrammarError::LeftRecursionDetected { rule } => {
                write!(
                    f,
                    "unguarded left-recursion cycle detected involving rule '{rule}'"
                )
            }
            GrammarError::InvalidRootRule { root } => {
                write!(f, "invalid root rule id {root}")
            }
        }
    }
}

impl std::error::Error for GrammarError {}

/// Finalized, immutable byte-level Pushdown Automaton grammar.
#[derive(Debug, Clone)]
pub struct Grammar {
    rules: Vec<Rule>,
    root: RuleId,
    /// Reverse lookup `LeafId -> RulePos`.
    leaf_positions: Vec<RulePos>,
    /// Total bytecode elements across all rules.
    total_elements: usize,
    /// Stable 64-bit FNV-1a fingerprint of the compiled bytecode.
    fingerprint: u64,
}

/// Mutable builder for constructing a validated [`Grammar`].
#[derive(Debug, Default)]
pub struct GrammarBuilder {
    rules: Vec<Rule>,
    total_elements: usize,
}

impl GrammarBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve a new empty rule slot (allows forward references for recursive schemas).
    pub fn declare_rule(&mut self, name: impl Into<String>) -> Result<RuleId, GrammarError> {
        if self.rules.len() >= MAX_GRAMMAR_RULES {
            return Err(GrammarError::TooManyRules {
                count: self.rules.len() + 1,
                max: MAX_GRAMMAR_RULES,
            });
        }
        let id = self.rules.len() as RuleId;
        self.rules.push(Rule {
            name: name.into(),
            alts: Vec::new(),
            leaf_ids: Vec::new(),
            alt_first: Vec::new(),
            alt_nullable: Vec::new(),
        });
        Ok(id)
    }

    /// Append an alternative sequence to a previously declared rule.
    pub fn add_alt(&mut self, rule: RuleId, seq: Vec<Element>) -> Result<(), GrammarError> {
        let next_total = self.total_elements.saturating_add(seq.len());
        if next_total > MAX_GRAMMAR_ELEMENTS {
            return Err(GrammarError::TooManyElements {
                count: next_total,
                max: MAX_GRAMMAR_ELEMENTS,
            });
        }
        self.total_elements = next_total;
        self.rules[rule as usize].alts.push(seq);
        Ok(())
    }

    /// Add a rule with a set of alternatives in one step.
    pub fn add_rule(
        &mut self,
        name: impl Into<String>,
        alts: Vec<Vec<Element>>,
    ) -> Result<RuleId, GrammarError> {
        let id = self.declare_rule(name)?;
        for alt in alts {
            self.add_alt(id, alt)?;
        }
        Ok(id)
    }

    /// Helper: convert a literal byte string into a sequence of single-byte `Element::Byte`.
    pub fn literal_elements(bytes: &[u8]) -> Vec<Element> {
        bytes
            .iter()
            .map(|&b| Element::Byte(ByteSet::from_byte(b)))
            .collect()
    }

    /// Validate references, check for unguarded left-recursion cycles, precompute
    /// `FIRST` sets & nullability, assign dense [`LeafId`]s, and return [`Grammar`].
    pub fn finalize(mut self, root: RuleId) -> Result<Grammar, GrammarError> {
        if (root as usize) >= self.rules.len() {
            return Err(GrammarError::InvalidRootRule { root });
        }

        // 1. Validate non-empty rules and rule reference bounds.
        let num_rules = self.rules.len();
        for rule in &self.rules {
            if rule.alts.is_empty() {
                return Err(GrammarError::EmptyRule {
                    rule: rule.name.clone(),
                });
            }
            for alt in &rule.alts {
                for elem in alt {
                    if let Element::RuleRef(target) = *elem {
                        if (target as usize) >= num_rules {
                            return Err(GrammarError::InvalidRuleRef {
                                from: rule.name.clone(),
                                target,
                            });
                        }
                    }
                }
            }
        }

        // 2. Compute nullable rules via fixed-point iteration.
        let mut nullable = vec![false; num_rules];
        loop {
            let mut changed = false;
            for (r_idx, rule) in self.rules.iter().enumerate() {
                if nullable[r_idx] {
                    continue;
                }
                let is_null = rule.alts.iter().any(|alt| {
                    alt.iter().all(|elem| match *elem {
                        Element::Byte(_) => false,
                        Element::RuleRef(t) => nullable[t as usize],
                    })
                });
                if is_null {
                    nullable[r_idx] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // 3. Build left-call graph and reject any left-recursion cycle.
        let mut left_edges: Vec<Vec<usize>> = vec![Vec::new(); num_rules];
        for (r_idx, rule) in self.rules.iter().enumerate() {
            for alt in &rule.alts {
                for elem in alt {
                    match *elem {
                        Element::Byte(_) => break,
                        Element::RuleRef(target) => {
                            let t = target as usize;
                            if !left_edges[r_idx].contains(&t) {
                                left_edges[r_idx].push(t);
                            }
                            if !nullable[t] {
                                break;
                            }
                        }
                    }
                }
            }
        }

        let mut color = vec![0u8; num_rules];
        fn dfs_cycle(
            u: usize,
            edges: &[Vec<usize>],
            color: &mut [u8],
            rules: &[Rule],
        ) -> Result<(), GrammarError> {
            color[u] = 1;
            for &v in &edges[u] {
                if color[v] == 1 {
                    return Err(GrammarError::LeftRecursionDetected {
                        rule: rules[v].name.clone(),
                    });
                }
                if color[v] == 0 {
                    dfs_cycle(v, edges, color, rules)?;
                }
            }
            color[u] = 2;
            Ok(())
        }
        for r in 0..num_rules {
            if color[r] == 0 {
                dfs_cycle(r, &left_edges, &mut color, &self.rules)?;
            }
        }

        // 4. Compute FIRST byte sets for every rule and every alternative (fixed-point).
        let mut rule_first = vec![ByteSet::empty(); num_rules];
        loop {
            let mut changed = false;
            for (r_idx, rule) in self.rules.iter().enumerate() {
                let mut acc = rule_first[r_idx];
                for alt in &rule.alts {
                    for elem in alt {
                        match *elem {
                            Element::Byte(bs) => {
                                acc = acc.union(bs);
                                break;
                            }
                            Element::RuleRef(t) => {
                                acc = acc.union(rule_first[t as usize]);
                                if !nullable[t as usize] {
                                    break;
                                }
                            }
                        }
                    }
                }
                if acc != rule_first[r_idx] {
                    rule_first[r_idx] = acc;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // 5. Assign dense LeafId, per-alt FIRST & nullability, and compute stable fingerprint.
        let mut leaf_positions = Vec::new();
        let mut hash: u64 = 0xcbf29ce484222325;
        let mix = |h: &mut u64, v: u64| {
            *h ^= v;
            *h = h.wrapping_mul(0x100000001b3);
        };
        mix(&mut hash, root as u64);
        mix(&mut hash, num_rules as u64);

        for (r_idx, rule) in self.rules.iter_mut().enumerate() {
            mix(&mut hash, rule.alts.len() as u64);
            rule.leaf_ids = Vec::with_capacity(rule.alts.len());
            rule.alt_first = Vec::with_capacity(rule.alts.len());
            rule.alt_nullable = Vec::with_capacity(rule.alts.len());

            for (a_idx, alt) in rule.alts.iter().enumerate() {
                mix(&mut hash, alt.len() as u64);
                let mut row = Vec::with_capacity(alt.len());
                let mut af = ByteSet::empty();
                let mut all_null = true;
                for (e_idx, elem) in alt.iter().enumerate() {
                    match *elem {
                        Element::Byte(bs) => {
                            let lid = leaf_positions.len() as LeafId;
                            leaf_positions.push(RulePos {
                                rule: r_idx as RuleId,
                                alt: a_idx as u16,
                                elem: e_idx as u16,
                            });
                            row.push(Some(lid));
                            if all_null {
                                af = af.union(bs);
                                all_null = false;
                            }
                            mix(&mut hash, 1);
                            for w in bs.0 {
                                mix(&mut hash, w);
                            }
                        }
                        Element::RuleRef(target) => {
                            row.push(None);
                            if all_null {
                                af = af.union(rule_first[target as usize]);
                                if !nullable[target as usize] {
                                    all_null = false;
                                }
                            }
                            mix(&mut hash, 2);
                            mix(&mut hash, target as u64);
                        }
                    }
                }
                rule.leaf_ids.push(row);
                rule.alt_first.push(af);
                rule.alt_nullable.push(all_null);
            }
        }

        Ok(Grammar {
            rules: self.rules,
            root,
            leaf_positions,
            total_elements: self.total_elements,
            fingerprint: hash,
        })
    }
}

impl Grammar {
    pub fn root(&self) -> RuleId {
        self.root
    }

    pub fn num_rules(&self) -> usize {
        self.rules.len()
    }

    pub fn num_leaf_positions(&self) -> usize {
        self.leaf_positions.len()
    }

    pub fn total_elements(&self) -> usize {
        self.total_elements
    }

    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    #[inline(always)]
    pub fn rule(&self, id: RuleId) -> &Rule {
        &self.rules[id as usize]
    }

    #[inline(always)]
    pub fn leaf_pos(&self, id: LeafId) -> RulePos {
        self.leaf_positions[id as usize]
    }

    #[inline(always)]
    pub fn leaf_id_at(&self, pos: RulePos) -> Option<LeafId> {
        self.rules[pos.rule as usize].leaf_ids[pos.alt as usize][pos.elem as usize]
    }

    #[inline(always)]
    pub fn byte_set_at(&self, pos: RulePos) -> Option<ByteSet> {
        match self.rules[pos.rule as usize].alts[pos.alt as usize].get(pos.elem as usize)? {
            Element::Byte(bs) => Some(*bs),
            Element::RuleRef(_) => None,
        }
    }

    /// Estimate static bytecode memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>();
        bytes += self.leaf_positions.capacity() * std::mem::size_of::<RulePos>();
        for r in &self.rules {
            bytes += std::mem::size_of::<Rule>() + r.name.capacity();
            bytes += r.alt_first.capacity() * std::mem::size_of::<ByteSet>();
            bytes += r.alt_nullable.capacity() * std::mem::size_of::<bool>();
            for alt in &r.alts {
                bytes += std::mem::size_of::<Vec<Element>>()
                    + alt.capacity() * std::mem::size_of::<Element>();
            }
            for row in &r.leaf_ids {
                bytes += std::mem::size_of::<Vec<Option<LeafId>>>()
                    + row.capacity() * std::mem::size_of::<Option<LeafId>>();
            }
        }
        bytes
    }
}

/// Deterministic checkpoint for rollback (e.g., speculative decoding rejection).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdaCheckpoint {
    leaves: Vec<Stack>,
}

/// Reusable scratch buffers to eliminate per-token heap churn during runtime PDA checks.
#[derive(Debug, Default)]
pub struct PdaScratch {
    cur: Vec<Stack>,
    next: Vec<Stack>,
    work: Vec<Stack>,
}

/// Active PDA state: deduplicated, canonically sorted set of expanded leaf stacks.
///
/// Every stack in `leaves` is either empty (`Accept` / valid EOS) or has a top
/// [`RulePos`] pointing directly at an [`Element::Byte`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdaState {
    leaves: Vec<Stack>,
    config: PdaConfig,
}

impl PdaState {
    /// Initialize PDA state at the grammar's root rule.
    pub fn new(grammar: &Grammar, config: PdaConfig) -> Self {
        let root = grammar.root();
        let mut work = Vec::new();
        for a in 0..grammar.rule(root).alts.len() {
            work.push(SmallStack::from_single(RulePos {
                rule: root,
                alt: a as u16,
                elem: 0,
            }));
        }
        let mut leaves = Vec::new();
        expand_worklist(grammar, &mut work, &mut leaves, config, None);
        Self { leaves, config }
    }

    pub fn config(&self) -> PdaConfig {
        self.config
    }

    /// All currently active expanded leaf stacks (empty stack denotes Accept/EOS).
    #[inline]
    pub fn leaves(&self) -> &[Stack] {
        &self.leaves
    }

    /// True if the grammar can legally terminate (emit EOS) at this position.
    #[inline]
    pub fn can_accept(&self) -> bool {
        self.leaves.iter().any(SmallStack::is_empty)
    }

    /// True if no valid transitions or acceptance remain.
    #[inline]
    pub fn is_dead(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Capture a rollback checkpoint (for speculative verify or backtracking).
    pub fn checkpoint(&self) -> PdaCheckpoint {
        PdaCheckpoint {
            leaves: self.leaves.clone(),
        }
    }

    /// Restore PDA state to a previously captured checkpoint.
    pub fn rollback(&mut self, cp: PdaCheckpoint) {
        self.leaves = cp.leaves;
    }

    /// Advance the PDA by a single byte. Returns `false` if rejected (leaving state unchanged).
    pub fn advance_byte(&mut self, grammar: &Grammar, byte: u8) -> bool {
        let mut work = Vec::new();
        for &stack in &self.leaves {
            let Some(top) = stack.last() else {
                continue;
            };
            if let Some(bs) = grammar.byte_set_at(top) {
                if bs.contains(byte) {
                    let mut next = stack;
                    next.last_mut().expect("non-empty").elem += 1;
                    work.push(next);
                }
            }
        }
        if work.is_empty() {
            return false;
        }
        let mut next_leaves = Vec::new();
        expand_worklist(grammar, &mut work, &mut next_leaves, self.config, None);
        if next_leaves.is_empty() {
            return false;
        }
        self.leaves = next_leaves;
        true
    }

    /// Advance the PDA by a multi-byte token slice atomically.
    /// Returns `false` and leaves state unchanged if the token is invalid.
    pub fn advance_bytes(&mut self, grammar: &Grammar, bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return false;
        }
        let mut cur = self.leaves.clone();
        let mut next = Vec::new();
        let mut work = Vec::new();
        for (idx, &b) in bytes.iter().enumerate() {
            let next_filter = bytes.get(idx + 1).copied();
            step_stacks_byte(
                grammar,
                &cur,
                b,
                &mut work,
                &mut next,
                self.config,
                next_filter,
            );
            if next.is_empty() {
                return false;
            }
            std::mem::swap(&mut cur, &mut next);
            next.clear();
        }
        self.leaves = cur;
        true
    }

    /// Pure predicate: returns `true` iff consuming `bytes` from the current state
    /// reaches at least one valid expanded leaf stack (or `Accept`).
    pub fn can_consume_bytes(
        &self,
        grammar: &Grammar,
        bytes: &[u8],
        scratch: &mut PdaScratch,
    ) -> bool {
        leaves_can_consume_bytes(grammar, &self.leaves, bytes, self.config, scratch)
    }
}

/// Pure predicate over any slice of expanded leaf stacks (zero heap allocation!).
pub fn leaves_can_consume_bytes(
    grammar: &Grammar,
    leaves: &[Stack],
    bytes: &[u8],
    config: PdaConfig,
    scratch: &mut PdaScratch,
) -> bool {
    if bytes.is_empty() {
        return false;
    }
    let first = bytes[0];
    scratch.work.clear();
    for &stack in leaves {
        let Some(top) = stack.last() else {
            continue;
        };
        if let Some(bs) = grammar.byte_set_at(top) {
            if bs.contains(first) {
                let mut next = stack;
                next.last_mut().expect("non-empty").elem += 1;
                scratch.work.push(next);
            }
        }
    }
    if scratch.work.is_empty() {
        return false;
    }
    if bytes.len() == 1 {
        return expand_has_any_live(grammar, &mut scratch.work, config);
    }

    scratch.cur.clear();
    expand_worklist(
        grammar,
        &mut scratch.work,
        &mut scratch.cur,
        config,
        Some(bytes[1]),
    );
    if scratch.cur.is_empty() {
        return false;
    }

    for idx in 1..bytes.len() {
        let b = bytes[idx];
        if idx + 1 == bytes.len() {
            // Final byte: only need to verify at least one live state exists!
            scratch.work.clear();
            for &stack in &scratch.cur {
                let Some(top) = stack.last() else {
                    continue;
                };
                if let Some(bs) = grammar.byte_set_at(top) {
                    if bs.contains(b) {
                        let mut next = stack;
                        next.last_mut().expect("non-empty").elem += 1;
                        scratch.work.push(next);
                    }
                }
            }
            if scratch.work.is_empty() {
                return false;
            }
            return expand_has_any_live(grammar, &mut scratch.work, config);
        }
        scratch.next.clear();
        step_stacks_byte(
            grammar,
            &scratch.cur,
            b,
            &mut scratch.work,
            &mut scratch.next,
            config,
            Some(bytes[idx + 1]),
        );
        if scratch.next.is_empty() {
            return false;
        }
        std::mem::swap(&mut scratch.cur, &mut scratch.next);
    }
    true
}

/// Check whether a single specific leaf stack `stack` can consume `bytes` under `config`.
pub fn stack_can_consume_bytes(
    grammar: &Grammar,
    stack: Stack,
    bytes: &[u8],
    config: PdaConfig,
    scratch: &mut PdaScratch,
) -> bool {
    let single = [stack];
    leaves_can_consume_bytes(grammar, &single, bytes, config, scratch)
}

/// Expand the caller stack `stack.without_top()` into `out_leaves`.
/// Returns `true` iff the expanded caller stack is non-empty (live).
pub fn expand_caller_stack(
    grammar: &Grammar,
    stack: Stack,
    config: PdaConfig,
    work: &mut Vec<Stack>,
    out_leaves: &mut Vec<Stack>,
) -> bool {
    if stack.is_empty() {
        out_leaves.clear();
        return false;
    }
    work.clear();
    work.push(stack.without_top());
    expand_worklist(grammar, work, out_leaves, config, None);
    !out_leaves.is_empty()
}

/// Fast O(1) liveness check: returns `true` as soon as ANY branch in `work`
/// reaches a `Byte` leaf or `Accept` (avoids expanding all remaining alternatives!).
fn expand_has_any_live(grammar: &Grammar, work: &mut Vec<Stack>, config: PdaConfig) -> bool {
    let max_depth = config.effective_depth();
    while let Some(mut stack) = work.pop() {
        let Some(top) = stack.last() else {
            return true; // Accept!
        };
        let rule = grammar.rule(top.rule);
        let alt = &rule.alts[top.alt as usize];
        let elem_idx = top.elem as usize;
        if elem_idx >= alt.len() {
            stack.pop();
            work.push(stack);
            continue;
        }
        match alt[elem_idx] {
            Element::Byte(_) => return true,
            Element::RuleRef(target) => {
                let is_tail = elem_idx + 1 == alt.len();
                let target_alts = grammar.rule(target).alts.len();
                if is_tail {
                    stack.pop();
                    for a in 0..target_alts {
                        let mut next = stack;
                        next.push(RulePos {
                            rule: target,
                            alt: a as u16,
                            elem: 0,
                        });
                        work.push(next);
                    }
                } else {
                    if stack.len() + 1 > max_depth {
                        continue;
                    }
                    stack.last_mut().expect("non-empty").elem += 1;
                    for a in 0..target_alts {
                        let mut next = stack;
                        next.push(RulePos {
                            rule: target,
                            alt: a as u16,
                            elem: 0,
                        });
                        work.push(next);
                    }
                }
            }
        }
    }
    false
}

/// Advance a slice of expanded leaf stacks by one byte `b`.
fn step_stacks_byte(
    grammar: &Grammar,
    cur: &[Stack],
    byte: u8,
    work: &mut Vec<Stack>,
    out_leaves: &mut Vec<Stack>,
    config: PdaConfig,
    next_byte_filter: Option<u8>,
) {
    work.clear();
    for &stack in cur {
        let Some(top) = stack.last() else {
            continue;
        };
        if let Some(bs) = grammar.byte_set_at(top) {
            if bs.contains(byte) {
                let mut next = stack;
                next.last_mut().expect("non-empty").elem += 1;
                work.push(next);
            }
        }
    }
    if !work.is_empty() {
        expand_worklist(grammar, work, out_leaves, config, next_byte_filter);
    } else {
        out_leaves.clear();
    }
}

/// Epsilon-expand all stacks in `work` into canonically sorted, deduplicated leaf stacks
/// (or empty `Accept` stack), applying TCO and optional next-byte lookahead pruning.
/// Zero heap allocations once `work` and `out_leaves` are warmed up!
pub(crate) fn expand_worklist(
    grammar: &Grammar,
    work: &mut Vec<Stack>,
    out_leaves: &mut Vec<Stack>,
    config: PdaConfig,
    next_byte_filter: Option<u8>,
) {
    out_leaves.clear();
    let max_depth = config.effective_depth();
    let cap = config.max_active_stacks;

    while let Some(mut stack) = work.pop() {
        let Some(top) = stack.last() else {
            if next_byte_filter.is_none() {
                out_leaves.push(SmallStack::new());
            }
            continue;
        };

        let alt = &grammar.rule(top.rule).alts[top.alt as usize];
        let elem_idx = top.elem as usize;

        if elem_idx >= alt.len() {
            stack.pop();
            work.push(stack);
            continue;
        }

        match alt[elem_idx] {
            Element::Byte(bs) => {
                if next_byte_filter.is_none_or(|nb| bs.contains(nb)) {
                    out_leaves.push(stack);
                    if out_leaves.len() > cap * 2 {
                        out_leaves.sort_unstable();
                        out_leaves.dedup();
                        if out_leaves.len() >= cap {
                            out_leaves.truncate(cap);
                            break;
                        }
                    }
                }
            }
            Element::RuleRef(target) => {
                let is_tail_call = elem_idx + 1 == alt.len();
                let target_rule = grammar.rule(target);
                let target_alts = target_rule.alts.len();
                if is_tail_call {
                    stack.pop();
                    for a in 0..target_alts {
                        if let Some(nb) = next_byte_filter {
                            if !target_rule.alt_nullable[a]
                                && !target_rule.alt_first[a].contains(nb)
                            {
                                continue;
                            }
                        }
                        let mut next = stack;
                        next.push(RulePos {
                            rule: target,
                            alt: a as u16,
                            elem: 0,
                        });
                        work.push(next);
                    }
                } else {
                    if stack.len() + 1 > max_depth {
                        continue;
                    }
                    stack.last_mut().expect("non-empty").elem += 1;
                    for a in 0..target_alts {
                        if let Some(nb) = next_byte_filter {
                            if !target_rule.alt_nullable[a]
                                && !target_rule.alt_first[a].contains(nb)
                            {
                                continue;
                            }
                        }
                        let mut next = stack;
                        next.push(RulePos {
                            rule: target,
                            alt: a as u16,
                            elem: 0,
                        });
                        work.push(next);
                    }
                }
            }
        }
    }

    out_leaves.sort_unstable();
    out_leaves.dedup();
    if out_leaves.len() > cap {
        out_leaves.truncate(cap);
    }
}

/// Inline `Copy` simulation stack used during XGrammar compile-time token classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct SimStack {
    pub frames: SmallStack,
    /// Maximum stack depth reached along this branch (1 means strictly inside leaf frame / TCO).
    pub max_len: u8,
}

/// Reusable scratch buffers for zero-allocation VocabTrie simulation.
#[derive(Default)]
pub(crate) struct SimScratch {
    pub work: Vec<SimStack>,
}

/// Expand `SimStack`s from a single-leaf root `[p]` (caller stack unknown),
/// returning `true` if any branch popped past frame 0.
pub(crate) fn expand_sim_worklist(
    grammar: &Grammar,
    work: &mut Vec<SimStack>,
    out_leaves: &mut Vec<SimStack>,
    config: PdaConfig,
) -> bool {
    out_leaves.clear();
    let mut popped_past_root = false;
    let max_depth = config.effective_depth();
    let cap = config.max_active_stacks;

    while let Some(mut sim) = work.pop() {
        let Some(top) = sim.frames.last() else {
            popped_past_root = true;
            continue;
        };

        let alt = &grammar.rule(top.rule).alts[top.alt as usize];
        let elem_idx = top.elem as usize;

        if elem_idx >= alt.len() {
            sim.frames.pop();
            work.push(sim);
            continue;
        }

        match alt[elem_idx] {
            Element::Byte(_) => {
                out_leaves.push(sim);
                if out_leaves.len() > cap * 2 {
                    out_leaves.sort_unstable();
                    out_leaves.dedup();
                    if out_leaves.len() >= cap {
                        out_leaves.truncate(cap);
                        break;
                    }
                }
            }
            Element::RuleRef(target) => {
                let is_tail_call = elem_idx + 1 == alt.len();
                let target_alts = grammar.rule(target).alts.len();
                if is_tail_call {
                    sim.frames.pop();
                    for a in 0..target_alts {
                        let mut frames = sim.frames;
                        frames.push(RulePos {
                            rule: target,
                            alt: a as u16,
                            elem: 0,
                        });
                        work.push(SimStack {
                            frames,
                            max_len: sim.max_len,
                        });
                    }
                } else {
                    if sim.frames.len() + 1 > max_depth {
                        continue;
                    }
                    sim.frames.last_mut().expect("non-empty").elem += 1;
                    let new_max = sim.max_len.max((sim.frames.len() + 1) as u8);
                    for a in 0..target_alts {
                        let mut frames = sim.frames;
                        frames.push(RulePos {
                            rule: target,
                            alt: a as u16,
                            elem: 0,
                        });
                        work.push(SimStack {
                            frames,
                            max_len: new_max,
                        });
                    }
                }
            }
        }
    }

    out_leaves.sort_unstable();
    out_leaves.dedup();
    if out_leaves.len() > cap {
        out_leaves.truncate(cap);
    }
    popped_past_root
}

/// Advance a slice of `SimStack`s by one byte `byte`.
pub(crate) fn step_sim_byte(
    grammar: &Grammar,
    cur: &[SimStack],
    byte: u8,
    config: PdaConfig,
    scratch: &mut SimScratch,
) -> (Vec<SimStack>, bool) {
    scratch.work.clear();
    for &sim in cur {
        let Some(top) = sim.frames.last() else {
            continue;
        };
        if let Some(bs) = grammar.byte_set_at(top) {
            if bs.contains(byte) {
                let mut frames = sim.frames;
                frames.last_mut().expect("non-empty").elem += 1;
                scratch.work.push(SimStack {
                    frames,
                    max_len: sim.max_len,
                });
            }
        }
    }
    if scratch.work.is_empty() {
        return (Vec::new(), false);
    }
    let mut out = Vec::with_capacity(scratch.work.len().max(4));
    let popped = expand_sim_worklist(grammar, &mut scratch.work, &mut out, config);
    (out, popped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_direct_and_indirect_left_recursion() {
        let mut b = GrammarBuilder::new();
        let a = b.declare_rule("A").unwrap();
        let c = b.declare_rule("B").unwrap();
        b.add_alt(
            a,
            vec![Element::RuleRef(c), Element::Byte(ByteSet::from_byte(b'x'))],
        )
        .unwrap();
        b.add_alt(c, vec![Element::RuleRef(a)]).unwrap();
        let err = b.finalize(a).unwrap_err();
        assert!(matches!(err, GrammarError::LeftRecursionDetected { .. }));
    }

    #[test]
    fn tco_keeps_stack_depth_o1_on_long_loops() {
        let mut b = GrammarBuilder::new();
        let loop_rule = b.declare_rule("loop").unwrap();
        b.add_alt(loop_rule, vec![Element::Byte(ByteSet::from_byte(b'!'))])
            .unwrap();
        b.add_alt(
            loop_rule,
            vec![
                Element::Byte(ByteSet::from_byte(b'a')),
                Element::RuleRef(loop_rule),
            ],
        )
        .unwrap();
        let g = b.finalize(loop_rule).unwrap();
        let cfg = PdaConfig {
            max_recursion_depth: 4,
            max_active_stacks: 32,
        };
        let mut state = PdaState::new(&g, cfg);
        let payload = vec![b'a'; 500];
        assert!(state.advance_bytes(&g, &payload));
        assert!(!state.can_accept());
        assert!(state.advance_byte(&g, b'!'));
        assert!(state.can_accept());
    }

    #[test]
    fn enforces_max_recursion_depth_on_nested_parens() {
        let mut b = GrammarBuilder::new();
        let expr = b.declare_rule("expr").unwrap();
        b.add_alt(expr, vec![Element::Byte(ByteSet::from_byte(b'x'))])
            .unwrap();
        b.add_alt(
            expr,
            vec![
                Element::Byte(ByteSet::from_byte(b'(')),
                Element::RuleRef(expr),
                Element::Byte(ByteSet::from_byte(b')')),
            ],
        )
        .unwrap();
        let g = b.finalize(expr).unwrap();
        let cfg = PdaConfig {
            max_recursion_depth: 3,
            max_active_stacks: 32,
        };
        let mut state = PdaState::new(&g, cfg);
        assert!(state.advance_bytes(&g, b"((x))"));
        assert!(state.can_accept());

        let mut state2 = PdaState::new(&g, cfg);
        assert!(!state2.advance_bytes(&g, b"(((x)))"));
    }

    #[test]
    fn checkpoint_and_rollback_restore_exact_state() {
        let mut b = GrammarBuilder::new();
        let r = b
            .add_rule("root", vec![GrammarBuilder::literal_elements(b"hello")])
            .unwrap();
        let g = b.finalize(r).unwrap();
        let mut state = PdaState::new(&g, PdaConfig::default());
        assert!(state.advance_bytes(&g, b"he"));
        let cp = state.checkpoint();
        assert!(state.advance_bytes(&g, b"ll"));
        assert!(!state.advance_bytes(&g, b"z"));
        state.rollback(cp);
        assert!(state.advance_bytes(&g, b"llo"));
        assert!(state.can_accept());
    }
}
