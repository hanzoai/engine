//! Zero-compute Suffix Automaton (SAM) & rolling n-gram prompt-lookup drafter.
//!
//! Matches the longest recent suffix of the active context (between `min_ngram` and `max_ngram`)
//! against prior prompt / tool-diff / generated tokens and proposes up to `max_draft` exact
//! continuation tokens with zero neural compute cost.

use std::collections::HashMap;

use crate::corpus::VOCAB_SIZE;
use crate::verify::LinearDraft;

/// One state in the incremental Suffix Automaton.
#[derive(Clone, Debug, Default)]
struct SamState {
    len: usize,
    link: Option<usize>,
    /// Most recent ending index in the indexed token stream for substrings in this state.
    last_end_pos: usize,
    next: HashMap<u32, usize>,
}

/// Proposal returned by [`PromptLookupDrafter`].
#[derive(Clone, Debug)]
pub struct LookupProposal {
    pub draft: LinearDraft,
    /// Length of the matched context suffix (`0` if no match >= `min_ngram`).
    pub matched_ngram_len: usize,
}

/// Incremental Suffix Automaton + exact n-gram lookup drafter.
#[derive(Clone, Debug)]
pub struct PromptLookupDrafter {
    min_ngram: usize,
    max_ngram: usize,
    states: Vec<SamState>,
    last_state: usize,
    tokens: Vec<u32>,
    /// Exact rolling n-gram index (`min_ngram..=max_ngram`) -> up to 2 most recent ending
    /// indices `[older, latest]` so the currently growing tail never shadows an earlier
    /// prompt/diff occurrence that has full multi-token continuations!
    ngram_recent_ends: HashMap<u64, (Option<usize>, usize)>,
}

impl PromptLookupDrafter {
    pub fn new(min_ngram: usize, max_ngram: usize) -> Self {
        let min_n = min_ngram.max(2);
        let max_n = max_ngram.max(min_n);
        let mut s = Self {
            min_ngram: min_n,
            max_ngram: max_n,
            states: Vec::with_capacity(256),
            last_state: 0,
            tokens: Vec::new(),
            ngram_recent_ends: HashMap::new(),
        };
        s.reset();
        s
    }

    pub fn reset(&mut self) {
        self.states.clear();
        self.states.push(SamState {
            len: 0,
            link: None,
            last_end_pos: 0,
            next: HashMap::new(),
        });
        self.last_state = 0;
        self.tokens.clear();
        self.ngram_recent_ends.clear();
    }

    /// Ingest a full token slice into the suffix automaton and n-gram recency index.
    pub fn extend_slice(&mut self, toks: &[u32]) {
        for &t in toks {
            self.push_token(t);
        }
    }

    /// Append a single token in amortized `O(max_ngram)` time.
    pub fn push_token(&mut self, tok: u32) {
        let pos = self.tokens.len();
        self.tokens.push(tok);

        // 1. Standard Blumer et al. incremental Suffix Automaton extension.
        let cur = self.states.len();
        self.states.push(SamState {
            len: self.states[self.last_state].len + 1,
            link: None,
            last_end_pos: pos,
            next: HashMap::new(),
        });

        let mut p = Some(self.last_state);
        while let Some(u) = p {
            if self.states[u].next.contains_key(&tok) {
                break;
            }
            self.states[u].next.insert(tok, cur);
            p = self.states[u].link;
        }

        if let Some(u) = p {
            let q = self.states[u].next[&tok];
            if self.states[u].len + 1 == self.states[q].len {
                self.states[cur].link = Some(q);
            } else {
                let clone = self.states.len();
                let mut cloned = self.states[q].clone();
                cloned.len = self.states[u].len + 1;
                self.states.push(cloned);
                let mut v = Some(u);
                while let Some(w) = v {
                    if self.states[w].next.get(&tok) == Some(&q) {
                        self.states[w].next.insert(tok, clone);
                        v = self.states[w].link;
                    } else {
                        break;
                    }
                }
                self.states[q].link = Some(clone);
                self.states[cur].link = Some(clone);
            }
        } else {
            self.states[cur].link = Some(0);
        }
        self.last_state = cur;

        let mut up = self.states[cur].link;
        let mut hops = 0usize;
        while let Some(u) = up {
            if self.states[u].last_end_pos == pos || hops >= self.max_ngram {
                break;
            }
            self.states[u].last_end_pos = pos;
            up = self.states[u].link;
            hops += 1;
        }

        // 2. Update rolling n-gram recency index for lengths [min_ngram..=max_ngram],
        //    retaining the previous occurrence whenever a new tail occurrence is recorded.
        if pos + 1 >= self.min_ngram {
            let max_n = self.max_ngram.min(pos + 1);
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for n in 1..=max_n {
                let t = self.tokens[pos + 1 - n];
                h ^= (t as u64).wrapping_add(0x9e37_79b9);
                h = h.wrapping_mul(0x1000_0000_01b3);
                if n >= self.min_ngram {
                    let key = h ^ ((n as u64) << 56);
                    self.ngram_recent_ends
                        .entry(key)
                        .and_modify(|entry| {
                            if entry.1 != pos {
                                entry.0 = Some(entry.1);
                                entry.1 = pos;
                            }
                        })
                        .or_insert((None, pos));
                }
            }
        }
    }

    /// Number of indexed tokens.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Number of states in the suffix automaton.
    pub fn num_sam_states(&self) -> usize {
        self.states.len()
    }

    /// Query the longest matching suffix of `context` (length in `min_ngram..=max_ngram`)
    /// that occurs strictly earlier in the indexed history with continuation tokens,
    /// returning up to `max_draft` continuation tokens.
    pub fn propose(&self, context: &[u32], max_draft: usize) -> LookupProposal {
        if max_draft == 0 || context.len() < self.min_ngram || self.tokens.len() <= self.min_ngram {
            return LookupProposal {
                draft: LinearDraft {
                    tokens: Vec::new(),
                    draft_probs: Vec::new(),
                },
                matched_ngram_len: 0,
            };
        }

        let max_n = self.max_ngram.min(context.len());
        let ctx_end = context.len() - 1;
        let tail_pos = self.tokens.len() - 1;

        let mut hashes = Vec::with_capacity(max_n);
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for n in 1..=max_n {
            let t = context[ctx_end + 1 - n];
            h ^= (t as u64).wrapping_add(0x9e37_79b9);
            h = h.wrapping_mul(0x1000_0000_01b3);
            hashes.push(h ^ ((n as u64) << 56));
        }

        for n in (self.min_ngram..=max_n).rev() {
            let key = hashes[n - 1];
            if let Some(&(older_opt, latest)) = self.ngram_recent_ends.get(&key) {
                // Prefer `latest` if it is strictly earlier than the active tail; otherwise use `older`!
                let candidates = [Some(latest), older_opt];
                for cand in candidates.into_iter().flatten() {
                    if cand < tail_pos && cand + 1 >= n {
                        let hist_slice = &self.tokens[cand + 1 - n..=cand];
                        let ctx_slice = &context[context.len() - n..];
                        if hist_slice == ctx_slice {
                            let start = cand + 1;
                            let avail = (self.tokens.len() - start).min(max_draft);
                            if avail > 0 {
                                let draft_tokens = self.tokens[start..start + avail].to_vec();
                                let draft_probs = draft_tokens
                                    .iter()
                                    .map(|&tok| peaked_lookup_distribution(tok))
                                    .collect();
                                return LookupProposal {
                                    draft: LinearDraft {
                                        tokens: draft_tokens,
                                        draft_probs,
                                    },
                                    matched_ngram_len: n,
                                };
                            }
                        }
                    }
                }
            }
        }

        // Secondary SAM walk fallback for any suffix match tracked in automaton transitions.
        let needle = &context[context.len() - max_n..];
        let mut state = 0usize;
        let mut matched = 0usize;
        for &tok in needle {
            while state != 0 && !self.states[state].next.contains_key(&tok) {
                state = self.states[state].link.unwrap_or(0);
                matched = self.states[state].len;
            }
            if let Some(&nxt) = self.states[state].next.get(&tok) {
                state = nxt;
                matched += 1;
            } else {
                state = 0;
                matched = 0;
            }
        }
        if matched >= self.min_ngram {
            let end_pos = self.states[state].last_end_pos;
            if end_pos + 1 < self.tokens.len() {
                let start = end_pos + 1;
                let avail = (self.tokens.len() - start).min(max_draft);
                if avail > 0 {
                    let draft_tokens = self.tokens[start..start + avail].to_vec();
                    let draft_probs = draft_tokens
                        .iter()
                        .map(|&tok| peaked_lookup_distribution(tok))
                        .collect();
                    return LookupProposal {
                        draft: LinearDraft {
                            tokens: draft_tokens,
                            draft_probs,
                        },
                        matched_ngram_len: matched,
                    };
                }
            }
        }

        LookupProposal {
            draft: LinearDraft {
                tokens: Vec::new(),
                draft_probs: Vec::new(),
            },
            matched_ngram_len: 0,
        }
    }
}

/// Peaked probability vector for stochastic rejection-sampling compatibility.
fn peaked_lookup_distribution(tok: u32) -> Vec<f32> {
    let eps = 1e-4_f32;
    let bg = eps / ((VOCAB_SIZE - 1) as f32);
    let mut probs = vec![bg; VOCAB_SIZE];
    if (tok as usize) < VOCAB_SIZE {
        probs[tok as usize] = 1.0 - eps;
    }
    probs
}
