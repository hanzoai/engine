//! Exact lossless speculative verification & rejection sampling:
//! - Greedy bit-exact linear verification (100.0% bit-identical to plain decoding)
//! - Exact Leviathan/Chen stochastic rejection sampling with residual resampling
//! - EAGLE-3 greedy tree verification (longest matching tree path in one parallel pass)
//! - Exact multi-branch stochastic tree verification (SpecInfer / EAGLE-3 sequential residual)

use crate::rng::{argmax_f32, SplitMix64};

/// A linear chain of `k` drafted candidate tokens plus their draft distributions.
#[derive(Clone, Debug)]
pub struct LinearDraft {
    pub tokens: Vec<u32>,
    /// Draft probability distribution `q_i` over vocabulary for each drafted token `i`.
    pub draft_probs: Vec<Vec<f32>>,
}

/// One node in an EAGLE-3 topological draft tree.
/// Node `0` is always the root anchor token (already committed to KV cache before verification).
#[derive(Clone, Debug)]
pub struct TreeNode {
    pub id: usize,
    /// `None` for root node `0`; `Some(parent_id)` with `parent_id < id` for all descendants.
    pub parent: Option<usize>,
    /// Depth from root (`0` for root, `1..=max_depth` for drafted candidates).
    pub depth: usize,
    /// Token ID at this node.
    pub token: u32,
    /// Cumulative log-probability score used during beam expansion.
    pub cumulative_log_prob: f32,
    /// Draft probability distribution `q` over vocabulary from which `token` was proposed
    /// (empty for root node `0`).
    pub draft_probs: Vec<f32>,
    /// Indices of direct children in `DraftTree::nodes`.
    pub children: Vec<usize>,
}

/// Topological draft tree with explicit causal ancestor attention mask.
#[derive(Clone, Debug)]
pub struct DraftTree {
    pub nodes: Vec<TreeNode>,
    /// Row-major `N x N` boolean mask where `attention_mask[i * N + j] == true` iff
    /// node `j` is an ancestor of node `i` (including `j == i`).
    pub attention_mask: Vec<bool>,
}

impl DraftTree {
    /// Build a single-node root-only tree (equivalent to plain decoding when no draft tokens exist).
    pub fn root_only(anchor_token: u32) -> Self {
        Self {
            nodes: vec![TreeNode {
                id: 0,
                parent: None,
                depth: 0,
                token: anchor_token,
                cumulative_log_prob: 0.0,
                draft_probs: Vec::new(),
                children: Vec::new(),
            }],
            attention_mask: vec![true],
        }
    }

    /// Wrap a linear draft chain `[d_1, ..., d_k]` rooted at `anchor_token` as a degenerate tree.
    pub fn from_linear(anchor_token: u32, linear: &LinearDraft) -> Self {
        let mut nodes = Vec::with_capacity(1 + linear.tokens.len());
        nodes.push(TreeNode {
            id: 0,
            parent: None,
            depth: 0,
            token: anchor_token,
            cumulative_log_prob: 0.0,
            draft_probs: Vec::new(),
            children: if linear.tokens.is_empty() {
                Vec::new()
            } else {
                vec![1]
            },
        });
        let mut cum_lp = 0.0_f32;
        for (i, &tok) in linear.tokens.iter().enumerate() {
            let id = i + 1;
            let probs = linear.draft_probs.get(i).cloned().unwrap_or_default();
            let p = probs.get(tok as usize).copied().unwrap_or(1e-6).max(1e-6);
            cum_lp += p.ln();
            nodes.push(TreeNode {
                id,
                parent: Some(i),
                depth: id,
                token: tok,
                cumulative_log_prob: cum_lp,
                draft_probs: probs,
                children: if id < linear.tokens.len() {
                    vec![id + 1]
                } else {
                    Vec::new()
                },
            });
        }
        Self::with_computed_mask(nodes)
    }

    /// Finalize children lists and build the exact `N x N` topological ancestor attention mask.
    pub fn with_computed_mask(mut nodes: Vec<TreeNode>) -> Self {
        let n = nodes.len();
        for node in &mut nodes {
            node.children.clear();
        }
        for i in 1..n {
            if let Some(p) = nodes[i].parent {
                debug_assert!(p < i, "tree nodes must be topologically ordered");
                nodes[p].children.push(i);
            }
        }
        let mut mask = vec![false; n * n];
        for i in 0..n {
            let mut cur = Some(i);
            while let Some(u) = cur {
                mask[i * n + u] = true;
                cur = nodes[u].parent;
            }
        }
        Self {
            nodes,
            attention_mask: mask,
        }
    }

    /// Number of nodes including root (`1 + drafted_tokens`).
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of drafted candidate tokens (excluding root node `0`).
    #[inline]
    pub fn num_draft_tokens(&self) -> usize {
        self.nodes.len().saturating_sub(1)
    }

    /// Maximum depth in the tree (`0` for root-only).
    pub fn max_depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }

    /// Return the ordered ancestor chain `[0, ..., node_id]` from root to `node_id`.
    pub fn ancestor_path(&self, node_id: usize) -> Vec<usize> {
        let mut rev = Vec::new();
        let mut cur = Some(node_id);
        while let Some(u) = cur {
            rev.push(u);
            cur = self.nodes[u].parent;
        }
        rev.reverse();
        rev
    }
}

/// Result of a single speculative verification pass (linear or tree).
#[derive(Clone, Debug)]
pub struct VerifyOutcome {
    /// Emitted tokens in order: `[accepted_draft_1, ..., accepted_draft_a, bonus_or_correction]`.
    /// Always has length `1 + accepted_draft_count >= 1`.
    pub emitted_tokens: Vec<u32>,
    /// Number of drafted tokens accepted (`0..=max_depth`).
    pub accepted_draft_count: usize,
    /// Tree node indices along the accepted path starting from root `0`:
    /// length is `1 + accepted_draft_count` (`accepted_path[0] == 0`).
    /// The last node in `accepted_path` is the parent whose logits produced the bonus/correction token.
    pub accepted_path: Vec<usize>,
}

/// Bit-exact greedy verification for a linear draft chain `[d_1, ..., d_k]`.
///
/// `target_logits` must have length `k + 1`: `target_logits[i]` predicts token `i + 1`
/// (with `target_logits[0]` predicting `d_1` from anchor, and `target_logits[k]` predicting
/// the bonus token if all `k` drafted tokens match).
pub fn verify_linear_greedy(draft_tokens: &[u32], target_logits: &[Vec<f32>]) -> VerifyOutcome {
    let k = draft_tokens.len();
    assert!(
        target_logits.len() > k,
        "verify_linear_greedy requires k + 1 target logit rows"
    );

    let mut emitted = Vec::with_capacity(k + 1);
    let mut accepted_path = Vec::with_capacity(k + 1);
    accepted_path.push(0);

    for (i, &draft_tok) in draft_tokens.iter().enumerate() {
        let target_tok = argmax_f32(&target_logits[i]) as u32;
        if target_tok == draft_tok {
            emitted.push(draft_tok);
            accepted_path.push(i + 1);
        } else {
            emitted.push(target_tok);
            return VerifyOutcome {
                emitted_tokens: emitted,
                accepted_draft_count: i,
                accepted_path,
            };
        }
    }

    // All k drafted tokens accepted: emit the free bonus token from target_logits[k].
    let bonus_tok = argmax_f32(&target_logits[k]) as u32;
    emitted.push(bonus_tok);
    VerifyOutcome {
        emitted_tokens: emitted,
        accepted_draft_count: k,
        accepted_path,
    }
}

/// Compute normalized residual distribution `norm(max(0, p - q))`.
/// Falls back to normalized `p` if the residual mass is numerically zero.
pub fn residual_distribution(p: &[f32], q: &[f32]) -> Vec<f32> {
    assert_eq!(p.len(), q.len());
    let mut res = Vec::with_capacity(p.len());
    let mut sum = 0.0_f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let diff = (pi - qi).max(0.0);
        res.push(diff);
        sum += diff as f64;
    }
    if sum > 1e-12 {
        let inv = 1.0 / sum;
        for r in &mut res {
            *r = ((*r as f64) * inv) as f32;
        }
        res
    } else {
        let p_sum: f64 = p.iter().map(|&x| x.max(0.0) as f64).sum();
        if p_sum > 1e-12 {
            p.iter()
                .map(|&x| ((x.max(0.0) as f64) / p_sum) as f32)
                .collect()
        } else {
            vec![1.0 / (p.len().max(1) as f32); p.len()]
        }
    }
}

/// Exact Leviathan et al. (2023) / Chen et al. (2023) lossless stochastic rejection sampling
/// for a linear draft chain.
///
/// Preserves the exact target distribution `p_target` at every position:
/// - Accept `d_i` with probability `min(1, p_i(d_i) / q_i(d_i))`.
/// - On first rejection, sample correction token from `norm(max(0, p_i - q_i))`.
/// - If all `k` are accepted, sample bonus token from `p_k`.
pub fn verify_linear_stochastic(
    draft: &LinearDraft,
    target_probs: &[Vec<f32>],
    rng: &mut SplitMix64,
) -> VerifyOutcome {
    let k = draft.tokens.len();
    assert!(
        target_probs.len() > k,
        "verify_linear_stochastic requires k + 1 target probability rows"
    );
    assert_eq!(draft.draft_probs.len(), k);

    let mut emitted = Vec::with_capacity(k + 1);
    let mut accepted_path = Vec::with_capacity(k + 1);
    accepted_path.push(0);

    for i in 0..k {
        let d_tok = draft.tokens[i] as usize;
        let p_i = &target_probs[i];
        let q_i = &draft.draft_probs[i];
        let p_d = p_i.get(d_tok).copied().unwrap_or(0.0).max(0.0) as f64;
        let q_d = q_i.get(d_tok).copied().unwrap_or(0.0).max(0.0) as f64;

        let accept_prob = if q_d <= 0.0 {
            0.0
        } else {
            (p_d / q_d).min(1.0)
        };

        if rng.next_f64() < accept_prob {
            emitted.push(d_tok as u32);
            accepted_path.push(i + 1);
        } else {
            let res = residual_distribution(p_i, q_i);
            let corr = rng.sample_categorical(&res) as u32;
            emitted.push(corr);
            return VerifyOutcome {
                emitted_tokens: emitted,
                accepted_draft_count: i,
                accepted_path,
            };
        }
    }

    let bonus = rng.sample_categorical(&target_probs[k]) as u32;
    emitted.push(bonus);
    VerifyOutcome {
        emitted_tokens: emitted,
        accepted_draft_count: k,
        accepted_path,
    }
}

/// Bit-exact EAGLE-3 greedy tree verification.
///
/// Starting at root node `0`, repeatedly computes `t = argmax(target_logits[curr])`
/// and descends into the matching child node if present. Terminates on the first depth
/// where no child matches `t`, emitting `t` as the exact bonus/correction token.
/// Produces 100.0% bit-identical tokens to plain greedy decoding.
pub fn verify_tree_greedy(tree: &DraftTree, target_logits: &[Vec<f32>]) -> VerifyOutcome {
    assert_eq!(
        tree.nodes.len(),
        target_logits.len(),
        "verify_tree_greedy requires one target logit row per tree node"
    );

    let mut emitted = Vec::new();
    let mut accepted_path = vec![0usize];
    let mut curr = 0usize;

    loop {
        let target_tok = argmax_f32(&target_logits[curr]) as u32;
        let matched_child = tree.nodes[curr]
            .children
            .iter()
            .copied()
            .find(|&cid| tree.nodes[cid].token == target_tok);

        if let Some(child_id) = matched_child {
            emitted.push(target_tok);
            accepted_path.push(child_id);
            curr = child_id;
        } else {
            emitted.push(target_tok);
            return VerifyOutcome {
                emitted_tokens: emitted,
                accepted_draft_count: accepted_path.len() - 1,
                accepted_path,
            };
        }
    }
}

/// Exact multi-branch stochastic tree verification (SpecInfer / EAGLE-3 sequential residual).
///
/// At each tree node `curr`, iterates through its children `c_1, ..., c_m` with proposal
/// distributions `q_c` while maintaining the exact residual target distribution `p_rem`
/// (initialized to `target_probs[curr]`):
/// - Accept child `c` (token `x`) with probability `min(1, p_rem(x) / q_c(x))`.
/// - On acceptance, descend to `curr = c`.
/// - On rejection, update `p_rem <- norm(max(0, p_rem - q_c))` and test the next sibling.
/// - If all siblings are rejected (or `curr` is a leaf), sample from `p_rem` and terminate.
///
/// By induction on `m`, the marginal output distribution at every tree level equals
/// `target_probs[curr]` exactly.
pub fn verify_tree_stochastic(
    tree: &DraftTree,
    target_probs: &[Vec<f32>],
    rng: &mut SplitMix64,
) -> VerifyOutcome {
    assert_eq!(
        tree.nodes.len(),
        target_probs.len(),
        "verify_tree_stochastic requires one target prob row per tree node"
    );

    let mut emitted = Vec::new();
    let mut accepted_path = vec![0usize];
    let mut curr = 0usize;

    loop {
        let mut p_rem = target_probs[curr].clone();
        let mut accepted_child = None;

        for &cid in &tree.nodes[curr].children {
            let child = &tree.nodes[cid];
            let tok = child.token as usize;
            let q_c = &child.draft_probs;
            if q_c.len() != p_rem.len() {
                continue;
            }
            let p_x = p_rem.get(tok).copied().unwrap_or(0.0).max(0.0) as f64;
            let q_x = q_c.get(tok).copied().unwrap_or(0.0).max(0.0) as f64;
            let accept_prob = if q_x <= 0.0 {
                0.0
            } else {
                (p_x / q_x).min(1.0)
            };

            if rng.next_f64() < accept_prob {
                accepted_child = Some(cid);
                break;
            }
            p_rem = residual_distribution(&p_rem, q_c);
        }

        if let Some(cid) = accepted_child {
            emitted.push(tree.nodes[cid].token);
            accepted_path.push(cid);
            curr = cid;
        } else {
            let fallback = rng.sample_categorical(&p_rem) as u32;
            emitted.push(fallback);
            return VerifyOutcome {
                emitted_tokens: emitted,
                accepted_draft_count: accepted_path.len() - 1,
                accepted_path,
            };
        }
    }
}
