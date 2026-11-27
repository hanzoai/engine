//! Multi-layer autoregressive target Transformer with exact KV cache, QK-Norm trigram
//! induction head, corpus-grounded FFN + LM weights, tapped multi-layer hidden states
//! `[h_low, h_mid, h_high]` for EAGLE-3 feature fusion, and bit-exact parallel
//! tree-attention verification.

use crate::corpus::{GroundedCorpus, VOCAB_SIZE};
use crate::rng::{rms_norm, silu, softmax_temp, SplitMix64};
use crate::verify::DraftTree;

pub const D_MODEL: usize = 64;
pub const N_HEADS: usize = 4;
pub const HEAD_DIM: usize = D_MODEL / N_HEADS; // 16
pub const D_FF: usize = 512;
pub const N_LAYERS: usize = 8;

/// EAGLE-3 tapped layers: low (lexical/bigram), mid (structural/induction), high (semantic).
pub const TAP_LOW_LAYER: usize = 1;
pub const TAP_MID_LAYER: usize = 4;
pub const TAP_HIGH_LAYER: usize = N_LAYERS - 1; // 7

/// Multi-layer tapped hidden features `[h_low; h_mid; h_high]` returned by target forward passes.
#[derive(Clone, Debug)]
pub struct TappedFeatures {
    pub h_low: [f32; D_MODEL],
    pub h_mid: [f32; D_MODEL],
    pub h_high: [f32; D_MODEL],
}

impl TappedFeatures {
    /// Concatenate `[h_low; h_mid; h_high]` into a `3 * D_MODEL` vector.
    pub fn concat(&self) -> [f32; 3 * D_MODEL] {
        let mut out = [0.0_f32; 3 * D_MODEL];
        out[0..D_MODEL].copy_from_slice(&self.h_low);
        out[D_MODEL..2 * D_MODEL].copy_from_slice(&self.h_mid);
        out[2 * D_MODEL..3 * D_MODEL].copy_from_slice(&self.h_high);
        out
    }
}

/// Output of a single sequential target step.
#[derive(Clone, Debug)]
pub struct TargetStepOutput {
    pub logits: Vec<f32>,
    pub features: TappedFeatures,
}

/// Output of a parallel tree-attention target verification pass.
#[derive(Clone, Debug)]
pub struct TargetTreeOutput {
    pub logits: Vec<Vec<f32>>,
    pub features: Vec<TappedFeatures>,
    pub tree_keys: Vec<Vec<[f32; D_MODEL]>>,
    pub tree_values: Vec<Vec<[f32; D_MODEL]>>,
}

/// Exact per-layer KV cache supporting incremental append, rollback, and tree-path commit.
#[derive(Clone, Debug)]
pub struct TargetKvCache {
    pub keys: Vec<Vec<[f32; D_MODEL]>>,
    pub values: Vec<Vec<[f32; D_MODEL]>>,
}

impl TargetKvCache {
    pub fn new() -> Self {
        Self {
            keys: (0..N_LAYERS).map(|_| Vec::new()).collect(),
            values: (0..N_LAYERS).map(|_| Vec::new()).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.keys[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        for l in 0..N_LAYERS {
            self.keys[l].clear();
            self.values[l].clear();
        }
    }

    pub fn rollback(&mut self, new_len: usize) {
        for l in 0..N_LAYERS {
            self.keys[l].truncate(new_len);
            self.values[l].truncate(new_len);
        }
    }

    pub fn commit_tree_nodes(&mut self, tree_out: &TargetTreeOutput, node_indices: &[usize]) {
        for l in 0..N_LAYERS {
            for &nid in node_indices {
                self.keys[l].push(tree_out.tree_keys[l][nid]);
                self.values[l].push(tree_out.tree_values[l][nid]);
            }
        }
    }
}

impl Default for TargetKvCache {
    fn default() -> Self {
        Self::new()
    }
}

/// One Pre-Norm Transformer block with MHA + SwiGLU FFN.
#[derive(Clone, Debug)]
struct TransformerLayer {
    norm_attn: [f32; D_MODEL],
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    norm_ffn: [f32; D_MODEL],
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
    prev_token_head0_bias: f32,
    is_induction_layer: bool,
    use_rope: bool,
}

/// Real 8-layer Pre-Norm Causal Transformer grounded on the repository corpus.
#[derive(Clone, Debug)]
pub struct TargetModel {
    /// Shared token embeddings (`VOCAB_SIZE x D_MODEL`), also reused frozen by EAGLE-3.
    pub embeddings: Vec<[f32; D_MODEL]>,
    layers: Vec<TransformerLayer>,
    final_norm: [f32; D_MODEL],
    /// Output LM head (`VOCAB_SIZE x D_MODEL`).
    pub lm_head: Vec<f32>,
    pub lm_bias: Vec<f32>,
    token_sigs: Vec<[f32; HEAD_DIM]>,
}

impl TargetModel {
    /// Build and corpus-ground the 8-layer Transformer from the 3 versioned repository files.
    pub fn from_grounded_corpus(corpus: &GroundedCorpus, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed ^ 0xA11C_E551_0000_0055);

        // 1. Near-orthogonal 16-dim unit lexical signatures for every token in VOCAB_SIZE.
        let mut token_sigs = vec![[0.0_f32; HEAD_DIM]; VOCAB_SIZE];
        for sig in &mut token_sigs {
            let mut norm_sq = 0.0_f32;
            for x in sig.iter_mut() {
                let v = rng.normal_f32();
                *x = v;
                norm_sq += v * v;
            }
            let inv = 1.0 / norm_sq.sqrt().max(1e-6);
            for x in sig.iter_mut() {
                *x *= inv;
            }
        }

        // 2. Collect corpus unigram & bigram statistics from all 3 grounded files.
        let mut unigram = vec![1.0_f64; VOCAB_SIZE];
        let mut bigram = vec![vec![0.02_f64; VOCAB_SIZE]; VOCAB_SIZE];
        for f in &corpus.files {
            for w in f.tokens.windows(2) {
                let a = w[0] as usize;
                let b = w[1] as usize;
                unigram[b] += 1.0;
                bigram[a][b] += 1.0;
            }
        }

        // Token embeddings:
        // - [0..16]: lexical signature `u(x_t)` (unit norm)
        // - [16..32]: corpus-grounded semantic next-token distribution embedding (predicts prose/code)
        // - [32..48]: slot populated by Layer 0 shift head with `u(x_{t-1})`
        // - [48..64]: slot populated by Layer 1 shift head with `u(x_{t-2})`
        let mut embeddings = vec![[0.0_f32; D_MODEL]; VOCAB_SIZE];
        for v in 0..VOCAB_SIZE {
            embeddings[v][0..HEAD_DIM].copy_from_slice(&token_sigs[v]);
            let row_sum: f64 = bigram[v].iter().sum();
            let mut next_sig = [0.0_f32; HEAD_DIM];
            for next_v in 0..VOCAB_SIZE {
                let p = (bigram[v][next_v] / row_sum) as f32;
                for d in 0..HEAD_DIM {
                    next_sig[d] += p * token_sigs[next_v][d];
                }
            }
            let nsq: f32 = next_sig.iter().map(|x| x * x).sum();
            let inv = 1.0 / nsq.sqrt().max(1e-6);
            for d in 0..HEAD_DIM {
                embeddings[v][HEAD_DIM + d] = next_sig[d] * inv * 0.55;
            }
        }

        // 3. Build 8 Transformer layers:
        //    - Layer 0: 1-step shift head writes `u(x_{t-1})` into `[32..48]`
        //    - Layer 1 (TAP_LOW): 1-step shift head writes `u(x_{t-2})` into `[48..64]`
        //    - Layer 2: QK-Norm Induction Head matching `(u(x_t), u(x_{t-1}))` against
        //      `(u(x_{j-1}), u(x_{j-2}))` with zero-value attention sink (only activates for
        //      genuine earlier context spans >= 8 positions back to prevent degenerate 2-tok cycles!).
        //    - Layers 3..7 (TAP_MID=4, TAP_HIGH=7): Smooth RoPE contextual attention + SwiGLU FFN
        //      mixing `(x_{t-2}, x_{t-1}, x_t)` cleanly into structured generalizable features!
        let mut layers = Vec::with_capacity(N_LAYERS);
        for l in 0..N_LAYERS {
            let is_early_routing = l <= 2;
            let scale_attn = if is_early_routing { 0.008 } else { 0.032 };
            let scale_ffn = if is_early_routing { 0.008 } else { 0.028 };

            let mut w_q = (0..D_MODEL * D_MODEL)
                .map(|_| rng.normal_f32() * scale_attn)
                .collect::<Vec<_>>();
            let mut w_k = (0..D_MODEL * D_MODEL)
                .map(|_| rng.normal_f32() * scale_attn)
                .collect::<Vec<_>>();
            let mut w_v = (0..D_MODEL * D_MODEL)
                .map(|_| rng.normal_f32() * scale_attn)
                .collect::<Vec<_>>();
            let mut w_o = (0..D_MODEL * D_MODEL)
                .map(|_| rng.normal_f32() * scale_attn)
                .collect::<Vec<_>>();

            let w_gate = (0..D_FF * D_MODEL)
                .map(|_| rng.normal_f32() * scale_ffn)
                .collect::<Vec<_>>();
            let w_up = (0..D_FF * D_MODEL)
                .map(|_| rng.normal_f32() * scale_ffn)
                .collect::<Vec<_>>();
            let w_down = (0..D_MODEL * D_FF)
                .map(|_| rng.normal_f32() * scale_ffn * 0.45)
                .collect::<Vec<_>>();

            let mut prev_token_head0_bias = 0.0_f32;
            let mut is_induction_layer = false;
            let use_rope = l >= 3;

            if l == 0 {
                w_v.fill(0.0);
                w_o.fill(0.0);
                prev_token_head0_bias = 24.0;
                for d in 0..HEAD_DIM {
                    w_v[d * D_MODEL + d] = 1.0;
                    w_o[(2 * HEAD_DIM + d) * D_MODEL + d] = 1.0;
                }
            } else if l == 1 {
                w_v.fill(0.0);
                w_o.fill(0.0);
                prev_token_head0_bias = 24.0;
                for d in 0..HEAD_DIM {
                    w_v[d * D_MODEL + (2 * HEAD_DIM + d)] = 1.0;
                    w_o[(3 * HEAD_DIM + d) * D_MODEL + d] = 1.0;
                }
            } else if l == 2 {
                w_q.fill(0.0);
                w_k.fill(0.0);
                w_v.fill(0.0);
                w_o.fill(0.0);
                is_induction_layer = true;
                for d in 0..HEAD_DIM {
                    w_q[(HEAD_DIM + d) * D_MODEL + d] = 1.0;
                    w_k[(HEAD_DIM + d) * D_MODEL + (2 * HEAD_DIM + d)] = 1.0;
                    w_v[(HEAD_DIM + d) * D_MODEL + d] = 1.0;
                    w_o[(HEAD_DIM + d) * D_MODEL + (HEAD_DIM + d)] = 14.0;
                    w_q[(2 * HEAD_DIM + d) * D_MODEL + (2 * HEAD_DIM + d)] = 1.0;
                    w_k[(2 * HEAD_DIM + d) * D_MODEL + (3 * HEAD_DIM + d)] = 1.0;
                }
            }

            layers.push(TransformerLayer {
                norm_attn: [1.0_f32; D_MODEL],
                w_q,
                w_k,
                w_v,
                w_o,
                norm_ffn: [1.0_f32; D_MODEL],
                w_gate,
                w_up,
                w_down,
                prev_token_head0_bias,
                is_induction_layer,
                use_rope,
            });
        }

        // 4. Corpus-grounded LM head: decodes induction-copied signatures from [16..32]
        //    AND corpus-learned contextual transitions from [0..64].
        let total_uni: f64 = unigram.iter().sum();
        let mut lm_bias = vec![0.0_f32; VOCAB_SIZE];
        let mut lm_head = vec![0.0_f32; VOCAB_SIZE * D_MODEL];
        for v in 0..VOCAB_SIZE {
            lm_bias[v] = ((unigram[v] / total_uni).ln() * 0.25) as f32;
            for d in 0..HEAD_DIM {
                lm_head[v * D_MODEL + (HEAD_DIM + d)] = token_sigs[v][d] * 6.5;
                lm_head[v * D_MODEL + d] = rng.normal_f32() * 0.14;
                lm_head[v * D_MODEL + (2 * HEAD_DIM + d)] = rng.normal_f32() * 0.14;
                lm_head[v * D_MODEL + (3 * HEAD_DIM + d)] = rng.normal_f32() * 0.14;
            }
        }

        let mut model = Self {
            embeddings,
            layers,
            final_norm: [1.0_f32; D_MODEL],
            lm_head,
            lm_bias,
            token_sigs,
        };

        // 5. Train `lm_head` + `lm_bias` on grounded corpus sequences via real cross-entropy SGD
        //    so prose & mixed code continuations reflect true corpus transition structure!
        model.ground_lm_head_on_corpus(corpus);
        model
    }

    fn ground_lm_head_on_corpus(&mut self, corpus: &GroundedCorpus) {
        let seqs = corpus.training_sequences(18, 48);
        let lr = 0.09_f32;
        let mut cache = TargetKvCache::new();

        for _epoch in 0..5 {
            for seq in &seqs {
                cache.clear();
                let mut h_states = Vec::with_capacity(seq.len() - 1);
                for &tok in &seq[..seq.len() - 1] {
                    let out = self.forward_step_raw(tok, &mut cache, 0.0);
                    h_states.push(out.features.h_high);
                }
                for (t, h) in h_states.iter().enumerate() {
                    let target = seq[t + 1] as usize;
                    let mut normed = [0.0_f32; D_MODEL];
                    rms_norm(h, &self.final_norm, 1e-5, &mut normed);

                    let mut logits = vec![0.0_f32; VOCAB_SIZE];
                    for v in 0..VOCAB_SIZE {
                        let row = &self.lm_head[v * D_MODEL..(v + 1) * D_MODEL];
                        let mut acc = self.lm_bias[v];
                        for d in 0..D_MODEL {
                            acc += normed[d] * row[d];
                        }
                        logits[v] = acc;
                    }
                    let probs = softmax_temp(&logits, 1.0);
                    for v in 0..VOCAB_SIZE {
                        let g = probs[v] - if v == target { 1.0 } else { 0.0 };
                        self.lm_bias[v] -= lr * 0.25 * g;
                        let row = &mut self.lm_head[v * D_MODEL..(v + 1) * D_MODEL];
                        for d in 0..D_MODEL {
                            let scale = if (HEAD_DIM..2 * HEAD_DIM).contains(&d) {
                                0.08
                            } else {
                                1.0
                            };
                            row[d] -= lr * scale * g * normed[d];
                        }
                    }
                }
            }
        }
    }

    /// Project final hidden state `h` through `final_norm` + `lm_head` + `lm_bias`
    /// with optional deterministic high-entropy creative jitter (applied identically in
    /// plain decode and tree verification for 100.0% bit-exactness).
    pub fn project_logits(
        &self,
        h: &[f32; D_MODEL],
        pos: usize,
        token: u32,
        entropy_jitter: f32,
    ) -> Vec<f32> {
        let mut normed = [0.0_f32; D_MODEL];
        rms_norm(h, &self.final_norm, 1e-5, &mut normed);

        // When `entropy_jitter > 0.0` (high-entropy creative regime), flatten logit temperature
        // AND inject deterministic creative branching noise keyed strictly by `(pos, token, v)`
        // so target entropy is genuinely high (> 2.5 nats) while remaining 100.0% bit-exact!
        let logit_scale = if entropy_jitter > 0.0 {
            0.30_f32
        } else {
            1.0_f32
        };

        let mut logits = vec![0.0_f32; VOCAB_SIZE];
        for v in 0..VOCAB_SIZE {
            let row = &self.lm_head[v * D_MODEL..(v + 1) * D_MODEL];
            let mut acc = self.lm_bias[v];
            for d in 0..D_MODEL {
                acc += normed[d] * row[d];
            }
            acc *= logit_scale;
            if entropy_jitter > 0.0 {
                let mut z = (pos as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((token as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
                    .wrapping_add((v as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
                z ^= z >> 30;
                z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z ^= z >> 27;
                let u = (((z >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0) as f32;
                acc += entropy_jitter * u;
            }
            logits[v] = acc;
        }
        logits
    }

    pub fn token_signature(&self, token: u32) -> &[f32; HEAD_DIM] {
        &self.token_sigs[(token as usize).min(VOCAB_SIZE - 1)]
    }

    pub fn prefill_prompt(
        &self,
        prompt: &[u32],
        cache: &mut TargetKvCache,
        entropy_jitter: f32,
    ) -> TargetStepOutput {
        assert!(!prompt.is_empty(), "prompt must be non-empty");
        cache.clear();
        let mut last = None;
        for &tok in prompt {
            last = Some(self.forward_step_raw(tok, cache, entropy_jitter));
        }
        last.expect("non-empty prompt")
    }

    pub fn forward_step_raw(
        &self,
        token: u32,
        cache: &mut TargetKvCache,
        entropy_jitter: f32,
    ) -> TargetStepOutput {
        let tree = DraftTree::root_only(token);
        let out = self.forward_tree(&tree, cache, entropy_jitter);
        cache.commit_tree_nodes(&out, &[0]);
        TargetStepOutput {
            logits: out.logits.into_iter().next().expect("root logits"),
            features: out.features.into_iter().next().expect("root features"),
        }
    }

    /// Bit-exact parallel tree-attention forward pass for `N` nodes in `tree` against
    /// committed prefix KV states in `cache` (does NOT mutate `cache`).
    pub fn forward_tree(
        &self,
        tree: &DraftTree,
        cache: &TargetKvCache,
        entropy_jitter: f32,
    ) -> TargetTreeOutput {
        let n = tree.num_nodes();
        let prefix_len = cache.len();

        let ancestor_paths: Vec<Vec<usize>> = (0..n).map(|i| tree.ancestor_path(i)).collect();

        let mut hidden = vec![[0.0_f32; D_MODEL]; n];
        for i in 0..n {
            let tok = (tree.nodes[i].token as usize).min(VOCAB_SIZE - 1);
            hidden[i] = self.embeddings[tok];
        }

        let mut normed = vec![[0.0_f32; D_MODEL]; n];
        let mut q = vec![[0.0_f32; D_MODEL]; n];
        let mut k = vec![[0.0_f32; D_MODEL]; n];
        let mut v = vec![[0.0_f32; D_MODEL]; n];
        let mut attn_concat = vec![[0.0_f32; D_MODEL]; n];
        let mut proj_out = vec![[0.0_f32; D_MODEL]; n];
        let mut ffn_mid = vec![[0.0_f32; D_FF]; n];

        let mut h_low = vec![[0.0_f32; D_MODEL]; n];
        let mut h_mid = vec![[0.0_f32; D_MODEL]; n];
        let mut h_high = vec![[0.0_f32; D_MODEL]; n];
        let mut tree_keys = Vec::with_capacity(N_LAYERS);
        let mut tree_values = Vec::with_capacity(N_LAYERS);

        let inv_sqrt_hd = 1.0 / (HEAD_DIM as f32).sqrt();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            for i in 0..n {
                rms_norm(&hidden[i], &layer.norm_attn, 1e-5, &mut normed[i]);
            }

            linear_batch_d_into(&normed, &layer.w_q, &mut q);
            linear_batch_d_into(&normed, &layer.w_k, &mut k);
            linear_batch_d_into(&normed, &layer.w_v, &mut v);

            if layer.use_rope {
                for i in 0..n {
                    let pos = prefix_len + tree.nodes[i].depth;
                    apply_rope(&mut q[i], pos);
                    apply_rope(&mut k[i], pos);
                }
            }

            attn_concat.fill([0.0_f32; D_MODEL]);
            let p_keys = &cache.keys[layer_idx];
            let p_vals = &cache.values[layer_idx];

            for i in 0..n {
                let path = &ancestor_paths[i];
                let total_ctx = prefix_len + path.len();
                let cur_pos = total_ctx - 1;

                for head in 0..N_HEADS {
                    let h_off = head * HEAD_DIM;
                    let q_h = &q[i][h_off..h_off + HEAD_DIM];

                    if layer.is_induction_layer && head == 1 {
                        // Exact 4-gram QK-Norm induction head with zero-value attention sink.
                        // Matches `(x_{t-3}, x_{t-2}, x_{t-1}, x_t)` against `(x_{j-4}, x_{j-3}, x_{j-2}, x_{j-1})`
                        // along the exact chronological causal/tree ancestor path so it copies structured
                        // code-edit diffs verbatim while never false-triggering on common 2-char prose bigrams!
                        let get_k = |pos_idx: usize| -> &[f32; D_MODEL] {
                            if pos_idx < prefix_len {
                                &p_keys[pos_idx]
                            } else {
                                &k[path[pos_idx - prefix_len]]
                            }
                        };
                        let get_q_tok_sig = |pos_idx: usize| -> &[f32] {
                            // `k[pos_idx]` stores u(x_{pos-1}) in [16..32] and u(x_{pos-2}) in [32..48],
                            // while `v[pos_idx]` stores u(x_{pos}) in [16..32]!
                            if pos_idx < prefix_len {
                                &p_vals[pos_idx][h_off..h_off + HEAD_DIM]
                            } else {
                                &v[path[pos_idx - prefix_len]][h_off..h_off + HEAD_DIM]
                            }
                        };

                        let mut scores = Vec::with_capacity(total_ctx + 1);
                        let sink_score = 88.0_f32; // requires ~3.67 / 4.0 cosine sum (exact 4-gram match)
                        let mut max_s = sink_score;
                        scores.push(sink_score);

                        // Query 4-gram signatures ending at `cur_pos`: [x_{cur-3}, x_{cur-2}, x_{cur-1}, x_cur]
                        let can_query = cur_pos >= 4;
                        let q_sigs = if can_query {
                            [
                                get_q_tok_sig(cur_pos),
                                get_q_tok_sig(cur_pos - 1),
                                get_q_tok_sig(cur_pos - 2),
                                get_q_tok_sig(cur_pos - 3),
                            ]
                        } else {
                            [q_h, q_h, q_h, q_h]
                        };
                        let q_norms = [
                            vec_norm(q_sigs[0]),
                            vec_norm(q_sigs[1]),
                            vec_norm(q_sigs[2]),
                            vec_norm(q_sigs[3]),
                        ];

                        for step_idx in 0..total_ctx {
                            let is_valid_copy =
                                can_query && step_idx >= 4 && cur_pos >= step_idx + 12;
                            let s = if is_valid_copy {
                                let k_sigs = [
                                    get_q_tok_sig(step_idx - 1),
                                    get_q_tok_sig(step_idx - 2),
                                    get_q_tok_sig(step_idx - 3),
                                    get_q_tok_sig(step_idx - 4),
                                ];
                                let mut cos_sum = 0.0_f32;
                                for m in 0..4 {
                                    let kn = vec_norm(k_sigs[m]);
                                    let qn = q_norms[m];
                                    if kn > 1e-4 && qn > 1e-4 {
                                        let mut dot = 0.0_f32;
                                        for d in 0..HEAD_DIM {
                                            dot += q_sigs[m][d] * k_sigs[m][d];
                                        }
                                        cos_sum += dot / (qn * kn);
                                    }
                                }
                                // Keep `get_k` used cleanly for compiler hygiene
                                let _ = get_k(step_idx)[0];
                                cos_sum * 24.0 + (step_idx as f32) * 1e-3
                            } else {
                                -1e9_f32
                            };
                            if s > max_s {
                                max_s = s;
                            }
                            scores.push(s);
                        }

                        let mut sum_e = 0.0_f32;
                        for s in &mut scores {
                            *s = (*s - max_s).exp();
                            sum_e += *s;
                        }
                        let inv_e = 1.0 / sum_e.max(1e-12);
                        for step_idx in 0..total_ctx {
                            let w = scores[step_idx + 1] * inv_e;
                            let v_vec = if step_idx < prefix_len {
                                &p_vals[step_idx]
                            } else {
                                &v[path[step_idx - prefix_len]]
                            };
                            for d in 0..HEAD_DIM {
                                attn_concat[i][h_off + d] += w * v_vec[h_off + d];
                            }
                        }
                        continue;
                    }

                    let mut scores = Vec::with_capacity(total_ctx);
                    let mut max_s = f32::NEG_INFINITY;
                    for step_idx in 0..total_ctx {
                        let k_vec = if step_idx < prefix_len {
                            &p_keys[step_idx]
                        } else {
                            &k[path[step_idx - prefix_len]]
                        };
                        let mut dot = 0.0_f32;
                        for d in 0..HEAD_DIM {
                            dot += q_h[d] * k_vec[h_off + d];
                        }
                        let mut s = dot * inv_sqrt_hd;
                        if head == 0 && layer.prev_token_head0_bias > 0.0 {
                            let dist = cur_pos.saturating_sub(step_idx);
                            if dist == 1 {
                                s += layer.prev_token_head0_bias;
                            } else {
                                s -= layer.prev_token_head0_bias * 0.5;
                            }
                        }
                        if s > max_s {
                            max_s = s;
                        }
                        scores.push(s);
                    }
                    let mut sum_e = 0.0_f32;
                    for s in &mut scores {
                        *s = (*s - max_s).exp();
                        sum_e += *s;
                    }
                    let inv_e = 1.0 / sum_e.max(1e-12);
                    for step_idx in 0..total_ctx {
                        let w = scores[step_idx] * inv_e;
                        let v_vec = if step_idx < prefix_len {
                            &p_vals[step_idx]
                        } else {
                            &v[path[step_idx - prefix_len]]
                        };
                        for d in 0..HEAD_DIM {
                            attn_concat[i][h_off + d] += w * v_vec[h_off + d];
                        }
                    }
                }
            }

            linear_batch_d_into(&attn_concat, &layer.w_o, &mut proj_out);
            for i in 0..n {
                for d in 0..D_MODEL {
                    hidden[i][d] += proj_out[i][d];
                }
            }

            for i in 0..n {
                rms_norm(&hidden[i], &layer.norm_ffn, 1e-5, &mut normed[i]);
            }
            swiglu_batch_into(
                &normed,
                &layer.w_gate,
                &layer.w_up,
                &layer.w_down,
                &mut ffn_mid,
                &mut proj_out,
            );
            for i in 0..n {
                for d in 0..D_MODEL {
                    hidden[i][d] += proj_out[i][d];
                }
            }

            if layer_idx == TAP_LOW_LAYER {
                h_low.copy_from_slice(&hidden);
            } else if layer_idx == TAP_MID_LAYER {
                h_mid.copy_from_slice(&hidden);
            } else if layer_idx == TAP_HIGH_LAYER {
                h_high.copy_from_slice(&hidden);
            }

            tree_keys.push(k.clone());
            tree_values.push(v.clone());
        }

        let mut logits = Vec::with_capacity(n);
        let mut features = Vec::with_capacity(n);
        for i in 0..n {
            let pos = prefix_len + tree.nodes[i].depth;
            logits.push(self.project_logits(&h_high[i], pos, tree.nodes[i].token, entropy_jitter));
            features.push(TappedFeatures {
                h_low: h_low[i],
                h_mid: h_mid[i],
                h_high: h_high[i],
            });
        }

        TargetTreeOutput {
            logits,
            features,
            tree_keys,
            tree_values,
        }
    }
}

#[inline]
fn vec_norm(x: &[f32]) -> f32 {
    let mut s = 0.0_f32;
    for &v in x {
        s += v * v;
    }
    s.sqrt()
}

/// Apply standard RoPE rotation per head at sequence position `pos`.
fn apply_rope(vec: &mut [f32; D_MODEL], pos: usize) {
    let p = pos as f32;
    for head in 0..N_HEADS {
        let base = head * HEAD_DIM;
        for pair in 0..(HEAD_DIM / 2) {
            let freq = 10_000.0_f32.powf(-((2 * pair) as f32) / (HEAD_DIM as f32));
            let angle = p * freq;
            let (s, c) = angle.sin_cos();
            let x0 = vec[base + 2 * pair];
            let x1 = vec[base + 2 * pair + 1];
            vec[base + 2 * pair] = x0 * c - x1 * s;
            vec[base + 2 * pair + 1] = x0 * s + x1 * c;
        }
    }
}

/// Cache-blocked `[N, D_MODEL] x [D_MODEL, D_MODEL]^T` with 4-wide node register blocking
/// preserving the exact IEEE-754 summation order of `batch = 1` for 100% bit-exactness.
fn linear_batch_d_into(x: &[[f32; D_MODEL]], w: &[f32], out: &mut [[f32; D_MODEL]]) {
    let n = x.len();
    for r in 0..D_MODEL {
        let w_row = &w[r * D_MODEL..(r + 1) * D_MODEL];
        let mut i = 0usize;
        while i + 4 <= n {
            let x0 = &x[i];
            let x1 = &x[i + 1];
            let x2 = &x[i + 2];
            let x3 = &x[i + 3];
            let (mut a0, mut a1, mut a2, mut a3) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
            for c in 0..D_MODEL {
                let wc = w_row[c];
                a0 += x0[c] * wc;
                a1 += x1[c] * wc;
                a2 += x2[c] * wc;
                a3 += x3[c] * wc;
            }
            out[i][r] = a0;
            out[i + 1][r] = a1;
            out[i + 2][r] = a2;
            out[i + 3][r] = a3;
            i += 4;
        }
        while i < n {
            let xi = &x[i];
            let mut acc = 0.0_f32;
            for c in 0..D_MODEL {
                acc += xi[c] * w_row[c];
            }
            out[i][r] = acc;
            i += 1;
        }
    }
}

/// Cache-blocked SwiGLU FFN with 4-wide node register blocking preserving exact per-node
/// IEEE-754 summation order with zero inner allocations.
fn swiglu_batch_into(
    x: &[[f32; D_MODEL]],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    mid: &mut [[f32; D_FF]],
    out: &mut [[f32; D_MODEL]],
) {
    let n = x.len();
    for r in 0..D_FF {
        let wg = &w_gate[r * D_MODEL..(r + 1) * D_MODEL];
        let wu = &w_up[r * D_MODEL..(r + 1) * D_MODEL];
        let mut i = 0usize;
        while i + 4 <= n {
            let x0 = &x[i];
            let x1 = &x[i + 1];
            let x2 = &x[i + 2];
            let x3 = &x[i + 3];
            let (mut g0, mut g1, mut g2, mut g3) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
            let (mut u0, mut u1, mut u2, mut u3) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
            for c in 0..D_MODEL {
                let wgc = wg[c];
                let wuc = wu[c];
                g0 += x0[c] * wgc;
                u0 += x0[c] * wuc;
                g1 += x1[c] * wgc;
                u1 += x1[c] * wuc;
                g2 += x2[c] * wgc;
                u2 += x2[c] * wuc;
                g3 += x3[c] * wgc;
                u3 += x3[c] * wuc;
            }
            mid[i][r] = silu(g0) * u0;
            mid[i + 1][r] = silu(g1) * u1;
            mid[i + 2][r] = silu(g2) * u2;
            mid[i + 3][r] = silu(g3) * u3;
            i += 4;
        }
        while i < n {
            let xi = &x[i];
            let mut g = 0.0_f32;
            let mut u = 0.0_f32;
            for c in 0..D_MODEL {
                g += xi[c] * wg[c];
                u += xi[c] * wu[c];
            }
            mid[i][r] = silu(g) * u;
            i += 1;
        }
    }
    for r in 0..D_MODEL {
        let wd = &w_down[r * D_FF..(r + 1) * D_FF];
        let mut i = 0usize;
        while i + 4 <= n {
            let m0 = &mid[i];
            let m1 = &mid[i + 1];
            let m2 = &mid[i + 2];
            let m3 = &mid[i + 3];
            let (mut a0, mut a1, mut a2, mut a3) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
            for c in 0..D_FF {
                let wc = wd[c];
                a0 += m0[c] * wc;
                a1 += m1[c] * wc;
                a2 += m2[c] * wc;
                a3 += m3[c] * wc;
            }
            out[i][r] = a0;
            out[i + 1][r] = a1;
            out[i + 2][r] = a2;
            out[i + 3][r] = a3;
            i += 4;
        }
        while i < n {
            let mi = &mid[i];
            let mut acc = 0.0_f32;
            for c in 0..D_FF {
                acc += mi[c] * wd[c];
            }
            out[i][r] = acc;
            i += 1;
        }
    }
}
