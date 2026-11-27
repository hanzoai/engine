//! Clean-room Apache-2.0 implementation of the EAGLE-3 feature-level speculative
//! draft head (arXiv 2503.01840):
//!
//! 1. **Multi-layer hidden-state fusion**: projects concatenated target hidden features
//!    `[h_low; h_mid; h_high]` (`3 * D_MODEL`) -> `D_MODEL` via learned `W_fc` + RMSNorm.
//! 2. **Token + feature combination**: projects `[e(x_{t+1}); feat_t]` (`2 * D_MODEL`) -> `D_MODEL`
//!    via `W_in`, reusing frozen target token embeddings.
//! 3. **Lightweight 1-layer causal draft Transformer**: single context-mixing + SiLU FFN
//!    layer operating autoregressively on its own output features `a_s`.
//! 4. **Shared frozen target LM head**: maps normalized draft features `rms_norm(a_s)` through
//!    the target model's frozen `lm_head` + `lm_bias` (exact EAGLE-3 architecture, preventing
//!    overfitting and keeping draft-head footprint minimal!).
//! 5. **Training-Time Test (TTT) multi-step unrolled cross-entropy training**: trains directly
//!    via multi-step unrolled KL/CE distillation (NO feature regression loss!) with analytical
//!    BPTT + Adam on on-policy target rollouts grounded in the repository corpus.
//! 6. **Linear chain & EAGLE-3 beam tree drafting**: emits linear chains (`draft_linear`),
//!    beam draft trees (`draft_tree`), and exact stochastic draft trees (`draft_tree_stochastic`).

use crate::corpus::VOCAB_SIZE;
use crate::rng::{argmax_f32, rms_norm, silu, softmax_temp, SplitMix64};
use crate::target::{TappedFeatures, TargetKvCache, TargetModel, D_MODEL};
use crate::verify::{DraftTree, LinearDraft, TreeNode};

pub const D_FUSE_IN: usize = 3 * D_MODEL; // 192
pub const D_COMBINE_IN: usize = 2 * D_MODEL; // 128
pub const D_DRAFT_FF: usize = 128;

/// Training telemetry recorded before and after EAGLE-3 TTT optimization.
#[derive(Clone, Debug)]
pub struct Eagle3TrainingReport {
    pub epochs: usize,
    pub unroll_steps: usize,
    pub training_windows: usize,
    pub untrained_loss: f64,
    pub trained_loss: f64,
    pub untrained_top1_acc: f64,
    pub trained_top1_acc: f64,
    pub untrained_step2_top1_acc: f64,
    pub trained_step2_top1_acc: f64,
    pub train_wall_ms: f64,
}

/// Lightweight per-parameter Adam state.
#[derive(Clone, Debug)]
struct AdamParam {
    w: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
}

impl AdamParam {
    fn new(w: Vec<f32>) -> Self {
        let n = w.len();
        Self {
            w,
            m: vec![0.0; n],
            v: vec![0.0; n],
            g: vec![0.0; n],
        }
    }

    fn zero_grad(&mut self) {
        self.g.fill(0.0);
    }

    fn step(&mut self, lr: f32, step_t: u32, weight_decay: f32) {
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let eps = 1e-8_f32;
        let bc1 = 1.0 - beta1.powi(step_t as i32);
        let bc2 = 1.0 - beta2.powi(step_t as i32);
        for i in 0..self.w.len() {
            let grad = self.g[i].clamp(-5.0, 5.0);
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * grad;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * grad * grad;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            self.w[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * self.w[i]);
        }
    }
}

/// Cached activations for one unrolled draft step (used for exact analytical BPTT in `train_ttt`).
struct StepCache {
    comb: [f32; D_COMBINE_IN],
    post_attn: [f32; D_MODEL],
    ffn_pre: [f32; D_DRAFT_FF],
    ffn_act: [f32; D_DRAFT_FF],
    feat_out: [f32; D_MODEL],
    normed_out: [f32; D_MODEL],
    inv_rms_out: f32,
    probs: Vec<f32>,
}

/// Clean-room EAGLE-3 feature-level draft head sharing frozen target embeddings & LM head.
#[derive(Clone, Debug)]
pub struct Eagle3DraftHead {
    /// Multi-layer fusion `W_fc: [D_MODEL, 3 * D_MODEL]` fusing `[h_low; h_mid; h_high]`.
    w_fc: AdamParam,
    /// Input combiner `W_in: [D_MODEL, 2 * D_MODEL]` combining `[e(x); prev_feat]`.
    w_in: AdamParam,
    /// Lightweight draft self-attention context projection `W_ctx: [D_MODEL, D_MODEL]`.
    w_ctx: AdamParam,
    /// Draft FFN layer 1 `W_ff1: [D_DRAFT_FF, D_MODEL]` and bias.
    w_ff1: AdamParam,
    b_ff1: AdamParam,
    /// Draft FFN layer 2 `W_ff2: [D_MODEL, D_DRAFT_FF]`.
    w_ff2: AdamParam,
    norm_ones: [f32; D_MODEL],
}

type TeacherTrajectory = (Vec<u32>, Vec<TappedFeatures>, Vec<Vec<f32>>);

impl Eagle3DraftHead {
    /// Initialize an untrained EAGLE-3 draft head (random small projections; embeddings and
    /// LM head are shared frozen from `TargetModel`, matching arXiv 2503.01840!).
    pub fn new_untrained(_target: &TargetModel, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed ^ 0xEA61_E300_0000_0055);
        let init_mat = |rows: usize, cols: usize, scale: f32, r: &mut SplitMix64| -> Vec<f32> {
            (0..rows * cols).map(|_| r.normal_f32() * scale).collect()
        };

        Self {
            w_fc: AdamParam::new(init_mat(D_MODEL, D_FUSE_IN, 0.08, &mut rng)),
            w_in: AdamParam::new(init_mat(D_MODEL, D_COMBINE_IN, 0.08, &mut rng)),
            w_ctx: AdamParam::new(init_mat(D_MODEL, D_MODEL, 0.05, &mut rng)),
            w_ff1: AdamParam::new(init_mat(D_DRAFT_FF, D_MODEL, 0.08, &mut rng)),
            b_ff1: AdamParam::new(vec![0.0_f32; D_DRAFT_FF]),
            w_ff2: AdamParam::new(init_mat(D_MODEL, D_DRAFT_FF, 0.06, &mut rng)),
            norm_ones: [1.0_f32; D_MODEL],
        }
    }

    /// Total trainable parameter count of the EAGLE-3 draft head (embeddings & LM head shared frozen).
    pub fn param_count(&self) -> usize {
        self.w_fc.w.len()
            + self.w_in.w.len()
            + self.w_ctx.w.len()
            + self.w_ff1.w.len()
            + self.b_ff1.w.len()
            + self.w_ff2.w.len()
    }

    /// Fuse `[h_low; h_mid; h_high]` (`3 * D_MODEL`) -> `g_fused` (`D_MODEL`).
    pub fn fuse_features(&self, tapped: &TappedFeatures) -> [f32; D_MODEL] {
        let cat = tapped.concat();
        let mut raw = [0.0_f32; D_MODEL];
        for r in 0..D_MODEL {
            let row = &self.w_fc.w[r * D_FUSE_IN..(r + 1) * D_FUSE_IN];
            let mut acc = 0.0_f32;
            for c in 0..D_FUSE_IN {
                acc += cat[c] * row[c];
            }
            raw[r] = acc;
        }
        let mut out = [0.0_f32; D_MODEL];
        rms_norm(&raw, &self.norm_ones, 1e-5, &mut out);
        out
    }

    /// Execute one autoregressive draft step given current token `tok`, previous feature `prev_feat`,
    /// and rolling causal context summary `ctx_summary`, projecting via shared frozen `target.lm_head`.
    fn forward_draft_step(
        &self,
        tok: u32,
        prev_feat: &[f32; D_MODEL],
        ctx_summary: &[f32; D_MODEL],
        target: &TargetModel,
    ) -> (StepCache, Vec<f32>) {
        let emb = &target.embeddings[(tok as usize).min(VOCAB_SIZE - 1)];
        let mut comb = [0.0_f32; D_COMBINE_IN];
        comb[0..D_MODEL].copy_from_slice(emb);
        comb[D_MODEL..D_COMBINE_IN].copy_from_slice(prev_feat);

        // Input combination u = W_in * [e(x); prev_feat] + 0.5 * prev_feat
        let mut u = [0.0_f32; D_MODEL];
        for r in 0..D_MODEL {
            let row = &self.w_in.w[r * D_COMBINE_IN..(r + 1) * D_COMBINE_IN];
            let mut acc = prev_feat[r] * 0.5;
            for c in 0..D_COMBINE_IN {
                acc += comb[c] * row[c];
            }
            u[r] = acc;
        }

        // Lightweight causal context mixing
        let mut post_attn = u;
        for r in 0..D_MODEL {
            let row = &self.w_ctx.w[r * D_MODEL..(r + 1) * D_MODEL];
            let mut acc = 0.0_f32;
            for c in 0..D_MODEL {
                acc += ctx_summary[c] * row[c];
            }
            post_attn[r] += acc;
        }

        // Lightweight draft FFN (SiLU) + residual
        let mut ffn_pre = [0.0_f32; D_DRAFT_FF];
        let mut ffn_act = [0.0_f32; D_DRAFT_FF];
        for r in 0..D_DRAFT_FF {
            let row = &self.w_ff1.w[r * D_MODEL..(r + 1) * D_MODEL];
            let mut acc = self.b_ff1.w[r];
            for c in 0..D_MODEL {
                acc += post_attn[c] * row[c];
            }
            ffn_pre[r] = acc;
            ffn_act[r] = silu(acc);
        }

        let mut feat_out = post_attn;
        for r in 0..D_MODEL {
            let row = &self.w_ff2.w[r * D_DRAFT_FF..(r + 1) * D_DRAFT_FF];
            let mut acc = 0.0_f32;
            for c in 0..D_DRAFT_FF {
                acc += ffn_act[c] * row[c];
            }
            feat_out[r] += acc;
        }

        // RMSNorm + shared frozen target LM head!
        let sum_sq: f32 = feat_out.iter().map(|x| x * x).sum();
        let inv_rms_out = 1.0 / ((sum_sq / (D_MODEL as f32)) + 1e-5).sqrt();
        let mut normed_out = [0.0_f32; D_MODEL];
        for d in 0..D_MODEL {
            normed_out[d] = feat_out[d] * inv_rms_out;
        }

        let mut logits = vec![0.0_f32; VOCAB_SIZE];
        for v in 0..VOCAB_SIZE {
            let row = &target.lm_head[v * D_MODEL..(v + 1) * D_MODEL];
            let mut acc = target.lm_bias[v];
            for d in 0..D_MODEL {
                acc += normed_out[d] * row[d];
            }
            logits[v] = acc;
        }
        let probs = softmax_temp(&logits, 1.0);

        let cache = StepCache {
            comb,
            post_attn,
            ffn_pre,
            ffn_act,
            feat_out,
            normed_out,
            inv_rms_out,
            probs,
        };
        (cache, logits)
    }

    fn eval_on_trajectories(
        &self,
        trajectories: &[TeacherTrajectory],
        target: &TargetModel,
        unroll_steps: usize,
    ) -> (f64, f64, f64) {
        let mut total_loss = 0.0_f64;
        let mut total_count = 0usize;
        let mut step1_hits = 0usize;
        let mut step1_total = 0usize;
        let mut step2_hits = 0usize;
        let mut step2_total = 0usize;

        for (seq, feats, tgt_probs) in trajectories {
            if seq.len() <= unroll_steps + 2 {
                continue;
            }
            let max_anchor = seq.len() - unroll_steps - 1;
            for t in 1..max_anchor {
                let mut cur_feat = self.fuse_features(&feats[t]);
                let mut ctx_sum = cur_feat;
                for s in 0..unroll_steps {
                    let in_tok = argmax_f32(&tgt_probs[t + s]) as u32;
                    let (sc, _) = self.forward_draft_step(in_tok, &cur_feat, &ctx_sum, target);
                    let tgt = &tgt_probs[t + 1 + s];
                    let tgt_top = argmax_f32(tgt);
                    let draft_top = argmax_f32(&sc.probs);

                    let mut ce = 0.0_f64;
                    for v in 0..VOCAB_SIZE {
                        let pt = tgt[v] as f64;
                        if pt > 1e-8 {
                            ce -= pt * (sc.probs[v].max(1e-8) as f64).ln();
                        }
                    }
                    total_loss += ce;
                    total_count += 1;

                    if s == 0 {
                        step1_total += 1;
                        if draft_top == tgt_top {
                            step1_hits += 1;
                        }
                    } else if s == 1 {
                        step2_total += 1;
                        if draft_top == tgt_top {
                            step2_hits += 1;
                        }
                    }

                    let inv = 1.0 / ((s + 2) as f32);
                    for d in 0..D_MODEL {
                        ctx_sum[d] = ctx_sum[d] * (1.0 - inv) + sc.feat_out[d] * inv;
                    }
                    cur_feat = sc.feat_out;
                }
            }
        }

        let loss = total_loss / (total_count.max(1) as f64);
        let acc1 = (step1_hits as f64) / (step1_total.max(1) as f64);
        let acc2 = (step2_hits as f64) / (step2_total.max(1) as f64);
        (loss, acc1, acc2)
    }

    /// Train the EAGLE-3 draft head directly on on-policy target rollouts grounded in the
    /// repository corpus via exact multi-step unrolled TTT cross-entropy BPTT + Adam
    /// (NO feature regression loss!).
    pub fn train_ttt(
        &mut self,
        target: &TargetModel,
        sequences: &[Vec<u32>],
        epochs: usize,
        unroll_steps: usize,
        lr: f32,
    ) -> Eagle3TrainingReport {
        let start_t = std::time::Instant::now();
        let unroll = unroll_steps.clamp(2, 4);

        // 1. Collect ON-POLICY teacher trajectories `[h_low; h_mid; h_high]` and target
        //    distributions by prefilling grounded corpus prompts and rolling `target` forward.
        let mut trajectories = Vec::with_capacity(sequences.len());
        let mut cache = TargetKvCache::new();
        for seq in sequences {
            if seq.len() <= unroll + 8 {
                continue;
            }
            cache.clear();
            let prompt_len = (seq.len() / 2).clamp(16, 28);
            let rollout_len = 28usize;

            let mut rollout_toks = Vec::with_capacity(rollout_len + 1);
            let mut feats = Vec::with_capacity(rollout_len + 1);
            let mut probs = Vec::with_capacity(rollout_len + 1);

            let last_prefill = target.prefill_prompt(&seq[..prompt_len], &mut cache, 0.0);
            let mut cur_tok = argmax_f32(&last_prefill.logits) as u32;
            rollout_toks.push(seq[prompt_len - 1]);
            feats.push(last_prefill.features);
            probs.push(softmax_temp(&last_prefill.logits, 1.0));

            for _ in 0..rollout_len {
                let out = target.forward_step_raw(cur_tok, &mut cache, 0.0);
                rollout_toks.push(cur_tok);
                feats.push(out.features);
                let p = softmax_temp(&out.logits, 1.0);
                cur_tok = argmax_f32(&p) as u32;
                probs.push(p);
            }
            trajectories.push((rollout_toks, feats, probs));
        }

        let (untrained_loss, untrained_top1_acc, untrained_step2_top1_acc) =
            self.eval_on_trajectories(&trajectories, target, unroll);

        // 2. Real multi-step unrolled BPTT + Adam updates through frozen target LM head
        let mut adam_step = 0u32;
        let step_weights = [1.0_f32, 0.80, 0.60, 0.45];

        for epoch in 0..epochs {
            let progress = (epoch as f32) / (epochs.max(1) as f32);
            let cur_lr =
                lr * (0.15 + 0.85 * (0.5 * (1.0 + (std::f32::consts::PI * progress).cos())));

            for (seq, feats, tgt_probs) in &trajectories {
                let max_anchor = seq.len().saturating_sub(unroll + 1);
                if max_anchor <= 1 {
                    continue;
                }

                self.zero_grads();
                let mut batch_steps = 0usize;

                for t in 1..max_anchor {
                    let cat_feat = feats[t].concat();
                    let mut raw_fuse = [0.0_f32; D_MODEL];
                    for r in 0..D_MODEL {
                        let row = &self.w_fc.w[r * D_FUSE_IN..(r + 1) * D_FUSE_IN];
                        let mut acc = 0.0_f32;
                        for c in 0..D_FUSE_IN {
                            acc += cat_feat[c] * row[c];
                        }
                        raw_fuse[r] = acc;
                    }
                    let sum_sq: f32 = raw_fuse.iter().map(|x| x * x).sum();
                    let inv_rms = 1.0 / ((sum_sq / (D_MODEL as f32)) + 1e-5).sqrt();
                    let mut g_fused = [0.0_f32; D_MODEL];
                    for d in 0..D_MODEL {
                        g_fused[d] = raw_fuse[d] * inv_rms;
                    }

                    // Forward unroll S steps feeding self-generated features `a_s` forward
                    let mut caches: Vec<StepCache> = Vec::with_capacity(unroll);
                    let mut ctx_hist: Vec<[f32; D_MODEL]> = Vec::with_capacity(unroll);
                    let mut cur_feat = g_fused;
                    let mut ctx_sum = g_fused;

                    for s in 0..unroll {
                        ctx_hist.push(ctx_sum);
                        let in_tok = argmax_f32(&tgt_probs[t + s]) as u32;
                        let (sc, _) = self.forward_draft_step(in_tok, &cur_feat, &ctx_sum, target);
                        let next_feat = sc.feat_out;
                        caches.push(sc);
                        let inv = 1.0 / ((s + 2) as f32);
                        for d in 0..D_MODEL {
                            ctx_sum[d] = ctx_sum[d] * (1.0 - inv) + next_feat[d] * inv;
                        }
                        cur_feat = next_feat;
                    }

                    // Backward BPTT from s = unroll - 1 down to 0
                    let mut d_next_feat = [0.0_f32; D_MODEL];
                    for s in (0..unroll).rev() {
                        let sc = &caches[s];
                        let tgt = &tgt_probs[t + 1 + s];
                        let w_s = step_weights[s];

                        // Backprop through shared frozen `target.lm_head` -> d_normed_out
                        let mut d_normed_out = [0.0_f32; D_MODEL];
                        for v in 0..VOCAB_SIZE {
                            let dl = w_s * (sc.probs[v] - tgt[v]);
                            let w_row = &target.lm_head[v * D_MODEL..(v + 1) * D_MODEL];
                            for d in 0..D_MODEL {
                                d_normed_out[d] += dl * w_row[d];
                            }
                        }

                        // Backprop through output RMSNorm -> add into d_feat_out
                        let dot_dn: f32 = (0..D_MODEL)
                            .map(|d| d_normed_out[d] * sc.normed_out[d])
                            .sum();
                        let mut d_feat_out = d_next_feat;
                        for d in 0..D_MODEL {
                            d_feat_out[d] += sc.inv_rms_out
                                * (d_normed_out[d]
                                    - sc.normed_out[d] * (dot_dn / (D_MODEL as f32)));
                        }

                        // Backprop through FFN residual: feat_out = post_attn + W_ff2 * ffn_act
                        let mut d_ffn_act = [0.0_f32; D_DRAFT_FF];
                        for r in 0..D_MODEL {
                            let df = d_feat_out[r];
                            let w_row = &self.w_ff2.w[r * D_DRAFT_FF..(r + 1) * D_DRAFT_FF];
                            let g_row = &mut self.w_ff2.g[r * D_DRAFT_FF..(r + 1) * D_DRAFT_FF];
                            for c in 0..D_DRAFT_FF {
                                g_row[c] += df * sc.ffn_act[c];
                                d_ffn_act[c] += df * w_row[c];
                            }
                        }

                        // Backprop through SiLU + W_ff1
                        let mut d_post_attn = d_feat_out;
                        for r in 0..D_DRAFT_FF {
                            let x = sc.ffn_pre[r];
                            let sig = 1.0 / (1.0 + (-x).exp());
                            let d_silu = sig * (1.0 + x * (1.0 - sig));
                            let d_pre = d_ffn_act[r] * d_silu;
                            self.b_ff1.g[r] += d_pre;
                            let w_row = &self.w_ff1.w[r * D_MODEL..(r + 1) * D_MODEL];
                            let g_row = &mut self.w_ff1.g[r * D_MODEL..(r + 1) * D_MODEL];
                            for c in 0..D_MODEL {
                                g_row[c] += d_pre * sc.post_attn[c];
                                d_post_attn[c] += d_pre * w_row[c];
                            }
                        }

                        // Backprop through ctx mixing: post_attn = u + W_ctx * ctx_sum
                        let ctx_in = &ctx_hist[s];
                        for r in 0..D_MODEL {
                            let dp = d_post_attn[r];
                            let g_row = &mut self.w_ctx.g[r * D_MODEL..(r + 1) * D_MODEL];
                            for c in 0..D_MODEL {
                                g_row[c] += dp * ctx_in[c];
                            }
                        }

                        // Backprop through W_in * [e(x); prev_feat] + 0.5 * prev_feat
                        let mut d_prev_feat = [0.0_f32; D_MODEL];
                        for d in 0..D_MODEL {
                            d_prev_feat[d] = 0.5 * d_post_attn[d];
                        }
                        for r in 0..D_MODEL {
                            let dp = d_post_attn[r];
                            let w_row = &self.w_in.w[r * D_COMBINE_IN..(r + 1) * D_COMBINE_IN];
                            let g_row = &mut self.w_in.g[r * D_COMBINE_IN..(r + 1) * D_COMBINE_IN];
                            for c in 0..D_COMBINE_IN {
                                g_row[c] += dp * sc.comb[c];
                            }
                            for c in 0..D_MODEL {
                                d_prev_feat[c] += dp * w_row[D_MODEL + c];
                            }
                        }

                        d_next_feat = d_prev_feat;
                    }

                    // Backprop from `d_next_feat` through input fusion RMSNorm into `W_fc`
                    let dot_dg: f32 = (0..D_MODEL).map(|d| d_next_feat[d] * g_fused[d]).sum();
                    for r in 0..D_MODEL {
                        let d_raw =
                            inv_rms * (d_next_feat[r] - g_fused[r] * (dot_dg / (D_MODEL as f32)));
                        let g_row = &mut self.w_fc.g[r * D_FUSE_IN..(r + 1) * D_FUSE_IN];
                        for c in 0..D_FUSE_IN {
                            g_row[c] += d_raw * cat_feat[c];
                        }
                    }

                    batch_steps += 1;
                }

                if batch_steps > 0 {
                    let inv_b = 1.0 / (batch_steps as f32);
                    self.scale_grads(inv_b);
                    adam_step += 1;
                    self.adam_step(cur_lr, adam_step);
                }
            }
        }

        let (trained_loss, trained_top1_acc, trained_step2_top1_acc) =
            self.eval_on_trajectories(&trajectories, target, unroll);

        Eagle3TrainingReport {
            epochs,
            unroll_steps: unroll,
            training_windows: trajectories.len(),
            untrained_loss,
            trained_loss,
            untrained_top1_acc,
            trained_top1_acc,
            untrained_step2_top1_acc,
            trained_step2_top1_acc,
            train_wall_ms: start_t.elapsed().as_secs_f64() * 1000.0,
        }
    }

    fn zero_grads(&mut self) {
        self.w_fc.zero_grad();
        self.w_in.zero_grad();
        self.w_ctx.zero_grad();
        self.w_ff1.zero_grad();
        self.b_ff1.zero_grad();
        self.w_ff2.zero_grad();
    }

    fn scale_grads(&mut self, scale: f32) {
        for p in [
            &mut self.w_fc,
            &mut self.w_in,
            &mut self.w_ctx,
            &mut self.w_ff1,
            &mut self.b_ff1,
            &mut self.w_ff2,
        ] {
            for g in &mut p.g {
                *g *= scale;
            }
        }
    }

    fn adam_step(&mut self, lr: f32, step_t: u32) {
        let wd = 8e-4_f32;
        self.w_fc.step(lr, step_t, wd);
        self.w_in.step(lr, step_t, wd);
        self.w_ctx.step(lr, step_t, wd);
        self.w_ff1.step(lr, step_t, wd);
        self.b_ff1.step(lr, step_t, 0.0);
        self.w_ff2.step(lr, step_t, wd);
    }

    /// Generate a linear chain of `depth` candidate tokens given `anchor_token` and
    /// `prev_tapped` features `[h_low; h_mid; h_high]`.
    pub fn draft_linear(
        &self,
        anchor_token: u32,
        prev_tapped: &TappedFeatures,
        target: &TargetModel,
        depth: usize,
    ) -> LinearDraft {
        if depth == 0 {
            return LinearDraft {
                tokens: Vec::new(),
                draft_probs: Vec::new(),
            };
        }

        let mut cur_feat = self.fuse_features(prev_tapped);
        let mut ctx_sum = cur_feat;
        let mut cur_tok = anchor_token;

        let mut tokens = Vec::with_capacity(depth);
        let mut draft_probs = Vec::with_capacity(depth);

        for s in 0..depth {
            let (sc, _) = self.forward_draft_step(cur_tok, &cur_feat, &ctx_sum, target);
            let next_tok = argmax_f32(&sc.probs) as u32;
            tokens.push(next_tok);
            draft_probs.push(sc.probs);

            let inv = 1.0 / ((s + 2) as f32);
            for d in 0..D_MODEL {
                ctx_sum[d] = ctx_sum[d] * (1.0 - inv) + sc.feat_out[d] * inv;
            }
            cur_feat = sc.feat_out;
            cur_tok = next_tok;
        }

        LinearDraft {
            tokens,
            draft_probs,
        }
    }

    /// Generate a multi-branch EAGLE-3 beam draft tree (`depth`, `fanout`, `max_nodes` including root)
    /// rooted at `anchor_token`.
    pub fn draft_tree(
        &self,
        anchor_token: u32,
        prev_tapped: &TappedFeatures,
        target: &TargetModel,
        depth: usize,
        fanout: usize,
        max_nodes: usize,
    ) -> DraftTree {
        let max_nodes = max_nodes.max(1);
        if depth == 0 || fanout == 0 || max_nodes == 1 {
            return DraftTree::root_only(anchor_token);
        }

        let fused = self.fuse_features(prev_tapped);
        let mut nodes = Vec::with_capacity(max_nodes);
        nodes.push(TreeNode {
            id: 0,
            parent: None,
            depth: 0,
            token: anchor_token,
            cumulative_log_prob: 0.0,
            draft_probs: Vec::new(),
            children: Vec::new(),
        });

        let mut frontier: Vec<(usize, [f32; D_MODEL], [f32; D_MODEL])> = vec![(0, fused, fused)];

        for d in 1..=depth {
            if nodes.len() >= max_nodes || frontier.is_empty() {
                break;
            }

            struct Cand {
                parent_id: usize,
                token: u32,
                cum_lp: f32,
                probs: Vec<f32>,
                next_feat: [f32; D_MODEL],
                next_ctx: [f32; D_MODEL],
            }

            let mut cands: Vec<Cand> = Vec::new();
            let inv = 1.0 / ((d + 1) as f32);

            for &(pid, ref feat, ref ctx_sum) in &frontier {
                let p_tok = nodes[pid].token;
                let p_lp = nodes[pid].cumulative_log_prob;
                let (sc, _) = self.forward_draft_step(p_tok, feat, ctx_sum, target);

                let top_k = top_k_indices(&sc.probs, fanout);
                let mut next_ctx = *ctx_sum;
                for dim in 0..D_MODEL {
                    next_ctx[dim] = next_ctx[dim] * (1.0 - inv) + sc.feat_out[dim] * inv;
                }

                for tok_idx in top_k {
                    let prob = sc.probs[tok_idx].max(1e-8);
                    cands.push(Cand {
                        parent_id: pid,
                        token: tok_idx as u32,
                        cum_lp: p_lp + prob.ln(),
                        probs: sc.probs.clone(),
                        next_feat: sc.feat_out,
                        next_ctx,
                    });
                }
            }

            cands.sort_by(|a, b| b.cum_lp.total_cmp(&a.cum_lp));
            let remaining = max_nodes - nodes.len();
            let level_budget = if d < depth {
                remaining.min(fanout.max(2) + 1).min(cands.len())
            } else {
                remaining.min(cands.len())
            };

            let mut next_frontier = Vec::new();
            for cand in cands.into_iter().take(level_budget) {
                let id = nodes.len();
                nodes.push(TreeNode {
                    id,
                    parent: Some(cand.parent_id),
                    depth: d,
                    token: cand.token,
                    cumulative_log_prob: cand.cum_lp,
                    draft_probs: cand.probs,
                    children: Vec::new(),
                });
                if next_frontier.len() < fanout.max(2) {
                    next_frontier.push((id, cand.next_feat, cand.next_ctx));
                }
            }
            frontier = next_frontier;
        }

        DraftTree::with_computed_mask(nodes)
    }

    /// Generate a stochastic multi-branch tree with exact conditional sibling distributions `q_j`.
    pub fn draft_tree_stochastic(
        &self,
        anchor_token: u32,
        prev_tapped: &TappedFeatures,
        target: &TargetModel,
        depth: usize,
        fanout: usize,
        rng: &mut SplitMix64,
    ) -> DraftTree {
        if depth == 0 || fanout == 0 {
            return DraftTree::root_only(anchor_token);
        }

        let fused = self.fuse_features(prev_tapped);
        let mut nodes = vec![TreeNode {
            id: 0,
            parent: None,
            depth: 0,
            token: anchor_token,
            cumulative_log_prob: 0.0,
            draft_probs: Vec::new(),
            children: Vec::new(),
        }];

        let mut frontier: Vec<(usize, [f32; D_MODEL], [f32; D_MODEL])> = vec![(0, fused, fused)];

        for d in 1..=depth {
            let inv = 1.0 / ((d + 1) as f32);
            let mut next_frontier = Vec::new();

            for &(pid, ref feat, ref ctx_sum) in &frontier {
                let (sc, _) = self.forward_draft_step(nodes[pid].token, feat, ctx_sum, target);
                let mut next_ctx = *ctx_sum;
                for dim in 0..D_MODEL {
                    next_ctx[dim] = next_ctx[dim] * (1.0 - inv) + sc.feat_out[dim] * inv;
                }

                let mut cur_q = sc.probs.clone();
                for _branch in 0..fanout {
                    let sum_q: f32 = cur_q.iter().sum();
                    if sum_q <= 1e-8 {
                        break;
                    }
                    let inv_q = 1.0 / sum_q;
                    for p in &mut cur_q {
                        *p *= inv_q;
                    }
                    let tok = rng.sample_categorical(&cur_q);
                    let id = nodes.len();
                    nodes.push(TreeNode {
                        id,
                        parent: Some(pid),
                        depth: d,
                        token: tok as u32,
                        cumulative_log_prob: nodes[pid].cumulative_log_prob
                            + cur_q[tok].max(1e-8).ln(),
                        draft_probs: cur_q.clone(),
                        children: Vec::new(),
                    });
                    if d < depth {
                        next_frontier.push((id, sc.feat_out, next_ctx));
                    }
                    cur_q[tok] = 0.0;
                }
            }
            frontier = next_frontier;
        }

        DraftTree::with_computed_mask(nodes)
    }
}

fn top_k_indices(probs: &[f32], k: usize) -> Vec<usize> {
    let mut idxs: Vec<usize> = (0..probs.len()).collect();
    idxs.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    idxs.truncate(k.min(probs.len()));
    idxs
}
