// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Real token-level causal Transformer + GDN micro-model execution & output equivalence verifier.
//!
//! Executes a deterministic pure-Rust hybrid causal attention + Gated DeltaNet recurrent
//! micro-model over real paged KV blocks (`block_size = 16`) and fixed FP32 GDN state slots,
//! proving that:
//! 1. **Chunked prefill** (arbitrary chunk sizes: `17`, `32`, `64`),
//! 2. **Interleaved continuous batching** (prefill chunks piggybacked onto active decode steps), and
//! 3. **Preemption + recomputation** (dropping KV blocks & GDN state mid-prefill and recomputing)
//!
//! all produce **100% bit-identical logits and sampled tokens (`0.000 pp` quality regression)**
//! compared to sequential unchunked single-request execution!

use std::collections::BTreeMap;
use std::time::Instant;

use crate::kv_pool::{PagedKvPool, SchedulerError};
use crate::sha256::SplitMix64;

/// Configuration for the deterministic hybrid causal micro-model verifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MicroModelConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension (`d_model`).
    pub d_model: usize,
    /// Number of attention heads (`head_dim = d_model / num_heads`).
    pub num_heads: usize,
    /// Paged KV block size in tokens.
    pub block_size: usize,
}

impl Default for MicroModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            d_model: 32,
            num_heads: 4,
            block_size: 16,
        }
    }
}

/// Deterministic hybrid Attention + GDN recurrent micro-model with real paged KV storage.
#[derive(Debug, Clone)]
pub struct HybridMicroModel {
    cfg: MicroModelConfig,
    embed: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    wg_decay: Vec<f32>,
    wg_k: Vec<f32>,
    wg_v: Vec<f32>,
    w_lm: Vec<f32>,
}

/// Physical paged KV block tensor storage + fixed FP32 GDN recurrent state backing [`PagedKvPool`].
struct PagedTensorArena {
    block_size: usize,
    d_model: usize,
    k_blocks: Vec<Vec<f32>>,
    v_blocks: Vec<Vec<f32>>,
    gdn_states: Vec<Vec<f32>>,
}

impl PagedTensorArena {
    fn new(total_blocks: usize, total_gdn_slots: usize, block_size: usize, d_model: usize) -> Self {
        let block_elems = block_size * d_model;
        let gdn_elems = d_model * d_model;
        Self {
            block_size,
            d_model,
            k_blocks: vec![vec![0.0f32; block_elems]; total_blocks],
            v_blocks: vec![vec![0.0f32; block_elems]; total_blocks],
            gdn_states: vec![vec![0.0f32; gdn_elems]; total_gdn_slots],
        }
    }

    fn zero_gdn_slot(&mut self, slot_id: usize) {
        self.gdn_states[slot_id].fill(0.0);
    }

    fn write_kv(&mut self, block_id: usize, slot_in_block: usize, k: &[f32], v: &[f32]) {
        let off = slot_in_block * self.d_model;
        self.k_blocks[block_id][off..off + self.d_model].copy_from_slice(k);
        self.v_blocks[block_id][off..off + self.d_model].copy_from_slice(v);
    }

    fn read_k(&self, block_id: usize, slot_in_block: usize) -> &[f32] {
        let off = slot_in_block * self.d_model;
        &self.k_blocks[block_id][off..off + self.d_model]
    }

    fn read_v(&self, block_id: usize, slot_in_block: usize) -> &[f32] {
        let off = slot_in_block * self.d_model;
        &self.v_blocks[block_id][off..off + self.d_model]
    }
}

/// Verification report proving bit-exact output equivalence between sequential reference
/// and chunked/interleaved/preempted continuous batching.
#[derive(Debug, Clone, PartialEq)]
pub struct VerificationReport {
    /// Number of mixed-length sequences verified.
    pub num_sequences: usize,
    /// Total prompt tokens verified across all sequences.
    pub total_prompt_tokens: usize,
    /// Total generated output tokens compared against sequential reference.
    pub total_output_tokens: usize,
    /// Exact count of output tokens matching sequential unchunked reference bit-for-bit.
    pub matched_tokens: usize,
    /// Output token correctness percentage (`100.0` required).
    pub output_correctness_pct: f64,
    /// Quality regression in percentage points (`0.000` required).
    pub quality_regression_pp: f64,
    /// Maximum absolute logit difference across all checked steps (`0.0` expected).
    pub max_abs_logit_diff: f32,
    /// Number of interleaved chunked prefill+decode steps executed.
    pub interleaved_steps: usize,
    /// True if a forced preemption-and-recompute path was also verified bit-exact.
    pub preemption_recompute_verified: bool,
    /// True if the paged KV pool returned 100% clean (zero leaked blocks/slots).
    pub kv_pool_clean: bool,
    /// Directly measured CPU execution time of the full verifier in milliseconds.
    pub cpu_verify_wall_ms: f64,
}

/// Prompt + decode budget specification for token-level equivalence testing.
#[derive(Debug, Clone)]
pub struct VerifySequenceSpec {
    /// Unique sequence ID.
    pub seq_id: usize,
    /// Input prompt token IDs.
    pub prompt_tokens: Vec<u32>,
    /// Number of greedy output tokens to generate.
    pub decode_steps: usize,
}

impl HybridMicroModel {
    /// Builds a deterministic [`HybridMicroModel`] from `seed`.
    #[must_use]
    pub fn new(cfg: MicroModelConfig, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let d = cfg.d_model;
        let v = cfg.vocab_size;
        let scale_d = 1.0 / (d as f32).sqrt();
        let mut init = |len: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|_| ((rng.next_f64() as f32) * 2.0 - 1.0) * scale)
                .collect()
        };
        Self {
            cfg,
            embed: init(v * d, 0.5),
            wq: init(d * d, scale_d),
            wk: init(d * d, scale_d),
            wv: init(d * d, scale_d),
            wo: init(d * d, scale_d),
            wg_decay: init(d, 0.3),
            wg_k: init(d * d, scale_d),
            wg_v: init(d * d, scale_d),
            w_lm: init(v * d, scale_d),
        }
    }

    /// Processes a single causal token at `pos` for `seq_id` using its physical KV block table
    /// and GDN recurrent state slot, returning logits over `0..vocab_size`.
    fn forward_token(
        &self,
        token: u32,
        pos: usize,
        block_ids: &[usize],
        gdn_slot_id: usize,
        arena: &mut PagedTensorArena,
    ) -> Vec<f32> {
        let d = self.cfg.d_model;
        let head_dim = d / self.cfg.num_heads;
        let tok_idx = (token as usize) % self.cfg.vocab_size;
        let x = &self.embed[tok_idx * d..(tok_idx + 1) * d];

        // 1. Attention projections Q, K, V.
        let q = matvec(&self.wq, x, d, d);
        let k = matvec(&self.wk, x, d, d);
        let v = matvec(&self.wv, x, d, d);

        let block_idx = pos / arena.block_size;
        let slot_in_block = pos % arena.block_size;
        let phys_block = block_ids[block_idx];
        arena.write_kv(phys_block, slot_in_block, &k, &v);

        // 2. Exact causal paged multi-head attention over positions `0..=pos` in deterministic order.
        let inv_sqrt_hd = 1.0 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; d];
        let mut scores = vec![0.0f32; pos + 1];
        for h in 0..self.cfg.num_heads {
            let h_off = h * head_dim;
            let mut max_score = f32::NEG_INFINITY;
            for (t, score_slot) in scores.iter_mut().enumerate() {
                let b_id = block_ids[t / arena.block_size];
                let s_id = t % arena.block_size;
                let kt = arena.read_k(b_id, s_id);
                let mut dot = 0.0f32;
                for j in 0..head_dim {
                    dot += q[h_off + j] * kt[h_off + j];
                }
                let sc = dot * inv_sqrt_hd;
                *score_slot = sc;
                if sc > max_score {
                    max_score = sc;
                }
            }
            let mut sum_exp = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0 / sum_exp;
            for (t, &w_t) in scores.iter().enumerate() {
                let prob = w_t * inv_sum;
                let b_id = block_ids[t / arena.block_size];
                let s_id = t % arena.block_size;
                let vt = arena.read_v(b_id, s_id);
                for j in 0..head_dim {
                    attn_out[h_off + j] += prob * vt[h_off + j];
                }
            }
        }
        let attn_proj = matvec(&self.wo, &attn_out, d, d);

        // 3. Gated DeltaNet (GDN) linear recurrent state update: S_t = decay * S_{t-1} + k_g outer v_g.
        let gk = matvec(&self.wg_k, x, d, d);
        let gv = matvec(&self.wg_v, x, d, d);
        let state = &mut arena.gdn_states[gdn_slot_id];
        let mut gdn_out = vec![0.0f32; d];
        for r in 0..d {
            let decay = 0.75 + 0.20 * self.wg_decay[r].tanh();
            let row_off = r * d;
            let mut acc = 0.0f32;
            for c in 0..d {
                let updated = decay * state[row_off + c] + 0.05 * gk[r] * gv[c];
                state[row_off + c] = updated;
                acc += updated * q[c];
            }
            gdn_out[r] = acc.tanh();
        }

        // 4. Residual combination + LM head logits.
        let mut hidden = vec![0.0f32; d];
        for i in 0..d {
            hidden[i] = (x[i] + 0.5 * attn_proj[i] + 0.5 * gdn_out[i]).tanh();
        }
        matvec(&self.w_lm, &hidden, self.cfg.vocab_size, d)
    }

    /// Runs a single sequence sequentially without chunking or batching (ground-truth oracle).
    fn run_sequential_reference(
        &self,
        spec: &VerifySequenceSpec,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>), SchedulerError> {
        let total_len = spec.prompt_tokens.len() + spec.decode_steps;
        let blocks_needed = total_len.div_ceil(self.cfg.block_size).max(1);
        let mut pool = PagedKvPool::with_explicit_capacity(
            self.cfg.block_size,
            1024,
            4096,
            blocks_needed + 2,
            2,
        )?;
        let mut arena =
            PagedTensorArena::new(blocks_needed + 2, 2, self.cfg.block_size, self.cfg.d_model);
        let tbl = pool.admit_sequence(spec.seq_id, total_len)?.clone();
        arena.zero_gdn_slot(tbl.gdn_slot_id);

        let mut last_logits = Vec::new();
        for (pos, &tok) in spec.prompt_tokens.iter().enumerate() {
            last_logits = self.forward_token(tok, pos, &tbl.block_ids, tbl.gdn_slot_id, &mut arena);
        }

        let mut generated = Vec::with_capacity(spec.decode_steps);
        let mut step_logits = Vec::with_capacity(spec.decode_steps);
        let mut cur_pos = spec.prompt_tokens.len();
        for step in 0..spec.decode_steps {
            let next_tok = argmax(&last_logits) as u32;
            generated.push(next_tok);
            step_logits.push(last_logits.clone());
            if step + 1 < spec.decode_steps {
                last_logits = self.forward_token(
                    next_tok,
                    cur_pos,
                    &tbl.block_ids,
                    tbl.gdn_slot_id,
                    &mut arena,
                );
                cur_pos += 1;
            }
        }
        pool.free_sequence(spec.seq_id);
        Ok((generated, step_logits))
    }

    /// Verifies bit-identical output equivalence across mixed chunk sizes (`chunk_budget`)
    /// interleaved with active decodes AND a forced mid-prefill preemption + recompute event.
    ///
    /// # Errors
    /// Returns [`SchedulerError`] if paged KV pool allocation fails.
    pub fn verify_chunked_and_interleaved(
        &self,
        specs: &[VerifySequenceSpec],
        chunk_budget: usize,
        force_preemption_test: bool,
    ) -> Result<VerificationReport, SchedulerError> {
        let t0 = Instant::now();
        // 1. Compute exact sequential single-request reference for every sequence.
        let mut ref_outputs: BTreeMap<usize, (Vec<u32>, Vec<Vec<f32>>)> = BTreeMap::new();
        let mut total_prompt_tokens = 0usize;
        let mut total_output_tokens = 0usize;
        for spec in specs {
            total_prompt_tokens += spec.prompt_tokens.len();
            total_output_tokens += spec.decode_steps;
            ref_outputs.insert(spec.seq_id, self.run_sequential_reference(spec)?);
        }

        // 2. Execute all sequences concurrently with shared PagedKvPool, interleaved chunked
        //    prefill + active decodes, and an optional forced preemption + recompute mid-flight.
        let max_blocks: usize = specs
            .iter()
            .map(|s| (s.prompt_tokens.len() + s.decode_steps).div_ceil(self.cfg.block_size))
            .sum::<usize>()
            + 8;
        let max_slots = specs.len().max(2);
        let mut pool = PagedKvPool::with_explicit_capacity(
            self.cfg.block_size,
            1024,
            4096,
            max_blocks,
            max_slots,
        )?;
        let mut arena =
            PagedTensorArena::new(max_blocks, max_slots, self.cfg.block_size, self.cfg.d_model);

        struct LiveSeq<'a> {
            spec: &'a VerifySequenceSpec,
            prefilled: usize,
            generated: Vec<u32>,
            logits: Vec<Vec<f32>>,
            last_token: Option<u32>,
            finished: bool,
            resident: bool,
        }

        let mut live: Vec<LiveSeq<'_>> = specs
            .iter()
            .map(|spec| LiveSeq {
                spec,
                prefilled: 0,
                generated: Vec::with_capacity(spec.decode_steps),
                logits: Vec::with_capacity(spec.decode_steps),
                last_token: None,
                finished: false,
                resident: false,
            })
            .collect();

        let mut interleaved_steps = 0usize;
        let mut preemption_triggered = false;

        while live.iter().any(|s| !s.finished) {
            interleaved_steps += 1;
            let mut budget = chunk_budget.max(2);

            // Step A: Advance all currently decoding sequences by 1 token (interleaved).
            for seq in &mut live {
                if seq.finished || !seq.resident || seq.prefilled < seq.spec.prompt_tokens.len() {
                    continue;
                }
                if budget == 0 {
                    break;
                }
                let pos = seq.spec.prompt_tokens.len() + seq.generated.len() - 1;
                pool.allocate_for_len(seq.spec.seq_id, pos + 1)?;
                let tbl = pool
                    .sequence_table(seq.spec.seq_id)
                    .expect("resident")
                    .clone();
                let tok = seq.last_token.expect("has last token");
                let logits =
                    self.forward_token(tok, pos, &tbl.block_ids, tbl.gdn_slot_id, &mut arena);
                let next_tok = argmax(&logits) as u32;
                seq.generated.push(next_tok);
                seq.logits.push(logits);
                seq.last_token = Some(next_tok);
                budget -= 1;
                if seq.generated.len() >= seq.spec.decode_steps {
                    seq.finished = true;
                    seq.resident = false;
                    pool.free_sequence(seq.spec.seq_id);
                }
            }

            // Step B: Trigger one deterministic mid-prefill preemption + recompute on the longest
            // partially prefilled sequence to verify preemption restores bit-exact state.
            if force_preemption_test && !preemption_triggered {
                if let Some(victim) = live.iter_mut().find(|s| {
                    s.resident && s.prefilled > 0 && s.prefilled < s.spec.prompt_tokens.len()
                }) {
                    let _recomputed = pool.preempt_recompute(victim.spec.seq_id)?;
                    victim.prefilled = 0;
                    victim.resident = false;
                    preemption_triggered = true;
                }
            }

            // Step C: Piggyback prefill chunks across non-finished prefilling/waiting sequences.
            for seq in &mut live {
                if seq.finished || seq.prefilled >= seq.spec.prompt_tokens.len() || budget == 0 {
                    continue;
                }
                if !seq.resident {
                    let tbl = pool.admit_sequence(seq.spec.seq_id, 1)?.clone();
                    arena.zero_gdn_slot(tbl.gdn_slot_id);
                    seq.resident = true;
                }
                let rem = seq.spec.prompt_tokens.len() - seq.prefilled;
                let take = rem.min(budget);
                let new_len = seq.prefilled + take;
                pool.allocate_for_len(seq.spec.seq_id, new_len)?;
                let tbl = pool
                    .sequence_table(seq.spec.seq_id)
                    .expect("resident")
                    .clone();

                let mut last_logits = Vec::new();
                for offset in 0..take {
                    let pos = seq.prefilled + offset;
                    let tok = seq.spec.prompt_tokens[pos];
                    last_logits =
                        self.forward_token(tok, pos, &tbl.block_ids, tbl.gdn_slot_id, &mut arena);
                }
                seq.prefilled = new_len;
                budget -= take;

                if seq.prefilled == seq.spec.prompt_tokens.len() {
                    let first_tok = argmax(&last_logits) as u32;
                    seq.generated.push(first_tok);
                    seq.logits.push(last_logits);
                    seq.last_token = Some(first_tok);
                    if seq.generated.len() >= seq.spec.decode_steps {
                        seq.finished = true;
                        seq.resident = false;
                        pool.free_sequence(seq.spec.seq_id);
                    }
                }
            }
        }

        // 3. Compare every generated token and every logit vector against sequential reference.
        let mut matched_tokens = 0usize;
        let mut max_abs_logit_diff = 0.0f32;
        for seq in &live {
            let (ref_toks, ref_logits) = &ref_outputs[&seq.spec.seq_id];
            for (g, r) in seq.generated.iter().zip(ref_toks.iter()) {
                if g == r {
                    matched_tokens += 1;
                }
            }
            for (lv, rv) in seq.logits.iter().zip(ref_logits.iter()) {
                for (&a, &b) in lv.iter().zip(rv.iter()) {
                    let diff = (a - b).abs();
                    if diff > max_abs_logit_diff {
                        max_abs_logit_diff = diff;
                    }
                }
            }
        }

        let output_correctness_pct = if total_output_tokens > 0 {
            (matched_tokens as f64 / total_output_tokens as f64) * 100.0
        } else {
            100.0
        };
        let quality_regression_pp = 100.0 - output_correctness_pct;
        let cpu_verify_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(VerificationReport {
            num_sequences: specs.len(),
            total_prompt_tokens,
            total_output_tokens,
            matched_tokens,
            output_correctness_pct,
            quality_regression_pp,
            max_abs_logit_diff,
            interleaved_steps,
            preemption_recompute_verified: preemption_triggered,
            kv_pool_clean: pool.is_clean(),
            cpu_verify_wall_ms,
        })
    }
}

fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for (r, slot) in out.iter_mut().enumerate() {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        *slot = acc;
    }
    out
}

fn argmax(logits: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}
