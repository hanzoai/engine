# Proof-of-Inference: forward-pass commitment design

Status: **design only — not yet built.** The Freivalds verifier core (`hanzo-engine/src/poi.rs`)
and the determinism prerequisites (lowest-id argmax in `sampler.rs`, canonical MoE expert
top-k in `ops.rs`) are built and unit-tested on `feat/poi-determinism-prereqs`. This document
specifies the next slice — the activation-trace commitment and the forward-pass integration —
that turns the verifier into an end-to-end proof a real model emits. No end-to-end proof is
emitted until that slice lands and a real zen model produces a transcript.

The wire constants (domains, field, encodings) MUST match the chain side byte-for-byte
(`~/work/lux/precompile/aivmbridge` style: keccak-256, raw-utf8 domain tags, no length prefix).

---

## 1. ActivationTraceMMR — the streaming commitment

A forward pass over a 70B model at 256 tokens is terabytes of activations; we never retain
them. A **Merkle Mountain Range** accumulates per-`(layer, token, tap)` leaves in execution
order and keeps only `O(log N)` peak hashes (~768 bytes of frontier). Activations are hashed
as produced, then dropped.

- **Leaf** (the tensor the kernel produced, in its native dtype):
  `leaf = keccak(DOMAIN_POI_ACT || u32be(layer) || u32be(token) || u8(tap) || u8(dtype) || shape_canon || tensor_bytes_row_major)`
  where `tap ∈ {matmul_in, matmul_out, block_out}` — exactly the operands a Freivalds check
  or a non-matmul recompute will open, no more. `shape_canon = u8(rank) || rank*u32be(dim)`.
- **Node**: `node = keccak(DOMAIN_POI_NODE || left || right)` — leaf/node domain separation
  (second-preimage hardening: 32-byte leaf preimage vs 64-byte node preimage).
- **Folding**: RFC-6962 lone-node **promotion** on odd levels (NOT Bitcoin duplicate-last —
  that is CVE-2012-2459 malleable; the existing `chains/aivm/quorum_merkle.go` uses the
  malleable form and should be migrated separately).
- **Close**: `activationTraceRoot = keccak(DOMAIN_POI || u64be(n_leaves) || mmr_root)`
  (the leaf count is bound so a subtree cannot be reinterpreted as the whole tree).
- **Opening**: `O(log N)` inclusion proof. The prover re-derives a challenged leaf by
  re-running the forward pass deterministically to that `(layer, token)` (the optimistic
  bargain: cheap in the happy path, a partial re-execution only under challenge); it never
  persists the full activation set.

Domains: `DOMAIN_POI = "lux/aivmbridge/poi/v1"`, `DOMAIN_POI_ACT = "lux/aivmbridge/poi/act/v1"`,
`DOMAIN_POI_NODE = "lux/aivmbridge/poi/node/v1"`, `DOMAIN_POI_WEIGHT = "lux/aivmbridge/poi/weight/v1"`.

## 2. Forward-pass hook — zero overhead when off

Thread `Option<&mut ProofTranscript>` through the forward context (`pipeline/normal.rs`
`ModelForwardContext`), NOT a boolean checked in hot loops. When `None`, each commit site is a
single not-taken branch: no hashing, no allocation, no tensor copy. The per-matmul hook lives
in the caller (`layers.rs` Linear/attention forward) around `QuantMethod::forward_raw`, which
holds the layer identity:

```text
let c = self.qmethod.forward_raw(a)?;
if let Some(t) = ctx.poi.as_mut() {
    t.commit_activation(self.layer_idx, tok, Tap::MatmulIn(self.role),  &a.as_quant_view());
    t.commit_activation(self.layer_idx, tok, Tap::MatmulOut(self.role), &c.as_quant_view());
}
```

`as_quant_view()` borrows the int8 activations + the i32 accumulator + scales (no copy). The
weight `B` is committed once at load into `modelWeightsRoot`, never re-hashed per token.

Proof-bearing runs set `set_moe_proof_deterministic(true)` (canonical routing) and pin
temp-0 greedy decode (lowest-id argmax) — the determinism contract.

## 3. transcriptRoot — the binding

`modelID := modelWeightsRoot` is **measured**, not a label: keccak Merkle over the quantized
weight bytes the engine actually loaded (`QuantizedSerde::serialize()` per layer, with the
quant-type discriminant in each leaf — re-quantizing to a different scheme gives a different
root). Then:

```text
transcriptRoot = keccak(
    DOMAIN_POI
 || modelID(32)             // measured weights root  (= chain modelSpecHash)
 || inputCommitment(32)     // keccak(canonical token ids)        — promptHash
 || quantizationSpec(32)    // keccak(quant scheme + per-tensor dtype + group sizes)
 || kernelVersion(32)       // keccak(kernel build id + reduction mode + sampler/seed)  — runtimeMeasurement
 || activationTraceRoot(32) // the MMR close
 || outputCommitment(32)    // keccak(canonical output token ids / mel frames)  — outputHash
)
```

This feeds the on-chain `reportData` (see `lux/dao/contracts/.../ComputeProofLib.sol`): the
binding chains challenge -> model -> input -> output, so an attestation cannot be replayed,
spliced across tasks, or pointed at a different model/input/output. `quantizationSpec` +
`kernelVersion` make the decoding regime part of identity (a genuine model at temp=2.0, or a
different reduction order, is rejected).

## 4. Challenge -> open one layer

The verifier derives `(layer_ell, r)` from an on-chain beacon AFTER the prover committed
(`seed = keccak(DOMAIN_POI || "freivalds" || openBlockHash || challengeId || activationTraceRoot || u32be(layer_ell))`,
`layer_ell` itself beacon-derived). The prover opens that layer's `A` (matmul_in), `B`
(weight ref + path under `modelWeightsRoot`), and `C` (matmul_out) with MMR inclusion proofs.
Windowed Freivalds (a random token sub-range) keeps the opened-operand payload small.

## 5. Freivalds verifies the opened layer

The verifier (`poi.rs::freivalds_verify_multi`, mirrored on-chain in a future
`chains/aivm/compute_proof.go`) recomputes the leaf hashes, checks the MMR inclusions against
the committed `activationTraceRoot`/`modelWeightsRoot`, re-derives `r`, and checks
`A*(B*r) == C*r` over F_p (p = 2^61-1) on the exact int8 accumulator — `O(tk+kn+tn+log N)`,
~`min(t,k,n)`x cheaper than recomputing the layer. Mismatch -> slash. Sparse cheats are pinned
by interactive **bisection** to the first divergent layer (`O(log L)` rounds), so soundness is
independent of how many layers are sampled.

## 6. Minimum tests for this slice

```text
TestForwardPassEmitsTranscriptRoot          forward pass with poi=Some emits a non-zero root
TestTranscriptRootBindsOutput               root changes iff outputCommitment changes
TestTranscriptRootChangesOnActivationMutation  flip one committed activation -> root changes
TestChallengeOpensCommittedLayer            beacon-derived layer; opening verifies against root
TestFreivaldsVerifiesOpenedLayer            honest opened layer passes (uses poi.rs core)
TestBadLayerOpeningRejected                 opening with a wrong operand fails the MMR/Freivalds
TestTranscriptCannotBeReusedAcrossTaskID    challenge binds taskId -> a transcript is task-scoped
TestQuantizationSpecBoundIntoTranscript     same weights, different quant scheme -> different root
```

End-to-end acceptance: run an **honest int8 zen MoE inference** (e.g. `quantized_qwen3_5_moe`)
end to end, emit the transcript, challenge a layer, verify it passes; then a faked / cheap-model
/ mis-routed run is **caught**. That green test on a real zen model is the first true "live".

## 7. Architecture extensions (beyond dense matmuls)

Two extensions cover the whole zen zoo; per-modality I/O commitment + deterministic
pre/post-processing covers the rest.

- **MoE routing proof** (Qwen3.5-MoE, GLM-MoE, DeepSeekV3, ...): the transcript additionally
  commits, per token, the router logits and the selected expert indices; the verifier
  recomputes the router and checks the **canonical** top-k (the `deterministic_topk_indices`
  selection) chose the same experts, then Freivalds-checks the selected experts' GEMMs only
  (sparse). Determinism prerequisite already landed.
- **Sequential-chaining proof** (Gated DeltaNet / Mamba / MiniMax linear-recurrent AND
  diffusion multi-step): per-step Freivalds plus binding consecutive states/latents in the
  commitment (step t's committed output is step t+1's committed input), so an intermediate
  cannot be swapped. One primitive, two payoffs (recurrent models + diffusion).
- **Multimodal**: vision ViT and audio conformer towers are GEMMs -> Freivalds; commit the
  preprocessed input (pixel tensor / audio features) by hash and pin preprocessing
  (resize/normalize; FFT->mel is commit-and-recompute, deterministic signal processing).
- **Floating point**: not via Freivalds-with-tolerance (unsound + false-rejecting). Quantized
  -> exact Freivalds; fp16/bf16/fp8 -> deterministic-fp (pin the reduction order) or TEE
  attestation. See `~/work/zoo/proofs/proof-of-inference-soundness.tex`.
