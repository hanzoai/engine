# Hanzo Engine: phase-aware resonance memory (experimental)

Status: native similarity kernel available as `hanzo_engine::resonance`. This
does **not** enable persistent memory, an HTTP endpoint, ANN indexing, or automatic
negation extraction. Existing embedding and search behavior is unchanged.

## Why

Dense embeddings often rank statements sharing vocabulary close together,
even if their factual polarity differs. A phase-aware **reranker** can offer
another scoring channel when an ingestion process provides verified semantic
operator metadata. We retain dense retrieval and lexical search for recall.

The math follows the public equation in A. Listopad, [Wave-Based Semantic Memory
with Resonance-Based Retrieval](https://arxiv.org/abs/2509.09691)
(arXiv:2509.09691v1). This module is a fresh implementation; it does not copy
ResonanceDB source code. The upstream project's code is source-available under
non-permissive commercial conditions; contact its author before considering a
separate upstream-code integration.

## Engine API

- `WavePattern::from_embedding(&[f32])` maps ordinary real embeddings into
  complex features with zero imaginary part and no extra semantic assumptions.
- `WavePattern::from_polar(&amplitudes, &phases)` takes explicit phase channels,
  and `from_cartesian` accepts precomputed complex features.
- `PhaseOperator::{Identity,Negate,Rotate(radians)}` applies **explicit** labels.
  Negate shifts the entire pattern by pi; it is not a general-purpose natural
  language negation parser.
- `WavePattern::resonance(&other)` returns normalized interference [0,1] and
  accounts for energy imbalance.
- `rerank_candidates(query, candidates, resonance_weight, top_k)` fuses
  interference with baseline dense cosine over a retrieved shortlist.
  Dense cosine is provided by callers for the original, untransformed
  embeddings; the default production search path remains unaffected.

Equal-norm, purely real wave patterns satisfy
`resonance(x,y) = (1 + cosine(x,y)) / 2`.
There is no semantic improvement from wrapping vectors in this representation
unless phase encodes **additional** information.

## Recommended production path

1. Introduce versioned atomic memory records: content, entity/relation/arguments,
   scope, asserted/negated status, epistemic stance, valid_from/valid_until,
   observed_at, source, tenant, embedding model, representation version.
2. Extract polarity only with grounded spans, strict negation scope, and a
   confidence gate. Preserve unknown/ambiguous labels instead of guessing.
   Negation of a proposition is not the same as multiplying every feature of
   an arbitrary paragraph by -1.
3. Continue using the existing index or BM25 + embedding ranker as the first
   stage. Pass 50-200 candidates to the phase-aware reranker for evaluation.
   Apply tenant, permissions, version and time-scope filters *before* scoring.
4. Add controlled access through an explicit experimental flag; implement
   serializable/versioned records and WAL or authoritative-store persistence
   outside this kernel. Do not confuse Engine's disk KV-token cache with
   long-term semantic memory.
5. Keep contradictions as separate time-stamped claims and resolve by
   provenance, recency and scope; interference is not itself a truth oracle.
6. Integrate with Kai/Enso for agent memory, tool routing, contradiction checks
   and fact-vs-hypothesis filtering only after passing the gates below.

## Retrieval path

    Query text + operator metadata
             |
       Existing embedder
             |
    Dense/BM25 top-K shortlist
             |
    Scope/ACL/time validity filters
             |
    Native WavePattern rerank
             |
    Optional cross-encoder / Kai decision
             |
    Cited memory snippets for agent context

## Minimum evaluation gates

- **Correctness:** self-match ~=1; anti-phase ~=0; orthogonal equal-energy
  real vectors ~=0.5; known unequal-energy penalties; dimension, empty,
  NaN/Infinity validation; shared global phase rotation invariance.
- **Negation:** real-world 500+ minimal-pair queries, including nested negation,
  quoted claims, uncertain statements and negation with different scopes.
  Track nDCG@10, Recall@20/100, contradiction false-positive rate.
- **Baselines:** existing BM25, embedding cosine, cross-encoder rerank,
  explicit metadata filter, phase-only, and hybrid. Ablate phase extraction
  to prove gains are not just a consequence of operator labels.
- **Workload:** 10K, 100K, 1M records, CPU and optionally GPU, dimension
  384/768/1024. Report p50/p95/p99, RSS and indexed vs scan-based throughput,
  along with ingestion/update costs; benchmark the same candidate set.
- **Safety:** no private memory leakage across tenant/ACL boundaries,
  unknown polarity cannot silently exclude relevant evidence, rollback flag,
  and deterministic replay including encoder/phase versions.

The cited paper's scalar Java baseline reports approximately 135 ms average
top-1 retrieval for 500K 1024-dimensional patterns on a 12-core CPU; it
does **not** show universal single-digit millisecond retrieval on millions
of patterns. Full scans are O(Nd), so scale requires indexing or shortlists.

## Release notes

- v0: pure Rust public kernel + hybrid shortlist reranker + unit tests.
- Next: add a phase-aware record schema, controlled memory APIs, persisted
  index adapter, observability, real contrastive evaluations and an A/B flag.
- **No** upstream ResonanceDB runtime or licensed code has been imported.
