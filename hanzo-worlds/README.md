# Native world inference

`hanzo-worlds` supplies `WorldInference` and multi-tenant `WorldRegistry` for
native Hanzo Engine hosts of `POST /v1/worlds/{world}/inference`. It consumes the
published canonical `hanzo-active-inference` crate from `hanzoai/ml` for exact
Bayesian filtering, risk/ambiguity and pragmatic/epistemic EFE decomposition,
policy posteriors, and Dirichlet parameter learning. It does not run a language model.

## Security & authority boundaries

- **Host-owned model & admissibility**: The host authenticates tenant identity,
  resolves `(tenant_id, world_id)`, and supplies the immutable versioned generative
  model, preference distribution, policy habits, precision, and hard
  `AdmissibilityPolicy`. Callers cannot supply or override these fields
  (`#[serde(deny_unknown_fields)]` rejects any injected keys).
- **Strict three-way separation**: Every `Response` explicitly separates:
  1. `inference` — filtered Bayesian posterior and exact EFE policy posterior;
  2. `recommendation` — strictly advisory policy ranking (`advisory_only: true`);
  3. `execution_authorized` — invariantly `false`.
- **Multi-tenant isolation**: `WorldRegistry` keys worlds by `(tenant_id, world_id)`.
  Missing tenant identity fails closed (`Unauthorized` / 403); cross-tenant lookups
  return `NotFound` (404) without revealing existence.
- **No actuator interface**: Real-world execution requires independent verified
  execution policies and domain qualification outside this library.

## Validation & benchmarks

```sh
cargo test -p hanzo-worlds
cargo bench -p hanzo-worlds
cargo clippy -p hanzo-worlds --all-targets -- -D warnings
```
