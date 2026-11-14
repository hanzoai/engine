# Native world inference

`hanzo-worlds` supplies `WorldInference::infer` for native engine hosts of
`/v1/worlds/{world}/inference`. It calls the canonical `hanzo-active-inference`
crate in hanzoai/ml: Bayesian state inference, risk/ambiguity and information-gain
EFE, policy priors/precision and Dirichlet learning. It does not run a language model.

A host owns authentication, tenant/world resolution, versioned generative models,
preference distributions, permitted policy masks and time/resource budgets. These are
not user-overridable request fields. A returned recommendation cannot authorize action.
Drones and business operations require separate verified execution policies and real
domain qualification; this package has no actuator or transaction interface.

This is an additive native library. It is not mounted in the production HTTP service.
The integration owner must wire its authenticated world store before mounting routes.

Release order: publish `hanzo-active-inference 0.1.0` from hanzoai/ml, then test and
publish this crate. For coordinated source review, test with a command-line local
Cargo patch (not a committed dependency override):

```sh
cargo test -p hanzo-worlds --config 'patch.crates-io.hanzo-active-inference.path="/absolute/path/to/ml/hanzo-active-inference"'
```
