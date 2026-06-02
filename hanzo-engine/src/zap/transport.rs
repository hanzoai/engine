//! ZAP transport wiring (capnp-rpc) -- the drop-in step.
//!
//! This module is intentionally a documentation/stub layer: it carries no
//! `capnp`/`capnp-rpc`/`capnpc` dependency so `cargo check -p hanzo-engine`
//! stays green on hosts without the `capnp` schema compiler installed. The
//! message + adapter surface (`super::{message,server,client}`) is fully
//! compiled and testable in-process today; flipping on the wire is mechanical.
//!
//! # Enabling the real wire
//!
//! 1. Add to `hanzo-engine/Cargo.toml`, gated by a `zap-capnp` sub-feature:
//!    ```toml
//!    [dependencies]
//!    capnp = { version = "0.20", optional = true }
//!    capnp-rpc = { version = "0.20", optional = true }
//!    [build-dependencies]
//!    capnpc = { version = "0.20", optional = true }
//!    [features]
//!    zap = []
//!    zap-capnp = ["zap", "dep:capnp", "dep:capnp-rpc", "dep:capnpc"]
//!    ```
//!    (Requires the `capnp` binary at build; this host does not have it, which
//!    is why the dep is off by default.)
//!
//! 2. In `build.rs`, compile the schema when `zap-capnp` is set:
//!    ```ignore
//!    #[cfg(feature = "zap-capnp")]
//!    capnpc::CompilerCommand::new()
//!        .file("schema/inference.capnp")
//!        .run()
//!        .expect("compile inference.capnp");
//!    ```
//!
//! 3. Include the generated module and impl the generated `inference::Server`
//!    trait by delegating to `ZapInferenceServer`:
//!    ```ignore
//!    pub mod inference_capnp {
//!        include!(concat!(env!("OUT_DIR"), "/inference_capnp.rs"));
//!    }
//!    // impl inference_capnp::inference::Server for ZapInferenceServer { ... }
//!    // each method reads the capnp params -> super::message::* -> calls the
//!    // matching ZapInferenceServer method -> writes the capnp results.
//!    ```
//!
//! 4. Serve over a `zap://` endpoint with `capnp_rpc::RpcSystem` on a Tokio TCP
//!    or unix listener (mirrors `hanzo-zap`'s planned `transport::connect`).
//!    The node side connects with `zap::Client` and bootstraps the `Inference`
//!    capability, replacing its `reqwest` POST to `/ai/chat/completions`.
//!
//! Until step 1 is taken, callers use `ZapInferenceServer` directly (it impls
//! `super::ZapInference`), exercising the full translation path in-process.

/// URL scheme for the ZAP inference endpoint. `transport::connect` in
/// `hanzo-zap` dispatches on this (`zap`, `zap+tcp`, `zap+unix`).
pub const ZAP_SCHEME: &str = "zap";

/// Builds the default `zap://` URL for a local engine endpoint.
pub fn default_endpoint(port: u16) -> String {
    format!("{ZAP_SCHEME}://127.0.0.1:{port}")
}
