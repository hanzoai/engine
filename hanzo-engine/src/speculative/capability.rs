//! The ONE seam for self-speculative decoding.
//!
//! A model that carries its own multi-token-prediction (MTP) head can act as its
//! OWN speculative draft: one forward drafts K tokens that the base verifies in a
//! single batched pass. That property is DATA the model owns — not a fact the
//! pipeline should hard-code per architecture.
//!
//! [`SelfSpeculative`] is that capability. The speculative pipeline attaches MTP by
//! *asking the model* for the trait object; it never names an architecture. Adding
//! a new self-speculative model (GLM-5.2's `nextn` head, a future MoE's MTP, …) is
//! therefore a capability impl next to that model — never an edit to the pipeline.
//!
//! Decomplect (Hickey): *what a model is* (it has an MTP head, loaded thus) is
//! separated from *how the pipeline runs speculation* (attach a proposer, verify a
//! batch). One way to attach self-speculation, for every model, forever.

use crate::speculative::{MtpConfig, SpeculativeProposer};
use hanzo_ml::Result;

/// A model whose weights include an MTP draft head and can therefore be its own
/// speculative draft.
///
/// [`attach_mtp`](SelfSpeculative::attach_mtp) loads the head as described by
/// `cfg` (a companion MTP-GGUF path and/or in-band `nextn` tensors, the model's
/// choice), wires it to the base model's shared embeddings + output head, and
/// returns a ready [`SpeculativeProposer`]. Models without an MTP head simply do
/// not implement this trait, and [`crate::pipeline`] reports that honestly.
pub trait SelfSpeculative {
    /// Build the self-speculative proposer for this model from `cfg`.
    fn attach_mtp(&self, cfg: &MtpConfig) -> Result<Box<dyn SpeculativeProposer + Send + Sync>>;
}
