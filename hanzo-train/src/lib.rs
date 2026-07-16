//! `hanzo-train` — native Rust LoRA fine-tuning for the Hanzo engine.
//!
//! A small, Tinker-shaped API over the engine's ML stack (`hanzo-ml` autograd,
//! `hanzo-nn` layers + AdamW). Phase 1 covers one vertical slice: LoRA supervised
//! fine-tuning of a Llama / Qwen2-family decoder (SmolLM2, Qwen2.5), with the base
//! weights frozen and only the LoRA factors trained. A saved adapter is written in
//! PEFT layout so the existing engine loads it for inference.
//!
//! The four primitives ([`create_lora_training_client`], [`TrainingClient::forward_backward`],
//! [`TrainingClient::optim_step`], [`TrainingClient::sample`],
//! [`TrainingClient::save_weights_and_get_sampling_client`]) mirror Thinking Machines' Tinker.

pub mod client;
pub mod config;
pub mod data;
pub mod distill;
pub mod loader;
pub mod lora;
pub mod loss;
pub mod model;
pub mod run;
pub mod save;
pub mod types;

pub use client::{
    create_lora_training_client, create_lora_training_client_from_engine, TrainingClient,
};
pub use distill::{run_distill, DistillConfig, DistillReport};
pub use loader::BaseModel;
pub use run::{run_sft, run_sft_with_client, SftConfig, SftReport};
pub use types::{AdamParams, Datum, ForwardBackwardOutput, LoraConfig, ModelInput, SamplingParams};
