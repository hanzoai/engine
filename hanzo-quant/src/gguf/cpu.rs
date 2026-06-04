//! CPU/Metal implementation of indexed MoE forward for GGUF quantized weights.
//!
//! This dequantizes the weights and delegates to UnquantLinear's gather_forward.

use hanzo_ml::{
    quantized::{QMatMul, QTensor},
    Result, Tensor,
};
use hanzo_nn::Linear;
use std::sync::Arc;

use crate::{QuantMethod, QuantMethodConfig, UnquantLinear};

/// Perform indexed MoE forward pass on a QTensor by dequantizing and using UnquantLinear.
///
/// # Arguments
/// * `qtensor` - The quantized weight tensor [num_experts, n, k]
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn qtensor_indexed_moe_forward(
    qtensor: &Arc<QTensor>,
    x: &Tensor,
    ids: &Tensor,
) -> Result<Tensor> {
    let device = x.device();

    // Dequantize all weights to f32
    let weights = qtensor.dequantize(device)?;

    // Create an UnquantLinear and use its gather_forward
    let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weights, None)))?;

    unquant.gather_forward(x, ids)
}

/// Perform indexed MoE forward pass on a QMatMul.
///
/// This is the main entry point for CPU/Metal GGUF quantized MoE forward.
///
/// # Arguments
/// * `qmatmul` - The quantized weight matrix
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn cpu_indexed_moe_forward(qmatmul: &QMatMul, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    match qmatmul {
        // Metal keeps the [E,n,k] GGUF bank quantized in unified memory and runs a per-expert,
        // router-gathered quant matvec straight out of it (QMetalStorage::indexed_moe_forward) -- no
        // whole-bank dequant, which on Metal materializes the multi-GB f32 bank per token and OOMs.
        // CPU has no resident-buffer win, so it stays on the dequantize-then-gather path.
        QMatMul::QTensor(qtensor) if qtensor.device().is_metal() => {
            qmatmul.indexed_moe_forward(x, ids)
        }
        QMatMul::QTensor(qtensor) => qtensor_indexed_moe_forward(qtensor, x, ids),
        #[cfg(feature = "vulkan")]
        QMatMul::VulkanQuant { qtensor, .. } => qtensor_indexed_moe_forward(qtensor, x, ids),
        // Resident MoE bank: delegate to the GPU keep-quantized path (per-expert banked matvec/matmul,
        // no whole-bank dequant). Dequantizing the [E,n,k] bank here would be the ~140GB f32 OOM.
        #[cfg(feature = "vulkan")]
        QMatMul::VulkanQuantBank { .. } => qmatmul.indexed_moe_forward(x, ids),
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
            // For non-quantized tensors, use UnquantLinear directly
            let unquant =
                UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(t.clone(), None)))?;
            unquant.gather_forward(x, ids)
        }
    }
}
