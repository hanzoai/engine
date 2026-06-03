#![allow(clippy::cast_possible_truncation)]

use hanzo_ml::{Result, Tensor};

/// Vulkan single-step gated delta rule recurrence (decode, seq_len==1).
///
/// Inputs (all contiguous f32, on the same Vulkan device):
///   q, k: [BH, K]  v: [BH, V]  g, beta: [BH]
///   state: [BH, K, V] (updated IN PLACE in VRAM; the caller's pool keeps it across tokens)
/// q must already be scaled by 1/sqrt(K), matching gated_delta_rule_recurrence and the CUDA wrapper.
///
/// Returns: output y [BH, V].
// Canonical hanzo-ml (candle 0.10.2) has not yet ported the GDN Vulkan kernels
// (VulkanStorage::gdn_step / gdn_conv1d_step). Bail until they land; only Qwen3.5/3.6
// hybrid-MoE decode uses these, so dense/standard models are unaffected.
#[cfg(feature = "vulkan")]
pub fn gdn_step_vulkan(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _g: &Tensor,
    _beta: &Tensor,
    _state: &mut Tensor,
) -> Result<Tensor> {
    hanzo_ml::bail!("gdn_step_vulkan: GDN Vulkan kernels not yet ported to canonical hanzo-ml")
}

/// Vulkan single-step causal depthwise conv1d (decode, seq_len==1, batch==1).
#[cfg(feature = "vulkan")]
pub fn gdn_conv1d_step_vulkan(
    _conv_state: &mut Tensor,
    _x: &Tensor,
    _weight: &Tensor,
) -> Result<Tensor> {
    hanzo_ml::bail!("gdn_conv1d_step_vulkan: GDN Vulkan kernels not yet ported to canonical hanzo-ml")
}

#[cfg(not(feature = "vulkan"))]
#[allow(unused)]
pub fn gdn_step_vulkan(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _g: &Tensor,
    _beta: &Tensor,
    _state: &mut Tensor,
) -> Result<Tensor> {
    hanzo_ml::bail!("gdn_step_vulkan requires the vulkan feature")
}

#[cfg(not(feature = "vulkan"))]
#[allow(unused)]
pub fn gdn_conv1d_step_vulkan(
    _conv_state: &mut Tensor,
    _x: &Tensor,
    _weight: &Tensor,
) -> Result<Tensor> {
    hanzo_ml::bail!("gdn_conv1d_step_vulkan requires the vulkan feature")
}
