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
#[cfg(feature = "vulkan")]
pub fn gdn_step_vulkan(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    use hanzo_ml::Storage;

    let (bh, k_dim) = q.dims2()?;
    let v_dim = v.dim(1)?;

    let (q_s, q_l) = q.storage_and_layout();
    let Storage::Vulkan(q_s) = &*q_s else {
        hanzo_ml::bail!("gdn_step: q must be a vulkan tensor");
    };
    let (k_s, k_l) = k.storage_and_layout();
    let Storage::Vulkan(k_s) = &*k_s else {
        hanzo_ml::bail!("gdn_step: k must be a vulkan tensor");
    };
    let (v_s, v_l) = v.storage_and_layout();
    let Storage::Vulkan(v_s) = &*v_s else {
        hanzo_ml::bail!("gdn_step: v must be a vulkan tensor");
    };
    let (g_s, g_l) = g.storage_and_layout();
    let Storage::Vulkan(g_s) = &*g_s else {
        hanzo_ml::bail!("gdn_step: g must be a vulkan tensor");
    };
    let (beta_s, beta_l) = beta.storage_and_layout();
    let Storage::Vulkan(beta_s) = &*beta_s else {
        hanzo_ml::bail!("gdn_step: beta must be a vulkan tensor");
    };
    let (state_s, state_l) = state.storage_and_layout();
    let Storage::Vulkan(state_s) = &*state_s else {
        hanzo_ml::bail!("gdn_step: state must be a vulkan tensor");
    };

    // The shader updates `state`'s buffer in place; the state Tensor already views that buffer, so
    // there is nothing to write back. `out` is a fresh owned storage, independent of the read guards.
    let out = q_s.gdn_step(
        q_l, k_s, k_l, v_s, v_l, g_s, g_l, beta_s, beta_l, state_s, state_l, bh, k_dim, v_dim,
    )?;
    Ok(Tensor::from((hanzo_ml::Storage::Vulkan(out), (bh, v_dim))))
}

/// Vulkan single-step causal depthwise conv1d (decode, seq_len==1, batch==1).
///
/// Inputs (contiguous f32, same Vulkan device):
///   conv_state: [conv_dim, k_size] (updated IN PLACE: drop oldest column, append x; kept across tokens)
///   x: [conv_dim] new column   weight: [conv_dim, k_size]
///
/// Returns: silu(conv) [conv_dim].
#[cfg(feature = "vulkan")]
pub fn gdn_conv1d_step_vulkan(
    conv_state: &mut Tensor,
    x: &Tensor,
    weight: &Tensor,
) -> Result<Tensor> {
    use hanzo_ml::Storage;

    let conv_dim = weight.dim(0)?;
    let k_size = weight.dim(1)?;

    let (cs_s, cs_l) = conv_state.storage_and_layout();
    let Storage::Vulkan(cs_s) = &*cs_s else {
        hanzo_ml::bail!("gdn_conv1d_step: conv_state must be a vulkan tensor");
    };
    let (x_s, x_l) = x.storage_and_layout();
    let Storage::Vulkan(x_s) = &*x_s else {
        hanzo_ml::bail!("gdn_conv1d_step: x must be a vulkan tensor");
    };
    let (w_s, w_l) = weight.storage_and_layout();
    let Storage::Vulkan(w_s) = &*w_s else {
        hanzo_ml::bail!("gdn_conv1d_step: weight must be a vulkan tensor");
    };

    let out = cs_s.gdn_conv1d_step(cs_l, x_s, x_l, w_s, w_l, conv_dim, k_size)?;
    Ok(Tensor::from((hanzo_ml::Storage::Vulkan(out), conv_dim)))
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
