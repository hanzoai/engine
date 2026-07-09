//! FASHN VTON v1.5 configuration. The released checkpoint ships no config.json (a single
//! `model.safetensors`), so these are the fixed architecture constants from the reference
//! `TryOnModel.__init__` (fashn-ai/fashn-vton-1.5, Apache-2.0).

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

#[derive(Debug, Clone)]
pub struct VtonConfig {
    pub height: usize,
    pub width: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub double_blocks_depth: usize,
    pub single_blocks_depth: usize,
    pub patch_mixer_depth: usize,
    pub mlp_ratio: f64,
    pub channels_in: usize,
    pub patch_size: usize,
    pub theta: usize,
    pub axes_dim: [usize; 3],
    pub n_classes: usize,
}

impl Default for VtonConfig {
    fn default() -> Self {
        Self {
            height: 864,
            width: 576,
            hidden_size: 1280,
            num_heads: 10,
            double_blocks_depth: 8,
            single_blocks_depth: 16,
            patch_mixer_depth: 4,
            mlp_ratio: 4.0,
            channels_in: 3,
            patch_size: 12,
            theta: 10000,
            axes_dim: [16, 56, 56],
            n_classes: 3,
        }
    }
}

impl VtonConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
    pub fn grid_h(&self) -> usize {
        self.height / self.patch_size
    }
    pub fn grid_w(&self) -> usize {
        self.width / self.patch_size
    }
    pub fn mlp_hidden(&self) -> usize {
        (self.hidden_size as f64 * self.mlp_ratio) as usize
    }
    pub fn x_in_channels(&self) -> usize {
        self.channels_in * 2 + 1
    }
    pub fn garment_in_channels(&self) -> usize {
        self.channels_in + 1
    }
    pub fn patch_out_dim(&self) -> usize {
        self.channels_in * self.patch_size * self.patch_size
    }
}
