use std::{
    str::FromStr,
    sync::{Arc, Mutex, MutexGuard},
};

use hanzo_ml::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::config::{KvCacheLayout, ModelConfigLike};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
pub enum PagedCacheType {
    #[default]
    Auto,
    F8E4M3,
    /// Symmetric per-token int8 KV cache (KIVI / KVQuant). Codes + a co-located per-token f32
    /// scale are packed 4-per-u32, so the storage dtype is `U32` (Vulkan has no 1-byte tensor);
    /// the packed layout is built by `calculate_{key,value}_block_shape`. See
    /// `hanzo_paged_attn::quant` for the reference/oracle and the Vulkan `*_q8` kernels.
    Int8,
}

impl PagedCacheType {
    pub fn to_dtype(&self, act_dtype: DType) -> DType {
        match self {
            PagedCacheType::F8E4M3 => DType::F8E4M3,
            // int8 codes are packed 4-per-u32 with a co-located f32 scale word; the storage
            // element is u32. `U32` is the unique marker for an int8 KV cache downstream.
            PagedCacheType::Int8 => DType::U32,
            PagedCacheType::Auto => act_dtype,
        }
    }
}

impl FromStr for PagedCacheType {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "f8e4m3" => Ok(Self::F8E4M3),
            "int8" => Ok(Self::Int8),
            other => Err(format!(
                "Unexpected `PagedCacheType`, got `{other}` but expected `auto`, `f8e4m3`, or `int8`."
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub cache_type: PagedCacheType,
}

pub type KVCache = (Tensor, Tensor);

pub struct CacheEngine {
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,
}

impl CacheEngine {
    pub fn new(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Self> {
        let dtype = cache_config.cache_type.to_dtype(dtype);
        Ok(Self {
            gpu_cache: Arc::new(Mutex::new(Self::allocate_gpu_cache(
                model_config,
                cache_config,
                dtype,
                device,
                layer_devices,
            )?)),
        })
    }

    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<KVCache>> {
        // Use blocking lock instead of busy-wait spin loop to avoid CPU waste
        // and potential thread starvation issues
        self.gpu_cache.lock().expect("KV cache mutex was poisoned")
    }

    fn allocate_gpu_cache(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Vec<KVCache>> {
        let mut gpu_cache = Vec::new();

        // One K/V pair per cached layer, on the device of the layer that reads it: for a hybrid
        // only the attention layers are here, and they reach their pair by its position in this
        // list.
        for layer_idx in model_config.kv_layers() {
            let device = Self::device_for(model_config, &layer_devices, device, layer_idx);
            let requested_kv_cache_layout = model_config.kv_cache_layout_for_layer(layer_idx);
            let kv_cache_layout =
                if matches!(requested_kv_cache_layout, KvCacheLayout::FlashInferHnd)
                    && !device.is_cuda()
                {
                    KvCacheLayout::Standard
                } else {
                    requested_kv_cache_layout
                };
            let (key_blocks, value_blocks) = match kv_cache_layout {
                KvCacheLayout::Standard | KvCacheLayout::StandardNoFlashInfer => {
                    let key_block_shape = Self::calculate_key_block_shape(
                        model_config,
                        dtype,
                        cache_config.block_size,
                        layer_idx,
                    );
                    let value_block_shape = Self::calculate_value_block_shape(
                        model_config,
                        dtype,
                        cache_config.block_size,
                        layer_idx,
                    );
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use hanzo_ml::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * key_block_shape.0
                                * key_block_shape.1
                                * key_block_shape.2
                                * key_block_shape.3;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                    key_block_shape.3,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                    key_block_shape.3,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    #[allow(unused)]
                    let value_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use hanzo_ml::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * value_block_shape.0
                                * value_block_shape.1
                                * value_block_shape.2;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "v_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    value_block_shape.0,
                                    value_block_shape.1,
                                    value_block_shape.2,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    value_block_shape.0,
                                    value_block_shape.1,
                                    value_block_shape.2,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    (key_blocks, value_blocks)
                }
                KvCacheLayout::FlashInferHnd => {
                    let key_block_shape = Self::calculate_flashinfer_block_shape(
                        model_config,
                        cache_config.block_size,
                        layer_idx,
                    );
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use hanzo_ml::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * key_block_shape.0
                                * key_block_shape.1
                                * key_block_shape.2;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    let value_blocks = unsafe {
                        Tensor::empty(
                            (
                                cache_config.num_gpu_blocks,
                                key_block_shape.0,
                                key_block_shape.1,
                                key_block_shape.2,
                            ),
                            dtype,
                            device,
                        )?
                    };
                    (key_blocks, value_blocks)
                }
                KvCacheLayout::Mla {
                    kv_lora_rank,
                    kpe_head_dim,
                } => {
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use hanzo_ml::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * cache_config.block_size
                                * kv_lora_rank;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kv_lora_rank,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kv_lora_rank,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    #[allow(unused)]
                    let value_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use hanzo_ml::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * cache_config.block_size
                                * kpe_head_dim;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "v_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kpe_head_dim,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kpe_head_dim,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    (key_blocks, value_blocks)
                }
            };
            gpu_cache.push((key_blocks, value_blocks));
        }
        Ok(gpu_cache)
    }

    /// The device for cache entry `layer_idx`: that of the decoder layer reading it
    /// ([`ModelConfigLike::kv_reader`]), or the base device when no decoder layer reads it or the
    /// map does not place that layer.
    fn device_for<'a>(
        model_config: &dyn ModelConfigLike,
        layer_devices: &'a [Option<Device>],
        device: &'a Device,
        layer_idx: usize,
    ) -> &'a Device {
        model_config
            .kv_reader(layer_idx)
            .and_then(|reader| layer_devices.get(reader))
            .and_then(Option::as_ref)
            .unwrap_or(device)
    }

    fn calculate_key_block_shape(
        model_config: &dyn ModelConfigLike,
        dtype: DType,
        block_size: usize,
        layer_idx: usize,
    ) -> (usize, usize, usize, usize) {
        let num_kv_heads = model_config.num_kv_heads_for_layer(layer_idx);
        let head_dim = model_config.k_head_dim_for_layer(layer_idx);
        if dtype == DType::U32 {
            // int8-packed key cache, token-major: [num_kv_heads, block_size, head_dim/4 + 1] u32
            // (head_dim/4 code words + 1 co-located per-token f32 scale word), rank-padded with a
            // trailing 1 so the Standard rank-5 allocation and block-shape readers still apply.
            // Element (block, head, token, word) sits at token*wpt + word, matching the `*_q8`
            // shaders whose strides are computed from head_size/block_size/num_kv_heads.
            let wpt = head_dim / 4 + 1;
            return (num_kv_heads, block_size, wpt, 1);
        }
        let element_size = dtype.size_in_bytes();
        let x = 16 / element_size;
        (num_kv_heads, head_dim / x, block_size, x)
    }

    fn calculate_value_block_shape(
        model_config: &dyn ModelConfigLike,
        dtype: DType,
        block_size: usize,
        layer_idx: usize,
    ) -> (usize, usize, usize) {
        let num_kv_heads = model_config.num_kv_heads_for_layer(layer_idx);
        if dtype == DType::U32 {
            // int8-packed value cache, token-major: [num_kv_heads, block_size, head_dim/4 + 1] u32,
            // the same per-token layout as the key cache (KIVI uses per-token grouping for values).
            let wpt = model_config.v_head_dim_for_layer(layer_idx) / 4 + 1;
            return (num_kv_heads, block_size, wpt);
        }
        (
            num_kv_heads,
            model_config.v_head_dim_for_layer(layer_idx),
            block_size,
        )
    }

    fn calculate_flashinfer_block_shape(
        model_config: &dyn ModelConfigLike,
        block_size: usize,
        layer_idx: usize,
    ) -> (usize, usize, usize) {
        (
            model_config.num_kv_heads_for_layer(layer_idx),
            block_size,
            model_config.k_head_dim_for_layer(layer_idx),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::tests::Indexed;
    use crate::paged_attention::{KvLayers, ModelConfigMetadata};

    #[test]
    fn entries_live_with_the_layer_that_reads_them() {
        let layer_devices = vec![Some(Device::Cpu); Indexed.num_layers()];
        let base = Device::Cpu;
        let at = |config: &dyn ModelConfigLike, layer_idx| {
            CacheEngine::device_for(config, &layer_devices, &base, layer_idx)
        };
        let mapped = |layer: usize| layer_devices[layer].as_ref().unwrap();

        // An attention layer and its index cache past the decoder depth share that layer's device.
        assert!(std::ptr::eq(at(&Indexed, 3), mapped(3)));
        assert!(std::ptr::eq(at(&Indexed, 48 + 3), mapped(3)));
        assert!(std::ptr::eq(at(&Indexed, 48 + 47), mapped(47)));

        // By default an entry past the depth, a proposer head, has no reader: the base device.
        let head = KvLayers::new(
            ModelConfigMetadata {
                max_seq_len: 4096,
                num_layers: 48,
                hidden_size: 2560,
                num_kv_heads: 2,
                num_attn_heads: 24,
                sliding_window: None,
                k_head_dim: 256,
                v_head_dim: 256,
                kv_cache_layout: KvCacheLayout::Standard,
            },
            vec![3, 7, 48],
        );
        assert!(std::ptr::eq(at(&head, 7), mapped(7)));
        assert!(std::ptr::eq(at(&head, 48), &base));

        // A reader the device map does not place falls back to the base device too.
        assert!(std::ptr::eq(
            CacheEngine::device_for(&Indexed, &layer_devices[..3], &base, 48 + 3),
            &base
        ));
    }
}
