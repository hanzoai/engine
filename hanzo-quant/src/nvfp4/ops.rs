use hanzo_ml::{DType, Device, Result, Tensor};
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use float8::F8E4M3;
#[cfg(feature = "cuda")]
use half::{bf16, f16};
#[cfg(feature = "cuda")]
use hanzo_ml::{CudaStorage, Shape, Storage};

#[cfg(feature = "cuda")]
use super::ffi;
#[cfg(feature = "cuda")]
use crate::utils::slice_ptr;

pub const NVFP4_BLOCK_SIZE: usize = 16;

pub const FP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Dequantize NVFP4 weights to `out_dtype`.
///
/// `w_q`: uint8 [..., N, K/2] packed FP4 (low nibble = even idx, high nibble = odd idx).
/// `w_s`: fp8 e4m3 [..., N, K/16] per-block scale.
/// `w_s2`: optional f32 scalar multiplying the per-block scale (ModelOpt).
pub fn nvfp4_dequantize(
    w_q: &Tensor,
    w_s: &Tensor,
    w_s2: Option<&Tensor>,
    out_dtype: DType,
) -> Result<Tensor> {
    let dev = w_q.device();
    let dims = w_q.dims();
    if dims.len() < 2 {
        hanzo_ml::bail!("Expected at least 2 dimensions for NVFP4 weight tensor, got {:?}", dims);
    }
    let n = dims[dims.len() - 2];
    let half_k = dims[dims.len() - 1];
    let k = half_k * 2;
    let num_blocks = k / NVFP4_BLOCK_SIZE;

    // Convert w_s to F32
    let s_f32 = w_s.to_dtype(DType::F32)?;
    let scale_f32 = if let Some(s2) = w_s2 {
        let s2_f32 = s2.to_dtype(DType::F32)?;
        s_f32.broadcast_mul(&s2_f32)?
    } else {
        s_f32
    };

    let w_q_cpu = w_q.to_device(&Device::Cpu)?;
    let scale_cpu = scale_f32.to_device(&Device::Cpu)?;

    let w_bytes: Vec<u8> = w_q_cpu.flatten_all()?.to_vec1()?;
    let scale_vals: Vec<f32> = scale_cpu.flatten_all()?.to_vec1()?;

    let total_elements = n * k;
    let mut out_data = vec![0.0f32; total_elements];

    out_data
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(row, row_out)| {
            let row_w_offset = row * half_k;
            let row_s_offset = row * num_blocks;

            for blk in 0..num_blocks {
                let blk_scale = scale_vals[row_s_offset + blk];
                let blk_w_offset = row_w_offset + blk * (NVFP4_BLOCK_SIZE / 2);
                let blk_out_offset = blk * NVFP4_BLOCK_SIZE;

                for byte_idx in 0..(NVFP4_BLOCK_SIZE / 2) {
                    let byte = w_bytes[blk_w_offset + byte_idx];
                    let low_nibble = (byte & 0x0F) as usize;
                    let high_nibble = ((byte >> 4) & 0x0F) as usize;

                    row_out[blk_out_offset + byte_idx * 2] = FP4_E2M1_LUT[low_nibble] * blk_scale;
                    row_out[blk_out_offset + byte_idx * 2 + 1] = FP4_E2M1_LUT[high_nibble] * blk_scale;
                }
            }
        });

    let shape = if dims.len() == 3 {
        vec![dims[0], n, k]
    } else {
        vec![n, k]
    };

    Tensor::from_vec(out_data, shape.as_slice(), &Device::Cpu)?
        .to_device(dev)?
        .to_dtype(out_dtype)
}

/// output = input @ weight.T + bias, weights read packed.
///
///   input:  [M, K] f16/bf16
///   weight: [N, K/2] u8, 2 E2M1 values per byte
///   scale:  [N, K/16] fp8 e4m3 block scales
///   global: the checkpoint's FP32 `weight_scale_2`
#[cfg(feature = "cuda")]
pub fn nvfp4_matmul(
    input: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    global: f32,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    if !ffi::HAVE_NVFP4_GEMM_KERNELS {
        hanzo_ml::bail!("NVFP4 GEMM kernels not available");
    }

    let input = input.contiguous()?;
    let weight = weight.contiguous()?;
    let scale = scale.contiguous()?;

    let input_dims = input.dims();
    if input_dims.len() != 2 {
        hanzo_ml::bail!("Expected input to be rank 2, got {:?}", input_dims);
    }
    let m = input_dims[0];
    let k = input_dims[1];
    let n = weight.dims()[0];

    if weight.dims()[1] != k / 2 {
        hanzo_ml::bail!(
            "Weight shape mismatch: expected [N, K/2] = [{n}, {}], got {:?}",
            k / 2,
            weight.dims()
        );
    }
    // One uint4 load covers two blocks, so the kernel steps K in units of 32.
    if k % (NVFP4_BLOCK_SIZE * 2) != 0 {
        hanzo_ml::bail!("NVFP4 matmul needs K divisible by 32, got {k}");
    }

    let dev = match input.device() {
        Device::Cuda(dev) => dev.clone(),
        _ => hanzo_ml::bail!("Expected CUDA device"),
    };

    let input_l = input.layout();
    let weight_l = weight.layout();
    let scale_l = scale.layout();

    let input_storage = input.storage_and_layout().0;
    let weight_storage = weight.storage_and_layout().0;
    let scale_storage = scale.storage_and_layout().0;

    let weight_s = match &*weight_storage {
        Storage::Cuda(s) => s.as_cuda_slice::<u8>()?,
        _ => hanzo_ml::bail!("Expected CUDA storage for weight"),
    };
    // Block scales are fp8 e4m3; the kernel decodes the raw bytes itself.
    let scale_s = match &*scale_storage {
        Storage::Cuda(s) => s.as_cuda_slice::<F8E4M3>()?,
        _ => hanzo_ml::bail!("Expected CUDA storage for scale"),
    };

    let (weight_ptr, _weight_guard) = slice_ptr(weight_s, weight_l.start_offset());
    let (scale_ptr, _scale_guard) = slice_ptr(scale_s, scale_l.start_offset());
    let has_bias = bias.is_some();

    macro_rules! launch {
        ($ty:ty, $launcher:ident) => {{
            let output = dev.alloc_zeros::<$ty>(m * n)?;

            let input_s = match &*input_storage {
                Storage::Cuda(s) => s.as_cuda_slice::<$ty>()?,
                _ => hanzo_ml::bail!("Expected CUDA storage for input"),
            };
            let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());
            let (output_ptr, _output_guard) = slice_ptr(&output, 0);

            let bias_ptr = match bias {
                Some(b) => {
                    let b_l = b.layout();
                    let b_storage = b.storage_and_layout().0;
                    let b_s = match &*b_storage {
                        Storage::Cuda(s) => s.as_cuda_slice::<$ty>()?,
                        _ => hanzo_ml::bail!("Expected CUDA storage for bias"),
                    };
                    let (ptr, _guard) = slice_ptr(b_s, b_l.start_offset());
                    ptr as *const $ty
                }
                None => std::ptr::null(),
            };

            unsafe {
                ffi::$launcher(
                    input_ptr as *const $ty,
                    weight_ptr as *const u8,
                    scale_ptr as *const u8,
                    global,
                    bias_ptr,
                    output_ptr as *mut $ty,
                    m as i32,
                    n as i32,
                    k as i32,
                    has_bias,
                    dev.cuda_stream().cu_stream(),
                );
            }

            drop(_output_guard);
            Ok(Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                Shape::from((m, n)),
            )))
        }};
    }

    match input.dtype() {
        DType::F16 => launch!(f16, launch_nvfp4_matmul_f16),
        DType::BF16 => launch!(bf16, launch_nvfp4_matmul_bf16),
        dtype => hanzo_ml::bail!("Unsupported dtype for NVFP4 matmul: {dtype:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvfp4_dequantize_basic() {
        let weight = Tensor::new(&[0x10u8, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], &Device::Cpu)
            .unwrap()
            .reshape((1, 8))
            .unwrap();
        let scale = Tensor::new(&[1.0f32], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap()
            .reshape((1, 1))
            .unwrap();
        let dequant = nvfp4_dequantize(&weight, &scale, None, DType::F32).unwrap();
        assert_eq!(dequant.dims(), &[1, 16]);
        let vals: Vec<f32> = dequant.flatten_all().unwrap().to_vec1().unwrap();
        for i in 0..16 {
            let expected = FP4_E2M1_LUT[i];
            assert!((vals[i] - expected).abs() < 1e-4, "Mismatch at {i}: got {}, expected {expected}", vals[i]);
        }
    }

    /// The packed kernel must agree with dequantize-then-matmul, which is the path it replaces.
    #[cfg(feature = "cuda")]
    #[test]
    fn nvfp4_matmul_matches_dequantized() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let (n, k) = (64usize, 128usize);
        let blocks = k / NVFP4_BLOCK_SIZE;
        let global = 0.75f32;

        let packed: Vec<u8> = (0..n * k / 2).map(|i| ((i * 37 + 11) % 256) as u8).collect();
        let scales: Vec<f32> = (0..n * blocks)
            .map(|i| [0.5f32, 1.0, 2.0, 0.25][i % 4])
            .collect();

        let w_q_cpu = Tensor::from_vec(packed, (n, k / 2), &Device::Cpu)?;
        let w_s_cpu = Tensor::from_vec(scales, (n, blocks), &Device::Cpu)?.to_dtype(DType::F8E4M3)?;
        let s2 = Tensor::new(global, &Device::Cpu)?;
        let w_ref = nvfp4_dequantize(&w_q_cpu, &w_s_cpu, Some(&s2), DType::F32)?.t()?.contiguous()?;

        let w_q = w_q_cpu.to_device(&dev)?;
        let w_s = w_s_cpu.to_device(&dev)?;

        for m in [1usize, 4, 17, 64] {
            let x: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
            let x_cpu = Tensor::from_vec(x, (m, k), &Device::Cpu)?;
            let want: Vec<f32> = x_cpu.matmul(&w_ref)?.flatten_all()?.to_vec1()?;

            let x_dev = x_cpu.to_device(&dev)?.to_dtype(DType::BF16)?;
            let got: Vec<f32> = nvfp4_matmul(&x_dev, &w_q, &w_s, global, None)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;

            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 0.02 * w.abs().max(1.0),
                    "m={m} i={i}: got {g}, want {w}"
                );
            }
        }
        Ok(())
    }
}
