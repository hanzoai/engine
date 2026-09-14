use hanzo_ml::{DType, Device, Result, Tensor};
use rayon::prelude::*;

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
}

