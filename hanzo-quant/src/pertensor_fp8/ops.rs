use float8::F8E4M3;
use hanzo_ml::{DType, Result, Tensor};

use crate::scalar_fp8::ops::{dtype_to_fp8, fp8_to_dtype};

/// Per-tensor FP8 dequantization: `x = q * scale`.
///
/// One scalar carries the whole tensor, so the scale broadcasts over every element.
pub fn fp8_pertensor_dequantize(
    weight: &Tensor,
    scale: &Tensor,
    out_dtype: DType,
) -> Result<Tensor> {
    let weight_f32 = fp8_to_dtype(weight, DType::F32)?;
    let scale_f32 = scale.to_dtype(DType::F32)?;
    (weight_f32.broadcast_mul(&scale_f32))?.to_dtype(out_dtype)
}

/// Per-tensor FP8 quantization against the scale its dequantization uses:
/// `q = clamp(x / scale, ±448)`, so `q * scale` lands back on `x` within the E4M3 grid.
/// Magnitudes past the range saturate instead of turning into infinities.
#[allow(dead_code)] // The FP8 GEMM is the only caller, and it is CUDA-only.
pub fn fp8_pertensor_quantize(x: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let max = F8E4M3::MAX.to_f64();
    let scaled = x.broadcast_div(&scale.to_dtype(x.dtype())?)?;
    dtype_to_fp8(&scaled.clamp(-max, max)?)
}

#[cfg(test)]
mod tests {
    use float8::F8E4M3;
    use hanzo_ml::{DType, Device, Result, Tensor};

    use super::{fp8_pertensor_dequantize, fp8_pertensor_quantize};
    use crate::{fp8::QuantizationResult, scalar_fp8::ops::fp8_to_dtype, FP8Linear};

    /// E4M3 keeps three mantissa bits, so round-to-nearest lands within 1/16 of a
    /// normal value.
    const E4M3_RELATIVE: f32 = 1. / 16.;

    fn max_relative_error(reference: &Tensor, got: &Tensor) -> Result<f32> {
        (reference - got)?
            .abs()?
            .div(&reference.abs()?)?
            .max_all()?
            .to_scalar::<f32>()
    }

    #[test]
    fn static_scale_round_trip() -> Result<()> {
        let dev = Device::Cpu;
        // Away from zero so relative error is the meaningful measure, and spanning two
        // decades so both ends of the E4M3 exponent range are exercised.
        let x = Tensor::rand(0.05f32, 4f32, (64, 128), &dev)?;
        let scale = Tensor::new(0.01f32, &dev)?;

        let q = fp8_pertensor_quantize(&x, &scale)?;
        assert_eq!(q.dtype(), DType::F8E4M3);

        let back = fp8_pertensor_dequantize(&q, &scale, DType::F32)?;
        assert!(max_relative_error(&x, &back)? <= E4M3_RELATIVE);
        Ok(())
    }

    #[test]
    fn static_scale_saturates() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::new(&[-1e4f32, 1e4, 1.], &dev)?;
        let scale = Tensor::new(1f32, &dev)?;

        let back =
            fp8_pertensor_dequantize(&fp8_pertensor_quantize(&x, &scale)?, &scale, DType::F32)?;

        let max = F8E4M3::MAX.to_f32();
        assert_eq!(back.to_vec1::<f32>()?, vec![-max, max, 1.]);
        Ok(())
    }

    #[test]
    fn dynamic_scale_round_trip() -> Result<()> {
        let dev = Device::Cpu;
        // The dynamic path rounds to BF16 before it picks a scale, so that is the
        // reference the E4M3 grid is measured against.
        let x = Tensor::rand(0.05f32, 4f32, (64, 128), &dev)?
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?;

        let QuantizationResult {
            qw,
            dequantize_scale,
            ..
        } = FP8Linear::quantize(&x, DType::F8E4M3)?;
        assert_eq!(qw.dtype(), DType::F8E4M3);

        let back = fp8_to_dtype(&qw, DType::F32)?.broadcast_mul(&dequantize_scale)?;
        assert!(max_relative_error(&x, &back)? <= E4M3_RELATIVE);
        Ok(())
    }
}
