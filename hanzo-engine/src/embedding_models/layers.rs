#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{DType, IndexOp, Result, Tensor, D};
use hanzo_nn::Module;
use serde::Deserialize;

fn default_true() -> bool {
    true
}

/// A sentence-transformers pooling mode.
#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PoolingMode {
    Cls,
    Max,
    Mean,
    #[serde(rename = "mean_sqrt_len_tokens")]
    MeanSqrtLen,
    #[serde(rename = "weightedmean")]
    WeightedMean,
    #[serde(rename = "lasttoken")]
    LastToken,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Modes {
    One(PoolingMode),
    Many(Vec<PoolingMode>),
}

/// `1_Pooling/config.json` in either layout sentence-transformers writes: v6's
/// `embedding_dimension` and `pooling_mode` (a mode or a list), or the older
/// `word_embedding_dimension` and one `pooling_mode_*` flag per mode.
#[derive(Deserialize)]
struct PoolingConfig {
    #[serde(alias = "word_embedding_dimension")]
    embedding_dimension: usize,
    pooling_mode: Option<Modes>,
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_max_tokens: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_mean_sqrt_len_tokens: bool,
    #[serde(default)]
    pooling_mode_weightedmean_tokens: bool,
    #[serde(default)]
    pooling_mode_lasttoken: bool,
    #[serde(default = "default_true")]
    include_prompt: bool,
}

/// Pooling layer: each mode's `[batch, dim]` vector, concatenated in order. A batch holds
/// sequences of one length and no padding (the scheduler buckets by length), so every token
/// position is real.
#[derive(Deserialize, Debug, Clone)]
#[serde(try_from = "PoolingConfig")]
pub struct Pooling {
    dimension: usize,
    modes: Vec<PoolingMode>,
}

impl TryFrom<PoolingConfig> for Pooling {
    type Error = String;

    fn try_from(c: PoolingConfig) -> std::result::Result<Self, String> {
        if !c.include_prompt {
            return Err("pooling with include_prompt = false is not supported".into());
        }
        let modes = match c.pooling_mode {
            Some(Modes::One(mode)) => vec![mode],
            Some(Modes::Many(modes)) => modes,
            // sentence-transformers' legacy order.
            None => [
                (c.pooling_mode_cls_token, PoolingMode::Cls),
                (c.pooling_mode_max_tokens, PoolingMode::Max),
                (c.pooling_mode_mean_tokens, PoolingMode::Mean),
                (
                    c.pooling_mode_mean_sqrt_len_tokens,
                    PoolingMode::MeanSqrtLen,
                ),
                (
                    c.pooling_mode_weightedmean_tokens,
                    PoolingMode::WeightedMean,
                ),
                (c.pooling_mode_lasttoken, PoolingMode::LastToken),
            ]
            .into_iter()
            .filter_map(|(on, mode)| on.then_some(mode))
            .collect(),
        };
        if modes.is_empty() {
            return Err("pooling config names no pooling mode".into());
        }
        Ok(Self {
            dimension: c.embedding_dimension,
            modes,
        })
    }
}

impl Module for Pooling {
    // https://github.com/huggingface/sentence-transformers/blob/v6.1.0/sentence_transformers/sentence_transformer/modules/pooling.py
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if xs.dim(D::Minus1)? != self.dimension {
            hanzo_ml::bail!("xs does not match the expected embedding dimension.");
        }
        let len = xs.dim(1)?;
        let outputs = self
            .modes
            .iter()
            .map(|mode| match mode {
                PoolingMode::Cls => xs.i((.., 0, ..)),
                PoolingMode::Max => xs.max(1),
                PoolingMode::Mean => xs.mean(1),
                PoolingMode::MeanSqrtLen => xs.sum(1)? / (len as f64).sqrt(),
                PoolingMode::WeightedMean => {
                    // Position i weighs i + 1.
                    let w = Tensor::arange(1u32, len as u32 + 1, xs.device())?
                        .to_dtype(DType::F32)?
                        .to_dtype(xs.dtype())?
                        .reshape((1, len, 1))?;
                    xs.broadcast_mul(&w)?.sum(1)? / ((len * (len + 1)) as f64 / 2.)
                }
                PoolingMode::LastToken => xs.i((.., len - 1, ..)),
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&outputs, 1)
    }
}

/// Normalize layer
#[derive(Deserialize, Debug, Clone)]
pub struct Normalize;

impl Module for Normalize {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let norm = (xs.sqr()?.sum(1)? + 1e-12)?.sqrt()?;

        xs.broadcast_div(&norm.unsqueeze(D::Minus1)?)
    }
}

#[derive(Deserialize, Debug, Clone)]
pub enum DenseActivation {
    #[serde(alias = "torch.nn.modules.linear.Identity")]
    Identity,
}

/// Dense layer
#[derive(Deserialize, Debug, Clone)]
pub struct Dense {
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
    pub activation_function: DenseActivation,
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::Device;

    fn parse(json: &str) -> std::result::Result<Pooling, serde_json::Error> {
        serde_json::from_str(json)
    }

    #[test]
    fn both_config_layouts_read_alike() {
        // google/embeddinggemma-2 (sentence-transformers 6) and all-MiniLM-L6-v2 (2.x).
        let v6 = parse(
            r#"{"embedding_dimension": 768, "pooling_mode": "mean", "include_prompt": true}"#,
        )
        .unwrap();
        let v2 = parse(r#"{"word_embedding_dimension": 768, "pooling_mode_mean_tokens": true}"#)
            .unwrap();
        assert_eq!(
            (v6.dimension, v6.modes.clone()),
            (768, vec![PoolingMode::Mean])
        );
        assert_eq!((v2.dimension, v2.modes), (768, vec![PoolingMode::Mean]));
        let many =
            parse(r#"{"embedding_dimension": 4, "pooling_mode": ["cls", "lasttoken"]}"#).unwrap();
        assert_eq!(many.modes, vec![PoolingMode::Cls, PoolingMode::LastToken]);
    }

    #[test]
    fn unsupported_configs_fail_at_load() {
        assert!(parse(
            r#"{"embedding_dimension": 4, "pooling_mode": "mean", "include_prompt": false}"#
        )
        .is_err());
        assert!(parse(r#"{"word_embedding_dimension": 4}"#).is_err());
        assert!(parse(r#"{"embedding_dimension": 4, "pooling_mode": "median"}"#).is_err());
    }

    #[test]
    fn modes_pool_as_sentence_transformers() -> Result<()> {
        // One sequence of three tokens, two dims.
        let xs = Tensor::new(&[[[1f32, -2.], [3., 4.], [5., 0.]]], &Device::Cpu)?;
        let pool = |mode: &str| -> Result<Vec<f32>> {
            let p = parse(&format!(
                r#"{{"embedding_dimension": 2, "pooling_mode": "{mode}"}}"#
            ))
            .unwrap();
            p.forward(&xs)?.squeeze(0)?.to_vec1()
        };
        assert_eq!(pool("cls")?, vec![1., -2.]);
        assert_eq!(pool("max")?, vec![5., 4.]);
        assert_eq!(pool("mean")?, vec![3., 2. / 3.]);
        assert_eq!(pool("lasttoken")?, vec![5., 0.]);
        let w = pool("weightedmean")?;
        assert!((w[0] - 22. / 6.).abs() < 1e-6 && (w[1] - 6. / 6.).abs() < 1e-6);
        let s = pool("mean_sqrt_len_tokens")?;
        assert!((s[0] - 9. / 3f32.sqrt()).abs() < 1e-6);
        Ok(())
    }
}
