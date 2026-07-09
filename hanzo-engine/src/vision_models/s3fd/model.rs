#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{Module, Result, Tensor};
use hanzo_nn::{Conv2d, Conv2dConfig};
use hanzo_quant::ShardedVarBuilder;

use crate::layers::conv2d;

const L2NORM_EPS: f64 = 1e-10;

fn conv(
    vb: &ShardedVarBuilder,
    name: &str,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Conv2d> {
    let cfg = Conv2dConfig {
        padding,
        stride,
        dilation,
        ..Default::default()
    };
    conv2d(cin, cout, k, cfg, vb.pp(name))
}

/// Per-channel L2 normalization with a learnable scale (`x / (||x||_2 + eps) * g`),
/// normalizing over the channel axis. Matches face_alignment's `L2Norm`.
struct L2Norm {
    weight: Tensor,
}

impl L2Norm {
    fn new(channels: usize, vb: &ShardedVarBuilder, name: &str) -> Result<Self> {
        Ok(Self {
            weight: vb.pp(name).get(channels, "weight")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let norm = (x.sqr()?.sum_keepdim(1)?.sqrt()? + L2NORM_EPS)?;
        let normed = x.broadcast_div(&norm)?;
        normed.broadcast_mul(&self.weight.reshape((1, (), 1, 1))?)
    }
}

struct Head {
    conf: Conv2d,
    loc: Conv2d,
}

impl Head {
    fn new(vb: &ShardedVarBuilder, prefix: &str, cin: usize, conf_out: usize) -> Result<Self> {
        Ok(Self {
            conf: conv(
                vb,
                &format!("{prefix}_mbox_conf"),
                cin,
                conf_out,
                3,
                1,
                1,
                1,
            )?,
            loc: conv(vb, &format!("{prefix}_mbox_loc"), cin, 4, 3, 1, 1, 1)?,
        })
    }
}

/// One decoded detection scale: post-net `conf` `[1, 2, H, W]` (already maxout-
/// reduced) and `loc` `[1, 4, H, W]`, plus the anchor stride.
pub(crate) struct ScaleOutput {
    pub conf: Tensor,
    pub loc: Tensor,
    pub stride: usize,
}

/// S3FD = VGG16 conv backbone + dilated fc6/fc7 + conv6/conv7 extra layers, with
/// 6 detection heads (conv3_3, conv4_3, conv5_3, fc7, conv6_2, conv7_2). conv3_3
/// uses max-out background (4 conf channels -> max of the 3 bg + the face score).
pub struct S3fdModel {
    conv1_1: Conv2d,
    conv1_2: Conv2d,
    conv2_1: Conv2d,
    conv2_2: Conv2d,
    conv3_1: Conv2d,
    conv3_2: Conv2d,
    conv3_3: Conv2d,
    conv4_1: Conv2d,
    conv4_2: Conv2d,
    conv4_3: Conv2d,
    conv5_1: Conv2d,
    conv5_2: Conv2d,
    conv5_3: Conv2d,
    fc6: Conv2d,
    fc7: Conv2d,
    conv6_1: Conv2d,
    conv6_2: Conv2d,
    conv7_1: Conv2d,
    conv7_2: Conv2d,
    norm3_3: L2Norm,
    norm4_3: L2Norm,
    norm5_3: L2Norm,
    head3_3: Head,
    head4_3: Head,
    head5_3: Head,
    head_fc7: Head,
    head6_2: Head,
    head7_2: Head,
}

impl S3fdModel {
    pub fn new(vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            conv1_1: conv(&vb, "conv1_1", 3, 64, 3, 1, 1, 1)?,
            conv1_2: conv(&vb, "conv1_2", 64, 64, 3, 1, 1, 1)?,
            conv2_1: conv(&vb, "conv2_1", 64, 128, 3, 1, 1, 1)?,
            conv2_2: conv(&vb, "conv2_2", 128, 128, 3, 1, 1, 1)?,
            conv3_1: conv(&vb, "conv3_1", 128, 256, 3, 1, 1, 1)?,
            conv3_2: conv(&vb, "conv3_2", 256, 256, 3, 1, 1, 1)?,
            conv3_3: conv(&vb, "conv3_3", 256, 256, 3, 1, 1, 1)?,
            conv4_1: conv(&vb, "conv4_1", 256, 512, 3, 1, 1, 1)?,
            conv4_2: conv(&vb, "conv4_2", 512, 512, 3, 1, 1, 1)?,
            conv4_3: conv(&vb, "conv4_3", 512, 512, 3, 1, 1, 1)?,
            conv5_1: conv(&vb, "conv5_1", 512, 512, 3, 1, 1, 1)?,
            conv5_2: conv(&vb, "conv5_2", 512, 512, 3, 1, 1, 1)?,
            conv5_3: conv(&vb, "conv5_3", 512, 512, 3, 1, 1, 1)?,
            fc6: conv(&vb, "fc6", 512, 1024, 3, 1, 3, 3)?,
            fc7: conv(&vb, "fc7", 1024, 1024, 1, 1, 0, 1)?,
            conv6_1: conv(&vb, "conv6_1", 1024, 256, 1, 1, 0, 1)?,
            conv6_2: conv(&vb, "conv6_2", 256, 512, 3, 2, 1, 1)?,
            conv7_1: conv(&vb, "conv7_1", 512, 128, 1, 1, 0, 1)?,
            conv7_2: conv(&vb, "conv7_2", 128, 256, 3, 2, 1, 1)?,
            norm3_3: L2Norm::new(256, &vb, "conv3_3_norm")?,
            norm4_3: L2Norm::new(512, &vb, "conv4_3_norm")?,
            norm5_3: L2Norm::new(512, &vb, "conv5_3_norm")?,
            head3_3: Head::new(&vb, "conv3_3_norm", 256, 4)?,
            head4_3: Head::new(&vb, "conv4_3_norm", 512, 2)?,
            head5_3: Head::new(&vb, "conv5_3_norm", 512, 2)?,
            head_fc7: Head::new(&vb, "fc7", 1024, 2)?,
            head6_2: Head::new(&vb, "conv6_2", 512, 2)?,
            head7_2: Head::new(&vb, "conv7_2", 256, 2)?,
        })
    }

    /// Input `[1, 3, H, W]` (BGR, mean-subtracted). Returns the 6 detection scales
    /// with strides [4, 8, 16, 32, 64, 128]; conf is pre-softmax, 2-channel.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Vec<ScaleOutput>> {
        let h = self.conv1_1.forward(x)?.relu()?;
        let h = self.conv1_2.forward(&h)?.relu()?.max_pool2d(2)?;
        let h = self.conv2_1.forward(&h)?.relu()?;
        let h = self.conv2_2.forward(&h)?.relu()?.max_pool2d(2)?;
        let h = self.conv3_1.forward(&h)?.relu()?;
        let h = self.conv3_2.forward(&h)?.relu()?;
        let f3 = self.conv3_3.forward(&h)?.relu()?;
        let h = f3.max_pool2d(2)?;
        let h = self.conv4_1.forward(&h)?.relu()?;
        let h = self.conv4_2.forward(&h)?.relu()?;
        let f4 = self.conv4_3.forward(&h)?.relu()?;
        let h = f4.max_pool2d(2)?;
        let h = self.conv5_1.forward(&h)?.relu()?;
        let h = self.conv5_2.forward(&h)?.relu()?;
        let f5 = self.conv5_3.forward(&h)?.relu()?;
        let h = f5.max_pool2d(2)?;
        let h = self.fc6.forward(&h)?.relu()?;
        let ffc7 = self.fc7.forward(&h)?.relu()?;
        let h = self.conv6_1.forward(&ffc7)?.relu()?;
        let f6 = self.conv6_2.forward(&h)?.relu()?;
        let h = self.conv7_1.forward(&f6)?.relu()?;
        let f7 = self.conv7_2.forward(&h)?.relu()?;

        let f3 = self.norm3_3.forward(&f3)?;
        let f4 = self.norm4_3.forward(&f4)?;
        let f5 = self.norm5_3.forward(&f5)?;

        // conv3_3 head: max-out background. 4 conf channels -> max of the first 3
        // (background) concatenated with the face channel.
        let conf3_raw = self.head3_3.conf.forward(&f3)?;
        let parts = conf3_raw.chunk(4, 1)?;
        let bg = parts[0].maximum(&parts[1])?.maximum(&parts[2])?;
        let conf3 = Tensor::cat(&[&bg, &parts[3]], 1)?;
        let loc3 = self.head3_3.loc.forward(&f3)?;

        Ok(vec![
            ScaleOutput {
                conf: conf3,
                loc: loc3,
                stride: 4,
            },
            ScaleOutput {
                conf: self.head4_3.conf.forward(&f4)?,
                loc: self.head4_3.loc.forward(&f4)?,
                stride: 8,
            },
            ScaleOutput {
                conf: self.head5_3.conf.forward(&f5)?,
                loc: self.head5_3.loc.forward(&f5)?,
                stride: 16,
            },
            ScaleOutput {
                conf: self.head_fc7.conf.forward(&ffc7)?,
                loc: self.head_fc7.loc.forward(&ffc7)?,
                stride: 32,
            },
            ScaleOutput {
                conf: self.head6_2.conf.forward(&f6)?,
                loc: self.head6_2.loc.forward(&f6)?,
                stride: 64,
            },
            ScaleOutput {
                conf: self.head7_2.conf.forward(&f7)?,
                loc: self.head7_2.loc.forward(&f7)?,
                stride: 128,
            },
        ])
    }
}
