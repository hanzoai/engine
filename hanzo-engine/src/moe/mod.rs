mod experts;
#[cfg(feature = "cuda")]
pub(crate) mod grouped;

use hanzo_quant::Shard;

pub use experts::{MoEExperts, MoEExpertsConfig};

pub fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard::Simple {
        dim,
        rank,
        world_size,
    }
}
