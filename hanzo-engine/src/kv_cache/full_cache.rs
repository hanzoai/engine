use std::sync::{Arc, Mutex, MutexGuard};

use hanzo_ml::Tensor;

use super::{Cache, HybridCache, NormalCache};

pub type LayerCaches = Vec<Option<(Tensor, Tensor)>>;

#[derive(Debug, Clone)]
pub enum EitherCache {
    Normal(Arc<Mutex<NormalCache>>),
    Full(Cache),
    Hybrid(Arc<Mutex<HybridCache>>),
}

impl EitherCache {
    /// Panics otherwise!
    pub fn full(&self) -> &Cache {
        match self {
            Self::Full(full) => full,
            Self::Normal(_) => panic!("Got normal cache, expected full cache."),
            Self::Hybrid(_) => panic!("Got hybrid cache, expected full cache."),
        }
    }

    /// Panics otherwise!
    pub fn normal(&self) -> MutexGuard<'_, NormalCache> {
        match self {
            Self::Normal(normal) => normal.lock().unwrap(),
            Self::Full(_) => panic!("Got full cache, expected normal cache."),
            Self::Hybrid(_) => panic!("Got hybrid cache, expected normal cache."),
        }
    }

    /// Clone the shared handle to the normal cache (interior-mutable), for callers
    /// that must snapshot/roll it back while the model is borrowed elsewhere — e.g.
    /// `NormalSpeculativeCacheAccess`. `None` for non-normal caches.
    pub fn normal_arc(&self) -> Option<Arc<Mutex<NormalCache>>> {
        match self {
            Self::Normal(normal) => Some(normal.clone()),
            _ => None,
        }
    }

    /// Panics otherwise!
    pub fn hybrid(&self) -> MutexGuard<'_, HybridCache> {
        match self {
            Self::Hybrid(hybrid) => hybrid.lock().unwrap(),
            Self::Normal(_) => panic!("Got normal cache, expected hybrid cache."),
            Self::Full(_) => panic!("Got full cache, expected hybrid cache."),
        }
    }

    pub fn is_hybrid(&self) -> bool {
        matches!(self, Self::Hybrid(_))
    }
}
