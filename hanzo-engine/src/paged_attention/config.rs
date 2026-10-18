use super::attention_backend::{
    AttentionBackend, AttentionBackendKind, AttentionLayerSpec, FlashInferAttentionBackend,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCacheLayout {
    Standard,
    StandardNoFlashInfer,
    FlashInferHnd,
    Mla {
        kv_lora_rank: usize,
        kpe_head_dim: usize,
    },
}

fn flashinfer_supported_for_model<M: ModelConfigLike + ?Sized>(config: &M) -> bool {
    let backend = FlashInferAttentionBackend;
    (0..config.num_layers())
        .all(|layer_idx| backend.supports_layer(config.attention_layer_spec(layer_idx)))
}

fn select_attention_backend<M: ModelConfigLike + ?Sized>(config: &M) -> AttentionBackendKind {
    let backend = FlashInferAttentionBackend;
    if flashinfer_supported_for_model(config) {
        backend.kind()
    } else {
        AttentionBackendKind::Standard
    }
}

fn select_attention_backend_for_layer<M: ModelConfigLike + ?Sized>(
    config: &M,
    layer_idx: usize,
) -> AttentionBackendKind {
    let backend = FlashInferAttentionBackend;
    if backend.supports_layer(config.attention_layer_spec(layer_idx)) {
        backend.kind()
    } else {
        AttentionBackendKind::Standard
    }
}

fn select_kv_cache_layout<M: ModelConfigLike + ?Sized>(
    requested_layout: KvCacheLayout,
    config: &M,
) -> KvCacheLayout {
    match requested_layout {
        KvCacheLayout::Mla { .. } => requested_layout,
        KvCacheLayout::StandardNoFlashInfer => KvCacheLayout::Standard,
        KvCacheLayout::FlashInferHnd | KvCacheLayout::Standard => {
            match config.attention_backend_kind() {
                AttentionBackendKind::FlashInfer => KvCacheLayout::FlashInferHnd,
                AttentionBackendKind::Standard => KvCacheLayout::Standard,
            }
        }
    }
}

fn select_kv_cache_layout_for_layer<M: ModelConfigLike + ?Sized>(
    requested_layout: KvCacheLayout,
    config: &M,
    layer_idx: usize,
) -> KvCacheLayout {
    match requested_layout {
        KvCacheLayout::Mla { .. } => requested_layout,
        KvCacheLayout::StandardNoFlashInfer => KvCacheLayout::Standard,
        KvCacheLayout::FlashInferHnd | KvCacheLayout::Standard => {
            match config.attention_backend_kind_for_layer(layer_idx) {
                AttentionBackendKind::FlashInfer => KvCacheLayout::FlashInferHnd,
                AttentionBackendKind::Standard => KvCacheLayout::Standard,
            }
        }
    }
}

pub trait ModelConfigLike {
    fn max_seq_len(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn num_attn_heads(&self) -> usize;
    fn k_head_dim(&self) -> usize;
    fn v_head_dim(&self) -> usize;
    fn num_attn_heads_for_layer(&self, _layer_idx: usize) -> usize {
        self.num_attn_heads()
    }
    fn num_kv_heads_for_layer(&self, _layer_idx: usize) -> usize {
        self.num_kv_heads()
    }
    fn k_head_dim_for_layer(&self, _layer_idx: usize) -> usize {
        self.k_head_dim()
    }
    fn v_head_dim_for_layer(&self, _layer_idx: usize) -> usize {
        self.v_head_dim()
    }
    fn uses_own_kv_cache_for_layer(&self, _layer_idx: usize) -> bool {
        true
    }
    /// The decoder layers that hold a paged K/V pair, in cache order. The cache holds one K/V
    /// tensor per entry, shaped from that layer and placed with its
    /// [`kv_reader`](Self::kv_reader), and an attention layer reaches its pair by its position
    /// here. Hybrids override this with their attention layers: linear layers keep their state in
    /// the recurrent pool and never touch the paged cache. A proposer head appends its own layer
    /// after them.
    fn kv_layers(&self) -> Vec<usize> {
        (0..self.num_layers()).collect()
    }
    /// The decoder layer that reads cache entry `layer_idx`, whose device the entry lives on. A
    /// decoder layer reads its own entry. An entry past the decoder depth -- a proposer head -- has
    /// no decoder layer and lives on the base device; a config that appends an entry some decoder
    /// layer reads, such as a per-layer side cache, names that layer here.
    fn kv_reader(&self, layer_idx: usize) -> Option<usize> {
        (layer_idx < self.num_layers()).then_some(layer_idx)
    }
    fn attention_layer_spec(&self, layer_idx: usize) -> AttentionLayerSpec {
        AttentionLayerSpec {
            q_heads: self.num_attn_heads_for_layer(layer_idx),
            kv_heads: self.num_kv_heads_for_layer(layer_idx),
            k_head_dim: self.k_head_dim_for_layer(layer_idx),
            v_head_dim: self.v_head_dim_for_layer(layer_idx),
        }
    }
    fn attention_backend_kind(&self) -> AttentionBackendKind {
        select_attention_backend(self)
    }
    fn attention_backend_kind_for_layer(&self, layer_idx: usize) -> AttentionBackendKind {
        select_attention_backend_for_layer(self, layer_idx)
    }
    fn kv_cache_layout(&self) -> KvCacheLayout {
        select_kv_cache_layout(KvCacheLayout::Standard, self)
    }
    fn kv_cache_layout_for_layer(&self, layer_idx: usize) -> KvCacheLayout {
        select_kv_cache_layout_for_layer(KvCacheLayout::Standard, self, layer_idx)
    }
    /// Elements one token takes in cache entry `layer_idx`, from that layer's own spec: K and V
    /// each at the wider head dim, or the latent row of an MLA layout.
    fn kv_cache_elements_per_token_for_layer(&self, layer_idx: usize) -> usize {
        match self.kv_cache_layout_for_layer(layer_idx) {
            KvCacheLayout::Standard
            | KvCacheLayout::StandardNoFlashInfer
            | KvCacheLayout::FlashInferHnd => {
                let spec = self.attention_layer_spec(layer_idx);
                2 * spec.kv_heads * spec.k_head_dim.max(spec.v_head_dim)
            }
            KvCacheLayout::Mla {
                kv_lora_rank,
                kpe_head_dim,
            } => kv_lora_rank + kpe_head_dim,
        }
    }
    /// Elements one token takes across the whole cache: every entry of
    /// [`kv_layers`](Self::kv_layers) at its own layer's size.
    fn kv_cache_elements_per_token(&self) -> usize {
        self.kv_layers()
            .into_iter()
            .map(|layer_idx| self.kv_cache_elements_per_token_for_layer(layer_idx))
            .sum()
    }
}

#[derive(Clone)]
pub struct ModelConfigMetadata {
    pub max_seq_len: usize,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_kv_heads: usize,
    pub num_attn_heads: usize,
    pub sliding_window: Option<usize>,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
    pub kv_cache_layout: KvCacheLayout,
}

impl ModelConfigLike for ModelConfigMetadata {
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn num_attn_heads(&self) -> usize {
        self.num_attn_heads
    }
    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    fn num_layers(&self) -> usize {
        self.num_layers
    }
    fn k_head_dim(&self) -> usize {
        self.k_head_dim
    }
    fn v_head_dim(&self) -> usize {
        self.v_head_dim
    }
    fn attention_backend_kind(&self) -> AttentionBackendKind {
        match self.kv_cache_layout {
            KvCacheLayout::Mla { .. } | KvCacheLayout::StandardNoFlashInfer => {
                AttentionBackendKind::Standard
            }
            KvCacheLayout::FlashInferHnd | KvCacheLayout::Standard => {
                select_attention_backend(self)
            }
        }
    }
    fn attention_backend_kind_for_layer(&self, layer_idx: usize) -> AttentionBackendKind {
        match self.kv_cache_layout {
            KvCacheLayout::Mla { .. } | KvCacheLayout::StandardNoFlashInfer => {
                AttentionBackendKind::Standard
            }
            KvCacheLayout::FlashInferHnd | KvCacheLayout::Standard => {
                select_attention_backend_for_layer(self, layer_idx)
            }
        }
    }
    fn kv_cache_layout(&self) -> KvCacheLayout {
        select_kv_cache_layout(self.kv_cache_layout, self)
    }
    fn kv_cache_layout_for_layer(&self, layer_idx: usize) -> KvCacheLayout {
        select_kv_cache_layout_for_layer(self.kv_cache_layout, self, layer_idx)
    }
}

/// zen6's QSA index in the paged cache: attention layer `l` of a `num_layers`-deep decoder keeps
/// its index entry at `num_layers + l`, one head `width` wide in K and in V, and reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexEntries {
    pub num_layers: usize,
    pub width: usize,
}

impl IndexEntries {
    /// The attention layer that reads cache entry `layer_idx`, when that entry is an index entry.
    pub fn reader(&self, layer_idx: usize) -> Option<usize> {
        layer_idx.checked_sub(self.num_layers)
    }

    /// The `attention` layers' entries, then each one's index entry.
    pub fn layers(&self, attention: &[usize]) -> Vec<usize> {
        attention
            .iter()
            .copied()
            .chain(attention.iter().map(|l| self.num_layers + l))
            .collect()
    }
}

/// A config whose model holds a paged K/V pair at only some of its decoder layers. Hybrids build
/// it from their layer schedule; everything the paged cache does -- sizing, allocation, the index
/// an attention layer reads -- follows [`ModelConfigLike::kv_layers`].
#[derive(Clone, Debug)]
pub struct KvLayers<C> {
    inner: C,
    layers: Vec<usize>,
    index: Option<IndexEntries>,
}

impl<C> KvLayers<C> {
    pub fn new(inner: C, layers: Vec<usize>) -> Self {
        Self {
            inner,
            layers,
            index: None,
        }
    }

    /// The `attention` layers' K/V and, after them, each one's QSA index entry.
    pub fn indexed(inner: C, attention: &[usize], index: IndexEntries) -> Self {
        Self {
            inner,
            layers: index.layers(attention),
            index: Some(index),
        }
    }

    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// The index entries' shape, when `layer_idx` is one of them.
    fn index_entry(&self, layer_idx: usize) -> Option<IndexEntries> {
        self.index.filter(|i| i.reader(layer_idx).is_some())
    }
}

impl<C: ModelConfigLike> ModelConfigLike for KvLayers<C> {
    fn kv_layers(&self) -> Vec<usize> {
        self.layers.clone()
    }
    fn kv_reader(&self, layer_idx: usize) -> Option<usize> {
        match self.index_entry(layer_idx) {
            Some(index) => index.reader(layer_idx),
            None => self.inner.kv_reader(layer_idx),
        }
    }
    fn max_seq_len(&self) -> usize {
        self.inner.max_seq_len()
    }
    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }
    fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }
    fn num_kv_heads(&self) -> usize {
        self.inner.num_kv_heads()
    }
    fn num_attn_heads(&self) -> usize {
        self.inner.num_attn_heads()
    }
    fn k_head_dim(&self) -> usize {
        self.inner.k_head_dim()
    }
    fn v_head_dim(&self) -> usize {
        self.inner.v_head_dim()
    }
    fn num_attn_heads_for_layer(&self, layer_idx: usize) -> usize {
        self.inner.num_attn_heads_for_layer(layer_idx)
    }
    fn num_kv_heads_for_layer(&self, layer_idx: usize) -> usize {
        match self.index_entry(layer_idx) {
            Some(_) => 1,
            None => self.inner.num_kv_heads_for_layer(layer_idx),
        }
    }
    fn k_head_dim_for_layer(&self, layer_idx: usize) -> usize {
        match self.index_entry(layer_idx) {
            Some(index) => index.width,
            None => self.inner.k_head_dim_for_layer(layer_idx),
        }
    }
    fn v_head_dim_for_layer(&self, layer_idx: usize) -> usize {
        match self.index_entry(layer_idx) {
            Some(index) => index.width,
            None => self.inner.v_head_dim_for_layer(layer_idx),
        }
    }
    fn uses_own_kv_cache_for_layer(&self, layer_idx: usize) -> bool {
        self.inner.uses_own_kv_cache_for_layer(layer_idx)
    }
    fn attention_backend_kind(&self) -> AttentionBackendKind {
        self.inner.attention_backend_kind()
    }
    // An index entry's backend, layout and size follow from its own shape, as for any config that
    // answers its per-layer shape itself.
    fn attention_backend_kind_for_layer(&self, layer_idx: usize) -> AttentionBackendKind {
        match self.index_entry(layer_idx) {
            Some(_) => select_attention_backend_for_layer(self, layer_idx),
            None => self.inner.attention_backend_kind_for_layer(layer_idx),
        }
    }
    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.inner.kv_cache_layout()
    }
    fn kv_cache_layout_for_layer(&self, layer_idx: usize) -> KvCacheLayout {
        match self.index_entry(layer_idx) {
            Some(_) => select_kv_cache_layout_for_layer(KvCacheLayout::Standard, self, layer_idx),
            None => self.inner.kv_cache_layout_for_layer(layer_idx),
        }
    }
    fn kv_cache_elements_per_token_for_layer(&self, layer_idx: usize) -> usize {
        match self.index_entry(layer_idx) {
            Some(index) => 2 * index.width,
            None => self.inner.kv_cache_elements_per_token_for_layer(layer_idx),
        }
    }
}
