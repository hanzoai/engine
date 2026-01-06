//! Hanzo Engine DSO Integration
//! 
//! High-performance semantic experience retrieval for ANY LLM using Candle.
//! 
//! # Features
//! 
//! - **Fast Retrieval**: Candle-based GPU-accelerated similarity search
//! - **BitDelta Decompression**: On-the-fly decompression of compressed experiences
//! - **Context Injection**: Automatic prepending of relevant experiences
//! - **Batch Processing**: Efficient handling of multiple queries
//! - **Cross-Model Support**: Works with ANY LLM via embedding alignment

pub mod retrieval;
pub mod context;
pub mod kernels;

use hanzo_experience_registry::{Experience, ExperienceRegistry, CompressedEmbedding, CANONICAL_DIM};
use hanzo_dso_aggregator::AggregatedExperience;
use candle_core::{Tensor, Device, DType};
use anyhow::{Result, Context as _};
use std::sync::Arc;
use parking_lot::RwLock;

/// DSO-enhanced inference engine
pub struct DSOEngine {
    /// Experience registry (local + network)
    registry: Arc<dyn ExperienceRegistry>,
    
    /// Embedding cache (for fast retrieval)
    embedding_cache: Arc<RwLock<EmbeddingCache>>,
    
    /// Device for computation (CPU/CUDA/Metal)
    device: Device,
    
    /// Configuration
    config: DSOConfig,
}

/// Configuration for DSO engine
#[derive(Debug, Clone)]
pub struct DSOConfig {
    /// Number of experiences to retrieve per query
    pub top_k: usize,
    
    /// Minimum confidence threshold
    pub min_confidence: f32,
    
    /// Enable GPU acceleration
    pub use_gpu: bool,
    
    /// Batch size for retrieval
    pub batch_size: usize,
    
    /// Domain filter (e.g., "code.rust", "math")
    pub domain: Option<String>,
}

impl Default for DSOConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_confidence: 0.5,
            use_gpu: true,
            batch_size: 32,
            domain: None,
        }
    }
}

/// Embedding cache for fast retrieval
pub struct EmbeddingCache {
    /// Experience IDs
    ids: Vec<uuid::Uuid>,
    
    /// Embeddings tensor [N, CANONICAL_DIM]
    embeddings: Option<Tensor>,
    
    /// Compressed embeddings (for network sync)
    compressed: Vec<CompressedEmbedding>,
    
    /// Last update timestamp
    last_update: std::time::Instant,
}

impl EmbeddingCache {
    pub fn new() -> Self {
        Self {
            ids: Vec::new(),
            embeddings: None,
            compressed: Vec::new(),
            last_update: std::time::Instant::now(),
        }
    }
    
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    
    pub fn len(&self) -> usize {
        self.ids.len()
    }
}

impl DSOEngine {
    /// Create new DSO engine
    pub fn new(
        registry: Arc<dyn ExperienceRegistry>,
        config: DSOConfig,
    ) -> Result<Self> {
        let device = if config.use_gpu {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).unwrap_or(Device::Cpu)
            }
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).unwrap_or(Device::Cpu)
            }
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            {
                Device::Cpu
            }
        } else {
            Device::Cpu
        };
        
        Ok(Self {
            registry,
            embedding_cache: Arc::new(RwLock::new(EmbeddingCache::new())),
            device,
            config,
        })
    }
    
    /// Retrieve relevant experiences for a query
    pub async fn retrieve(
        &self,
        query_embedding: &[f32],
    ) -> Result<Vec<Experience>> {
        // Ensure cache is populated
        self.ensure_cache_updated().await?;
        
        // Convert query to Candle tensor
        let query_tensor = Tensor::from_vec(
            query_embedding.to_vec(),
            (1, CANONICAL_DIM),
            &self.device
        )?;
        
        // Compute similarities
        let cache = self.embedding_cache.read();
        
        if cache.is_empty() {
            return Ok(Vec::new());
        }
        
        let embeddings = cache.embeddings.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Embeddings not cached"))?;
        
        // Cosine similarity: (query · emb) / (||query|| * ||emb||)
        let similarities = self.compute_similarities(&query_tensor, embeddings)?;
        
        // Get top-k indices
        let top_k_indices = self.topk_indices(&similarities, self.config.top_k)?;
        
        // Retrieve experiences
        let mut results = Vec::new();
        for idx in top_k_indices {
            if idx < cache.ids.len() {
                let exp_id = cache.ids[idx];
                if let Ok(Some(exp)) = self.registry.get(&exp_id).await {
                    if exp.confidence >= self.config.min_confidence {
                        results.push(exp);
                    }
                }
            }
        }
        
        drop(cache);  // Release lock
        
        Ok(results)
    }
    
    /// Format experiences as context prompt
    pub fn format_context(&self, experiences: &[Experience]) -> String {
        if experiences.is_empty() {
            return String::new();
        }
        
        let mut context = String::from("# Learned Experiences\n\n");
        
        for (i, exp) in experiences.iter().enumerate() {
            context.push_str(&format!(
                "[E{}] {}\n",
                i + 1,
                exp.text
            ));
        }
        
        context.push('\n');
        context
    }
    
    /// Inject experiences into system prompt
    pub async fn inject_context(
        &self,
        query_embedding: &[f32],
        original_prompt: &str,
    ) -> Result<String> {
        let experiences = self.retrieve(query_embedding).await?;
        
        if experiences.is_empty() {
            return Ok(original_prompt.to_string());
        }
        
        let context = self.format_context(&experiences);
        
        // Prepend experiences to original prompt
        Ok(format!("{}{}", context, original_prompt))
    }
    
    /// Batch retrieval for multiple queries
    pub async fn retrieve_batch(
        &self,
        query_embeddings: &[Vec<f32>],
    ) -> Result<Vec<Vec<Experience>>> {
        use rayon::prelude::*;
        
        // Process in parallel using rayon
        let results: Vec<_> = query_embeddings
            .par_iter()
            .map(|emb| {
                // Use blocking runtime for async
                tokio::runtime::Runtime::new()
                    .unwrap()
                    .block_on(self.retrieve(emb))
                    .unwrap_or_default()
            })
            .collect();
        
        Ok(results)
    }
    
    /// Update cache from registry
    async fn ensure_cache_updated(&self) -> Result<()> {
        let cache = self.embedding_cache.read();
        
        // Check if cache needs update (every 60 seconds)
        if !cache.is_empty() && cache.last_update.elapsed().as_secs() < 60 {
            return Ok(());
        }
        
        drop(cache);  // Release read lock
        
        // Acquire write lock and update
        let mut cache = self.embedding_cache.write();
        
        // Double-check after acquiring write lock
        if !cache.is_empty() && cache.last_update.elapsed().as_secs() < 60 {
            return Ok(());
        }
        
        // Fetch all experiences from registry
        let experiences = self.registry.list_all().await?;
        
        // Filter by domain if specified
        let filtered: Vec<_> = if let Some(ref domain) = self.config.domain {
            experiences.into_iter()
                .filter(|exp| exp.domain.starts_with(domain))
                .collect()
        } else {
            experiences
        };
        
        // Extract IDs and embeddings
        cache.ids.clear();
        cache.compressed.clear();
        
        let mut embeddings_vec = Vec::new();
        
        for exp in filtered {
            cache.ids.push(exp.id);
            
            // Decompress embedding
            if let Some(ref compressed) = exp.compressed_embedding {
                cache.compressed.push(compressed.clone());
                
                let decompressed = self.decompress_embedding(compressed)?;
                embeddings_vec.extend(decompressed);
            }
        }
        
        // Create embeddings tensor
        if !embeddings_vec.is_empty() {
            let n = cache.ids.len();
            cache.embeddings = Some(Tensor::from_vec(
                embeddings_vec,
                (n, CANONICAL_DIM),
                &self.device
            )?);
        }
        
        cache.last_update = std::time::Instant::now();
        
        tracing::info!(
            "Updated embedding cache: {} experiences",
            cache.ids.len()
        );
        
        Ok(())
    }
    
    /// Decompress BitDelta embedding
    fn decompress_embedding(&self, compressed: &CompressedEmbedding) -> Result<Vec<f32>> {
        Ok(compressed
            .signs
            .iter()
            .map(|&sign| sign as f32 * compressed.scale)
            .collect())
    }
    
    /// Compute cosine similarities
    fn compute_similarities(&self, query: &Tensor, embeddings: &Tensor) -> Result<Tensor> {
        // Normalize query
        let query_norm = query.sqr()?.sum_all()?.sqrt()?;
        let query_normalized = query.broadcast_div(&query_norm)?;
        
        // Normalize embeddings (per row)
        let emb_norms = embeddings.sqr()?.sum_keepdim(1)?.sqrt()?;
        let emb_normalized = embeddings.broadcast_div(&emb_norms)?;
        
        // Compute dot products (cosine similarity)
        let similarities = emb_normalized.matmul(&query_normalized.t()?)?;
        
        Ok(similarities)
    }
    
    /// Get top-k indices from similarities
    fn topk_indices(&self, similarities: &Tensor, k: usize) -> Result<Vec<usize>> {
        // Convert to Vec
        let sim_vec: Vec<f32> = similarities.to_vec1()?;
        
        // Get top-k indices
        let mut indexed: Vec<_> = sim_vec.iter().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        
        Ok(indexed.into_iter().take(k).map(|(i, _)| i).collect())
    }
}

/// DSO-enhanced prompt builder
pub struct DSOPromptBuilder {
    engine: Arc<DSOEngine>,
}

impl DSOPromptBuilder {
    pub fn new(engine: Arc<DSOEngine>) -> Self {
        Self { engine }
    }
    
    /// Build prompt with DSO context injection
    pub async fn build(
        &self,
        query: &str,
        query_embedding: &[f32],
        system_prompt: Option<&str>,
    ) -> Result<String> {
        let experiences = self.engine.retrieve(query_embedding).await?;
        
        let mut prompt = String::new();
        
        // Add system prompt if provided
        if let Some(sys) = system_prompt {
            prompt.push_str(&format!("System: {}\n\n", sys));
        }
        
        // Add experiences
        if !experiences.is_empty() {
            prompt.push_str("# Learned Experiences\n\n");
            for (i, exp) in experiences.iter().enumerate() {
                prompt.push_str(&format!(
                    "[E{}] {} (confidence: {:.2})\n",
                    i + 1,
                    exp.text,
                    exp.confidence
                ));
            }
            prompt.push_str("\n---\n\n");
        }
        
        // Add user query
        prompt.push_str(&format!("User: {}\n\nAssistant:", query));
        
        Ok(prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_experience_registry::LocalExperienceRegistry;
    use std::collections::HashMap;
    
    #[tokio::test]
    async fn test_dso_engine() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        
        let registry = Arc::new(
            LocalExperienceRegistry::new(db_path.to_str().unwrap()).unwrap()
        ) as Arc<dyn ExperienceRegistry>;
        
        let config = DSOConfig::default();
        let engine = DSOEngine::new(registry, config).unwrap();
        
        // Test retrieval with dummy query
        let query_emb = vec![0.1; CANONICAL_DIM];
        let results = engine.retrieve(&query_emb).await.unwrap();
        
        // Should return empty (no experiences in registry)
        assert!(results.is_empty());
    }
    
    #[test]
    fn test_format_context() {
        use chrono::Utc;
        use uuid::Uuid;
        use hanzo_experience_registry::VoteMetadata;
        
        let registry = Arc::new(
            LocalExperienceRegistry::new(":memory:").unwrap()
        ) as Arc<dyn ExperienceRegistry>;
        
        let config = DSOConfig::default();
        let engine = DSOEngine::new(registry, config).unwrap();
        
        let experiences = vec![
            Experience {
                id: Uuid::new_v4(),
                text: "Test experience 1".to_string(),
                embedding: None,
                compressed_embedding: None,
                confidence: 0.9,
                domain: "test".to_string(),
                source_model: "test-model".to_string(),
                original_dim: CANONICAL_DIM,
                creator_node: "node1".to_string(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                usage_count: 0,
                votes: VoteMetadata::default(),
                metadata: HashMap::new(),
            }
        ];
        
        let context = engine.format_context(&experiences);
        
        assert!(context.contains("# Learned Experiences"));
        assert!(context.contains("Test experience 1"));
    }
}
