use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// Represents a fixed-size memory page for LoRA weights.
pub struct LoraPage {
    pub id: usize,
    pub offset: usize,
}

/// Manages the pool of GPU memory pages allocated for LoRA adapters.
pub struct LoraPager {
    pub total_pages: usize,
    pub page_size: usize,
    available_pages: RwLock<Vec<usize>>,
    // Maps adapter_id to its allocated pages
    pub adapter_mappings: RwLock<HashMap<String, Vec<LoraPage>>>,
}

impl LoraPager {
    pub fn new(total_memory_mb: usize, page_size_kb: usize) -> Self {
        let page_size = page_size_kb * 1024;
        let total_pages = (total_memory_mb * 1024 * 1024) / page_size;
        let available_pages = (0..total_pages).rev().collect();

        Self {
            total_pages,
            page_size,
            available_pages: RwLock::new(available_pages),
            adapter_mappings: RwLock::new(HashMap::new()),
        }
    }

    pub fn allocate_for_adapter(&self, adapter_id: &str, required_size: usize) -> Result<(), String> {
        let pages_needed = (required_size + self.page_size - 1) / self.page_size;
        let mut available = self.available_pages.write();
        
        if available.len() < pages_needed {
            return Err("Out of GPU memory for LoRA pages".to_string());
        }

        let mut allocated = Vec::new();
        for _ in 0..pages_needed {
            let page_id = available.pop().unwrap();
            allocated.push(LoraPage {
                id: page_id,
                offset: page_id * self.page_size,
            });
        }

        self.adapter_mappings.write().insert(adapter_id.to_string(), allocated);
        Ok(())
    }

    pub fn deallocate_adapter(&self, adapter_id: &str) {
        if let Some(pages) = self.adapter_mappings.write().remove(adapter_id) {
            let mut available = self.available_pages.write();
            for page in pages {
                available.push(page.id);
            }
        }
    }

    pub fn get_pages(&self, adapter_id: &str) -> Option<Vec<usize>> {
        self.adapter_mappings.read().get(adapter_id).map(|pages| {
            pages.iter().map(|p| p.offset).collect()
        })
    }
}
