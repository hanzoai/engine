use std::collections::HashMap;
use std::sync::Arc;
use hashbrown::HashSet;
use parking_lot::RwLock;
use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::Automaton;

/// Represents a compiled grammar state machine.
pub struct CompiledGrammar {
    pub dfa: DFA<Vec<u32>>,
    pub vocab_size: usize,
}

/// A cache for bitmasks to achieve near-zero overhead during generation.
/// Maps (FSM State, Context) -> Allowed Token Bitmask.
pub struct BitmaskCache {
    cache: RwLock<HashMap<u32, Arc<Vec<u8>>>>,
}

impl BitmaskCache {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn get_mask(&self, state: u32) -> Option<Arc<Vec<u8>>> {
        let read = self.cache.read();
        read.get(&state).cloned()
    }

    pub fn insert_mask(&self, state: u32, mask: Vec<u8>) {
        let mut write = self.cache.write();
        write.insert(state, Arc::new(mask));
    }
}

/// The XGrammar-2 Logit Processor.
/// Accelerates structured text output in generative Zen models.
pub struct XGrammarLogitProcessor {
    grammar: CompiledGrammar,
    cache: BitmaskCache,
    current_state: u32,
    vocab: Arc<Vec<String>>,
}

impl XGrammarLogitProcessor {
    pub fn new(json_schema: &str, vocab: Arc<Vec<String>>) -> Self {
        // In a full implementation, this would compile JSON Schema to a Regex/DFA.
        // For the prototype, we assume a simplified DFA construction.
        let dfa = DFA::new(json_schema).expect("Failed to compile grammar DFA");
        
        Self {
            grammar: CompiledGrammar {
                dfa,
                vocab_size: vocab.len(),
            },
            cache: BitmaskCache::new(),
            current_state: 0, // Initial DFA state
            vocab,
        }
    }

    /// Applies the grammar constraint to the logits.
    /// Near-zero overhead is achieved by retrieving the bitmask from the cache.
    pub fn apply(&mut self, logits: &mut [f32]) {
        let state = self.current_state;
        
        let mask = if let Some(cached_mask) = self.cache.get_mask(state) {
            cached_mask
        } else {
            let new_mask = self.compute_bitmask(state);
            self.cache.insert_mask(state, new_mask.clone());
            Arc::new(new_mask)
        };

        // Apply mask: set logits of invalid tokens to -infinity
        for (idx, logit) in logits.iter_mut().enumerate() {
            let byte_idx = idx / 8;
            let bit_idx = idx % 8;
            if (mask[byte_idx] & (1 << bit_idx)) == 0 {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    /// Computes which tokens are valid for the current FSM state.
    fn compute_bitmask(&self, state: u32) -> Vec<u8> {
        let mut mask = vec![0u8; (self.grammar.vocab_size + 7) / 8];
        
        for (token_id, token_str) in self.vocab.iter().enumerate() {
            let mut next_state = state;
            let mut valid = true;
            
            // Check if the token string is valid for the current DFA state
            for &byte in token_str.as_bytes() {
                next_state = self.grammar.dfa.next_state(next_state, byte);
                if self.grammar.dfa.is_dead_state(next_state) {
                    valid = false;
                    break;
                }
            }
            
            if valid {
                let byte_idx = token_id / 8;
                let bit_idx = token_id % 8;
                mask[byte_idx] |= 1 << bit_idx;
            }
        }
        mask
    }

    /// Updates the FSM state based on the selected token.
    pub fn update_state(&mut self, token_id: usize) {
        let token_str = &self.vocab[token_id];
        for &byte in token_str.as_bytes() {
            self.current_state = self.grammar.dfa.next_state(self.current_state, byte);
        }
    }
}
