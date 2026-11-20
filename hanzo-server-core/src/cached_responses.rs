//! ## Response caching functionality for the Responses API.
//!
//! Every entry belongs to the org that stored it (`tenant::org`), and every lookup names the
//! caller's org, so another org's response ID reads as absent.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::{Arc, RwLock};

use crate::openai::Message;
use crate::responses_types::ResponseResource;

/// Trait for caching responses. `owner` is the caller's org; `None` is a caller with no org.
pub trait ResponseCache: Send + Sync {
    /// Store a response object with the given ID
    fn store_response(
        &self,
        owner: Option<&str>,
        id: String,
        response: ResponseResource,
    ) -> Result<()>;

    /// Retrieve a response object by ID
    fn get_response(&self, owner: Option<&str>, id: &str) -> Result<Option<ResponseResource>>;

    /// Delete a response object by ID
    fn delete_response(&self, owner: Option<&str>, id: &str) -> Result<bool>;

    /// Store conversation history for a response
    fn store_conversation_history(
        &self,
        owner: Option<&str>,
        id: String,
        messages: Vec<Message>,
    ) -> Result<()>;

    /// Retrieve conversation history for a response
    fn get_conversation_history(
        &self,
        owner: Option<&str>,
        id: &str,
    ) -> Result<Option<Vec<Message>>>;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Key {
    owner: Option<String>,
    id: String,
}

impl Key {
    fn new(owner: Option<&str>, id: &str) -> Self {
        Self {
            owner: owner.map(str::to_owned),
            id: id.to_owned(),
        }
    }
}

/// In-memory implementation of ResponseCache
pub struct InMemoryResponseCache {
    responses: Arc<RwLock<HashMap<Key, ResponseResource>>>,
    conversation_histories: Arc<RwLock<HashMap<Key, Vec<Message>>>>,
}

impl InMemoryResponseCache {
    /// Create a new in-memory cache
    pub fn new() -> Self {
        Self {
            responses: Arc::new(RwLock::new(HashMap::new())),
            conversation_histories: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryResponseCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseCache for InMemoryResponseCache {
    fn store_response(
        &self,
        owner: Option<&str>,
        id: String,
        response: ResponseResource,
    ) -> Result<()> {
        let mut responses = self.responses.write().unwrap();
        responses.insert(Key::new(owner, &id), response);
        Ok(())
    }

    fn get_response(&self, owner: Option<&str>, id: &str) -> Result<Option<ResponseResource>> {
        let responses = self.responses.read().unwrap();
        Ok(responses.get(&Key::new(owner, id)).cloned())
    }

    fn delete_response(&self, owner: Option<&str>, id: &str) -> Result<bool> {
        // IMPORTANT: Lock ordering must be maintained to prevent deadlocks.
        // Order: responses -> conversation_histories
        // All methods that acquire multiple locks must follow this order.
        //
        // We acquire all locks before any modifications to ensure atomicity.
        // The locks are released in reverse order when dropped at end of scope.
        let mut responses = self.responses.write().unwrap();
        let mut histories = self.conversation_histories.write().unwrap();

        let key = Key::new(owner, id);
        let response_removed = responses.remove(&key).is_some();
        let history_removed = histories.remove(&key).is_some();

        Ok(response_removed || history_removed)
    }

    fn store_conversation_history(
        &self,
        owner: Option<&str>,
        id: String,
        messages: Vec<Message>,
    ) -> Result<()> {
        let mut histories = self.conversation_histories.write().unwrap();
        histories.insert(Key::new(owner, &id), messages);
        Ok(())
    }

    fn get_conversation_history(
        &self,
        owner: Option<&str>,
        id: &str,
    ) -> Result<Option<Vec<Message>>> {
        let histories = self.conversation_histories.read().unwrap();
        Ok(histories.get(&Key::new(owner, id)).cloned())
    }
}

/// Global response cache instance
pub static RESPONSE_CACHE: LazyLock<Arc<dyn ResponseCache>> =
    LazyLock::new(|| Arc::new(InMemoryResponseCache::new()));

/// Helper function to get the global cache instance
pub fn get_response_cache() -> Arc<dyn ResponseCache> {
    RESPONSE_CACHE.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses_types::{ItemStatus, OutputContent, OutputItem, ResponseStatus};

    #[test]
    fn test_in_memory_cache() {
        let cache = InMemoryResponseCache::new();

        // Create a test response
        let response =
            ResponseResource::new("test-id".to_string(), "test-model".to_string(), 1234567890)
                .with_status(ResponseStatus::Completed)
                .with_output(vec![OutputItem::message(
                    "msg-1".to_string(),
                    vec![OutputContent::text("Hello".to_string())],
                    ItemStatus::Completed,
                )]);

        // Store and retrieve
        cache
            .store_response(None, "test-id".to_string(), response.clone())
            .unwrap();
        let retrieved = cache.get_response(None, "test-id").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test-id");

        // Delete
        let deleted = cache.delete_response(None, "test-id").unwrap();
        assert!(deleted);
        let retrieved = cache.get_response(None, "test-id").unwrap();
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_conversation_history() {
        let cache = InMemoryResponseCache::new();

        let messages = vec![Message {
            content: Some(crate::openai::MessageContent::from_text(
                "Hello".to_string(),
            )),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        cache
            .store_conversation_history(None, "test-id".to_string(), messages.clone())
            .unwrap();

        let retrieved = cache.get_conversation_history(None, "test-id").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().len(), 1);
    }

    mod kv_isolation {
        use super::*;

        const ID: &str = "resp_shared";

        fn message(text: &str) -> Message {
            Message {
                content: Some(crate::openai::MessageContent::from_text(text.to_string())),
                role: "user".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }
        }

        fn acme_cache() -> InMemoryResponseCache {
            let cache = InMemoryResponseCache::new();
            let response = ResponseResource::new(ID.into(), "m".into(), 0);
            cache
                .store_response(Some("acme"), ID.into(), response)
                .unwrap();
            cache
                .store_conversation_history(Some("acme"), ID.into(), vec![message("acme's prompt")])
                .unwrap();
            cache
        }

        #[test]
        fn another_org_cannot_read_a_stored_response() {
            let cache = acme_cache();
            for caller in [Some("evil"), None, Some("ACME"), Some("acme ")] {
                assert!(
                    cache.get_response(caller, ID).unwrap().is_none(),
                    "{caller:?}"
                );
                assert!(
                    cache
                        .get_conversation_history(caller, ID)
                        .unwrap()
                        .is_none(),
                    "{caller:?}"
                );
            }
            assert!(cache.get_response(Some("acme"), ID).unwrap().is_some());
            assert!(cache
                .get_conversation_history(Some("acme"), ID)
                .unwrap()
                .is_some());
        }

        #[test]
        fn another_org_cannot_delete_a_stored_response() {
            let cache = acme_cache();
            assert!(!cache.delete_response(Some("evil"), ID).unwrap());
            assert!(!cache.delete_response(None, ID).unwrap());
            assert!(cache.get_response(Some("acme"), ID).unwrap().is_some());
            assert!(cache.delete_response(Some("acme"), ID).unwrap());
            assert!(cache.get_response(Some("acme"), ID).unwrap().is_none());
        }

        #[test]
        fn the_same_id_under_two_orgs_stays_two_entries() {
            let cache = acme_cache();
            cache
                .store_conversation_history(Some("evil"), ID.into(), vec![message("evil's prompt")])
                .unwrap();
            let read = |org| {
                let history = cache
                    .get_conversation_history(Some(org), ID)
                    .unwrap()
                    .unwrap();
                serde_json::to_string(&history).unwrap()
            };
            assert!(read("acme").contains("acme's prompt"));
            assert!(read("evil").contains("evil's prompt"));
            assert!(!read("acme").contains("evil's prompt"));
        }
    }
}
