use sha2::{Sha256, Digest};
use hex;

pub fn hash_public_key(pubkey_hex: &str) -> String {
    let mut hasher = Sha256::new();
    let bytes = hex::decode(pubkey_hex).unwrap_or_default();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}
