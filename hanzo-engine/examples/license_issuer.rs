//! License issuer: mint a Hanzo-signed engine license token.
//!
//! This is how Hanzo issues per-app tokens. It loads the PRIVATE Ed25519 signing key from a file or
//! env (NEVER hardcoded), constructs the claims, signs them, and prints the compact wire token to
//! stdout. The verifying half is `hanzo_engine::verify_license` / `load_and_verify`.
//!
//! Run:
//!   # from a base64url (no-pad) 32-byte seed in a file (e.g. the gitignored dev key):
//!   cargo run -p hanzo-engine --example license_issuer -- \
//!       --signing-key-file license-dev-key/dev_signing_key.b64 \
//!       --app hanzo --holder desktop-prod --days 365 --features inference,embeddings
//!
//!   # or via env (CI / KMS material exported to env):
//!   HANZO_LICENSE_SIGNING_KEY=<base64url-seed> cargo run -p hanzo-engine --example license_issuer -- \
//!       --app zoo --holder zoo-desktop-prod --days 365
//!
//! The signing key is accepted as a base64url (no padding) OR standard-base64 encoding of the raw
//! 32-byte Ed25519 seed.

use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use base64::Engine as _;
use clap::Parser;
use ed25519_dalek::SigningKey;
use hanzo_engine::license::{encode_license, verify_license, License, LICENSE_SCHEMA_VERSION};
use std::time::{SystemTime, UNIX_EPOCH};

/// Env var carrying the private signing seed (base64). File flag takes precedence if both are set.
const SIGNING_KEY_ENV: &str = "HANZO_LICENSE_SIGNING_KEY";

#[derive(Parser, Debug)]
#[command(
    name = "license_issuer",
    about = "Mint a Hanzo-signed engine license token (Ed25519)."
)]
struct Args {
    /// Path to a file containing the base64 private Ed25519 seed (32 bytes). Overrides the env var.
    #[arg(long)]
    signing_key_file: Option<String>,

    /// App this token is scoped to.
    #[arg(long, value_parser = ["hanzo", "lux", "zoo", "hanzo-dev"])]
    app: String,

    /// Who the entitlement is issued to (build channel / org / user).
    #[arg(long)]
    holder: String,

    /// Days the token is valid from now (must be >= 1).
    #[arg(long, default_value_t = 365, value_parser = clap::value_parser!(i64).range(1..))]
    days: i64,

    /// Comma-separated capability features (optional).
    #[arg(long, value_delimiter = ',', default_value = "")]
    features: Vec<String>,

    /// Override issued-at (unix seconds). Defaults to now. For reproducible/testing use.
    #[arg(long)]
    iat: Option<i64>,
}

fn load_signing_key(args: &Args) -> Result<SigningKey, String> {
    let raw = match &args.signing_key_file {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read signing key file {path}: {e}"))?,
        None => std::env::var(SIGNING_KEY_ENV).map_err(|_| {
            format!("no signing key: pass --signing-key-file or set {SIGNING_KEY_ENV}")
        })?,
    };
    let raw = raw.trim();
    let seed = URL_SAFE_NO_PAD
        .decode(raw)
        .or_else(|_| STANDARD.decode(raw))
        .map_err(|_| "signing key is not valid base64url/base64".to_string())?;
    let seed: [u8; 32] = seed
        .try_into()
        .map_err(|_| "signing key must decode to exactly 32 bytes".to_string())?;
    Ok(SigningKey::from_bytes(&seed))
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn random_nonce() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let signing_key = load_signing_key(&args)?;

    let iat = args.iat.unwrap_or_else(now_unix);
    let exp = iat + args.days * 86_400;
    let features: Vec<String> = args
        .features
        .iter()
        .filter(|f| !f.is_empty())
        .cloned()
        .collect();

    let license = License {
        v: LICENSE_SCHEMA_VERSION,
        app_id: args.app.clone(),
        holder: args.holder.clone(),
        iat,
        exp,
        features,
        nonce: random_nonce(),
    };

    let token = encode_license(&license, &signing_key);

    // Self-check: a token we just signed must verify against our own public key + claimed app.
    let pubkey = signing_key.verifying_key().to_bytes();
    verify_license(&token, &pubkey, &args.app, iat)
        .map_err(|e| format!("internal error: freshly-minted token failed to verify: {e}"))?;

    eprintln!(
        "issued license: app={} holder={} iat={} exp={} ({} days) features={:?}",
        license.app_id, license.holder, license.iat, license.exp, args.days, license.features
    );
    println!("{token}");
    Ok(())
}
