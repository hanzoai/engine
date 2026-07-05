//! Engine-side license verification.
//!
//! The proprietary engine refuses to serve unless it is handed a valid, unexpired, correctly-scoped
//! Ed25519-signed license token at startup (see SECURE-ENGINE-DISTRIBUTION-DESIGN.md). This module is
//! the offline verification core: it decodes a compact token, checks the signature against an
//! embedded public key, and validates the claims (app, time window). It deliberately depends only on
//! `ed25519-dalek` + `base64` + `serde_json` so it builds and tests without any of the heavy
//! rocm/cuda/metal backends.
//!
//! Token wire format (a minimal JWT-ish token, no header):
//!   base64url(JSON(payload)) "." base64url(ed25519_sig[64])
//! The signature is computed over the ASCII bytes of the base64url-encoded payload (the substring
//! before the '.'), so verification never has to re-canonicalize JSON.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use ed25519_dalek::{Signature, Verifier, VerifyingKey, SIGNATURE_LENGTH};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Env var holding the raw token (highest priority source).
pub const LICENSE_TOKEN_ENV: &str = "HANZO_ENGINE_LICENSE_TOKEN";
/// Env var holding a path to a file whose contents are the token (fallback source).
pub const LICENSE_FILE_ENV: &str = "HANZO_ENGINE_LICENSE_FILE";

/// Current token schema version. `verify_license` rejects anything else.
pub const LICENSE_SCHEMA_VERSION: u8 = 1;

/// Clock skew tolerance for the `iat` (issued-at) check, in seconds. A token may legitimately have an
/// `iat` slightly in the future relative to this host's clock; allow a small window before rejecting.
pub const IAT_SKEW_SECS: i64 = 300;

/// Hanzo's Ed25519 PUBLIC verification key (32 raw bytes).
///
/// DEV KEY -- replace with prod pubkey from KMS for release. The matching private seed lives only in
/// the gitignored `license-dev-key/` for tests/issuer; it is NEVER committed. For a real release this
/// const must be swapped for the public key whose private half is held in the KMS / CI secret.
pub const HANZO_LICENSE_PUBKEY: [u8; 32] = [
    0x6e, 0x79, 0xb8, 0x50, 0x79, 0xfe, 0xbd, 0x9e, 0xfc, 0x35, 0xbf, 0x4d, 0x8e, 0x0a, 0x6e, 0x86,
    0x27, 0x03, 0x1b, 0x87, 0x2f, 0xc6, 0xb7, 0x61, 0xe2, 0x8a, 0xf9, 0x8c, 0xed, 0xe4, 0x1e, 0xf7,
];

/// The `app_id` this build is allowed to run, fixed at compile time via cargo features so a Zoo token
/// cannot run a Hanzo-built engine. If no `license-app-*` feature is set, defaults to the permissive
/// dev value `"hanzo-dev"` (release builds are expected to pick exactly one).
pub const EXPECTED_APP_ID: &str = expected_app_id();

const fn expected_app_id() -> &'static str {
    #[cfg(feature = "license-app-hanzo")]
    {
        "hanzo"
    }
    #[cfg(all(feature = "license-app-lux", not(feature = "license-app-hanzo")))]
    {
        "lux"
    }
    #[cfg(all(
        feature = "license-app-zoo",
        not(feature = "license-app-hanzo"),
        not(feature = "license-app-lux")
    ))]
    {
        "zoo"
    }
    #[cfg(not(any(
        feature = "license-app-hanzo",
        feature = "license-app-lux",
        feature = "license-app-zoo"
    )))]
    {
        "hanzo-dev"
    }
}

/// Signed license claims. The serialized form of this struct is exactly what gets signed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct License {
    /// Schema version.
    pub v: u8,
    /// One of "hanzo" | "lux" | "zoo" (or a dev value). Must equal the build's expected app.
    pub app_id: String,
    /// Who this entitlement was issued to (build channel / org / user).
    pub holder: String,
    /// Issued-at, unix seconds.
    pub iat: i64,
    /// Expiry, unix seconds.
    pub exp: i64,
    /// Capability scoping (optional, may be empty).
    #[serde(default)]
    pub features: Vec<String>,
    /// 128-bit random hex, for revocation-by-id / log correlation.
    pub nonce: String,
}

/// Distinct verification failure modes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LicenseError {
    /// No token was supplied via env or file.
    Missing,
    /// Token could not be parsed (bad base64, bad JSON, wrong segment count, wrong sig length, ...).
    Malformed,
    /// Ed25519 signature did not verify against the embedded public key.
    BadSig,
    /// `now >= exp`.
    Expired,
    /// `now < iat - skew` (token issued in the future; likely a clock problem or forgery).
    NotYetValid,
    /// Token's `app_id` does not match this build's expected app.
    WrongApp,
}

impl std::fmt::Display for LicenseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            LicenseError::Missing => "no license token supplied (set HANZO_ENGINE_LICENSE_TOKEN or HANZO_ENGINE_LICENSE_FILE)",
            LicenseError::Malformed => "license token is malformed",
            LicenseError::BadSig => "license token signature is invalid",
            LicenseError::Expired => "license token has expired",
            LicenseError::NotYetValid => "license token is not yet valid (issued-at is in the future)",
            LicenseError::WrongApp => "license token app_id does not match this build",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for LicenseError {}

/// Verify a token against an explicit public key, expected app, and clock.
///
/// Steps: split on '.', base64url-decode both halves, verify the Ed25519 signature over the *encoded*
/// payload bytes, JSON-decode the payload, then check schema version, app match, and time window.
pub fn verify_license(
    token: &str,
    pubkey: &[u8; 32],
    expected_app: &str,
    now_unix: i64,
) -> Result<License, LicenseError> {
    let token = token.trim();
    let (payload_b64, sig_b64) = token.split_once('.').ok_or(LicenseError::Malformed)?;
    if payload_b64.is_empty() || sig_b64.is_empty() || sig_b64.contains('.') {
        return Err(LicenseError::Malformed);
    }

    let sig_bytes = URL_SAFE_NO_PAD
        .decode(sig_b64)
        .map_err(|_| LicenseError::Malformed)?;
    let sig_arr: [u8; SIGNATURE_LENGTH] =
        sig_bytes.try_into().map_err(|_| LicenseError::Malformed)?;
    let signature = Signature::from_bytes(&sig_arr);

    let verifying_key = VerifyingKey::from_bytes(pubkey).map_err(|_| LicenseError::Malformed)?;
    verifying_key
        .verify(payload_b64.as_bytes(), &signature)
        .map_err(|_| LicenseError::BadSig)?;

    let payload_json = URL_SAFE_NO_PAD
        .decode(payload_b64)
        .map_err(|_| LicenseError::Malformed)?;
    let license: License =
        serde_json::from_slice(&payload_json).map_err(|_| LicenseError::Malformed)?;

    if license.v != LICENSE_SCHEMA_VERSION {
        return Err(LicenseError::Malformed);
    }
    if license.app_id != expected_app {
        return Err(LicenseError::WrongApp);
    }
    if now_unix >= license.exp {
        return Err(LicenseError::Expired);
    }
    if now_unix < license.iat - IAT_SKEW_SECS {
        return Err(LicenseError::NotYetValid);
    }

    Ok(license)
}

/// Sign claims with a private Ed25519 signing key, producing a wire token.
///
/// This is the issuer-side counterpart to `verify_license`. The engine itself never signs; this lives
/// here so the verify/sign pair is tested together and so the issuer binary can reuse it. The private
/// key must come from a file/env/KMS, never hardcoded.
pub fn encode_license(license: &License, signing_key: &ed25519_dalek::SigningKey) -> String {
    use ed25519_dalek::Signer as _;
    let payload_json = serde_json::to_vec(license).expect("License serializes to JSON");
    let payload_b64 = URL_SAFE_NO_PAD.encode(payload_json);
    let signature = signing_key.sign(payload_b64.as_bytes());
    let sig_b64 = URL_SAFE_NO_PAD.encode(signature.to_bytes());
    format!("{payload_b64}.{sig_b64}")
}

fn current_unix_time() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Read the token from `HANZO_ENGINE_LICENSE_TOKEN`, falling back to the file named by
/// `HANZO_ENGINE_LICENSE_FILE`, and verify it against the embedded pubkey + this build's expected app
/// + the current system time.
pub fn load_and_verify() -> Result<License, LicenseError> {
    let token = match std::env::var(LICENSE_TOKEN_ENV) {
        Ok(t) if !t.trim().is_empty() => t,
        _ => match std::env::var(LICENSE_FILE_ENV) {
            Ok(path) if !path.trim().is_empty() => {
                std::fs::read_to_string(path.trim()).map_err(|_| LicenseError::Missing)?
            }
            _ => return Err(LicenseError::Missing),
        },
    };
    verify_license(
        &token,
        &HANZO_LICENSE_PUBKEY,
        EXPECTED_APP_ID,
        current_unix_time(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    const DEV_APP: &str = "hanzo";
    const ONE_DAY: i64 = 86_400;

    // The dev signing seed (base64url, 32 bytes) mints test tokens; its public half is the
    // HANZO_LICENSE_PUBKEY embedded above. The seed is a secret (never committed), read at
    // runtime from $DEV_SIGNING_SEED or the gitignored license-dev-key file -- so builds
    // without it still compile and the seed-dependent tests skip (see `dev_signing_key!`).
    fn dev_signing_key_opt() -> Option<SigningKey> {
        let b64 = std::env::var("DEV_SIGNING_SEED").ok().or_else(|| {
            std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../license-dev-key/dev_signing_key.b64"
            ))
            .ok()
        })?;
        let seed: [u8; 32] = URL_SAFE_NO_PAD.decode(b64.trim()).ok()?.try_into().ok()?;
        Some(SigningKey::from_bytes(&seed))
    }

    // Yields the dev SigningKey, or returns from the test early when the seed isn't available
    // (e.g. CI without the secret) — the test is skipped rather than failing.
    macro_rules! dev_signing_key {
        () => {{
            match dev_signing_key_opt() {
                Some(k) => k,
                None => return,
            }
        }};
    }

    fn make_license(app_id: &str, iat: i64, exp: i64) -> License {
        License {
            v: LICENSE_SCHEMA_VERSION,
            app_id: app_id.to_string(),
            holder: "test-holder".to_string(),
            iat,
            exp,
            features: vec!["inference".to_string(), "embeddings".to_string()],
            nonce: "00112233445566778899aabbccddeeff".to_string(),
        }
    }

    // Sanity: the embedded public key const is exactly the public half of the dev signing seed.
    #[test]
    fn dev_keypair_matches_embedded_pubkey() {
        let vk = dev_signing_key!().verifying_key();
        assert_eq!(vk.to_bytes(), HANZO_LICENSE_PUBKEY);
    }

    // (a) a freshly-issued valid token verifies OK.
    #[test]
    fn valid_token_verifies() {
        let now = 1_700_000_000;
        let lic = make_license(DEV_APP, now, now + ONE_DAY);
        let token = encode_license(&lic, &dev_signing_key!());

        let verified =
            verify_license(&token, &HANZO_LICENSE_PUBKEY, DEV_APP, now).expect("should verify");
        assert_eq!(verified, lic);
        assert_eq!(verified.holder, "test-holder");
        assert_eq!(verified.features, vec!["inference", "embeddings"]);
    }

    // (b) tampered signature -> BadSig.
    #[test]
    fn tampered_signature_is_bad_sig() {
        let now = 1_700_000_000;
        let lic = make_license(DEV_APP, now, now + ONE_DAY);
        let token = encode_license(&lic, &dev_signing_key!());

        // Flip one base64url char in the signature segment (after the '.').
        let (payload_b64, sig_b64) = token.split_once('.').unwrap();
        let mut sig: Vec<char> = sig_b64.chars().collect();
        sig[0] = if sig[0] == 'A' { 'B' } else { 'A' };
        let tampered = format!("{payload_b64}.{}", sig.into_iter().collect::<String>());

        let err = verify_license(&tampered, &HANZO_LICENSE_PUBKEY, DEV_APP, now).unwrap_err();
        assert_eq!(err, LicenseError::BadSig);
    }

    // A tampered *payload* (claims changed after signing) must also fail the signature check.
    #[test]
    fn tampered_payload_is_bad_sig() {
        let now = 1_700_000_000;
        let lic = make_license(DEV_APP, now, now + ONE_DAY);
        let token = encode_license(&lic, &dev_signing_key!());
        let (_payload_b64, sig_b64) = token.split_once('.').unwrap();

        // Re-encode a payload that grants extra time, keep the original signature.
        let forged = make_license(DEV_APP, now, now + 365 * ONE_DAY);
        let forged_payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&forged).unwrap());
        let forged_token = format!("{forged_payload}.{sig_b64}");

        let err = verify_license(&forged_token, &HANZO_LICENSE_PUBKEY, DEV_APP, now).unwrap_err();
        assert_eq!(err, LicenseError::BadSig);
    }

    // (c) expired token -> Expired.
    #[test]
    fn expired_token_is_expired() {
        let iat = 1_700_000_000;
        let exp = iat + ONE_DAY;
        let lic = make_license(DEV_APP, iat, exp);
        let token = encode_license(&lic, &dev_signing_key!());

        let now = exp + 1; // one second past expiry
        let err = verify_license(&token, &HANZO_LICENSE_PUBKEY, DEV_APP, now).unwrap_err();
        assert_eq!(err, LicenseError::Expired);
    }

    // (d) wrong app_id vs expected_app -> WrongApp.
    #[test]
    fn wrong_app_is_wrong_app() {
        let now = 1_700_000_000;
        let lic = make_license("zoo", now, now + ONE_DAY);
        let token = encode_license(&lic, &dev_signing_key!());

        // Build expects "hanzo"; the token is for "zoo".
        let err = verify_license(&token, &HANZO_LICENSE_PUBKEY, "hanzo", now).unwrap_err();
        assert_eq!(err, LicenseError::WrongApp);
    }

    // (e) malformed token -> Malformed (several shapes).
    #[test]
    fn malformed_tokens_are_malformed() {
        let now = 1_700_000_000;
        for bad in [
            "",                // empty
            "no-dot-here",     // missing '.'
            ".",               // empty halves
            "abc.",            // empty sig
            ".abc",            // empty payload
            "!!!.@@@",         // invalid base64url
            "a.b.c",           // too many segments
            "eyJhIjoxfQ.AAAA", // valid b64 but wrong sig length
        ] {
            let err = verify_license(bad, &HANZO_LICENSE_PUBKEY, DEV_APP, now).unwrap_err();
            assert_eq!(err, LicenseError::Malformed, "input was {bad:?}");
        }
    }

    // not-yet-valid: iat far in the future beyond skew.
    #[test]
    fn future_dated_token_is_not_yet_valid() {
        let now = 1_700_000_000;
        let iat = now + 10 * IAT_SKEW_SECS;
        let lic = make_license(DEV_APP, iat, iat + ONE_DAY);
        let token = encode_license(&lic, &dev_signing_key!());

        let err = verify_license(&token, &HANZO_LICENSE_PUBKEY, DEV_APP, now).unwrap_err();
        assert_eq!(err, LicenseError::NotYetValid);
    }

    // Wrong schema version is treated as malformed.
    #[test]
    fn wrong_version_is_malformed() {
        let now = 1_700_000_000;
        let mut lic = make_license(DEV_APP, now, now + ONE_DAY);
        lic.v = 99;
        let token = encode_license(&lic, &dev_signing_key!());

        let err = verify_license(&token, &HANZO_LICENSE_PUBKEY, DEV_APP, now).unwrap_err();
        assert_eq!(err, LicenseError::Malformed);
    }
}
