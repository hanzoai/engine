//! The one check that loads extensions (`crate::extension`) into the engine's process: a license
//! token, verified by `hanzo_engine::license` against the key this build trusts, whose claims name
//! `local_kai_enso`. A [`Gate`] exists only once that check has passed, and extensions load only
//! from one. The check runs again whenever the license it last held has expired, from a token
//! read afresh, so a short-lived token keeps them serving only while it is renewed.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use hanzo_engine::license::{
    current_unix_time, read_token, verify_entitlement, LOCAL_KAI_ENSO,
};
use hanzo_engine::{LicenseError, HANZO_LICENSE_PUBKEY, LICENSE_EXPECTED_APP_ID};

/// Where a token comes from: the environment in production, a test's own in tests.
type Source = Box<dyn Fn() -> Result<String, LicenseError> + Send + Sync>;

pub struct Gate {
    key: [u8; 32],
    source: Source,
    exp: AtomicI64,
}

impl Gate {
    /// The production gate: the token from `HANZO_ENGINE_LICENSE_TOKEN` or the file
    /// `HANZO_ENGINE_LICENSE_FILE` names, against the embedded key and this build's app.
    pub fn open() -> Result<Arc<Gate>, LicenseError> {
        Gate::with(HANZO_LICENSE_PUBKEY, read_token, current_unix_time())
    }

    /// A gate over `source`'s token, checked against `key` at `now`.
    pub fn with(
        key: [u8; 32],
        source: impl Fn() -> Result<String, LicenseError> + Send + Sync + 'static,
        now: i64,
    ) -> Result<Arc<Gate>, LicenseError> {
        let exp = check(&key, &source()?, now)?;
        Ok(Arc::new(Gate {
            key,
            source: Box::new(source),
            exp: AtomicI64::new(exp),
        }))
    }

    /// Whether the license holds at `now`: the one held until it expires, then a token read
    /// afresh and checked as at load.
    pub fn holds(&self, now: i64) -> bool {
        if now < self.exp.load(Ordering::Acquire) {
            return true;
        }
        match (self.source)().and_then(|t| check(&self.key, &t, now)) {
            Ok(exp) => {
                self.exp.store(exp, Ordering::Release);
                true
            }
            Err(_) => false,
        }
    }

    /// `routes`, answered while the license holds; past it every request is 403 `unlicensed`,
    /// a fixed message that names no model, path or token.
    pub fn guard(self: &Arc<Self>, routes: Router) -> Router {
        let gate = self.clone();
        routes.layer(middleware::from_fn(move |req: Request, next: Next| {
            let gate = gate.clone();
            async move {
                match gate.holds(current_unix_time()) {
                    true => next.run(req).await,
                    false => unlicensed(),
                }
            }
        }))
    }
}

fn check(key: &[u8; 32], token: &str, now: i64) -> Result<i64, LicenseError> {
    verify_entitlement(token, key, LICENSE_EXPECTED_APP_ID, now, LOCAL_KAI_ENSO).map(|l| l.exp)
}

fn unlicensed() -> Response {
    let body = serde_json::json!({"error": {"code": "unlicensed",
        "message": "this engine holds no license for this model"}});
    (StatusCode::FORBIDDEN, Json(body)).into_response()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;
    use ed25519_dalek::{Signer as _, SigningKey};
    use std::sync::Mutex;

    /// A key no build trusts: these tests hand its public half to the gate in place of Hanzo's.
    pub fn key() -> SigningKey {
        SigningKey::from_bytes(&[0x24; 32])
    }

    /// A token in `hanzo_engine::license`'s wire format, signed by [`key`], valid `[iat, exp)`.
    pub fn token(features: &[&str], iat: i64, exp: i64) -> String {
        let claims = serde_json::json!({"v": 1, "app_id": LICENSE_EXPECTED_APP_ID,
            "holder": "test", "iat": iat, "exp": exp, "features": features,
            "nonce": "00112233445566778899aabbccddeeff"});
        let payload = URL_SAFE_NO_PAD.encode(claims.to_string());
        let sig = URL_SAFE_NO_PAD.encode(key().sign(payload.as_bytes()).to_bytes());
        format!("{payload}.{sig}")
    }

    pub fn pubkey() -> [u8; 32] {
        key().verifying_key().to_bytes()
    }

    const NOW: i64 = 1_800_000_000;

    #[test]
    fn a_gate_opens_only_on_the_entitlement() {
        let ok = token(&[LOCAL_KAI_ENSO], NOW, NOW + 60);
        assert!(Gate::with(pubkey(), move || Ok(ok.clone()), NOW).is_ok());
        let plain = token(&["inference"], NOW, NOW + 60);
        let refused = Gate::with(pubkey(), move || Ok(plain.clone()), NOW).err();
        assert_eq!(refused, Some(LicenseError::NotEntitled));
        let expired = token(&[LOCAL_KAI_ENSO], NOW - 120, NOW - 60);
        let refused = Gate::with(pubkey(), move || Ok(expired.clone()), NOW).err();
        assert_eq!(refused, Some(LicenseError::Expired));
        let missing = Gate::with(pubkey(), || Err(LicenseError::Missing), NOW).err();
        assert_eq!(missing, Some(LicenseError::Missing));
        // Hanzo's own key refuses a token the test key signed.
        let ours = token(&[LOCAL_KAI_ENSO], NOW, NOW + 60);
        let refused = Gate::with(HANZO_LICENSE_PUBKEY, move || Ok(ours.clone()), NOW).err();
        assert_eq!(refused, Some(LicenseError::BadSig));
    }

    /// A gate holds its license until it expires, then holds only a renewed one.
    #[test]
    fn an_expired_license_is_checked_again() {
        let current = Arc::new(Mutex::new(token(&[LOCAL_KAI_ENSO], NOW, NOW + 60)));
        let read = current.clone();
        let gate = Gate::with(pubkey(), move || Ok(read.lock().unwrap().clone()), NOW).unwrap();
        assert!(gate.holds(NOW + 59));
        assert!(!gate.holds(NOW + 60), "expired and not renewed");
        *current.lock().unwrap() = token(&[LOCAL_KAI_ENSO], NOW + 60, NOW + 120);
        assert!(gate.holds(NOW + 61), "renewed");
        assert!(gate.holds(NOW + 119));
        *current.lock().unwrap() = token(&["inference"], NOW + 120, NOW + 180);
        assert!(!gate.holds(NOW + 120), "renewed without the entitlement");
    }
}
