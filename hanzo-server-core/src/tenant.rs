//! The prefix-cache partition a request runs in.

use axum::http::HeaderMap;
use hanzo_engine::Request;

/// The caller's org, set by the gateway in front of the engine from the validated principal.
pub const ORG_HEADER: &str = "x-org-id";

/// The cache salt for a request: its caller's org, then the salt the request asked for within
/// that org. Both parts are encoded together so a requested salt can never equal another org's.
pub fn cache_salt(headers: &HeaderMap, requested: Option<String>) -> Option<String> {
    let org = headers
        .get(ORG_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|org| !org.is_empty());
    let requested = requested.filter(|salt| !salt.is_empty());
    if org.is_none() && requested.is_none() {
        return None;
    }
    Some(serde_json::json!([org, requested]).to_string())
}

/// Puts `request` in the prefix-cache partition `salt` from [`cache_salt`].
pub fn partition(request: &mut Request, salt: Option<String>) {
    if let Request::Normal(request) = request {
        request.cache_salt = salt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn org(org: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(ORG_HEADER, org.parse().unwrap());
        headers
    }

    #[test]
    fn no_org_and_no_salt_shares_the_unsalted_cache() {
        assert_eq!(cache_salt(&HeaderMap::new(), None), None);
        assert_eq!(cache_salt(&org(" "), Some(String::new())), None);
    }

    #[test]
    fn orgs_are_partitioned() {
        let acme = cache_salt(&org("acme"), None);
        assert!(acme.is_some());
        assert_ne!(acme, cache_salt(&org("evil"), None));
        assert_eq!(acme, cache_salt(&org("acme"), None));
    }

    #[test]
    fn a_requested_salt_cannot_reach_another_org() {
        let acme = cache_salt(&org("acme"), None);
        let forged = [
            cache_salt(&HeaderMap::new(), Some("acme".into())),
            cache_salt(&HeaderMap::new(), acme.clone()),
            cache_salt(&org("evil"), Some("acme".into())),
            cache_salt(&org("evil"), acme.clone()),
        ];
        for salt in forged {
            assert_ne!(salt, acme);
        }
    }

    #[test]
    fn a_requested_salt_narrows_within_the_org() {
        let base = cache_salt(&org("acme"), None);
        let narrowed = cache_salt(&org("acme"), Some("user-7".into()));
        assert_ne!(narrowed, base);
        assert_eq!(narrowed, cache_salt(&org("acme"), Some("user-7".into())));
    }
}
