//! `/v1/worlds/{world}/inference` — native finite-state active-inference route.
//!
//! Enforces:
//! - Authenticated tenant isolation via `x-org-id` (`crate::tenant::org`); missing tenant
//!   identity fails closed with HTTP 403.
//! - Host-owned versioned generative models and hard admissibility constraints via
//!   [`hanzo_worlds::WorldRegistry`]; callers cannot override model parameters, preferences,
//!   habits, admissibility, or tenant identity.
//! - Strict JSON schema (`#[serde(deny_unknown_fields)]`) and bounded request size (64 KiB);
//!   malformed JSON, unknown fields, invalid beliefs/observations/policies, or path traversal
//!   fail with HTTP 400.
//! - Cross-tenant lookups return HTTP 404 (identical to missing worlds; zero existence leak).
//! - All-inadmissible candidate policies fail with HTTP 422.
//! - Strict separation between `inference`, `recommendation` (`advisory_only: true`), and
//!   `execution_authorized: false`.

use axum::{
    body::Bytes,
    extract::Path,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Extension, Json,
};
use hanzo_worlds::{RegistryError, Request, WorldRegistry};
use serde_json::json;

use crate::tenant;

/// Maximum request payload size (64 KiB) for finite-state world inference.
const MAX_WORLD_INFERENCE_BYTES: usize = 64 * 1024;

fn error_response(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "type": kind,
                "message": message.into(),
            }
        })),
    )
        .into_response()
}

/// `POST /v1/worlds/{world}/inference`: run authenticated, tenant-isolated finite-state
/// belief filtering and exact expected free-energy policy inference.
pub async fn world_inference(
    headers: HeaderMap,
    Path(world): Path<String>,
    Extension(registry): Extension<WorldRegistry>,
    body: Bytes,
) -> Response {
    let Some(org) = tenant::org(&headers) else {
        return error_response(
            StatusCode::FORBIDDEN,
            "auth_error",
            "authenticated tenant identity required (x-org-id)",
        );
    };
    if body.len() > MAX_WORLD_INFERENCE_BYTES {
        return error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "invalid_request_error",
            "world inference request exceeds 64 KiB",
        );
    }
    let request: Request = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(err) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("malformed world inference request: {err}"),
            );
        }
    };
    match registry.infer(&org, &world, &request) {
        Ok(resp) => {
            debug_assert!(!resp.execution_authorized);
            debug_assert!(resp.recommendation.advisory_only);
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(RegistryError::Unauthorized(msg)) => {
            error_response(StatusCode::FORBIDDEN, "auth_error", msg)
        }
        Err(RegistryError::NotFound) => {
            error_response(StatusCode::NOT_FOUND, "not_found_error", "world not found")
        }
        Err(RegistryError::MalformedInput(msg)) => {
            error_response(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
        }
        Err(RegistryError::Inadmissible(msg)) => error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "policy_inadmissible_error",
            msg,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::post, Router};
    use hanzo_worlds::{AdmissibilityPolicy, GenerativeModel, WorldInference};
    use tokio::net::TcpListener;

    use crate::route_registry::WORLDS_INFERENCE_ROUTE;

    fn sample_registry() -> WorldRegistry {
        let id = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let swap = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let model =
            GenerativeModel::new(vec![id.clone(), id.clone()], vec![id, swap], vec![0.9, 0.1])
                .unwrap();
        let engine = WorldInference::with_version(model, 4.0, "two-state-sim-v1").unwrap();
        let reg = WorldRegistry::new();
        reg.register(
            "tenant-alpha",
            "world_1",
            engine,
            AdmissibilityPolicy::allow_actions([0, 1]).with_forbidden_prefix(vec![0]),
        )
        .unwrap();
        reg
    }

    async fn spawn_worlds_app(registry: WorldRegistry) -> String {
        let app = Router::new()
            .route(WORLDS_INFERENCE_ROUTE.path, post(world_inference))
            .layer(Extension(registry));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn authenticated_inference_enforces_admissibility_and_no_execution() {
        let base = spawn_worlds_app(sample_registry()).await;
        let client = reqwest::Client::new();
        let res = client
            .post(format!("{base}/v1/worlds/world_1/inference"))
            .header(tenant::ORG_HEADER, "tenant-alpha")
            .json(&json!({
                "belief": [0.5, 0.5],
                "observation": [0, 0],
                "policies": [[0], [1]]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body: serde_json::Value = res.json().await.unwrap();
        assert_eq!(body["world_id"], "world_1");
        assert_eq!(body["model_version"], "two-state-sim-v1");
        assert_eq!(body["inference"]["posterior"]["belief"], json!([1.0, 0.0]));
        assert_eq!(
            body["inference"]["policies"]["probabilities"],
            json!([0.0, 1.0])
        );
        assert_eq!(body["recommendation"]["policy_index"], 1);
        assert_eq!(body["recommendation"]["actions"], json!([1]));
        assert_eq!(body["recommendation"]["advisory_only"], true);
        assert_eq!(body["execution_authorized"], false);
    }

    #[tokio::test]
    async fn rejects_unauthenticated_and_cross_tenant_requests() {
        let base = spawn_worlds_app(sample_registry()).await;
        let client = reqwest::Client::new();
        let payload = json!({"belief": [0.5, 0.5], "policies": [[1]]});

        // Missing x-org-id -> 403 Forbidden.
        let anon = client
            .post(format!("{base}/v1/worlds/world_1/inference"))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(anon.status(), StatusCode::FORBIDDEN);

        // Cross-tenant lookup -> 404 Not Found (no existence leak).
        let cross = client
            .post(format!("{base}/v1/worlds/world_1/inference"))
            .header(tenant::ORG_HEADER, "tenant-other")
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(cross.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn rejects_malformed_inputs_and_override_attempts_with_400_and_inadmissible_with_422() {
        let base = spawn_worlds_app(sample_registry()).await;
        let client = reqwest::Client::new();

        for bad in [
            json!({"belief": [0.5, 0.5], "policies": [[1]], "admissible": [true]}),
            json!({"belief": [0.5, 0.5], "policies": [[1]], "preferences": [0.1, 0.9]}),
            json!({"belief": [0.5, 0.5], "policies": [[1]], "execution_authorized": true}),
            json!({"belief": [0.2, 0.2], "policies": [[1]]}),
            json!({"belief": [0.5, 0.5], "policies": []}),
        ] {
            let res = client
                .post(format!("{base}/v1/worlds/world_1/inference"))
                .header(tenant::ORG_HEADER, "tenant-alpha")
                .json(&bad)
                .send()
                .await
                .unwrap();
            assert_eq!(res.status(), StatusCode::BAD_REQUEST, "payload={bad}");
        }

        // All candidate policies forbidden by host prefix rule -> 422.
        let blocked = client
            .post(format!("{base}/v1/worlds/world_1/inference"))
            .header(tenant::ORG_HEADER, "tenant-alpha")
            .json(&json!({"belief": [1.0, 0.0], "policies": [[0], [0, 1]]}))
            .send()
            .await
            .unwrap();
        assert_eq!(blocked.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }
}
