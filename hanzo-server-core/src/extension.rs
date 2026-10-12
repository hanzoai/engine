//! What a build of the engine outside this repository adds to its server: routes served beside
//! the engine's and a pick every request passes first, each an [`Extension`]. Extensions load
//! only through the one license check ([`Gate`]): [`Licensed::load`] opens the gate once and only
//! then calls the loaders, so a refused license reads nothing of theirs; their routes answer and
//! their picks run only while the license holds.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use axum::extract::Request;
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::Router;
use hanzo_engine::license::current_unix_time;
use hanzo_engine::LicenseError;
use hanzo_ml::Device;

use crate::gate::Gate;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// What a pick did with a request.
pub enum Pick {
    /// Answered it; the engine never sees it.
    Answer(Response),
    /// Handed it back for the engine to answer.
    Pass(Request),
}

pub trait Extension: Send + Sync + 'static {
    /// Called once at build with the engine's own router (its routes, no extension's pick), for
    /// an extension that answers through the engine in process.
    fn bind(&self, _engine: &Router) {}

    /// Routes served beside the engine's; one the engine serves panics at build.
    fn routes(&self) -> Router {
        Router::new()
    }

    /// Asked first for every request.
    fn pick(&self, req: Request) -> BoxFuture<'_, Pick> {
        Box::pin(async move { Pick::Pass(req) })
    }
}

/// Builds an extension on the engine's device. Called only once the license holds.
pub type Load = Box<dyn FnOnce(&Device) -> Result<Arc<dyn Extension>> + Send>;

/// Extensions loaded under a license, as the router builder takes them.
pub struct Licensed {
    gate: Arc<Gate>,
    extensions: Vec<Arc<dyn Extension>>,
}

impl Licensed {
    /// Runs `loads` on `device` when `gate` passed (in production `Gate::open()`). The refusal
    /// names the license's failure and nothing of the extensions: no model, path or token.
    ///
    /// Loaders may block reading weights: call it outside the async runtime, or in
    /// `spawn_blocking`.
    pub fn load(
        gate: Result<Arc<Gate>, LicenseError>,
        loads: Vec<Load>,
        device: &Device,
    ) -> Result<Licensed> {
        let gate = gate.map_err(|e| anyhow::anyhow!("extensions are not loaded: {e}"))?;
        let extensions = loads
            .into_iter()
            .map(|load| load(device))
            .collect::<Result<_>>()?;
        Ok(Licensed { gate, extensions })
    }

    /// `engine` with each extension bound to it, its routes merged under the gate, and its pick
    /// asked first for every request while the license holds.
    pub fn serve(&self, engine: Router) -> Router {
        for e in &self.extensions {
            e.bind(&engine);
        }
        let mut router = engine;
        for e in &self.extensions {
            router = router.merge(self.gate.guard(e.routes()));
        }
        let (gate, extensions) = (self.gate.clone(), self.extensions.clone());
        router.layer(middleware::from_fn(move |req: Request, next: Next| {
            let (gate, extensions) = (gate.clone(), extensions.clone());
            async move { pick(&gate, &extensions, req, next).await }
        }))
    }
}

async fn pick(
    gate: &Gate,
    extensions: &[Arc<dyn Extension>],
    mut req: Request,
    next: Next,
) -> Response {
    if !gate.holds(current_unix_time()) {
        return next.run(req).await;
    }
    for e in extensions {
        match e.pick(req).await {
            Pick::Answer(r) => return r,
            Pick::Pass(r) => req = r,
        }
    }
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::tests::{pubkey, token};
    use axum::body::Body;
    use axum::routing::{get, post};
    use hanzo_engine::license::LOCAL_KAI_ENSO;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Mutex, OnceLock};

    /// Serves `/v1/echo`, answers every request whose path ends in `/mine`, and on `/v1/through`
    /// answers by calling the engine's own `/v1/engine` in process.
    struct Echo {
        engine: OnceLock<Router>,
    }

    impl Extension for Echo {
        fn bind(&self, engine: &Router) {
            let _ = self.engine.set(engine.clone());
        }

        fn routes(&self) -> Router {
            Router::new().route("/v1/echo", get(|| async { "echo" }))
        }

        fn pick(&self, req: Request) -> BoxFuture<'_, Pick> {
            Box::pin(async move {
                use tower::ServiceExt;
                match req.uri().path() {
                    p if p.ends_with("/mine") => Pick::Answer(Response::new(Body::from("picked"))),
                    "/v1/through" => {
                        let inner = Request::post("/v1/engine").body(Body::empty()).unwrap();
                        let engine = self.engine.get().unwrap().clone();
                        Pick::Answer(engine.oneshot(inner).await.unwrap())
                    }
                    _ => Pick::Pass(req),
                }
            })
        }
    }

    fn engine() -> Router {
        Router::new()
            .route("/v1/engine", post(|| async { "engine" }))
            .route("/v1/models", get(|| async { "engine models" }))
    }

    fn echo(loads: Arc<AtomicUsize>) -> Load {
        Box::new(move |_: &Device| {
            loads.fetch_add(1, Ordering::Relaxed);
            Ok(Arc::new(Echo { engine: OnceLock::new() }) as Arc<dyn Extension>)
        })
    }

    async fn ask(app: &Router, method: &str, path: &str) -> (u16, String) {
        use tower::ServiceExt;
        let req = Request::builder().method(method).uri(path).body(Body::empty()).unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        let status = res.status().as_u16();
        let body = axum::body::to_bytes(res.into_body(), 1 << 20).await.unwrap();
        (status, String::from_utf8_lossy(&body).into_owned())
    }

    /// Refused, no loader runs and the message names only the license's failure.
    #[test]
    fn a_refused_license_loads_nothing() {
        let loads = Arc::new(AtomicUsize::new(0));
        let now = current_unix_time();
        let plain = token(&["inference"], now, now + 60);
        let refused = Licensed::load(
            Gate::with(pubkey(), move || Ok(plain.clone()), now),
            vec![echo(loads.clone())],
            &Device::Cpu,
        )
        .err()
        .unwrap();
        assert_eq!(
            refused.to_string(),
            "extensions are not loaded: license token does not grant this entitlement"
        );
        assert_eq!(loads.load(Ordering::Relaxed), 0);
        let missing = Licensed::load(Gate::open(), vec![echo(loads.clone())], &Device::Cpu);
        assert!(missing.is_err(), "no token in the environment, or none Hanzo's key signed");
        assert_eq!(loads.load(Ordering::Relaxed), 0);
    }

    /// Licensed: the extension's routes answer beside the engine's, its pick answers what it
    /// claims (also through the engine in process) and passes the rest; once the license
    /// expires unrenewed, its routes refuse and its pick is not asked.
    #[tokio::test]
    async fn a_licensed_extension_routes_and_picks_until_expiry() {
        let loads = Arc::new(AtomicUsize::new(0));
        let now = current_unix_time();
        let current = Arc::new(Mutex::new(token(&[LOCAL_KAI_ENSO], now, now + 1)));
        let read = current.clone();
        let gate = Gate::with(pubkey(), move || Ok(read.lock().unwrap().clone()), now);
        let licensed = Licensed::load(gate, vec![echo(loads.clone())], &Device::Cpu).unwrap();
        assert_eq!(loads.load(Ordering::Relaxed), 1);
        let app = licensed.serve(engine());

        assert_eq!(ask(&app, "GET", "/v1/echo").await, (200, "echo".into()));
        assert_eq!(ask(&app, "POST", "/v1/x/mine").await, (200, "picked".into()));
        assert_eq!(ask(&app, "POST", "/v1/through").await, (200, "engine".into()));
        assert_eq!(ask(&app, "GET", "/v1/models").await, (200, "engine models".into()));

        *current.lock().unwrap() = token(&["inference"], now, now + 600);
        tokio::time::sleep(std::time::Duration::from_millis(1_100)).await;
        let (status, body) = ask(&app, "GET", "/v1/echo").await;
        assert_eq!(status, 403);
        assert!(body.contains("unlicensed"), "{body}");
        assert_eq!(ask(&app, "POST", "/v1/x/mine").await.0, 404, "not picked: the engine's");
        assert_eq!(ask(&app, "GET", "/v1/models").await, (200, "engine models".into()));
    }
}
