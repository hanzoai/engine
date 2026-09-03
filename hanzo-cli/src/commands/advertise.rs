//! mDNS/DNS-SD advertisement of a running `hanzo serve` engine on the LAN.
//!
//! Publishes `_hanzo-engine._tcp.local.` so peers (Hanzo Desktop, `dev`) can
//! discover the engine, its version, and its loaded models without config.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use mdns_sd::{ServiceDaemon, ServiceInfo};
use tracing::{info, warn};

const SERVICE_TYPE: &str = "_hanzo-engine._tcp.local.";
const OBSERVE_INTERVAL: Duration = Duration::from_secs(5);
const FALLBACK_HOSTNAME: &str = "hanzo-engine";

/// A live mDNS registration. Holds the responder daemon and the stable identity
/// used to re-publish on model changes and to send a goodbye on shutdown.
pub struct Advertiser {
    daemon: ServiceDaemon,
    instance: String,
    host_name: String,
    port: u16,
    fullname: String,
}

impl Advertiser {
    pub fn start(port: u16, models: &[String]) -> Result<Arc<Self>> {
        let daemon = ServiceDaemon::new()?;
        let instance = hostname();
        // The A record for `<machine>.local` belongs to the system responder.
        // Point the SRV target at a name this daemon alone owns, so the two
        // responders never contend for one name.
        let host_name = format!("{instance}-engine.local.");
        let info = build_info(&instance, &host_name, port, models)?;
        let fullname = info.get_fullname().to_string();
        daemon.register(info)?;
        info!(
            "mDNS: advertising {SERVICE_TYPE} as '{instance}' on port {port} ({} models)",
            models.len()
        );
        Ok(Arc::new(Self {
            daemon,
            instance,
            host_name,
            port,
            fullname,
        }))
    }

    /// Re-publish with a new model list. Registering the same fullname updates
    /// the TXT record and re-announces.
    pub fn advertise(&self, models: &[String]) -> Result<()> {
        let info = build_info(&self.instance, &self.host_name, self.port, models)?;
        self.daemon.register(info)?;
        Ok(())
    }

    /// Send a goodbye packet and stop the responder. Best-effort.
    pub fn shutdown(&self) {
        let _ = self.daemon.unregister(&self.fullname);
        let _ = self.daemon.shutdown();
        info!("mDNS: deregistered '{}'", self.instance);
    }
}

fn build_info(
    instance: &str,
    host_name: &str,
    port: u16,
    models: &[String],
) -> Result<ServiceInfo> {
    let props = HashMap::from([
        ("v".to_string(), env!("CARGO_PKG_VERSION").to_string()),
        ("models".to_string(), models.join(",")),
        ("port".to_string(), port.to_string()),
    ]);
    Ok(ServiceInfo::new(SERVICE_TYPE, instance, host_name, "", port, props)?.enable_addr_auto())
}

fn hostname() -> String {
    hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| FALLBACK_HOSTNAME.to_string())
}

/// Poll a model-list source and re-advertise when the set changes. Decoupled
/// from the engine type: `list` yields the current served model ids.
pub fn spawn_model_observer<F>(list: F, adv: Arc<Advertiser>, initial: Vec<String>)
where
    F: Fn() -> Vec<String> + Send + 'static,
{
    let mut last = initial;
    last.sort();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(OBSERVE_INTERVAL).await;
            let mut now = list();
            now.sort();
            if now != last {
                match adv.advertise(&now) {
                    Ok(()) => info!(
                        "mDNS: model list changed, re-advertised ({} models)",
                        now.len()
                    ),
                    Err(e) => warn!("mDNS: re-advertise failed: {e}"),
                }
                last = now;
            }
        }
    });
}
