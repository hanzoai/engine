//! ## Model registry (M1 single-slot stub).
//!
//! Holds the set of registered "experts" that the server can route to. In M1 this
//! is intentionally a flat in-memory map with a single in-process model plus
//! optional metadata-only entries for remote-HTTP and subprocess backends. There
//! is no router yet: requests continue to dispatch via the existing
//! `model_id`-keyed pipeline in `MistralRs::get_sender`. The registry's only
//! M1 jobs are
//!
//!   1. Surface every registered expert in `/v1/models` with a capabilities
//!      array, so clients (Zoo Desktop) can discover what the server can do.
//!   2. Persist enough metadata (`modalities`, `resident_size_mb`, `quant_label`,
//!      backend kind) for later milestones (M2 ModelRegistry, M4 ModalityRouter)
//!      to build on without churning this surface.
//!
//! Future milestones (per `MULTIMODAL_ROUTER_DESIGN.md`):
//!   * M2 attaches per-expert async queues, VRAM budgeting, and real
//!     multi-model loading.
//!   * M4 adds a `ModalityRouter` on top of `lookup`.
//!   * M5 wires `RemoteHttp` to actually proxy `/v1/chat/completions` traffic.
//!
//! The structures here are deliberately minimal so M2 can extend them without
//! breaking the public shape.
//!
//! ### Modality bits
//!
//! `ModalitySet` is a small manual `u32` bitset. We avoid pulling in the
//! `bitflags` crate to keep `mistralrs-server-core`'s dependency surface
//! unchanged. The constants intentionally mirror the names used in the
//! design doc (`TEXT | IMAGE_IN | IMAGE_OUT | AUDIO_IN | AUDIO_OUT | VIDEO |
//! MODEL3D | TOOL`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Capability bitset for a registered expert.
///
/// The bit values are stable; new modalities must be appended (next free bit
/// is `1 << 8`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ModalitySet(pub u32);

impl ModalitySet {
    pub const NONE: Self = Self(0);
    pub const TEXT: Self = Self(1 << 0);
    pub const IMAGE_IN: Self = Self(1 << 1);
    pub const IMAGE_OUT: Self = Self(1 << 2);
    pub const AUDIO_IN: Self = Self(1 << 3);
    pub const AUDIO_OUT: Self = Self(1 << 4);
    pub const VIDEO: Self = Self(1 << 5);
    pub const MODEL3D: Self = Self(1 << 6);
    pub const TOOL: Self = Self(1 << 7);

    /// Union of two sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Does `self` contain every bit in `other`?
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Returns the kebab-case names of the bits that are set, in stable order.
    ///
    /// This is what `/v1/models` publishes as the `capabilities` array, so the
    /// names match the strings in `MULTIMODAL_ROUTER_DESIGN.md` §"Zoo Desktop
    /// integration".
    pub fn capability_names(self) -> Vec<&'static str> {
        const ALL: &[(ModalitySet, &str)] = &[
            (ModalitySet::TEXT, "text"),
            (ModalitySet::IMAGE_IN, "image_in"),
            (ModalitySet::IMAGE_OUT, "image_out"),
            (ModalitySet::AUDIO_IN, "audio_in"),
            (ModalitySet::AUDIO_OUT, "audio_out"),
            (ModalitySet::VIDEO, "video"),
            (ModalitySet::MODEL3D, "model3d"),
            (ModalitySet::TOOL, "tool"),
        ];
        ALL.iter()
            .filter(|(bit, _)| self.contains(*bit))
            .map(|(_, name)| *name)
            .collect()
    }
}

impl std::ops::BitOr for ModalitySet {
    type Output = ModalitySet;
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for ModalitySet {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// How an expert is reached.
///
/// * `InProcess` — served by the same `MistralRs` instance that owns this
///   registry. The associated id is what gets passed to `MistralRs::get_sender`.
/// * `RemoteHttp` — an OpenAI-compatible server reachable over loopback HTTP
///   (e.g. `zen5-server`). M1 parses and lists it but does not yet proxy
///   requests; that ships in M5.
/// * `Subprocess` — locally-managed child process. M1 parses but is otherwise
///   inert; M6 wires it up.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ExpertBackend {
    InProcess,
    RemoteHttp { url: String },
    Subprocess { cmd: PathBuf },
}

impl ExpertBackend {
    /// Short tag used in tracing/diagnostics.
    pub fn kind(&self) -> &'static str {
        match self {
            ExpertBackend::InProcess => "inprocess",
            ExpertBackend::RemoteHttp { .. } => "proxy",
            ExpertBackend::Subprocess { .. } => "subprocess",
        }
    }
}

/// A single registered expert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredExpert {
    pub id: String,
    pub backend: ExpertBackend,
    pub modalities: ModalitySet,
    pub resident_size_mb: u64,
    pub quant_label: String,
}

/// Spec passed to `ModelRegistry::register`. Fields mirror `RegisteredExpert`
/// but with sensible defaults so callers don't have to know everything up
/// front. M1 currently uses defaults for `resident_size_mb` (`0`) and
/// `quant_label` (`"unknown"`); M2 will populate them from real model
/// metadata when loading happens.
#[derive(Debug, Clone)]
pub struct ExpertSpec {
    pub id: String,
    pub backend: ExpertBackend,
    pub modalities: ModalitySet,
    pub resident_size_mb: u64,
    pub quant_label: String,
}

impl ExpertSpec {
    /// Convenience constructor mirroring the typical M1 call site.
    pub fn new(id: impl Into<String>, backend: ExpertBackend, modalities: ModalitySet) -> Self {
        Self {
            id: id.into(),
            backend,
            modalities,
            resident_size_mb: 0,
            quant_label: "unknown".to_string(),
        }
    }
}

/// Single-slot M1 registry.
///
/// In M2 this grows per-expert queues and a VRAM budget. The public API
/// (`register` / `lookup` / `list`) is shaped so those additions are
/// backward-compatible.
#[derive(Debug, Default)]
pub struct ModelRegistry {
    experts: HashMap<String, RegisteredExpert>,
}

impl ModelRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an expert. Returns an error if `spec.id` is already taken,
    /// the id is empty, or the backend payload is obviously malformed.
    pub fn register(&mut self, spec: ExpertSpec) -> Result<()> {
        if spec.id.is_empty() {
            return Err(anyhow!("expert id must not be empty"));
        }
        if self.experts.contains_key(&spec.id) {
            return Err(anyhow!("expert id `{}` is already registered", spec.id));
        }
        match &spec.backend {
            ExpertBackend::RemoteHttp { url } if url.is_empty() => {
                return Err(anyhow!(
                    "expert `{}`: RemoteHttp backend requires a non-empty URL",
                    spec.id
                ));
            }
            ExpertBackend::Subprocess { cmd } if cmd.as_os_str().is_empty() => {
                return Err(anyhow!(
                    "expert `{}`: Subprocess backend requires a non-empty cmd path",
                    spec.id
                ));
            }
            _ => {}
        }

        self.experts.insert(
            spec.id.clone(),
            RegisteredExpert {
                id: spec.id,
                backend: spec.backend,
                modalities: spec.modalities,
                resident_size_mb: spec.resident_size_mb,
                quant_label: spec.quant_label,
            },
        );
        Ok(())
    }

    /// Look up an expert by id.
    pub fn lookup(&self, id: &str) -> Option<&RegisteredExpert> {
        self.experts.get(id)
    }

    /// List every registered expert. Order is unspecified (HashMap iteration).
    pub fn list(&self) -> Vec<&RegisteredExpert> {
        self.experts.values().collect()
    }

    /// Is this id an in-process expert? Used by handlers that need to decide
    /// whether to dispatch locally or (M5+) proxy out.
    pub fn is_in_process(&self, id: &str) -> bool {
        matches!(
            self.experts.get(id).map(|e| &e.backend),
            Some(ExpertBackend::InProcess)
        )
    }
}

/// Process-global registry, populated once at server startup. Using a
/// `OnceLock` keeps M1 from having to thread a new state parameter through
/// every handler signature. M2 will likely move this into the
/// `SharedMistralRsState` along with the per-expert queues.
static GLOBAL_REGISTRY: OnceLock<ModelRegistry> = OnceLock::new();

/// Install the registry. Returns an error if called more than once.
pub fn set_global(registry: ModelRegistry) -> Result<()> {
    GLOBAL_REGISTRY
        .set(registry)
        .map_err(|_| anyhow!("ModelRegistry already initialized"))
}

/// Borrow the process-global registry, if any.
pub fn global() -> Option<&'static ModelRegistry> {
    GLOBAL_REGISTRY.get()
}

/// Parse a `--register ID:KIND:LOCATION` spec.
///
/// * `KIND` ∈ {`inprocess`, `proxy`, `subprocess`}.
/// * For `inprocess`, `LOCATION` is `auto` (the server's already-loaded model).
/// * For `proxy`, `LOCATION` is a URL.
/// * For `subprocess`, `LOCATION` is a path.
///
/// The split is on the first two colons only, so URLs containing `:` (e.g.
/// `http://127.0.0.1:8001`) round-trip cleanly.
pub fn parse_register_spec(raw: &str) -> Result<ExpertSpec> {
    let (id, rest) = raw
        .split_once(':')
        .ok_or_else(|| anyhow!("expected ID:KIND:LOCATION, got `{}`", raw))?;
    let (kind, location) = rest
        .split_once(':')
        .ok_or_else(|| anyhow!("expected ID:KIND:LOCATION, got `{}`", raw))?;
    if id.is_empty() {
        return Err(anyhow!("expected non-empty ID in `{}`", raw));
    }

    let backend = match kind {
        "inprocess" => {
            if location != "auto" {
                return Err(anyhow!(
                    "inprocess backend currently only supports LOCATION=auto (got `{}`)",
                    location
                ));
            }
            ExpertBackend::InProcess
        }
        "proxy" => {
            if location.is_empty() {
                return Err(anyhow!("proxy backend requires a URL in `{}`", raw));
            }
            ExpertBackend::RemoteHttp {
                url: location.to_string(),
            }
        }
        "subprocess" => {
            if location.is_empty() {
                return Err(anyhow!("subprocess backend requires a path in `{}`", raw));
            }
            ExpertBackend::Subprocess {
                cmd: PathBuf::from(location),
            }
        }
        other => {
            return Err(anyhow!(
                "unknown backend kind `{}` (expected inprocess, proxy, or subprocess)",
                other
            ));
        }
    };

    // M1: every in-process expert is assumed to be text+tool capable until M2
    // wires real per-model introspection. Remote/subprocess experts default
    // to text only; callers can override after construction.
    let modalities = match &backend {
        ExpertBackend::InProcess => ModalitySet::TEXT | ModalitySet::TOOL,
        _ => ModalitySet::TEXT,
    };

    Ok(ExpertSpec::new(id, backend, modalities))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_inprocess_auto() {
        let spec = parse_register_spec("default:inprocess:auto").unwrap();
        assert_eq!(spec.id, "default");
        assert!(matches!(spec.backend, ExpertBackend::InProcess));
        assert!(spec.modalities.contains(ModalitySet::TEXT));
        assert!(spec.modalities.contains(ModalitySet::TOOL));
    }

    #[test]
    fn parse_proxy_with_url() {
        let spec = parse_register_spec("vision:proxy:http://127.0.0.1:8001/v1").unwrap();
        assert_eq!(spec.id, "vision");
        match spec.backend {
            ExpertBackend::RemoteHttp { url } => assert_eq!(url, "http://127.0.0.1:8001/v1"),
            other => panic!("unexpected backend: {other:?}"),
        }
    }

    #[test]
    fn parse_subprocess_path() {
        let spec = parse_register_spec("local:subprocess:/usr/local/bin/zen-foley").unwrap();
        match spec.backend {
            ExpertBackend::Subprocess { cmd } => {
                assert_eq!(cmd, PathBuf::from("/usr/local/bin/zen-foley"))
            }
            other => panic!("unexpected backend: {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_unknown_kind() {
        assert!(parse_register_spec("x:nope:auto").is_err());
    }

    #[test]
    fn parse_rejects_inprocess_non_auto() {
        assert!(parse_register_spec("x:inprocess:/some/path").is_err());
    }

    #[test]
    fn register_and_lookup() {
        let mut reg = ModelRegistry::new();
        reg.register(ExpertSpec::new(
            "default",
            ExpertBackend::InProcess,
            ModalitySet::TEXT | ModalitySet::TOOL,
        ))
        .unwrap();
        assert!(reg.is_in_process("default"));
        assert_eq!(reg.list().len(), 1);
        assert!(reg.lookup("default").is_some());
        assert!(reg.lookup("missing").is_none());
    }

    #[test]
    fn register_rejects_duplicates() {
        let mut reg = ModelRegistry::new();
        let spec = ExpertSpec::new("x", ExpertBackend::InProcess, ModalitySet::TEXT);
        reg.register(spec.clone()).unwrap();
        assert!(reg.register(spec).is_err());
    }

    #[test]
    fn capability_names_are_stable() {
        let bits = ModalitySet::TEXT | ModalitySet::IMAGE_IN | ModalitySet::TOOL;
        assert_eq!(bits.capability_names(), vec!["text", "image_in", "tool"]);
    }
}
