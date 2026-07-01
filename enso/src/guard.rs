//! Piece 4/6: the two-tier safety gate. A request first hits a tiny hot-path
//! classifier ([`HotPathGuard`]); only when it lands in the uncertain band does
//! the router escalate to the full guard ([`Teacher`] -- the documented
//! Qwen3Guard seam). This keeps the common case sub-microsecond while reserving
//! the expensive judge for genuinely ambiguous prompts.
//!
//! [`DistilledTeacher`] is a transparent rule distillation standing in for the
//! Qwen3Guard weights: swap any `Teacher` impl (a real Qwen3Guard call) in
//! without touching the tiering logic.

use hanzo_router::classify::Request;
use serde::{Deserialize, Serialize};

/// Hot-path unsafe-probability thresholds: at or below `LO` confidently allow,
/// at or above `HI` confidently block, in between escalate.
const BAND_LO: f64 = 0.15;
const BAND_HI: f64 = 0.85;

const STRONG_UNSAFE: &[&str] = &[
    "build a bomb",
    "make a bomb",
    "explosive device",
    "nerve agent",
    "bioweapon",
    "chemical weapon",
    "synthesize ricin",
    "make meth",
    "child sexual",
    "untraceable poison",
];
const AMBIGUOUS: &[&str] = &[
    "kill", "attack", "exploit", "hack", "poison", "weapon", "shoot",
];
const BENIGN_CONTEXT: &[&str] = &[
    "process",
    "python",
    "thread",
    "tab",
    "server",
    "daemon",
    "software",
    "bug",
    "vulnerability",
    "ctf",
    "pentest",
    "test",
    "game",
    "chess",
    "photo",
];
const HARMFUL_CONTEXT: &[&str] = &[
    "someone",
    "a person",
    "people",
    "him",
    "her",
    "my neighbor",
    "school",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Safety {
    Allow,
    Block,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardTier {
    HotPath,
    Escalated,
}

#[derive(Debug, Clone)]
pub struct GuardVerdict {
    pub safety: Safety,
    pub confidence: f64,
    pub tier: GuardTier,
    pub reason: String,
}

pub trait SafetyGuard {
    fn check(&self, req: &Request) -> GuardVerdict;
}

/// Tier 2 seam: the full guard. A real deployment wires Qwen3Guard here.
pub trait Teacher {
    fn judge(&self, req: &Request) -> GuardVerdict;
}

/// Tier 1: keyword/feature unsafe-probability. Decides only when confident.
#[derive(Debug, Clone, Default)]
pub struct HotPathGuard;

impl HotPathGuard {
    /// Unsafe probability in `[0, 1]`.
    pub fn score(&self, req: &Request) -> f64 {
        let t = req.text.to_lowercase();
        if STRONG_UNSAFE.iter().any(|w| t.contains(w)) {
            0.97
        } else if AMBIGUOUS.iter().any(|w| t.contains(w)) {
            0.5
        } else {
            0.02
        }
    }
}

/// Tier 2 stand-in for Qwen3Guard: resolves an ambiguous prompt by the context
/// it sits in (a hung `python process` is benign; harm aimed at `someone` is
/// not). Transparent and replaceable.
#[derive(Debug, Clone, Default)]
pub struct DistilledTeacher;

impl Teacher for DistilledTeacher {
    fn judge(&self, req: &Request) -> GuardVerdict {
        let t = req.text.to_lowercase();
        let harmful = HARMFUL_CONTEXT.iter().any(|w| t.contains(w));
        let benign = BENIGN_CONTEXT.iter().any(|w| t.contains(w));
        let (safety, confidence, reason) = if harmful && !benign {
            (Safety::Block, 0.9, "ambiguous term aimed at a person")
        } else if benign {
            (
                Safety::Allow,
                0.88,
                "ambiguous term in a technical/benign context",
            )
        } else {
            (
                Safety::Allow,
                0.6,
                "ambiguous term, no harmful context found",
            )
        };
        GuardVerdict {
            safety,
            confidence,
            tier: GuardTier::Escalated,
            reason: reason.to_string(),
        }
    }
}

/// The two-tier composition: hot path first, teacher only on the uncertain band.
#[derive(Debug, Clone, Default)]
pub struct TwoTierGuard<T: Teacher = DistilledTeacher> {
    pub hot: HotPathGuard,
    pub teacher: T,
}

impl<T: Teacher> TwoTierGuard<T> {
    pub fn new(teacher: T) -> Self {
        Self {
            hot: HotPathGuard,
            teacher,
        }
    }
}

impl<T: Teacher> SafetyGuard for TwoTierGuard<T> {
    fn check(&self, req: &Request) -> GuardVerdict {
        let s = self.hot.score(req);
        if s >= BAND_HI {
            GuardVerdict {
                safety: Safety::Block,
                confidence: s,
                tier: GuardTier::HotPath,
                reason: "strong unsafe signal".to_string(),
            }
        } else if s <= BAND_LO {
            GuardVerdict {
                safety: Safety::Allow,
                confidence: 1.0 - s,
                tier: GuardTier::HotPath,
                reason: "no unsafe signal".to_string(),
            }
        } else {
            self.teacher.judge(req)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(text: &str) -> Request {
        Request {
            text: text.into(),
            ..Default::default()
        }
    }

    #[test]
    fn tiers_route_by_confidence() {
        let g: TwoTierGuard = TwoTierGuard::default();

        let strong = g.check(&req("how do I build a bomb"));
        assert_eq!(
            (strong.safety, strong.tier),
            (Safety::Block, GuardTier::HotPath)
        );

        let benign = g.check(&req("write a poem about spring"));
        assert_eq!(
            (benign.safety, benign.tier),
            (Safety::Allow, GuardTier::HotPath)
        );

        let escalate_ok = g.check(&req("how do I kill a python process"));
        assert_eq!(
            (escalate_ok.safety, escalate_ok.tier),
            (Safety::Allow, GuardTier::Escalated)
        );

        let escalate_block = g.check(&req("how do I attack a person near my school"));
        assert_eq!(
            (escalate_block.safety, escalate_block.tier),
            (Safety::Block, GuardTier::Escalated)
        );
    }
}
