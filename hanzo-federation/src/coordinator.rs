//! Federation coordinator — receives canonical BF16 delta blobs from
//! workers each round, aggregates via DeltaSoup trim-mean, serves the
//! consensus delta back.
//!
//! Stateless w.r.t. model weights: workers hold their own copies and apply
//! the consensus delta locally. The coordinator only ever sees bytes.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Notify;
use tokio::time::sleep;

use crate::codec::{bf16_to_f32, decode_delta, encode_delta_with_meta, f32_to_bf16, TensorMeta};
use crate::scheduler::{Assignment, Scheduler};
use crate::topology::Lab;

/// DeltaSoup-style aggregation modes.
#[derive(Debug, Clone, Copy)]
pub enum AggregationMethod {
    /// Plain element-wise mean.
    Mean,
    /// Per-element median.
    Median,
    /// Trim-mean: drop top+bottom 1 if N≥4, else fall back to mean.
    ByzantineRobust,
}

impl AggregationMethod {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "mean" => Ok(Self::Mean),
            "median" => Ok(Self::Median),
            "byzantine_robust" => Ok(Self::ByzantineRobust),
            other => Err(anyhow!("unknown aggregation method {other:?}")),
        }
    }
}

/// Combine N worker deltas into one consensus delta.
///
/// `deltas[i][name]` is the byte payload for tensor `name` from worker i. The
/// payload format is per-tensor codec-tagged via [`TensorMeta::codec`] — `None`
/// or `Some("bf16")` means raw bf16, `Some("bitdelta")` means BitDelta (only
/// when the `compression` feature is built). Workers may mix codecs across
/// tensors; the consensus output is always emitted as bf16 so that downstream
/// applications keep their existing apply paths unchanged.
///
/// Validates tensor name/shape agreement; returns Err rather than producing
/// silently-wrong output.
pub fn aggregate(
    deltas: &[HashMap<String, (TensorMeta, Vec<u8>)>],
    method: AggregationMethod,
) -> Result<Vec<(String, Vec<u64>, Vec<u8>)>> {
    if deltas.is_empty() {
        return Err(anyhow!("aggregate called with zero deltas"));
    }
    // Preserve insertion order via a BTreeMap keyed by name from the first
    // delta (matches the iteration order Python uses).
    let first = &deltas[0];
    let mut names: Vec<&str> = first.keys().map(|s| s.as_str()).collect();
    names.sort(); // stable order

    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let (meta_0, _bytes_0) = first
            .get(name)
            .ok_or_else(|| anyhow!("first delta missing tensor {name:?}"))?;
        let element_count: usize = meta_0.element_count() as usize;

        // Stack: rows = workers, cols = elements.
        let n = deltas.len();
        let mut stack: Vec<Vec<f32>> = Vec::with_capacity(n);
        for d in deltas {
            let (meta, raw) = d
                .get(name)
                .ok_or_else(|| anyhow!("worker delta missing tensor {name:?}"))?;
            if meta.shape != meta_0.shape {
                return Err(anyhow!(
                    "tensor {name:?} shape mismatch: {:?} vs {:?}",
                    meta.shape,
                    meta_0.shape
                ));
            }
            let values = decode_tensor_to_f32(name, meta, raw)?;
            if values.len() != element_count {
                return Err(anyhow!(
                    "tensor {name:?} element-count mismatch: {} vs {}",
                    values.len(),
                    element_count
                ));
            }
            stack.push(values);
        }

        let agg_f32 = combine_columns(&stack, element_count, method);
        let agg_bytes = f32_to_bf16(&agg_f32);
        out.push((name.to_string(), meta_0.shape.clone(), agg_bytes));
    }
    Ok(out)
}

/// Per-tensor decoder dispatch — see [`crate::codec::decode_delta_to_f32`] for
/// the equivalent whole-blob entry point.
fn decode_tensor_to_f32(name: &str, meta: &TensorMeta, raw: &[u8]) -> Result<Vec<f32>> {
    let codec = meta.codec.as_deref().unwrap_or("bf16");
    match codec {
        "bf16" => Ok(bf16_to_f32(raw)),
        "bitdelta" => {
            #[cfg(feature = "compression")]
            {
                crate::codec_bitdelta::decode_bitdelta_tensor(meta, raw)
                    .map_err(|e| anyhow!("bitdelta decode of {name:?}: {e}"))
            }
            #[cfg(not(feature = "compression"))]
            {
                let _ = (name, meta, raw);
                Err(anyhow!(
                    "tensor {name:?} uses codec=\"bitdelta\" but hanzo-federation was \
                     built without the `compression` feature"
                ))
            }
        }
        other => Err(anyhow!(
            "tensor {name:?} uses unknown codec {other:?}; expected \"bf16\" or \"bitdelta\""
        )),
    }
}

/// Apply the aggregation method along the worker axis (rows of `stack`).
fn combine_columns(stack: &[Vec<f32>], cols: usize, method: AggregationMethod) -> Vec<f32> {
    let n = stack.len();
    let mut out = Vec::with_capacity(cols);
    let mut col = vec![0.0f32; n];
    for c in 0..cols {
        for r in 0..n {
            col[r] = stack[r][c];
        }
        let v = match method {
            AggregationMethod::Mean => col.iter().sum::<f32>() / n as f32,
            AggregationMethod::Median => median_inplace(&mut col),
            AggregationMethod::ByzantineRobust => {
                if n >= 4 {
                    // Sort + drop top/bottom 1, mean the rest.
                    col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let trimmed = &col[1..n - 1];
                    trimmed.iter().sum::<f32>() / trimmed.len() as f32
                } else {
                    col.iter().sum::<f32>() / n as f32
                }
            }
        };
        out.push(v);
    }
    out
}

fn median_inplace(xs: &mut [f32]) -> f32 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

// ── per-round bookkeeping ───────────────────────────────────────────────────

#[derive(Debug)]
struct RoundState {
    round_id: u64,
    expected: HashSet<String>,
    received: BTreeMap<String, Vec<u8>>, // worker → raw blob
    losses: BTreeMap<String, f64>,
    aggregate: Option<Vec<u8>>,
    started_at: Instant,
    started_at_unix: f64,
    completed_at: Option<Instant>,
    /// Notifies any awaiters of `get_aggregate`.
    ready: Arc<Notify>,
}

impl RoundState {
    fn new(round_id: u64, expected: HashSet<String>) -> Self {
        Self {
            round_id,
            expected,
            received: BTreeMap::new(),
            losses: BTreeMap::new(),
            aggregate: None,
            started_at: Instant::now(),
            started_at_unix: now_unix_seconds_f64(),
            completed_at: None,
            ready: Arc::new(Notify::new()),
        }
    }
}

fn now_unix_seconds_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ── coordinator state object ────────────────────────────────────────────────

/// Shared mutable state for the coordinator. Cheap to clone (Arc internally).
#[derive(Clone)]
pub struct CoordinatorState {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for CoordinatorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoordinatorState")
            .field("lab_nodes", &self.inner.lab.nodes.len())
            .field("aggregation", &self.inner.lab.aggregation)
            .finish()
    }
}

struct Inner {
    lab: Lab,
    assignment: Assignment,
    method: AggregationMethod,
    rounds: Mutex<BTreeMap<u64, RoundState>>,
    /// Subset of `lab.workers` that actually has an HMAC secret.
    secrets: HashMap<String, String>,
}

impl CoordinatorState {
    pub fn new(lab: Lab) -> Result<Self> {
        let assignment = Scheduler::new(&lab).plan(&[])?;
        let method = AggregationMethod::parse(&lab.aggregation)?;
        let secrets = lab.secrets();
        tracing::info!(
            workers = lab.workers().count(),
            "coordinator initialized"
        );
        tracing::info!(?assignment.data_weights, "data weights");
        Ok(Self {
            inner: Arc::new(Inner {
                lab,
                assignment,
                method,
                rounds: Mutex::new(BTreeMap::new()),
                secrets,
            }),
        })
    }

    pub fn lab(&self) -> &Lab {
        &self.inner.lab
    }

    pub fn secrets(&self) -> &HashMap<String, String> {
        &self.inner.secrets
    }

    pub fn assignment(&self) -> &Assignment {
        &self.inner.assignment
    }

    /// Topology view — matches the `topology_view` JSON in coordinator.py.
    pub fn topology_view(&self) -> serde_json::Value {
        let workers: Vec<_> = self
            .inner
            .lab
            .workers()
            .map(|n| {
                serde_json::json!({
                    "name": n.name,
                    "host": n.host,
                    "backend": n.backend_hint,
                    "memory_gb": n.memory_gb,
                    "pin_experts": n.pin_experts,
                })
            })
            .collect();
        serde_json::json!({
            "workers": workers,
            "data_weights": self.inner.assignment.data_weights,
            "expert_pins": self.inner.assignment.expert_pins,
            "aggregation": self.inner.lab.aggregation,
            "sync_interval_steps": self.inner.lab.sync_interval_steps,
        })
    }

    /// Metrics — single JSON surface, matches `metrics()` in coordinator.py.
    pub fn metrics(&self) -> serde_json::Value {
        let rounds = self.inner.rounds.lock().unwrap();
        let recent: Vec<_> = rounds
            .iter()
            .rev()
            .take(50)
            .map(|(_, r)| {
                let mut expected: Vec<_> = r.expected.iter().cloned().collect();
                expected.sort();
                let received: Vec<_> = r.received.keys().cloned().collect();
                let duration_s = r
                    .completed_at
                    .map(|c| c.duration_since(r.started_at).as_secs_f64());
                serde_json::json!({
                    "round_id": r.round_id,
                    "expected": expected,
                    "received": received,
                    "losses": r.losses,
                    "aggregated": r.aggregate.is_some(),
                    "started_at": r.started_at_unix,
                    "completed_at": r.completed_at.map(|c| {
                        r.started_at_unix + c.duration_since(r.started_at).as_secs_f64()
                    }),
                    "duration_s": duration_s,
                })
            })
            .collect();
        // Reverse so output is oldest→newest like Python does.
        let rounds_vec: Vec<_> = recent.into_iter().rev().collect();
        let current = rounds.keys().last().copied().map(|x| x as i64).unwrap_or(-1);
        serde_json::json!({
            "topology": self.topology_view(),
            "rounds": rounds_vec,
            "current_round": current,
        })
    }

    /// Worker pushes a delta. Triggers aggregation when the final expected
    /// worker arrives.
    pub fn put_delta(&self, round_id: u64, worker: &str, blob: Vec<u8>) -> Result<()> {
        let (ready, do_agg) = {
            let mut rounds = self.inner.rounds.lock().unwrap();
            let r = rounds.entry(round_id).or_insert_with(|| {
                let expected: HashSet<String> =
                    self.inner.lab.workers().map(|n| n.name.clone()).collect();
                RoundState::new(round_id, expected)
            });
            r.received.insert(worker.to_string(), blob);
            let received_n = r.received.len();
            let expected_n = r.expected.len();
            tracing::info!(
                round = round_id,
                worker = worker,
                "{received_n}/{expected_n} deltas in"
            );
            let do_agg = r.aggregate.is_none()
                && r.received.keys().cloned().collect::<HashSet<_>>() == r.expected;
            (r.ready.clone(), do_agg)
        };

        if do_agg {
            self.run_aggregation(round_id)?;
            ready.notify_waiters();
        }
        Ok(())
    }

    /// Block (async) until the aggregate for `round_id` is ready; returns
    /// the canonical blob. 10-minute timeout matching Python.
    pub async fn get_aggregate(&self, round_id: u64) -> Result<Vec<u8>> {
        // Fast path.
        let notify = {
            let rounds = self.inner.rounds.lock().unwrap();
            if let Some(r) = rounds.get(&round_id) {
                if let Some(b) = &r.aggregate {
                    return Ok(b.clone());
                }
                r.ready.clone()
            } else {
                // No round at all yet — create one and wait.
                drop(rounds);
                let mut rounds = self.inner.rounds.lock().unwrap();
                let r = rounds.entry(round_id).or_insert_with(|| {
                    let expected: HashSet<String> =
                        self.inner.lab.workers().map(|n| n.name.clone()).collect();
                    RoundState::new(round_id, expected)
                });
                r.ready.clone()
            }
        };
        // Wait up to 10 minutes for completion.
        let deadline = Instant::now() + Duration::from_secs(600);
        loop {
            let waited = tokio::time::timeout(Duration::from_secs(1), notify.notified()).await;
            {
                let rounds = self.inner.rounds.lock().unwrap();
                if let Some(r) = rounds.get(&round_id) {
                    if let Some(b) = &r.aggregate {
                        return Ok(b.clone());
                    }
                }
            }
            if waited.is_err() && Instant::now() >= deadline {
                return Err(anyhow!("round {round_id} not aggregated within 10 min"));
            }
            // tiny yield to avoid busy spinning if notify resolved spuriously
            sleep(Duration::from_millis(10)).await;
        }
    }

    pub fn end_round(&self, round_id: u64, worker: &str, loss: Option<f64>, _step: Option<i64>) {
        let mut rounds = self.inner.rounds.lock().unwrap();
        if let Some(r) = rounds.get_mut(&round_id) {
            if let Some(l) = loss {
                r.losses.insert(worker.to_string(), l);
            }
        }
    }

    fn run_aggregation(&self, round_id: u64) -> Result<()> {
        // Decode each blob under the lock briefly to clone bytes out, then
        // release for the heavy math.
        let blobs: Vec<Vec<u8>> = {
            let rounds = self.inner.rounds.lock().unwrap();
            let r = rounds
                .get(&round_id)
                .ok_or_else(|| anyhow!("round {round_id} vanished mid-aggregate"))?;
            r.received.values().cloned().collect()
        };
        tracing::info!(round = round_id, n = blobs.len(), "aggregating");

        let decoded: Vec<HashMap<String, (TensorMeta, Vec<u8>)>> = blobs
            .iter()
            .map(|b| {
                decode_delta(b).map(|triples| {
                    triples
                        .into_iter()
                        .map(|(name, meta, raw)| (name, (meta, raw)))
                        .collect()
                })
            })
            .collect::<Result<_>>()?;

        let agg = aggregate(&decoded, self.inner.method)?;
        let items: Vec<(String, &[u8], Vec<u64>)> = agg
            .iter()
            .map(|(name, shape, bytes)| (name.clone(), bytes.as_slice(), shape.clone()))
            .collect();
        let blob = encode_delta_with_meta(&items);

        let mut rounds = self.inner.rounds.lock().unwrap();
        if let Some(r) = rounds.get_mut(&round_id) {
            r.aggregate = Some(blob);
            r.completed_at = Some(Instant::now());
            let dur = r
                .completed_at
                .unwrap()
                .duration_since(r.started_at)
                .as_secs_f64();
            tracing::info!(round = round_id, secs = dur, "round done");
        }
        Ok(())
    }
}

/// Thin façade matching the Python `Coordinator` class — keeps the example
/// binary readable.
#[derive(Debug, Clone)]
pub struct Coordinator {
    pub state: CoordinatorState,
}

impl Coordinator {
    pub fn new(lab: Lab) -> Result<Self> {
        Ok(Self {
            state: CoordinatorState::new(lab)?,
        })
    }

    /// Serve forever on `bind`. Returns when the server exits.
    pub async fn serve(self, bind: std::net::SocketAddr) -> Result<()> {
        crate::transport::server::serve(self.state, bind).await
    }
}

// re-export for convenience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndRoundPayload {
    #[serde(default)]
    pub loss: Option<f64>,
    #[serde(default)]
    pub step: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::f32_to_bf16;
    use std::collections::HashMap;

    fn one_tensor(name: &str, vals: &[f32]) -> HashMap<String, (TensorMeta, Vec<u8>)> {
        let raw = f32_to_bf16(vals);
        let meta = TensorMeta {
            dtype: "BF16".into(),
            shape: vec![vals.len() as u64],
            offsets: [0, raw.len() as u64],
            codec: None,
        };
        let mut m = HashMap::new();
        m.insert(name.to_string(), (meta, raw));
        m
    }

    #[test]
    fn mean_of_two() {
        let a = one_tensor("x", &[1.0, 2.0, 3.0]);
        let b = one_tensor("x", &[3.0, 4.0, 5.0]);
        let out = aggregate(&[a, b], AggregationMethod::Mean).unwrap();
        let f = bf16_to_f32(&out[0].2);
        // bf16 truncation tolerance
        for (got, want) in f.iter().zip(&[2.0, 3.0, 4.0]) {
            assert!((got - want).abs() < 0.05, "got {got}, want {want}");
        }
    }

    #[test]
    fn trim_mean_n4_drops_extremes() {
        let workers: Vec<_> = [100.0_f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|v| one_tensor("x", &[*v]))
            .collect();
        let out = aggregate(&workers, AggregationMethod::ByzantineRobust).unwrap();
        let f = bf16_to_f32(&out[0].2);
        // After dropping 100.0 and 2.0, mean(3.0, 4.0) = 3.5
        assert!((f[0] - 3.5).abs() < 0.05, "got {}", f[0]);
    }

    #[test]
    fn trim_mean_n3_falls_back_to_mean() {
        let workers: Vec<_> = [1.0_f32, 2.0, 3.0]
            .iter()
            .map(|v| one_tensor("x", &[*v]))
            .collect();
        let out = aggregate(&workers, AggregationMethod::ByzantineRobust).unwrap();
        let f = bf16_to_f32(&out[0].2);
        assert!((f[0] - 2.0).abs() < 0.05);
    }

    #[test]
    fn shape_mismatch_errors() {
        let a = one_tensor("x", &[1.0, 2.0]);
        let b = one_tensor("x", &[1.0, 2.0, 3.0]);
        assert!(aggregate(&[a, b], AggregationMethod::Mean).is_err());
    }
}
