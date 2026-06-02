//! Worker — local federated trainer.
//!
//! Pure Rust port of `worker.py`. Generic over the user's step function and
//! parameter source: this crate doesn't know what model you're training, only
//! how to push and pull canonical bf16 delta blobs.
//!
//! Usage:
//!
//! ```no_run
//! use hanzo_federation::{Worker, WorkerConfig};
//! use std::sync::{Arc, Mutex};
//!
//! # async fn run() -> anyhow::Result<()> {
//! let state = Arc::new(Mutex::new(vec![0u8; 64]));
//! let s_step = state.clone();
//! let s_params = state.clone();
//! let s_apply = state.clone();
//!
//! let config = WorkerConfig {
//!     coordinator_url: "http://localhost:8443".into(),
//!     worker_name: "m1".into(),
//!     secret: None,
//!     steps_per_round: 8,
//!     total_rounds: 1,
//! };
//! let worker = Worker::new(config);
//! worker.run(
//!     // step_fn: returns loss
//!     move |_batch| {
//!         let mut s = s_step.lock().unwrap();
//!         for b in s.iter_mut() { *b = b.wrapping_add(1); }
//!         0.5
//!     },
//!     // params_iter: returns (name, bf16 LE bytes, shape) per call
//!     move || {
//!         let s = s_params.lock().unwrap();
//!         vec![("x".to_string(), s.clone(), vec![s.len() as u64 / 2])]
//!     },
//!     // apply_fn: write back consensus delta
//!     move |delta| {
//!         let mut s = s_apply.lock().unwrap();
//!         for (_, _, bytes) in delta {
//!             *s = bytes;
//!         }
//!     },
//!     // data_iter: infinite batch iterator
//!     || Box::new(std::iter::repeat(())),
//! ).await
//! # }
//! ```

use anyhow::Result;
use std::time::Instant;

use crate::codec::{decode_delta, encode_delta_with_meta, TensorMeta};
use crate::transport::TransportClient;

/// Static config for a worker.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub coordinator_url: String,
    pub worker_name: String,
    pub secret: Option<String>,
    pub steps_per_round: u32,
    pub total_rounds: u32,
}

/// A single round of work. Calls are sequential: step_fn × N, then push/pull.
pub struct Worker {
    config: WorkerConfig,
    transport: TransportClient,
}

impl std::fmt::Debug for Worker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Worker")
            .field("name", &self.config.worker_name)
            .field("coordinator", &self.config.coordinator_url)
            .finish()
    }
}

impl Worker {
    pub fn new(config: WorkerConfig) -> Self {
        let transport = TransportClient::with_secret(
            config.coordinator_url.clone(),
            config.worker_name.clone(),
            config.secret.clone(),
        );
        Self { config, transport }
    }

    pub fn client(&self) -> &TransportClient {
        &self.transport
    }

    /// Run the training loop.
    ///
    /// * `step_fn(batch) -> loss`
    /// * `params_iter() -> Vec<(name, bf16 LE bytes, shape)>`
    /// * `apply_fn(delta: Vec<(name, meta, bytes)>)`
    /// * `data_iter() -> Box<dyn Iterator<Item = Batch>>`
    pub async fn run<Batch, StepFn, ParamsFn, ApplyFn, DataFn>(
        &self,
        mut step_fn: StepFn,
        mut params_iter: ParamsFn,
        mut apply_fn: ApplyFn,
        mut data_iter: DataFn,
    ) -> Result<()>
    where
        StepFn: FnMut(&Batch) -> f32,
        ParamsFn: FnMut() -> Vec<(String, Vec<u8>, Vec<u64>)>,
        ApplyFn: FnMut(Vec<(String, TensorMeta, Vec<u8>)>),
        DataFn: FnMut() -> Box<dyn Iterator<Item = Batch>>,
    {
        // Sanity: coordinator alive.
        self.transport.healthz().await?;
        let topo = self.transport.topology().await?;
        let my_weight = topo
            .get("data_weights")
            .and_then(|w| w.get(&self.config.worker_name))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        tracing::info!(
            worker = %self.config.worker_name,
            weight = my_weight,
            "starting training loop"
        );

        let mut data = data_iter();
        for round_id in 0..self.config.total_rounds as u64 {
            let round_start = Instant::now();
            let mut losses: Vec<f32> = Vec::with_capacity(self.config.steps_per_round as usize);
            for _ in 0..self.config.steps_per_round {
                let Some(batch) = data.next() else {
                    break;
                };
                losses.push(step_fn(&batch));
            }

            // Snapshot params and push the delta.
            let snapshot = params_iter();
            let items: Vec<(String, &[u8], Vec<u64>)> = snapshot
                .iter()
                .map(|(n, b, s)| (n.clone(), b.as_slice(), s.clone()))
                .collect();
            let blob = encode_delta_with_meta(&items);
            let blob_len = blob.len();
            let push_start = Instant::now();
            self.transport.put_delta(round_id, blob).await?;
            let push_secs = push_start.elapsed().as_secs_f64();

            // Pull the aggregate.
            let agg = self.transport.get_aggregate(round_id).await?;
            let decoded = decode_delta(&agg)?;
            apply_fn(decoded);

            let mean = if losses.is_empty() {
                0.0_f32
            } else {
                losses.iter().sum::<f32>() / losses.len() as f32
            };
            self.transport
                .end_round(
                    round_id,
                    mean as f64,
                    ((round_id + 1) * self.config.steps_per_round as u64) as i64,
                )
                .await?;
            tracing::info!(
                round = round_id,
                steps = self.config.steps_per_round,
                push_s = push_secs,
                delta_mb = blob_len as f64 / 1024.0 / 1024.0,
                loss = mean,
                "round done in {:.1}s",
                round_start.elapsed().as_secs_f64()
            );
        }
        Ok(())
    }
}
