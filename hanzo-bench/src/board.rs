//! A run directory, scored.
//!
//! scripts/dossier.sh leaves, per shape, `hanzo_<tag>.json` (this binary's raw per-repetition
//! `[wall_s, tokens]`) and `llama_<tag>.json` (`llama-bench -o json`) beside a manifest that pins
//! both engines, the model and the box. Everything published is a pure function of those samples:
//! the board, and the evidence filed in Hanzo Research, which the site reads. A board is only ever
//! written, never read back.
//!
//! ```text
//! tok/s    = tokens / wall_s / concurrency     ours; the rival reports its own
//! 95% CI   = t(0.975, n-1) * s / sqrt(n)       Student t, sample deviation
//! CV%      = 100 * s / mean                    flagged above 5
//! best     = the fastest repetition taken, a discarded warmup included
//! ratio    = ours / theirs, relative intervals added in quadrature
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

// ---- values ---------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Prefill,
    Decode,
}

/// What a cell measures. Ordered as a board reads: prefill by length, then decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Shape {
    pub phase: Phase,
    /// Prompt tokens for prefill; generated tokens for decode.
    pub n: usize,
    pub concurrency: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Verdict {
    Win,
    Loss,
    Parity,
}

/// Deviation, 95% half-interval and coefficient of variation: what two or more samples have.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Spread {
    pub std: f64,
    pub ci: f64,
    pub cv: f64,
}

/// One engine on one shape. `samples` are the scored repetitions in tok/s and every other field
/// is a function of them, except that `best` also counts a repetition discarded as warmup.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Stats {
    pub mean: f64,
    pub n: usize,
    pub best: f64,
    #[serde(flatten)]
    pub spread: Option<Spread>,
    pub samples: Vec<f64>,
}

/// Ours over theirs. An interval wholly above 1 is a win, wholly below a loss.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Ratio {
    pub ratio: f64,
    pub ci: f64,
    pub lo: f64,
    pub hi: f64,
    pub verdict: Verdict,
    pub best: f64,
}

/// A variance above this, in percent, says the box was not quiet.
pub const NOISY_CV: f64 = 5.0;

/// Student t, two-sided 0.975, by degrees of freedom; the normal 1.96 past the table.
const T975: [f64; 30] = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
    2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
    2.052, 2.048, 2.045, 2.042,
];

impl Stats {
    pub fn of(samples: &[f64]) -> Option<Self> {
        let n = samples.len();
        let best = samples.iter().copied().reduce(f64::max)?;
        let mean = samples.iter().sum::<f64>() / n as f64;
        let spread = (n > 1).then(|| {
            let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            let std = var.sqrt();
            let t = T975.get(n - 2).copied().unwrap_or(1.96);
            Spread {
                std,
                ci: t * std / (n as f64).sqrt(),
                cv: 100.0 * std / mean,
            }
        });
        Some(Self {
            mean,
            n,
            best,
            spread,
            samples: samples.to_vec(),
        })
    }

    pub fn noisy(&self) -> bool {
        self.spread.as_ref().is_some_and(|s| s.cv > NOISY_CV)
    }

    fn relative_ci(&self) -> f64 {
        self.spread.as_ref().map_or(0.0, |s| s.ci / self.mean)
    }
}

impl Ratio {
    pub fn of(ours: &Stats, theirs: &Stats) -> Self {
        let ratio = ours.mean / theirs.mean;
        let ci = ratio * ours.relative_ci().hypot(theirs.relative_ci());
        let (lo, hi) = (ratio - ci, ratio + ci);
        let verdict = if lo > 1.0 {
            Verdict::Win
        } else if hi < 1.0 {
            Verdict::Loss
        } else {
            Verdict::Parity
        };
        Self {
            ratio,
            ci,
            lo,
            hi,
            verdict,
            best: ours.best / theirs.best,
        }
    }
}

impl Verdict {
    pub fn name(self) -> &'static str {
        match self {
            Self::Win => "WIN",
            Self::Loss => "LOSS",
            Self::Parity => "PARITY",
        }
    }
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

// ---- sources: what each engine leaves in a run directory ------------------------------------

/// One shape as this binary timed it: per repetition, the wall seconds of the whole concurrency
/// batch and the tokens it scored (prompt tokens for prefill, generated tokens for decode). The
/// clock is ours rather than the response's self-reported rate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Timed {
    #[serde(flatten)]
    pub shape: Shape,
    pub per_rep: Vec<(f64, usize)>,
}

/// What `--json` writes: a run's raw samples and what produced them.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Samples {
    pub engine_version: String,
    pub backend: String,
    #[serde(default)]
    pub sampler: String,
    pub model_id: String,
    pub results: Vec<Timed>,
}

/// One row of `llama-bench -o json`.
#[derive(Deserialize)]
struct LlamaRow {
    #[serde(default)]
    n_prompt: usize,
    #[serde(default)]
    n_gen: usize,
    #[serde(default)]
    samples_ts: Vec<f64>,
}

/// A result file, as the shapes it measured.
trait Source: DeserializeOwned {
    fn cells(&self) -> Vec<(Shape, Stats)>;
}

fn rates(per_rep: &[(f64, usize)], concurrency: usize) -> Vec<f64> {
    per_rep
        .iter()
        .filter(|(secs, toks)| *secs > 0.0 && *toks > 0)
        .map(|(secs, toks)| *toks as f64 / secs / concurrency as f64)
        .collect()
}

impl Timed {
    /// The first repetition is warmup when at least three were taken: this binary warms a shape
    /// once, so its first large repetition still pays pipeline compilation, while llama-bench
    /// warms every shape itself. `best` counts it all the same: a best is a best.
    pub fn stats(&self) -> Option<Stats> {
        let scored = match self.per_rep.len() {
            0..=2 => &self.per_rep[..],
            _ => &self.per_rep[1..],
        };
        let best = Stats::of(&rates(&self.per_rep, self.shape.concurrency))?.best;
        Some(Stats {
            best,
            ..Stats::of(&rates(scored, self.shape.concurrency))?
        })
    }
}

impl Source for Samples {
    fn cells(&self) -> Vec<(Shape, Stats)> {
        self.results
            .iter()
            .filter_map(|t| Some((t.shape, t.stats()?)))
            .collect()
    }
}

impl Source for Vec<LlamaRow> {
    fn cells(&self) -> Vec<(Shape, Stats)> {
        self.iter()
            .filter_map(|row| {
                let (phase, n) = if row.n_prompt > 0 && row.n_gen == 0 {
                    (Phase::Prefill, row.n_prompt)
                } else {
                    (Phase::Decode, row.n_gen)
                };
                let shape = Shape {
                    phase,
                    n,
                    concurrency: 1,
                };
                Some((shape, Stats::of(&row.samples_ts)?))
            })
            .collect()
    }
}

// ---- a run, scored --------------------------------------------------------------------------

/// The rival's side of a cell. A ratio exists exactly when the rival measured the shape.
#[derive(Clone, Debug, Serialize)]
pub struct Rival {
    pub llama: Stats,
    pub llama_src: String,
    pub ratio: Ratio,
}

#[derive(Clone, Debug, Serialize)]
pub struct Cell {
    #[serde(flatten)]
    pub shape: Shape,
    pub hanzo: Stats,
    pub hanzo_src: String,
    #[serde(flatten)]
    pub rival: Option<Rival>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Run {
    pub run: String,
    pub backend: String,
    pub model: String,
    pub engine_version: String,
    pub cells: Vec<Cell>,
    #[serde(skip)]
    pub manifest: Option<Manifest>,
}

fn parse<T: DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_str(&std::fs::read_to_string(path)?)
        .with_context(|| path.display().to_string())
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// `<prefix>*.json` in `dir`, by name.
fn named(dir: &Path, prefix: &str) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| dir.display().to_string())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = file_name(p);
            name.starts_with(prefix) && name.ends_with(".json")
        })
        .collect();
    paths.sort();
    Ok(paths)
}

fn by_shape<S: Source>(files: &[(String, S)]) -> BTreeMap<Shape, (Stats, String)> {
    files
        .iter()
        .flat_map(|(src, file)| {
            file.cells()
                .into_iter()
                .map(move |(shape, stats)| (shape, (stats, src.clone())))
        })
        .collect()
}

/// Read and score a run directory. Engines pair by shape, not by file name.
pub fn read(dir: &Path) -> Result<Run> {
    let ours = named(dir, "hanzo_")?
        .iter()
        .map(|p| Ok((file_name(p), parse::<Samples>(p)?)))
        .collect::<Result<Vec<_>>>()?;
    // Where the rival did not run, the harness leaves a marker object; it is not a result.
    let theirs: Vec<(String, Vec<LlamaRow>)> = named(dir, "llama_")?
        .iter()
        .filter_map(|p| parse(p).ok().map(|rows| (file_name(p), rows)))
        .collect();
    let Some((_, head)) = ours.first() else {
        bail!("{}: no hanzo_*.json", dir.display());
    };
    let rival = by_shape(&theirs);
    let cells = by_shape(&ours)
        .into_iter()
        .map(|(shape, (hanzo, hanzo_src))| {
            let rival = rival.get(&shape).map(|(llama, llama_src)| Rival {
                ratio: Ratio::of(&hanzo, llama),
                llama: llama.clone(),
                llama_src: llama_src.clone(),
            });
            Cell {
                shape,
                hanzo,
                hanzo_src,
                rival,
            }
        })
        .collect();
    let manifest = dir.join("manifest.json");
    Ok(Run {
        run: file_name(dir),
        backend: head.backend.clone(),
        model: head.model_id.clone(),
        engine_version: head.engine_version.clone(),
        cells,
        manifest: manifest.is_file().then(|| parse(&manifest)).transpose()?,
    })
}

/// Score a run directory in place: board.md for a reader, board.json for a machine, and the
/// paper's results-data.tex and board.tex.
pub fn score(dir: &Path) -> Result<Run> {
    let run = read(dir)?;
    std::fs::write(dir.join("board.md"), markdown(&run))?;
    std::fs::write(dir.join("board.json"), serde_json::to_string_pretty(&run)?)?;
    std::fs::write(dir.join("results-data.tex"), macros(&run))?;
    std::fs::write(dir.join("board.tex"), tabular(&run))?;
    Ok(run)
}

// ---- renderings -----------------------------------------------------------------------------

/// `mean±ci`, flagged `!` where the variance says the box was not quiet.
fn plus_minus(s: &Stats, pm: &str, flag: bool) -> String {
    match &s.spread {
        None => format!("{:.1}", s.mean),
        Some(sp) => {
            let flag = if flag && s.noisy() { "!" } else { "" };
            format!("{:.1}{pm}{:.1}{flag}", s.mean, sp.ci)
        }
    }
}

fn or_dash(s: Option<String>) -> String {
    s.unwrap_or_else(|| "--".into())
}

pub fn markdown(run: &Run) -> String {
    let model = run.model.rsplit('/').next().unwrap_or("");
    let model: String = model.chars().take(20).collect();
    let mut out = String::from(
        "| model | backend | phase | n | conc | hanzo t/s | llama t/s | ratio | verdict | best hanzo | best llama | best ratio |\n\
         |---|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    for c in &run.cells {
        let r = c.rival.as_ref();
        out += &format!(
            "| {model} | {} | {} | {} | {} | {} | {} | {} | {} | {:.2} | {} | {} |\n",
            run.backend,
            c.shape.phase.name(),
            c.shape.n,
            c.shape.concurrency,
            plus_minus(&c.hanzo, "±", true),
            or_dash(r.map(|r| plus_minus(&r.llama, "±", true))),
            or_dash(r.map(|r| format!("{:.3}±{:.3}", r.ratio.ratio, r.ratio.ci))),
            or_dash(r.map(|r| r.ratio.verdict.name().into())),
            c.hanzo.best,
            or_dash(r.map(|r| format!("{:.2}", r.llama.best))),
            or_dash(r.map(|r| format!("{:.3}", r.ratio.best))),
        );
    }
    out
}

// LaTeX control sequences are letters only, so a cell's macro key is the backend's title and a
// spelled role: canonical shapes are words, any other length spells its digits.
fn spell(n: usize) -> String {
    n.to_string()
        .bytes()
        .map(|d| b"ZOTHFVSNEI"[(d - b'0') as usize] as char)
        .collect()
}

fn title(backend: &str) -> String {
    let mut letters = backend.chars().filter(char::is_ascii_alphabetic);
    match letters.next() {
        Some(first) => first
            .to_uppercase()
            .chain(letters.flat_map(char::to_lowercase))
            .collect(),
        None => String::new(),
    }
}

fn macro_key(backend: &str, shape: Shape) -> String {
    let role = match (shape.phase, shape.n) {
        (Phase::Decode, _) => "Decode".to_string(),
        (Phase::Prefill, 512) => "Prefill".to_string(),
        (Phase::Prefill, 500) => "PrefillRagged".to_string(),
        (Phase::Prefill, 2048) => "PrefillLong".to_string(),
        (Phase::Prefill, 4096) => "PrefillMax".to_string(),
        (Phase::Prefill, n) => format!("Prefill{}", spell(n)),
    };
    let streams = match shape.concurrency {
        1 => String::new(),
        c => format!("C{}", spell(c)),
    };
    format!("r{}{role}{streams}", title(backend))
}

/// results-data.tex: one `\renewcommand` per figure the paper quotes.
pub fn macros(run: &Run) -> String {
    let mut lines = BTreeSet::new();
    let mut set = |key: String, value: String| {
        lines.insert(format!("\\renewcommand{{\\{key}}}{{{value}}}"));
    };
    // Decode is memory-bound, so tok/s times the bytes streamed per token is an effective
    // bandwidth, with the model file an upper bound on those bytes. Both engines stream the same
    // weights, so the reach is exact whatever the byte estimate.
    let model_gb = run.manifest.as_ref().and_then(|m| m.model_bytes);
    let model_gb = model_gb.map(|b| b as f64 / 1e9);
    for c in &run.cells {
        let key = macro_key(&run.backend, c.shape);
        set(format!("{key}H"), format!("{:.1}", c.hanzo.mean));
        if let Some(sp) = &c.hanzo.spread {
            set(format!("{key}Hci"), format!("{:.1}", sp.ci));
        }
        let Some(r) = &c.rival else { continue };
        set(format!("{key}L"), format!("{:.1}", r.llama.mean));
        set(format!("{key}R"), format!("{:.2}", r.ratio.ratio));
        set(format!("{key}V"), r.ratio.verdict.name().into());
        if let (Phase::Decode, 1, Some(gb)) = (c.shape.phase, c.shape.concurrency, model_gb) {
            let b = title(&run.backend);
            set(format!("r{b}EffBW"), format!("{:.0}", c.hanzo.mean * gb));
            set(format!("r{b}RoofBW"), format!("{:.0}", r.llama.mean * gb));
            set(
                format!("r{b}Reach"),
                format!("{:.0}\\%", 100.0 * r.ratio.ratio),
            );
        }
    }
    let mut out = String::from(
        "% GENERATED by hanzo-bench score -- do not hand-edit.\n\
         % Every number is a pure function of the raw per-rep samples in this run dir.\n\
         % '!' in board.md marks CV>5% (timing variance; box was not fully quiet).\n",
    );
    for line in lines {
        out += &line;
        out += "\n";
    }
    out
}

/// board.tex: a standalone booktabs tabular, for `\input` at float level.
pub fn tabular(run: &Run) -> String {
    let mut out = String::from(
        "\\begin{tabular}{@{}llrrrrl@{}}\n\\toprule\n\
         backend & phase & $n$ & hanzo t/s & llama t/s & ratio & verdict \\\\\n\\midrule\n",
    );
    for c in &run.cells {
        let r = c.rival.as_ref();
        out += &format!(
            "{} & {} & {} & {} & {} & {} & {} \\\\\n",
            run.backend,
            c.shape.phase.name(),
            c.shape.n,
            plus_minus(&c.hanzo, "$\\pm$", false),
            or_dash(r.map(|r| plus_minus(&r.llama, "$\\pm$", false))),
            or_dash(r.map(|r| format!("{:.2}$\\pm${:.2}", r.ratio.ratio, r.ratio.ci))),
            or_dash(r.map(|r| format!("\\textsc{{{}}}", r.ratio.verdict.name().to_lowercase()))),
        );
    }
    out + "\\bottomrule\n\\end{tabular}\n"
}

// ---- the manifest: what a run pins before its first sample ----------------------------------

#[derive(Clone, Debug, Default, Serialize, Deserialize, clap::Args)]
#[serde(default)]
pub struct Params {
    #[arg(long)]
    pub prompt_sizes: Option<String>,
    #[arg(long, default_value_t = 0)]
    pub n_gen: u64,
    #[arg(long, default_value_t = 0)]
    pub reps: u64,
    #[arg(long, default_value_t = 1)]
    pub concurrency: u64,
    #[arg(long, default_value_t = 0)]
    pub max_ctx: u64,
    /// Arguments that reached hanzo-bench verbatim.
    #[arg(long, allow_hyphen_values = true, default_value = "")]
    pub hanzo_args: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, clap::Args)]
#[serde(default)]
pub struct Env {
    #[arg(long, allow_hyphen_values = true)]
    pub backend_env: Option<String>,
}

/// What the harness knows about a run and passes in; `pin` adds what is read off the box.
#[derive(Clone, Debug, Default, Serialize, Deserialize, clap::Args)]
#[serde(default)]
pub struct Pins {
    #[arg(long)]
    pub backend: Option<String>,
    #[arg(long)]
    pub sampler: Option<String>,
    #[arg(long)]
    pub engine_git_sha: Option<String>,
    #[arg(long, action = clap::ArgAction::Set, default_value_t = false)]
    pub engine_git_dirty: bool,
    #[arg(long)]
    pub engine_version: Option<String>,
    #[arg(long)]
    pub hanzo_ml: Option<String>,
    #[arg(long)]
    pub hanzo_rocm_kernels: Option<String>,
    #[arg(long)]
    pub hanzo_metal_kernels: Option<String>,
    #[arg(long)]
    pub llama_sha: Option<String>,
    #[arg(long, action = clap::ArgAction::Set, default_value_t = false)]
    pub llama_git_dirty: bool,
    #[arg(long)]
    pub hanzo_bench_bin: Option<String>,
    #[arg(long)]
    pub llama_bench_bin: Option<String>,
    #[arg(long)]
    pub model_path: Option<String>,
    #[arg(long)]
    pub gpu: Option<String>,
    #[command(flatten)]
    pub params: Params,
    #[command(flatten)]
    pub env: Env,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Manifest {
    pub host: Option<String>,
    pub uname: Option<String>,
    pub timestamp: f64,
    pub iso: Option<String>,
    #[serde(flatten)]
    pub pins: Pins,
    pub hanzo_bench_bin_sha256: Option<String>,
    pub llama_bench_bin_sha256: Option<String>,
    pub model_bytes: Option<u64>,
    pub model_sha256: Option<String>,
}

fn sha256(path: Option<&String>) -> Result<Option<String>> {
    use sha2::Digest;
    let Some(path) = path.filter(|p| Path::new(p).is_file()) else {
        return Ok(None);
    };
    let mut hasher = sha2::Sha256::new();
    std::io::copy(&mut std::fs::File::open(path)?, &mut hasher)?;
    Ok(Some(format!("{:x}", hasher.finalize())))
}

fn uname() -> (String, String) {
    // SAFETY: utsname is plain data the kernel fills; a failed call leaves it zeroed.
    let mut u: libc::utsname = unsafe { std::mem::zeroed() };
    unsafe { libc::uname(&mut u) };
    let field = |b: &[libc::c_char]| -> String {
        let bytes: Vec<u8> = b
            .iter()
            .take_while(|c| **c != 0)
            .map(|c| *c as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    };
    let all = [
        &u.sysname[..],
        &u.nodename,
        &u.release,
        &u.version,
        &u.machine,
    ]
    .map(field);
    (all[1].clone(), all.join(" "))
}

/// Pin a run: the harness's facts, the box's name and clock, and digests of both binaries and
/// the model, so a binary built from an uncommitted tree cannot pass for its commit.
pub fn pin(pins: Pins) -> Result<Manifest> {
    let (host, uname) = uname();
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
    let model = pins.model_path.as_ref();
    Ok(Manifest {
        host: Some(host),
        uname: Some(uname),
        timestamp: now.as_secs_f64(),
        iso: Some(
            chrono::Local::now()
                .format("%Y-%m-%dT%H:%M:%S%z")
                .to_string(),
        ),
        hanzo_bench_bin_sha256: sha256(pins.hanzo_bench_bin.as_ref())?,
        llama_bench_bin_sha256: sha256(pins.llama_bench_bin.as_ref())?,
        model_bytes: model
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| m.len()),
        model_sha256: sha256(model)?,
        pins,
    })
}

/// `/x/Qwen_Qwen3-1.7B-Q4_K_M.gguf` is `Qwen3-1.7B-Q4_K_M`: the file's stem, less a vendor
/// prefix that only repeats the family name.
pub fn model_name(path: &str) -> String {
    let stem = Path::new(path).file_stem();
    let stem = stem
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    match stem.split_once('_') {
        Some((vendor, rest))
            if !vendor.is_empty()
                && vendor.bytes().all(|b| b.is_ascii_alphanumeric())
                && rest.starts_with(vendor) =>
        {
            rest.to_string()
        }
        _ => stem,
    }
}

// ---- evidence: a run as Hanzo Research files it ---------------------------------------------

pub const KIND: &str = "kernel-perf";

#[derive(Debug, Serialize)]
pub struct Versus {
    pub subject: String,
    #[serde(flatten)]
    pub ratio: Ratio,
}

#[derive(Debug, Serialize)]
pub struct Meta {
    pub run: String,
    pub host: Option<String>,
    /// The backend as the engine names it (`ROCm`), and the model by its file.
    pub backend: String,
    pub model: String,
    pub model_sha256: Option<String>,
    pub model_bytes: Option<u64>,
    pub gpu: Option<String>,
    pub sampler: Option<String>,
    pub reps: u64,
    pub concurrency: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<String>,
    pub ci95: Option<f64>,
    pub cv_pct: Option<f64>,
    pub std: Option<f64>,
    pub samples: Vec<f64>,
    pub noisy: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub versus: Option<Versus>,
}

/// One `POST /v1/research/experiments` member. The stable id is
/// `kernel-perf:<engine>/<backend>/<host>/<model>:<phase>-<n>`; research keeps versions under it
/// and is idempotent by content, so a later revision appends and a repeat adds nothing.
#[derive(Debug, Serialize)]
pub struct Experiment {
    pub id: String,
    pub kind: &'static str,
    pub subject: String,
    pub task: String,
    pub metric: &'static str,
    pub value: f64,
    pub n: usize,
    pub meta: Meta,
    pub git_sha: String,
    pub git_dirty: bool,
    pub lib_versions: BTreeMap<&'static str, String>,
    pub ts: i64,
}

/// Who measured: the subject a result is filed under and the commit that produced it.
struct Engine {
    subject: String,
    git_sha: String,
    git_dirty: bool,
    lib_versions: BTreeMap<&'static str, String>,
    args: Option<String>,
}

/// Every cell of every run as evidence: ours, and the rival's beside it where it ran.
pub fn evidence(runs: &[Run]) -> Result<Vec<Experiment>> {
    let filed = runs.iter().map(filed).collect::<Result<Vec<_>>>()?;
    Ok(filed.into_iter().flatten().collect())
}

fn filed(run: &Run) -> Result<Vec<Experiment>> {
    let Some(m) = &run.manifest else {
        bail!("{}: no manifest.json; nothing pins this run", run.run);
    };
    let pins = &m.pins;
    let text = |s: &Option<String>| s.clone().unwrap_or_default();
    let model = model_name(&text(&pins.model_path));
    let place = format!("{}/{}/{model}", text(&pins.backend), text(&m.host));
    let ours = Engine {
        subject: format!("hanzo-engine/{place}"),
        git_sha: text(&pins.engine_git_sha),
        git_dirty: pins.engine_git_dirty,
        lib_versions: [
            ("hanzo-engine", &pins.engine_version),
            ("hanzo-ml", &pins.hanzo_ml),
            ("hanzo-rocm-kernels", &pins.hanzo_rocm_kernels),
            ("hanzo-metal-kernels", &pins.hanzo_metal_kernels),
        ]
        .into_iter()
        .filter_map(|(k, v)| Some((k, v.clone().filter(|v| !v.is_empty())?)))
        .collect(),
        args: Some(pins.params.hanzo_args.clone()),
    };
    let theirs = Engine {
        subject: format!("llama.cpp/{place}"),
        git_sha: text(&pins.llama_sha),
        git_dirty: pins.llama_git_dirty,
        lib_versions: BTreeMap::new(),
        args: None,
    };
    let file = |by: &Engine, shape: Shape, stats: &Stats, noisy: bool, versus: Option<Versus>| {
        let task = format!("{}-{}", shape.phase.name(), shape.n);
        let spread = stats.spread.as_ref();
        Experiment {
            id: format!("{KIND}:{}:{task}", by.subject),
            kind: KIND,
            subject: by.subject.clone(),
            task,
            metric: "tok/s",
            value: stats.mean,
            n: stats.n,
            meta: Meta {
                run: run.run.clone(),
                host: m.host.clone(),
                backend: run.backend.clone(),
                model: model.clone(),
                model_sha256: m.model_sha256.clone(),
                model_bytes: m.model_bytes,
                gpu: pins.gpu.clone(),
                sampler: pins.sampler.clone(),
                reps: pins.params.reps,
                concurrency: pins.params.concurrency,
                args: by.args.clone(),
                ci95: spread.map(|s| s.ci),
                cv_pct: spread.map(|s| s.cv),
                std: spread.map(|s| s.std),
                samples: stats.samples.clone(),
                noisy,
                versus,
            },
            git_sha: by.git_sha.clone(),
            git_dirty: by.git_dirty,
            lib_versions: by.lib_versions.clone(),
            ts: m.timestamp as i64,
        }
    };
    Ok(run
        .cells
        .iter()
        .flat_map(|c| {
            let r = c.rival.as_ref();
            let noisy = c.hanzo.noisy() || r.is_some_and(|r| r.llama.noisy());
            let versus = r.map(|r| Versus {
                subject: theirs.subject.clone(),
                ratio: r.ratio.clone(),
            });
            let mine = file(&ours, c.shape, &c.hanzo, noisy, versus);
            std::iter::once(mine).chain(r.map(|r| file(&theirs, c.shape, &r.llama, noisy, None)))
        })
        .collect())
}

/// File evidence at `POST /v1/research/experiments`, bearer `$HANZO_API_KEY`, 200 to a batch.
pub fn post(url: &str, evidence: &[Experiment]) -> Result<()> {
    let key = std::env::var("HANZO_API_KEY")
        .ok()
        .filter(|k| !k.is_empty());
    let key = key.context("publishing needs $HANZO_API_KEY")?;
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;
    for batch in evidence.chunks(200) {
        let resp = client
            .post(url)
            .bearer_auth(&key)
            .json(&serde_json::json!({"experiments": batch, "attempts": []}))
            .send()?;
        let status = resp.status();
        println!("{}", resp.text()?);
        if !status.is_success() {
            bail!("{url} answered {status}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(phase: Phase, n: usize, concurrency: usize) -> Shape {
        Shape {
            phase,
            n,
            concurrency,
        }
    }

    #[test]
    fn student_t_interval_and_the_variance_flag() {
        let s = Stats::of(&[10.0, 11.0, 12.0]).unwrap();
        let sp = s.spread.clone().unwrap();
        assert!((s.mean - 11.0).abs() < 1e-12 && (sp.std - 1.0).abs() < 1e-12);
        assert!((sp.ci - 4.303 / 3f64.sqrt()).abs() < 1e-9); // t(0.975, 2) = 4.303
        assert!(s.best == 12.0 && s.noisy());
        let one = Stats::of(&[7.0]).unwrap();
        assert!(one.spread.is_none() && one.n == 1 && one.best == 7.0 && !one.noisy());
        assert!(Stats::of(&[]).is_none());
    }

    #[test]
    fn warmup_leaves_the_mean_and_stays_in_best() {
        let timed = |per_rep: Vec<(f64, usize)>| Timed {
            shape: shape(Phase::Decode, 128, 1),
            per_rep,
        };
        let three = timed(vec![(1.0, 200), (1.0, 100), (1.0, 100)])
            .stats()
            .unwrap();
        assert_eq!((three.samples, three.best), (vec![100.0, 100.0], 200.0));
        let two = timed(vec![(1.0, 200), (1.0, 100)]).stats().unwrap();
        assert_eq!(two.samples, vec![200.0, 100.0]);
        assert!(timed(vec![(0.0, 0)]).stats().is_none());
    }

    #[test]
    fn a_verdict_needs_the_whole_interval() {
        let h = Stats::of(&[9.9, 10.0, 10.1]).unwrap();
        let l = Stats::of(&[4.95, 5.0, 5.05]).unwrap();
        let r = Ratio::of(&h, &l);
        assert!((r.ratio - 2.0).abs() < 1e-12 && (r.best - 2.0).abs() < 1e-12);
        assert!((r.ci - 2.0 * h.relative_ci().hypot(l.relative_ci())).abs() < 1e-12);
        assert_eq!(
            (r.verdict, Ratio::of(&l, &h).verdict),
            (Verdict::Win, Verdict::Loss)
        );
        // Three samples a unit apart are an interval wider than the gap.
        let wide = Ratio::of(
            &Stats::of(&[9.0, 10.0, 11.0]).unwrap(),
            &Stats::of(&[4.0, 5.0, 6.0]).unwrap(),
        );
        assert_eq!(wide.verdict, Verdict::Parity);
    }

    #[test]
    fn shapes_read_prefill_by_length_then_decode() {
        let mut shapes = [
            shape(Phase::Decode, 128, 1),
            shape(Phase::Prefill, 2048, 1),
            shape(Phase::Prefill, 500, 1),
        ];
        shapes.sort();
        let lengths: Vec<usize> = shapes.iter().map(|s| s.n).collect();
        assert_eq!(lengths, vec![500, 2048, 128]);
    }

    #[test]
    fn macro_keys_are_letters_only() {
        assert_eq!(
            macro_key("ROCm", shape(Phase::Prefill, 512, 1)),
            "rRocmPrefill"
        );
        assert_eq!(
            macro_key("CUDA", shape(Phase::Prefill, 500, 1)),
            "rCudaPrefillRagged"
        );
        assert_eq!(
            macro_key("Metal", shape(Phase::Decode, 128, 4)),
            "rMetalDecodeCF"
        );
        assert_eq!(
            macro_key("Vulkan", shape(Phase::Prefill, 1024, 1)),
            "rVulkanPrefillOZTF"
        );
    }

    #[test]
    fn a_model_is_named_by_its_file() {
        assert_eq!(
            model_name("/x/Qwen_Qwen3-1.7B-Q4_K_M.gguf"),
            "Qwen3-1.7B-Q4_K_M"
        );
        assert_eq!(
            model_name("/x/Qwen3.8-27B-ROCmFP4-FAST.gguf"),
            "Qwen3.8-27B-ROCmFP4-FAST"
        );
        assert_eq!(
            model_name("/x/unsloth_Qwen3-4B-Q4_K_M.gguf"),
            "unsloth_Qwen3-4B-Q4_K_M"
        );
    }
}
