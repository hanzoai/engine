//! The request ledger: process-wide totals over finished requests, and one log line per finished
//! request (Halogen spec §10.2, §11).
//!
//! Every text generation finishes in `finish_or_add_toks_to_seq`, which hands its `Usage` to
//! [`record`]. The totals only grow, so `/metrics` serves them as Prometheus counters, and an
//! engine restarted inside the process does not reset them. Each choice of an `n > 1` request is
//! recorded as a request of its own, since each prefills and decodes on its own.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::sync::atomic::{AtomicU64, Ordering};

use crate::response::Usage;

/// Below this many prefilled tokens the prefill is too short to time, so the ledger line shows
/// the count instead of a rate (Halogen spec §11).
const RATED_PREFILL: usize = 2048;

static COUNTERS: Counters = Counters::new();

/// Counters over finished requests. Times are kept in microseconds so they add exactly.
struct Counters {
    requests: AtomicU64,
    prompt_tokens: AtomicU64,
    prompt_us: AtomicU64,
    generated: AtomicU64,
    decode_us: AtomicU64,
    cached: AtomicU64,
    drafted: AtomicU64,
    accepted: AtomicU64,
    structured: AtomicU64,
}

/// A read of the totals.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Totals {
    /// Finished requests.
    pub requests: u64,
    /// Prompt tokens prefilled: each prompt less the part the prefix cache served.
    pub prompt_tokens: u64,
    /// Seconds spent prefilling.
    pub prompt_seconds: f64,
    /// Tokens generated.
    pub generated: u64,
    /// Seconds spent decoding.
    pub decode_seconds: f64,
    /// Prompt tokens the prefix cache served.
    pub cached: u64,
    /// Tokens the speculative proposer drafted.
    pub drafted: u64,
    /// Drafted tokens the target accepted.
    pub accepted: u64,
    /// Requests decoded under a constraint: a JSON schema, a regex, a Lark or llguidance grammar.
    pub structured: u64,
}

fn micros(seconds: f32) -> u64 {
    (f64::from(seconds.max(0.0)) * 1e6).round() as u64
}

impl Counters {
    const fn new() -> Self {
        Self {
            requests: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
            prompt_us: AtomicU64::new(0),
            generated: AtomicU64::new(0),
            decode_us: AtomicU64::new(0),
            cached: AtomicU64::new(0),
            drafted: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            structured: AtomicU64::new(0),
        }
    }

    fn add(&self, usage: &Usage, structured: bool) {
        let cached = usage.cached_prompt_tokens.min(usage.prompt_tokens);
        let add = |counter: &AtomicU64, value: u64| {
            counter.fetch_add(value, Ordering::Relaxed);
        };
        add(&self.prompt_tokens, (usage.prompt_tokens - cached) as u64);
        add(&self.prompt_us, micros(usage.total_prompt_time_sec));
        add(&self.generated, usage.completion_tokens as u64);
        add(&self.decode_us, micros(usage.total_completion_time_sec));
        add(&self.cached, cached as u64);
        add(&self.drafted, usage.draft_tokens as u64);
        add(&self.accepted, usage.draft_accepted as u64);
        add(&self.structured, u64::from(structured));
        // Last, so a reader that sees the request also sees its tokens.
        add(&self.requests, 1);
    }

    fn read(&self) -> Totals {
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        Totals {
            requests: load(&self.requests),
            prompt_tokens: load(&self.prompt_tokens),
            prompt_seconds: load(&self.prompt_us) as f64 / 1e6,
            generated: load(&self.generated),
            decode_seconds: load(&self.decode_us) as f64 / 1e6,
            cached: load(&self.cached),
            drafted: load(&self.drafted),
            accepted: load(&self.accepted),
            structured: load(&self.structured),
        }
    }
}

/// The totals over every request this process has finished.
pub fn totals() -> Totals {
    COUNTERS.read()
}

/// A request finished: add it to the totals and log its ledger line. `drafter` names the
/// speculative proposer attached to the pipeline; `structured` says the request decoded under a
/// constraint.
pub(crate) fn record(usage: &Usage, drafter: Option<&str>, structured: bool) {
    COUNTERS.add(usage, structured);
    tracing::info!("{}", line(usage, drafter));
}

/// The ledger line, in Halogen's shape so `bench-serving.py`'s pattern reads it (spec §11):
///
/// `serve_api: {name} {n} tok in {s}s = {rate} | [{r} rounds, commit {c}/round | ]prompt {p}[ ({k}
/// cached, {pct}%)], prefill {s}s` and then ` = {rate} t/s` when at least 2048 tokens were
/// prefilled, else ` ({new} new)`.
///
/// `name` is the drafter when it drafted for this request, else `batch`. Decode time is the decode
/// window only, from the end of prefill to the last token. A verify round commits its accepted
/// drafts plus one token and a plain decode step commits one, and the first token comes from the
/// prefill, so the decode ran `n - 1 - accepted` rounds and committed `(n - 1) / rounds` a round.
fn line(usage: &Usage, drafter: Option<&str>) -> String {
    let generated = usage.completion_tokens;
    let decode = f64::from(usage.total_completion_time_sec);
    let rate = if decode > 0.0 && generated > 1 {
        format!("{:.2} t/s", generated as f64 / decode)
    } else {
        "n/a".to_string()
    };
    let drafted = usage.draft_tokens > 0;
    let name = match drafter {
        Some(name) if drafted => name,
        None if drafted => "spec",
        _ => "batch",
    };
    let mut line = format!("serve_api: {name} {generated} tok in {decode:.2}s = {rate} | ");
    if drafted {
        let decoded = generated.saturating_sub(1);
        let rounds = decoded.saturating_sub(usage.draft_accepted).max(1);
        line += &format!(
            "{rounds} rounds, commit {:.2}/round | ",
            decoded as f64 / rounds as f64
        );
    }
    let prompt = usage.prompt_tokens;
    let cached = usage.cached_prompt_tokens.min(prompt);
    line += &format!("prompt {prompt}");
    if cached > 0 {
        line += &format!(
            " ({cached} cached, {:.1}%)",
            100.0 * cached as f64 / prompt as f64
        );
    }
    let prefill = f64::from(usage.total_prompt_time_sec);
    let new = prompt - cached;
    line += &format!(", prefill {prefill:.2}s");
    if new >= RATED_PREFILL && prefill > 0.0 {
        line += &format!(" = {:.0} t/s", new as f64 / prefill);
    } else {
        line += &format!(" ({new} new)");
    }
    line
}

#[cfg(test)]
mod tests {
    use super::{line, Counters, Totals};
    use crate::response::Usage;

    /// `bench-serving.py`'s ledger pattern (Halogen spec §11), verbatim.
    const BENCH: &str = concat!(
        r"serve_api: (\w+) (\d+) tok in ([\d.]+)s = ([\d.]+) t/s \| ",
        r"(\d+) rounds, commit ([\d.]+)/round \| prompt (\d+)",
        r"(?: \((\d+) cached(?:, [\d.]+%)?\))?, prefill ([\d.]+)s"
    );

    fn usage(prompt: usize, cached: usize, generated: usize) -> Usage {
        Usage {
            prompt_tokens: prompt,
            cached_prompt_tokens: cached,
            completion_tokens: generated,
            total_tokens: prompt + generated,
            total_prompt_time_sec: 0.5,
            total_completion_time_sec: 2.0,
            ..Usage::default()
        }
    }

    fn fields(line: &str) -> Vec<String> {
        let re = regex::Regex::new(BENCH).unwrap();
        let caps = re
            .captures(line)
            .unwrap_or_else(|| panic!("unparsed: {line}"));
        caps.iter()
            .skip(1)
            .map(|m| m.map_or(String::new(), |m| m.as_str().to_string()))
            .collect()
    }

    /// A speculative request: rounds and commit per round are derived from the accepted drafts,
    /// and the bench's pattern reads every field.
    #[test]
    fn a_drafted_request_reads_as_the_bench_expects() {
        let usage = Usage {
            draft_tokens: 240,
            draft_accepted: 150,
            ..usage(100, 0, 201)
        };
        let text = line(&usage, Some("mtp"));
        assert_eq!(
            text,
            "serve_api: mtp 201 tok in 2.00s = 100.50 t/s | 50 rounds, commit 4.00/round \
             | prompt 100, prefill 0.50s (100 new)"
        );
        assert_eq!(
            fields(&text),
            ["mtp", "201", "2.00", "100.50", "50", "4.00", "100", "", "0.50"]
        );
    }

    /// A cached prompt shows its cached share, and a long prefill its rate.
    #[test]
    fn a_cached_prompt_and_a_long_prefill() {
        let usage = Usage {
            draft_tokens: 8,
            draft_accepted: 4,
            ..usage(10_000, 4_000, 11)
        };
        let text = line(&usage, Some("dflash"));
        assert_eq!(
            text,
            "serve_api: dflash 11 tok in 2.00s = 5.50 t/s | 6 rounds, commit 1.67/round \
             | prompt 10000 (4000 cached, 40.0%), prefill 0.50s = 12000 t/s"
        );
        assert_eq!(fields(&text)[7], "4000");
    }

    /// No drafts: `batch`, no rounds, so the bench's pattern (which needs rounds) skips it, as it
    /// skips Halogen's serial requests. One token has no decode rate.
    #[test]
    fn an_undrafted_request_is_batch() {
        let text = line(&usage(7, 0, 1), Some("mtp"));
        assert_eq!(
            text,
            "serve_api: batch 1 tok in 2.00s = n/a | prompt 7, prefill 0.50s (7 new)"
        );
        assert!(!regex::Regex::new(BENCH).unwrap().is_match(&text));
        // Drafts with no attached drafter recorded still name speculation.
        let drafted = Usage {
            draft_tokens: 3,
            ..usage(7, 0, 5)
        };
        assert!(line(&drafted, None).starts_with("serve_api: spec 5 tok"));
    }

    /// The totals add prefilled (not cached) prompt tokens, times in seconds, and the rest as sent.
    #[test]
    fn totals_add_each_request() {
        let counters = Counters::new();
        assert_eq!(counters.read(), Totals::default());
        counters.add(
            &Usage {
                draft_tokens: 12,
                draft_accepted: 9,
                ..usage(100, 40, 20)
            },
            true,
        );
        counters.add(&usage(50, 0, 5), false);
        assert_eq!(
            counters.read(),
            Totals {
                requests: 2,
                prompt_tokens: 110,
                prompt_seconds: 1.0,
                generated: 25,
                decode_seconds: 4.0,
                cached: 40,
                drafted: 12,
                accepted: 9,
                structured: 1,
            }
        );
    }
}
