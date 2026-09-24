//! `/health`, `/metrics` and `/cache`: what this server is and what its engine is doing (Halogen
//! spec §10). Each body is built from plain values here, so the rules are tested without a model.

#![allow(clippy::cast_precision_loss)]

use std::fmt::Write;

use hanzo_engine::{
    ledger::Totals,
    speculative::{SpeculativeAttachInfo, SpeculativeAttachKind},
    Beat, Load, Phase, PrefixMode, PrefixStats,
};
use serde_json::{json, Map, Value};

use crate::defaults::Defaults;

/// The content type of `/metrics`: Prometheus text format 0.0.4.
pub const METRICS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// Request fields this server honors beyond the OpenAI basics (Halogen spec §4, §10.1).
const SUPPORTED: &[&str] = &[
    "reasoning_effort",
    "enable_thinking",
    "preserve_thinking",
    "max_thinking_tokens",
    "thinking_budget_tokens",
    "thinking_budget",
    "thinking_token_budget",
    "reasoning",
    "thinking",
    "chat_template_kwargs",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "stream",
    "stream_options",
    "drafter",
];

/// The three names of one token budget, reasoning included (spec §4.1.1).
const TOKEN_BUDGET_ALIASES: &[&str] = &["max_tokens", "max_completion_tokens", "max_output_tokens"];

/// The spellings of a think-block budget and of the thinking switch (spec §4.1.3).
const THINKING_CONTROL_ALIASES: &[&str] = &[
    "thinking_budget_tokens",
    "thinking_budget",
    "thinking_token_budget",
    "reasoning.{enabled,effort,max_tokens}",
    "thinking.{type,budget_tokens}",
];

/// `reasoning_effort` values, sorted; `none` turns thinking off (spec §5.2).
const REASONING_EFFORT_VALUES: &[&str] = &["high", "low", "medium", "minimal", "none", "xhigh"];

/// The effort the chat template renders when neither the request nor the server sets one.
const TEMPLATE_REASONING_EFFORT: &str = "medium";

/// Keys `chat_template_kwargs` may carry (spec §4.1.2).
const CHAT_TEMPLATE_KWARGS: &[&str] = &["reasoning_effort", "enable_thinking", "preserve_thinking"];

/// Sampling controls the engine applies.
const SAMPLING: &[&str] = &[
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "n",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_sequence_breakers",
];

/// JSON Schema keywords llguidance compiles into the grammar, so the output obeys them.
const SCHEMA_ENFORCED: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "patternProperties",
    "minProperties",
    "maxProperties",
    "items",
    "prefixItems",
    "additionalItems",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "enum",
    "const",
    "anyOf",
    "allOf",
    "$ref",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
];

/// `format` values llguidance knows; any other refuses the schema.
const SCHEMA_FORMATS: &[&str] = &[
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "ipv4",
    "ipv6",
    "uuid",
    "uri",
    "unknown",
];

/// Annotations llguidance reads past without constraining anything.
const SCHEMA_ANNOTATIONS: &[&str] = &[
    "$defs",
    "definitions",
    "$schema",
    "$id",
    "id",
    "$anchor",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "readOnly",
    "writeOnly",
    "contentMediaType",
    "contentEncoding",
];

/// JSON Schema keywords llguidance refuses; a schema that uses one is rejected, never loosened.
const SCHEMA_REFUSED: &[&str] = &[
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "propertyNames",
    "dependentRequired",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "unevaluatedItems",
];

/// How the chat template writes a tool call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolFormat {
    /// A JSON body inside the template's call markers.
    Json,
    /// Qwen XML: `<function=NAME><parameter=KEY>VALUE</parameter></function>`.
    QwenXml,
    /// OpenAI Harmony channels.
    Harmony,
}

/// What `/health` reports about the served engine.
pub struct Snapshot {
    /// The model id responses carry.
    pub model: String,
    /// The `/v1` routes this server registers, sorted.
    pub endpoints: Vec<&'static str>,
    /// Whether the engine thread still runs.
    pub alive: bool,
    pub beat: Beat,
    pub load: Load,
    /// Wall clock in ms since the Unix epoch, to age the oldest running request.
    pub now: u64,
    /// The longest sequence the engine admits.
    pub context: Option<usize>,
    /// Sequences the engine runs at once.
    pub slots: usize,
    pub drafter: Option<SpeculativeAttachInfo>,
    pub indexer_budget: Option<usize>,
    pub checkpoint_format: &'static str,
    pub tool_format: ToolFormat,
    pub prefix: PrefixStats,
}

fn phase_name(phase: Phase) -> &'static str {
    match phase {
        Phase::Idle => "idle",
        Phase::Loop => "loop",
        Phase::Prefill => "prefill",
        Phase::Decode => "decode",
    }
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// The `/health` body, and whether the engine answers. An engine that does not answer gets the
/// same body with `status: engine_unresponsive`, served as a 503 (Halogen spec §10.1). The probe
/// never waits on the engine: the engine loop beats a heartbeat, and it has stopped answering when
/// its thread has exited or it has overstayed its phase (`Phase::budget`).
pub fn health(s: &Snapshot, d: &Defaults) -> (bool, Value) {
    let stalled = s.beat.stalled();
    let responds = s.alive && !stalled;
    let mut engine = json!({
        "responds": responds,
        "phase": phase_name(s.beat.phase),
        "phase_s": round1(s.beat.elapsed.as_secs_f64()),
    });
    if !s.alive {
        engine["detail"] = json!("the engine thread has exited");
    } else if let (true, Some(budget)) = (stalled, s.beat.phase.budget()) {
        engine["detail"] = json!(format!(
            "no progress for {:.1}s in {}, past its {}s budget",
            s.beat.elapsed.as_secs_f64(),
            phase_name(s.beat.phase),
            budget.as_secs()
        ));
    }

    let drafter = s.drafter.as_ref().map(SpeculativeAttachInfo::name);
    let mut drafters = vec!["serial"];
    drafters.extend(drafter);
    let prompt_lookup = match s.drafter.as_ref().map(|d| &d.kind) {
        Some(SpeculativeAttachKind::PromptLookup {
            ngram_min,
            ngram_max,
            gamma,
        }) => json!({
            "ngram": ngram_max,
            "ngram_min": ngram_min,
            "chain": gamma,
            "applies_to": "every request on the spec drafter",
        }),
        _ => json!("off"),
    };
    let prompt_cache = match s.prefix.mode {
        PrefixMode::Off => json!({"enabled": false}),
        PrefixMode::Sequence => json!({"enabled": true, "mode": "sequences"}),
        PrefixMode::Blocks(block_size) => {
            json!({"enabled": true, "mode": "blocks", "block_size": block_size})
        }
    };
    let xml = s.tool_format == ToolFormat::QwenXml;
    let tool_calls = json!({
        "wire_format": match s.tool_format {
            ToolFormat::Json => "json",
            ToolFormat::QwenXml => "qwen-xml (<function=>/<parameter=>)",
            ToolFormat::Harmony => "harmony",
        },
        "streaming": true,
        "tool_choice": ["auto", "none", "required", "{type: function, function: {name}}"],
        "forced_call_is_a_prefill": false,
        "parallel_tool_calls": true,
        "constrained_decoding": !xml,
        "argument_types_need_schema": xml,
    });
    let busy_for_s = s.load.oldest.map_or(0.0, |oldest| {
        round1(s.now.saturating_sub(oldest) as f64 / 1000.0)
    });
    let temperature = d.temperature.map_or("1 (the engine's)".to_string(), |t| {
        format!("{t} (--temperature)")
    });

    let mut body = Map::new();
    let mut put = |key: &str, value: Value| {
        body.insert(key.to_string(), value);
    };
    put(
        "status",
        json!(if responds {
            "ok"
        } else {
            "engine_unresponsive"
        }),
    );
    put("model", json!(s.model));
    put("endpoints", json!(s.endpoints));
    put("cache_counters", json!("/cache"));
    put("metrics", json!("/metrics"));
    put("context", json!(s.context));
    put("rope_scaling", Value::Null);
    if let Some(budget) = s.indexer_budget {
        put("indexer_budget", json!(budget));
    }
    put(
        "version",
        json!({
            "api": env!("CARGO_PKG_VERSION"),
            "engine": hanzo_engine::VERSION,
            "match": env!("CARGO_PKG_VERSION") == hanzo_engine::VERSION,
        }),
    );
    put("checkpoint_format", json!(s.checkpoint_format));
    put("engine", engine);
    put("busy", json!(s.load.running >= s.slots));
    put("slots", json!(s.slots));
    if let Some((_, positions)) = s.load.kv {
        put("kv_pool_positions", json!(positions));
    }
    put("in_flight", json!(s.load.running));
    put("busy_for_s", json!(busy_for_s));
    put("queued", json!(s.load.waiting));
    put(
        "decode",
        json!(format!(
            "sampled at the request's temperature, else at {temperature}; temperature 0 is greedy"
        )),
    );
    put("drafters_available", json!(drafters));
    put("drafter_default", json!(drafter.unwrap_or("serial")));
    put("prompt_lookup", prompt_lookup);
    put("prompt_cache", prompt_cache);
    put("tool_calls", tool_calls);
    put("supported", json!(SUPPORTED));
    put("token_budget_aliases", json!(TOKEN_BUDGET_ALIASES));
    put("max_tokens_default", json!(d.max_tokens));
    put("reasoning_effort_values", json!(REASONING_EFFORT_VALUES));
    put(
        "reasoning_effort_default",
        json!(d
            .reasoning_effort
            .as_deref()
            .unwrap_or(TEMPLATE_REASONING_EFFORT)),
    );
    put("token_budget_covers_reasoning", json!(true));
    put("max_thinking_tokens_default", json!(d.max_thinking_tokens));
    put(
        "thinking_answer_room",
        d.answer_room
            .map_or(json!("max(1024, 15% of max_tokens)"), |room| json!(room)),
    );
    put("thinking_control_aliases", json!(THINKING_CONTROL_ALIASES));
    put("server_defaults", Value::Object(server_defaults(d)));
    put(
        "server_defaults_rule",
        json!(
            "a field the request sends always wins; a default fills only a field the request omits"
        ),
    );
    put("chat_template_kwargs", json!(CHAT_TEMPLATE_KWARGS));
    put("error_format", json!("openai"));
    put(
        "structured_output",
        json!({
            "enabled": true,
            "response_format": ["json_schema", "json_object"],
            "text.format": ["json_schema", "json_object"],
            "grammar": ["regex", "lark", "json_schema", "llguidance"],
            "routes": ["/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses"],
            "sampling": "any sampling applies; the constraint masks the distribution before a token is drawn",
            "thinking": "the constraint binds every generated token from the first, the think block included",
            "enforced": SCHEMA_ENFORCED,
            "formats": SCHEMA_FORMATS,
            "accepted_not_enforced": SCHEMA_ANNOTATIONS,
            "refused": SCHEMA_REFUSED,
        }),
    );
    put(
        "sampling",
        json!({"implemented": SAMPLING, "greedy": "temperature 0 takes the argmax"}),
    );
    put("max_tokens_cap", json!(d.max_tokens_cap));
    put("max_tokens_over_cap", json!("400"));
    (responds, Value::Object(body))
}

/// The request defaults that are set, under the `hanzo serve` flags that set them.
fn server_defaults(d: &Defaults) -> Map<String, Value> {
    let Defaults {
        served_name: _,
        max_tokens: _,
        completion_max_tokens: _,
        max_tokens_cap: _,
        temperature,
        top_p,
        top_k,
        min_p,
        presence_penalty,
        frequency_penalty,
        reasoning_effort,
        enable_thinking,
        max_thinking_tokens,
        answer_room,
    } = d;
    [
        ("--temperature", temperature.map(|v| json!(v))),
        ("--top-p", top_p.map(|v| json!(v))),
        ("--top-k", top_k.map(|v| json!(v))),
        ("--min-p", min_p.map(|v| json!(v))),
        ("--presence-penalty", presence_penalty.map(|v| json!(v))),
        ("--frequency-penalty", frequency_penalty.map(|v| json!(v))),
        (
            "--reasoning-effort",
            reasoning_effort.as_ref().map(|v| json!(v)),
        ),
        ("--enable-thinking", enable_thinking.map(|v| json!(v))),
        (
            "--max-thinking-tokens",
            max_thinking_tokens.map(|v| json!(v)),
        ),
        ("--thinking-answer-room", answer_room.map(|v| json!(v))),
    ]
    .into_iter()
    .filter_map(|(flag, value)| Some((flag.to_string(), value?)))
    .collect()
}

/// A float the way `%.6g` prints it: six significant digits, no trailing zeros, and an exponent
/// below 1e-4 or from 1e6 on (spec §10.2).
fn g6(x: f64) -> String {
    if x == 0.0 || !x.is_finite() {
        return if x.is_nan() {
            "NaN".to_string()
        } else if x.is_infinite() {
            if x > 0.0 { "+Inf" } else { "-Inf" }.to_string()
        } else {
            "0".to_string()
        };
    }
    let sci = format!("{x:.5e}");
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let trim = |s: &str| {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if !(-4..6).contains(&exponent) {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mantissa), exponent.abs())
    } else {
        let decimals = usize::try_from(5 - exponent).unwrap_or(0);
        trim(&format!("{x:.decimals$}"))
    }
}

/// Tokens per second over a window, or 0 when the window timed nothing.
fn rate(tokens: u64, seconds: f64) -> f64 {
    if seconds > 0.0 {
        tokens as f64 / seconds
    } else {
        0.0
    }
}

/// The `/metrics` body (Halogen spec §10.2): the llama-server rows, then the engine's own. Counters
/// are the process-wide ledger totals; the two rate gauges cover the requests that finished since
/// `previous`, the totals at the last scrape. Load is summed over every loaded engine, and the KV
/// rows appear only when some engine has a paged pool.
pub fn metrics(totals: &Totals, previous: &Totals, loads: &[Load]) -> String {
    let mut out = String::new();
    let mut row = |name: &str, kind: &str, help: &str, value: String| {
        let _ = write!(
            out,
            "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {value}\n"
        );
    };
    let counter = "counter";
    let gauge = "gauge";
    row(
        "llamacpp:prompt_tokens_total",
        counter,
        "Number of prompt tokens processed.",
        totals.prompt_tokens.to_string(),
    );
    row(
        "llamacpp:prompt_seconds_total",
        counter,
        "Prompt process time",
        g6(totals.prompt_seconds),
    );
    row(
        "llamacpp:tokens_predicted_total",
        counter,
        "Number of generation tokens processed.",
        totals.generated.to_string(),
    );
    row(
        "llamacpp:tokens_predicted_seconds_total",
        counter,
        "Predict process time",
        g6(totals.decode_seconds),
    );
    row(
        "llamacpp:prompt_tokens_seconds",
        gauge,
        "Average prompt throughput in tokens/s.",
        g6(rate(
            totals.prompt_tokens.saturating_sub(previous.prompt_tokens),
            totals.prompt_seconds - previous.prompt_seconds,
        )),
    );
    row(
        "llamacpp:predicted_tokens_seconds",
        gauge,
        "Average generation throughput in tokens/s.",
        g6(rate(
            totals.generated.saturating_sub(previous.generated),
            totals.decode_seconds - previous.decode_seconds,
        )),
    );
    let running: usize = loads.iter().map(|l| l.running).sum();
    let waiting: usize = loads.iter().map(|l| l.waiting).sum();
    row(
        "llamacpp:requests_processing",
        gauge,
        "Number of requests processing.",
        running.to_string(),
    );
    row(
        "llamacpp:requests_deferred",
        gauge,
        "Number of requests deferred.",
        waiting.to_string(),
    );
    let pools: Vec<(usize, usize)> = loads.iter().filter_map(|l| l.kv).collect();
    let (used, positions) = pools.iter().fold((0, 0), |(u, p), (used, positions)| {
        (u + used, p + positions)
    });
    if !pools.is_empty() {
        row(
            "llamacpp:kv_cache_tokens",
            gauge,
            "KV-cache tokens.",
            used.to_string(),
        );
        row(
            "llamacpp:kv_cache_usage_ratio",
            gauge,
            "KV-cache usage. 1 means 100 percent usage.",
            g6((used as f64 / positions as f64).min(1.0)),
        );
    }
    row(
        "hanzo:requests_total",
        counter,
        "Requests finished.",
        totals.requests.to_string(),
    );
    row(
        "hanzo:prompt_tokens_cached_total",
        counter,
        "Prompt tokens served from the prefix cache.",
        totals.cached.to_string(),
    );
    row(
        "hanzo:draft_tokens_total",
        counter,
        "Tokens the speculative drafter proposed.",
        totals.drafted.to_string(),
    );
    row(
        "hanzo:draft_tokens_accepted_total",
        counter,
        "Drafted tokens the target accepted.",
        totals.accepted.to_string(),
    );
    row(
        "hanzo:structured_requests_total",
        counter,
        "Requests decoded under a JSON schema or grammar.",
        totals.structured.to_string(),
    );
    if !pools.is_empty() {
        row(
            "hanzo:kv_pool_positions",
            gauge,
            "Token positions in the paged KV pool.",
            positions.to_string(),
        );
    }
    out
}

/// A share rounded to four decimals, or null when nothing was counted.
fn share(part: u64, whole: u64) -> Value {
    if whole == 0 {
        Value::Null
    } else {
        json!((part as f64 / whole as f64 * 1e4).round() / 1e4)
    }
}

/// The `/cache` body (Halogen spec §10.3), or `None` when prefix caching is off, which the route
/// serves as a 501. Entries are sequences, or KV blocks of `block_size` tokens in block mode.
/// `token_hit_rate` is the share of prompt tokens the cache served, over every finished request.
pub fn cache(prefix: &PrefixStats, totals: &Totals) -> Option<Value> {
    let (mode, block_size) = match prefix.mode {
        PrefixMode::Off => return None,
        PrefixMode::Sequence => ("sequences", None),
        PrefixMode::Blocks(block_size) => ("blocks", Some(block_size)),
    };
    let mut body = json!({
        "mode": mode,
        "entries": prefix.entries,
        "hits": prefix.hits,
        "misses": prefix.misses,
        "stores": prefix.stores,
        "evicted": prefix.evicted,
        "prompt_tokens_saved": prefix.tokens_saved,
        "hit_rate": share(prefix.hits, prefix.hits + prefix.misses),
        "token_hit_rate": share(totals.cached, totals.prompt_tokens + totals.cached),
    });
    if let Some(block_size) = block_size {
        body["block_size"] = json!(block_size);
    }
    Some(body)
}

/// The OpenAI error envelope (Halogen spec §2.4).
pub fn error(message: &str) -> Value {
    json!({"error": {"message": message, "type": "server_error", "param": null, "code": null}})
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use hanzo_engine::{
        ledger::Totals, speculative::SpeculativeAttachInfo, Beat, Load, Phase, PrefixMode,
        PrefixStats,
    };
    use serde_json::{json, Value};

    use super::{cache, g6, health, metrics, Snapshot, ToolFormat};
    use crate::defaults::Defaults;

    fn prefix(mode: PrefixMode) -> PrefixStats {
        PrefixStats {
            mode,
            hits: 3,
            misses: 1,
            tokens_saved: 4096,
            entries: 12,
            stores: 20,
            evicted: 8,
        }
    }

    fn snapshot() -> Snapshot {
        Snapshot {
            model: "zen".to_string(),
            endpoints: vec!["/v1/chat/completions", "/v1/models"],
            alive: true,
            beat: Beat {
                phase: Phase::Decode,
                elapsed: Duration::from_millis(40),
            },
            load: Load {
                running: 1,
                waiting: 0,
                oldest: Some(1_000_000),
                kv: Some((2048, 655_360)),
            },
            now: 1_012_340,
            context: Some(655_360),
            slots: 2,
            drafter: Some(SpeculativeAttachInfo::mtp("self".into(), 3)),
            indexer_budget: None,
            checkpoint_format: "gguf",
            tool_format: ToolFormat::QwenXml,
            prefix: prefix(PrefixMode::Blocks(16)),
        }
    }

    /// The keys the bench reads (spec §13), and the load, drafter and cache as the engine has them.
    #[test]
    fn a_healthy_engine() {
        let (responds, body) = health(&snapshot(), &Defaults::default());
        assert!(responds);
        assert_eq!(body["status"], "ok");
        assert_eq!(body["model"], "zen");
        assert_eq!(body["context"], 655_360);
        assert_eq!(body["drafter_default"], "mtp");
        assert_eq!(body["drafters_available"], json!(["serial", "mtp"]));
        assert_eq!(
            body["endpoints"],
            json!(["/v1/chat/completions", "/v1/models"])
        );
        assert_eq!(body["checkpoint_format"], "gguf");
        assert_eq!(body["busy"], false);
        assert_eq!(body["slots"], 2);
        assert_eq!(body["in_flight"], 1);
        assert_eq!(body["queued"], 0);
        assert_eq!(body["busy_for_s"], 12.3);
        assert_eq!(body["kv_pool_positions"], 655_360);
        assert_eq!(body["rope_scaling"], Value::Null);
        assert!(body.get("indexer_budget").is_none());
        assert_eq!(body["prompt_lookup"], "off");
        assert_eq!(
            body["prompt_cache"],
            json!({"enabled": true, "mode": "blocks", "block_size": 16})
        );
        assert_eq!(
            body["engine"],
            json!({"responds": true, "phase": "decode", "phase_s": 0.0})
        );
        assert_eq!(
            body["tool_calls"]["wire_format"],
            "qwen-xml (<function=>/<parameter=>)"
        );
        assert_eq!(body["tool_calls"]["constrained_decoding"], false);
        assert_eq!(body["max_tokens_default"], 8192);
        assert_eq!(body["max_tokens_cap"], 65536);
        assert_eq!(body["max_tokens_over_cap"], "400");
        assert_eq!(body["reasoning_effort_default"], "medium");
        assert_eq!(body["max_thinking_tokens_default"], Value::Null);
        assert_eq!(body["thinking_answer_room"], "max(1024, 15% of max_tokens)");
        assert_eq!(body["server_defaults"], json!({}));
        assert_eq!(body["error_format"], "openai");
        assert!(body["structured_output"]["refused"]
            .as_array()
            .unwrap()
            .contains(&json!("oneOf")));
        // The first keys keep Halogen's order.
        let keys: Vec<_> = body.as_object().unwrap().keys().take(3).cloned().collect();
        assert_eq!(keys, ["status", "model", "endpoints"]);
    }

    /// Nothing drafts and nothing runs: the serial drafter, no age, no pool, and a budget when
    /// the model has an indexer.
    #[test]
    fn an_idle_engine_without_a_drafter() {
        let snapshot = Snapshot {
            drafter: None,
            load: Load::default(),
            indexer_budget: Some(2048),
            tool_format: ToolFormat::Json,
            prefix: prefix(PrefixMode::Off),
            ..snapshot()
        };
        let (responds, body) = health(&snapshot, &Defaults::default());
        assert!(responds);
        assert_eq!(body["drafter_default"], "serial");
        assert_eq!(body["drafters_available"], json!(["serial"]));
        assert_eq!(body["busy_for_s"], 0.0);
        assert_eq!(body["indexer_budget"], 2048);
        assert!(body.get("kv_pool_positions").is_none());
        assert_eq!(body["prompt_cache"], json!({"enabled": false}));
        assert_eq!(body["tool_calls"]["wire_format"], "json");
        assert_eq!(body["tool_calls"]["constrained_decoding"], true);
    }

    /// A full house is busy.
    #[test]
    fn every_slot_running_is_busy() {
        let snapshot = Snapshot {
            load: Load {
                running: 2,
                waiting: 5,
                ..snapshot().load
            },
            ..snapshot()
        };
        let (_, body) = health(&snapshot, &Defaults::default());
        assert_eq!(body["busy"], true);
        assert_eq!(body["queued"], 5);
    }

    /// Prompt lookup names its n-grams and chain, under the `spec` drafter.
    #[test]
    fn prompt_lookup_is_described() {
        let snapshot = Snapshot {
            drafter: Some(SpeculativeAttachInfo::prompt_lookup(3, 7, 8)),
            ..snapshot()
        };
        let (_, body) = health(&snapshot, &Defaults::default());
        assert_eq!(body["drafter_default"], "spec");
        assert_eq!(
            body["prompt_lookup"],
            json!({"ngram": 7, "ngram_min": 3, "chain": 8, "applies_to": "every request on the spec drafter"})
        );
    }

    /// A dead thread, or a loop past its phase's budget, is unresponsive: the same body, a 503.
    #[test]
    fn an_unresponsive_engine() {
        let dead = Snapshot {
            alive: false,
            ..snapshot()
        };
        let (responds, body) = health(&dead, &Defaults::default());
        assert!(!responds);
        assert_eq!(body["status"], "engine_unresponsive");
        assert_eq!(body["engine"]["detail"], "the engine thread has exited");
        assert_eq!(body["model"], "zen");
        assert_eq!(body["context"], 655_360);

        let stuck = Snapshot {
            beat: Beat {
                phase: Phase::Loop,
                elapsed: Duration::from_secs(45),
            },
            ..snapshot()
        };
        let (responds, body) = health(&stuck, &Defaults::default());
        assert!(!responds);
        assert_eq!(
            body["engine"]["detail"],
            "no progress for 45.0s in loop, past its 30s budget"
        );

        // A long prefill is work, not a stall.
        let prefilling = Snapshot {
            beat: Beat {
                phase: Phase::Prefill,
                elapsed: Duration::from_secs(600),
            },
            ..snapshot()
        };
        assert!(health(&prefilling, &Defaults::default()).0);
    }

    /// Server defaults appear under their flags, only when set, and fill the matching keys.
    #[test]
    fn server_defaults_that_are_set() {
        let defaults = Defaults {
            temperature: Some(0.6),
            top_k: Some(20),
            reasoning_effort: Some("low".to_string()),
            enable_thinking: Some(false),
            max_thinking_tokens: Some(4096),
            answer_room: Some(0),
            ..Defaults::default()
        };
        let (_, body) = health(&snapshot(), &defaults);
        assert_eq!(
            body["server_defaults"],
            json!({
                "--temperature": 0.6,
                "--top-k": 20,
                "--reasoning-effort": "low",
                "--enable-thinking": false,
                "--max-thinking-tokens": 4096,
                "--thinking-answer-room": 0,
            })
        );
        assert_eq!(body["reasoning_effort_default"], "low");
        assert_eq!(body["max_thinking_tokens_default"], 4096);
        assert_eq!(body["thinking_answer_room"], 0);
        assert!(body["decode"]
            .as_str()
            .unwrap()
            .contains("0.6 (--temperature)"));
    }

    #[test]
    fn floats_print_as_percent_six_g() {
        for (x, want) in [
            (0.0, "0"),
            (1.0, "1"),
            (12.5, "12.5"),
            (100.0, "100"),
            (0.1 + 0.2, "0.3"),
            (123_456.7, "123457"),
            (1_234_567.0, "1.23457e+06"),
            (0.0001, "0.0001"),
            (0.000_012_3, "1.23e-05"),
            (2.0 / 3.0, "0.666667"),
            (999_999.6, "1e+06"),
        ] {
            assert_eq!(g6(x), want, "{x}");
        }
    }

    fn totals(requests: u64, prompt: u64, prompt_s: f64, generated: u64, decode_s: f64) -> Totals {
        Totals {
            requests,
            prompt_tokens: prompt,
            prompt_seconds: prompt_s,
            generated,
            decode_seconds: decode_s,
            cached: 512,
            drafted: 40,
            accepted: 30,
            structured: 1,
        }
    }

    /// Every row in order, three lines each, no labels; the rate gauges cover only the requests
    /// that finished since the last scrape.
    #[test]
    fn metrics_rows() {
        let previous = totals(1, 1000, 1.0, 100, 2.0);
        let now = totals(3, 3000, 2.0, 400, 5.0);
        let loads = [
            Load {
                running: 1,
                waiting: 2,
                oldest: Some(1),
                kv: Some((1000, 4000)),
            },
            Load {
                running: 1,
                waiting: 0,
                oldest: None,
                kv: Some((3000, 4000)),
            },
        ];
        let text = metrics(&now, &previous, &loads);
        let want = "\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 3000
# HELP llamacpp:prompt_seconds_total Prompt process time
# TYPE llamacpp:prompt_seconds_total counter
llamacpp:prompt_seconds_total 2
# HELP llamacpp:tokens_predicted_total Number of generation tokens processed.
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 400
# HELP llamacpp:tokens_predicted_seconds_total Predict process time
# TYPE llamacpp:tokens_predicted_seconds_total counter
llamacpp:tokens_predicted_seconds_total 5
# HELP llamacpp:prompt_tokens_seconds Average prompt throughput in tokens/s.
# TYPE llamacpp:prompt_tokens_seconds gauge
llamacpp:prompt_tokens_seconds 2000
# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s.
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 100
# HELP llamacpp:requests_processing Number of requests processing.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 2
# HELP llamacpp:requests_deferred Number of requests deferred.
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 2
# HELP llamacpp:kv_cache_tokens KV-cache tokens.
# TYPE llamacpp:kv_cache_tokens gauge
llamacpp:kv_cache_tokens 4000
# HELP llamacpp:kv_cache_usage_ratio KV-cache usage. 1 means 100 percent usage.
# TYPE llamacpp:kv_cache_usage_ratio gauge
llamacpp:kv_cache_usage_ratio 0.5
# HELP hanzo:requests_total Requests finished.
# TYPE hanzo:requests_total counter
hanzo:requests_total 3
# HELP hanzo:prompt_tokens_cached_total Prompt tokens served from the prefix cache.
# TYPE hanzo:prompt_tokens_cached_total counter
hanzo:prompt_tokens_cached_total 512
# HELP hanzo:draft_tokens_total Tokens the speculative drafter proposed.
# TYPE hanzo:draft_tokens_total counter
hanzo:draft_tokens_total 40
# HELP hanzo:draft_tokens_accepted_total Drafted tokens the target accepted.
# TYPE hanzo:draft_tokens_accepted_total counter
hanzo:draft_tokens_accepted_total 30
# HELP hanzo:structured_requests_total Requests decoded under a JSON schema or grammar.
# TYPE hanzo:structured_requests_total counter
hanzo:structured_requests_total 1
# HELP hanzo:kv_pool_positions Token positions in the paged KV pool.
# TYPE hanzo:kv_pool_positions gauge
hanzo:kv_pool_positions 8000
";
        assert_eq!(text, want);
    }

    /// No request since the last scrape rates 0; without a paged pool the KV rows are left out.
    #[test]
    fn an_idle_scrape_without_a_pool() {
        let same = totals(3, 3000, 2.0, 400, 5.0);
        let text = metrics(&same, &same, &[Load::default()]);
        assert!(text.contains("\nllamacpp:prompt_tokens_seconds 0\n"));
        assert!(text.contains("\nllamacpp:predicted_tokens_seconds 0\n"));
        assert!(text.contains("\nllamacpp:requests_processing 0\n"));
        assert!(!text.contains("kv_cache"));
        assert!(!text.contains("kv_pool"));
        assert!(text.ends_with('\n'));
        assert_eq!(text.lines().count(), 13 * 3);
    }

    #[test]
    fn cache_counters() {
        let totals = totals(3, 3000, 2.0, 400, 5.0);
        assert_eq!(cache(&prefix(PrefixMode::Off), &totals), None);
        assert_eq!(
            cache(&prefix(PrefixMode::Blocks(16)), &totals),
            Some(json!({
                "mode": "blocks",
                "entries": 12,
                "hits": 3,
                "misses": 1,
                "stores": 20,
                "evicted": 8,
                "prompt_tokens_saved": 4096,
                "hit_rate": 0.75,
                "token_hit_rate": 0.1458,
                "block_size": 16,
            }))
        );
        let fresh = PrefixStats {
            hits: 0,
            misses: 0,
            ..prefix(PrefixMode::Sequence)
        };
        let body = cache(&fresh, &Totals::default()).unwrap();
        assert_eq!(body["mode"], "sequences");
        assert_eq!(body["hit_rate"], Value::Null);
        assert_eq!(body["token_hit_rate"], Value::Null);
        assert!(body.get("block_size").is_none());
    }
}
