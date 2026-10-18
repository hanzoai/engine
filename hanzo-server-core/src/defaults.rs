//! Server-wide request defaults: a value a request omits comes from here, and a value the request
//! sends always wins. The fields are `hanzo serve` flags.

use anyhow::Result;
use serde::Deserialize;

/// Largest whole-number default a flag may carry.
const MOST: usize = 1 << 30;

/// Defaults applied to generation requests, injected as an axum `Extension`.
#[derive(Clone, Debug, clap::Args, Deserialize)]
#[serde(default)]
pub struct Defaults {
    /// The model id `/v1/models` lists and every response carries; unset keeps the engine's id.
    #[arg(long = "served-model-name")]
    pub served_name: Option<String>,
    /// Token budget (reasoning plus answer) for chat, Responses and Messages requests that send none.
    #[arg(long, default_value_t = 8192)]
    pub max_tokens: usize,
    /// Token budget for `/v1/completions` requests that send none.
    #[arg(long, default_value_t = 128)]
    pub completion_max_tokens: usize,
    /// Largest budget a request may ask for; more is refused, never clamped.
    #[arg(long, default_value_t = 65536)]
    pub max_tokens_cap: usize,
    /// Sampling temperature (0..2) for requests that send none; unset decodes greedy.
    #[arg(long)]
    pub temperature: Option<f64>,
    /// Nucleus mass (0..1) for requests that send none.
    #[arg(long)]
    pub top_p: Option<f64>,
    /// Top-k (0 disables it) for requests that send none.
    #[arg(long)]
    pub top_k: Option<usize>,
    /// Min-p (0..1) for requests that send none.
    #[arg(long)]
    pub min_p: Option<f64>,
    /// Presence penalty (-2..2) for requests that send none.
    #[arg(long, allow_hyphen_values = true)]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty (-2..2) for requests that send none.
    #[arg(long, allow_hyphen_values = true)]
    pub frequency_penalty: Option<f32>,
    /// Reasoning effort for requests that send none: minimal, low, medium, high or xhigh. Stored
    /// mapped: minimal is low, high is xhigh.
    #[arg(long)]
    pub reasoning_effort: Option<String>,
    /// Whether requests that do not say think: 1/true/yes/on or 0/false/no/off.
    #[arg(long, value_parser = clap::builder::BoolishValueParser::new())]
    pub enable_thinking: Option<bool>,
    /// Think-block budget for requests that send none.
    #[arg(long)]
    pub max_thinking_tokens: Option<usize>,
    /// Tokens kept for the answer when the think-block budget is derived from `max_tokens`.
    /// Unset is the policy `max(1024, 15% of max_tokens)`; 0 keeps no room.
    #[arg(long = "thinking-answer-room")]
    pub answer_room: Option<usize>,
}

impl Default for Defaults {
    fn default() -> Self {
        Self {
            served_name: None,
            max_tokens: 8192,
            completion_max_tokens: 128,
            max_tokens_cap: 65536,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            reasoning_effort: None,
            enable_thinking: None,
            max_thinking_tokens: None,
            answer_room: None,
        }
    }
}

/// Refuses a value outside `lo..=hi`: defaults are never clamped.
fn within<T: PartialOrd + std::fmt::Display>(
    flag: &str,
    value: Option<T>,
    lo: T,
    hi: T,
) -> Result<()> {
    match value {
        Some(v) if !(v >= lo && v <= hi) => anyhow::bail!(
            "--{flag} {v} is outside {lo}..{hi}; values are not clamped, fix the flag"
        ),
        _ => Ok(()),
    }
}

impl Defaults {
    /// Checks every default at startup and stores `reasoning_effort` mapped. A default out of
    /// range stops the server; none is clamped.
    pub fn validated(mut self) -> Result<Self> {
        within("max-tokens", Some(self.max_tokens), 1, MOST)?;
        within(
            "completion-max-tokens",
            Some(self.completion_max_tokens),
            1,
            MOST,
        )?;
        within("max-tokens-cap", Some(self.max_tokens_cap), 1, MOST)?;
        for (flag, budget) in [
            ("max-tokens", self.max_tokens),
            ("completion-max-tokens", self.completion_max_tokens),
        ] {
            if budget > self.max_tokens_cap {
                anyhow::bail!(
                    "--{flag} {budget} is over --max-tokens-cap {}; values are not clamped, fix the flag",
                    self.max_tokens_cap
                );
            }
        }
        within("temperature", self.temperature, 0.0, 2.0)?;
        within("top-p", self.top_p, 0.0, 1.0)?;
        within("top-k", self.top_k, 0, MOST)?;
        within("min-p", self.min_p, 0.0, 1.0)?;
        within("presence-penalty", self.presence_penalty, -2.0, 2.0)?;
        within("frequency-penalty", self.frequency_penalty, -2.0, 2.0)?;
        within("max-thinking-tokens", self.max_thinking_tokens, 1, MOST)?;
        within("thinking-answer-room", self.answer_room, 0, MOST)?;
        if let Some(effort) = &self.reasoning_effort {
            let mapped = match effort.trim().to_lowercase().as_str() {
                "minimal" | "low" => "low",
                "medium" => "medium",
                "high" | "xhigh" => "xhigh",
                "none" => anyhow::bail!(
                    "--reasoning-effort none is not a default effort; to turn thinking off by default use --enable-thinking 0"
                ),
                _ => anyhow::bail!(
                    "--reasoning-effort {effort} is not one of minimal, low, medium, high, xhigh"
                ),
            };
            self.reasoning_effort = Some(mapped.to_string());
        }
        if let Some(name) = &self.served_name {
            if name.trim().is_empty() {
                anyhow::bail!("--served-model-name must not be empty");
            }
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[derive(Parser)]
    struct Flags {
        #[command(flatten)]
        defaults: Defaults,
    }

    fn parse(args: &[&str]) -> Result<Defaults> {
        let mut argv = vec!["serve"];
        argv.extend_from_slice(args);
        Flags::try_parse_from(argv)?.defaults.validated()
    }

    #[test]
    fn flags_have_their_documented_defaults() {
        let d = parse(&[]).unwrap();
        assert_eq!(d.max_tokens, 8192);
        assert_eq!(d.completion_max_tokens, 128);
        assert_eq!(d.max_tokens_cap, 65536);
        assert!(d.served_name.is_none() && d.temperature.is_none() && d.enable_thinking.is_none());
        let empty = Defaults::default();
        assert_eq!(empty.max_tokens, d.max_tokens);
        assert_eq!(empty.completion_max_tokens, d.completion_max_tokens);
        assert_eq!(empty.max_tokens_cap, d.max_tokens_cap);
    }

    #[test]
    fn flags_parse_and_map() {
        let d = parse(&[
            "--served-model-name",
            "zen",
            "--temperature",
            "0.6",
            "--top-k",
            "20",
            "--presence-penalty",
            "-1.5",
            "--reasoning-effort",
            "High",
            "--enable-thinking",
            "off",
            "--thinking-answer-room",
            "0",
        ])
        .unwrap();
        assert_eq!(d.served_name.as_deref(), Some("zen"));
        assert_eq!(d.temperature, Some(0.6));
        assert_eq!(d.top_k, Some(20));
        assert_eq!(d.presence_penalty, Some(-1.5));
        assert_eq!(d.reasoning_effort.as_deref(), Some("xhigh"));
        assert_eq!(d.enable_thinking, Some(false));
        assert_eq!(d.answer_room, Some(0));
        assert_eq!(
            parse(&["--reasoning-effort", "minimal"])
                .unwrap()
                .reasoning_effort
                .as_deref(),
            Some("low")
        );
        assert_eq!(
            parse(&["--enable-thinking", "1"]).unwrap().enable_thinking,
            Some(true)
        );
    }

    #[test]
    fn out_of_range_defaults_are_refused() {
        let refused = [
            (
                vec!["--temperature", "2.5"],
                "--temperature 2.5 is outside 0..2; values are not clamped, fix the flag",
            ),
            (
                vec!["--top-p", "1.5"],
                "--top-p 1.5 is outside 0..1; values are not clamped, fix the flag",
            ),
            (
                vec!["--frequency-penalty", "-3"],
                "--frequency-penalty -3 is outside -2..2; values are not clamped, fix the flag",
            ),
            (
                vec!["--max-tokens", "0"],
                "--max-tokens 0 is outside 1..1073741824; values are not clamped, fix the flag",
            ),
            (
                vec!["--max-thinking-tokens", "0"],
                "--max-thinking-tokens 0 is outside 1..1073741824; values are not clamped, fix the flag",
            ),
            (
                vec!["--max-tokens", "70000"],
                "--max-tokens 70000 is over --max-tokens-cap 65536; values are not clamped, fix the flag",
            ),
            (
                vec!["--max-tokens-cap", "100"],
                "--max-tokens 8192 is over --max-tokens-cap 100; values are not clamped, fix the flag",
            ),
            (
                vec!["--reasoning-effort", "none"],
                "--reasoning-effort none is not a default effort; to turn thinking off by default use --enable-thinking 0",
            ),
            (
                vec!["--reasoning-effort", "max"],
                "--reasoning-effort max is not one of minimal, low, medium, high, xhigh",
            ),
        ];
        for (args, message) in refused {
            let e = parse(&args).expect_err("must be refused");
            assert_eq!(e.to_string(), message, "{args:?}");
        }
        assert!(parse(&["--enable-thinking", "maybe"]).is_err());
        assert!(parse(&["--max-tokens", "128", "--max-tokens-cap", "128"]).is_ok());
    }
}
