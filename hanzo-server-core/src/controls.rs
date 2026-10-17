//! Request controls shared by the generation routes: the token budget, the reasoning controls,
//! the sampling checks, `tool_choice` and the drafter.
//!
//! Every function here refuses rather than clamps: a request is served as asked or gets a 400
//! that says why.

use std::collections::HashMap;

use hanzo_engine::{speculative::SpeculativeAttachInfo, ReasoningEffort, Tool, ToolChoice};
use serde_json::{Map, Value};

use crate::{defaults::Defaults, error::ApiError};

/// Most `logit_bias` entries one request may carry.
pub const LOGIT_BIAS_CAP: usize = 20_480;

/// Python's `repr()` of a JSON value: the error messages quote values with it.
fn repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::String(s) => format!("'{s}'"),
        other => other.to_string(),
    }
}

/// A JSON whole number (`5` or `5.0`, never `true` or `"5"`).
fn whole(value: &Value) -> Option<i64> {
    match value {
        Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64)),
        _ => None,
    }
}

/// Folds the three names of the token budget. Equal values agree;
/// different values are refused, whatever the order.
pub fn fold_budget(names: &[(&str, Option<i64>)]) -> Result<Option<i64>, ApiError> {
    let mut sent: Vec<(&str, i64)> = names
        .iter()
        .filter_map(|(name, value)| value.map(|v| (*name, v)))
        .collect();
    let Some(&(_, first)) = sent.first() else {
        return Ok(None);
    };
    if sent.iter().any(|&(_, v)| v != first) {
        sent.sort_by_key(|&(name, _)| name);
        let pairs = sent
            .iter()
            .map(|(name, v)| format!("{name}={v}"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(ApiError::invalid(format!(
            "conflicting token budgets ({pairs}); they are the same budget under three names, send one"
        )));
    }
    Ok(Some(first))
}

/// The budget a request runs with: the request's, else the route's
/// default. Over the cap is refused, never clamped; zero or below becomes 1.
pub fn resolve_budget(
    requested: Option<i64>,
    default: usize,
    cap: usize,
) -> Result<usize, ApiError> {
    let want = requested.unwrap_or(default as i64);
    if want > cap as i64 {
        return Err(ApiError::invalid(format!(
            "max_tokens {want} exceeds this server's cap of {cap}. The cap is server policy, not a model limit: one long request holds its slot for its whole run and every other client queues behind it. Raise it with --max-tokens-cap."
        )));
    }
    Ok(want.max(1) as usize)
}

/// The reasoning controls a request sends, after their aliases are folded in. Field names are
/// the request's.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Reasoning {
    pub reasoning_effort: Option<String>,
    pub enable_thinking: Option<bool>,
    pub preserve_thinking: Option<bool>,
    pub max_thinking_tokens: Option<i64>,
}

/// The aliases of the reasoning controls, as the request sent them.
#[derive(Debug, Clone, Copy, Default)]
pub struct Aliases<'a> {
    pub chat_template_kwargs: Option<&'a Value>,
    pub thinking_budget_tokens: Option<&'a Value>,
    pub thinking_budget: Option<&'a Value>,
    pub thinking_token_budget: Option<&'a Value>,
    pub reasoning: Option<&'a Value>,
    pub thinking: Option<&'a Value>,
}

impl Reasoning {
    /// Folds every alias into the four controls: `chat_template_kwargs` first, then the thinking
    /// aliases. Duplicates that agree are accepted; duplicates that differ are refused.
    pub fn fold(mut self, aliases: Aliases<'_>) -> Result<Self, ApiError> {
        if let Some(budget) = self.max_thinking_tokens {
            if budget < 0 {
                return Err(ApiError::invalid(
                    "max_thinking_tokens must be a whole number >= 0 (0 means no budget)",
                ));
            }
        }
        self.fold_template_kwargs(aliases.chat_template_kwargs)?;
        for (name, value) in [
            ("thinking_budget_tokens", aliases.thinking_budget_tokens),
            ("thinking_budget", aliases.thinking_budget),
            ("thinking_token_budget", aliases.thinking_token_budget),
        ] {
            let Some(value) = value.filter(|v| !v.is_null()) else {
                continue;
            };
            let budget = positive(value, &format!("{name} must be a whole number >= 1"))?;
            self.take("max_thinking_tokens", name, Value::from(budget))?;
        }
        if let Some(reasoning) = aliases.reasoning.filter(|v| !v.is_null()) {
            self.fold_reasoning(reasoning)?;
        }
        if let Some(thinking) = aliases.thinking.filter(|v| !v.is_null()) {
            self.fold_thinking(thinking)?;
        }
        Ok(self)
    }

    /// `chat_template_kwargs` may carry exactly the three template controls, each typed as its
    /// top-level field, and the types are checked.
    fn fold_template_kwargs(&mut self, kwargs: Option<&Value>) -> Result<(), ApiError> {
        let Some(kwargs) = kwargs.filter(|v| !v.is_null()) else {
            return Ok(());
        };
        let Value::Object(kwargs) = kwargs else {
            return Err(ApiError::invalid("chat_template_kwargs must be an object"));
        };
        const ALLOWED: [&str; 3] = ["reasoning_effort", "enable_thinking", "preserve_thinking"];
        let mut unknown: Vec<&String> = kwargs
            .keys()
            .filter(|k| !ALLOWED.contains(&k.as_str()))
            .collect();
        if !unknown.is_empty() {
            unknown.sort();
            let keys = unknown
                .iter()
                .map(|k| format!("'{k}'"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(ApiError::invalid(format!(
                "chat_template_kwargs contains unsupported key(s) {keys}; this server honours reasoning_effort, enable_thinking, preserve_thinking, which may also be sent as top-level fields"
            )));
        }
        for key in ALLOWED {
            let Some(nested) = kwargs.get(key).filter(|v| !v.is_null()) else {
                continue;
            };
            let well_typed = match key {
                "reasoning_effort" => nested.is_string(),
                _ => nested.is_boolean(),
            };
            if !well_typed {
                let kind = if key == "reasoning_effort" {
                    "a string"
                } else {
                    "true or false"
                };
                return Err(ApiError::invalid(format!(
                    "chat_template_kwargs.{key} must be {kind}"
                )));
            }
            if let Some(top) = self.get(key) {
                if &top != nested {
                    return Err(ApiError::invalid(format!(
                        "conflicting {key}: {} at the top level and {} in chat_template_kwargs; they are the same control under two names, send one",
                        repr(&top),
                        repr(nested)
                    )));
                }
            }
            self.set(key, nested);
        }
        Ok(())
    }

    /// OpenRouter's `reasoning: {effort, max_tokens, enabled}`.
    fn fold_reasoning(&mut self, reasoning: &Value) -> Result<(), ApiError> {
        let Value::Object(reasoning) = reasoning else {
            return Err(ApiError::invalid(
                "reasoning must be an object ({enabled, effort, max_tokens})",
            ));
        };
        if let Some(effort) = reasoning.get("effort").filter(|v| !v.is_null()) {
            if !effort.is_string() {
                return Err(ApiError::invalid("reasoning.effort must be a string"));
            }
            self.take("reasoning_effort", "reasoning.effort", effort.clone())?;
        }
        if let Some(budget) = reasoning.get("max_tokens").filter(|v| !v.is_null()) {
            let budget = positive(budget, "reasoning.max_tokens must be a whole number >= 1")?;
            self.take(
                "max_thinking_tokens",
                "reasoning.max_tokens",
                Value::from(budget),
            )?;
        }
        if let Some(enabled) = reasoning.get("enabled").filter(|v| !v.is_null()) {
            if !enabled.is_boolean() {
                return Err(ApiError::invalid("reasoning.enabled must be true or false"));
            }
            self.take("enable_thinking", "reasoning.enabled", enabled.clone())?;
        }
        Ok(())
    }

    /// Anthropic's `thinking: {type, budget_tokens}`, or a bare level string.
    fn fold_thinking(&mut self, thinking: &Value) -> Result<(), ApiError> {
        match thinking {
            Value::String(_) => self.take("reasoning_effort", "thinking", thinking.clone()),
            Value::Object(thinking) => {
                match thinking.get("type").and_then(Value::as_str) {
                    Some("enabled") => {
                        self.take("enable_thinking", "thinking.type", Value::Bool(true))?
                    }
                    Some("disabled") => {
                        self.take("enable_thinking", "thinking.type", Value::Bool(false))?
                    }
                    Some("adaptive") => {}
                    _ => {
                        return Err(ApiError::invalid(
                            "thinking.type must be enabled, disabled or adaptive",
                        ))
                    }
                }
                if let Some(budget) = thinking.get("budget_tokens").filter(|v| !v.is_null()) {
                    let budget =
                        positive(budget, "thinking.budget_tokens must be a whole number >= 1")?;
                    self.take(
                        "max_thinking_tokens",
                        "thinking.budget_tokens",
                        Value::from(budget),
                    )?;
                }
                Ok(())
            }
            _ => Err(ApiError::invalid(
                "thinking must be an object ({type, budget_tokens}) or a level",
            )),
        }
    }

    /// Sets `field` from an alias; a different value already there is refused.
    fn take(&mut self, field: &str, name: &str, value: Value) -> Result<(), ApiError> {
        if let Some(current) = self.get(field) {
            if current != value {
                return Err(ApiError::invalid(format!(
                    "conflicting {field}: {} and {} (as {name}); they are the same control under two names, send one",
                    repr(&current),
                    repr(&value)
                )));
            }
        }
        self.set(field, &value);
        Ok(())
    }

    fn get(&self, field: &str) -> Option<Value> {
        match field {
            "reasoning_effort" => self.reasoning_effort.clone().map(Value::String),
            "enable_thinking" => self.enable_thinking.map(Value::Bool),
            "preserve_thinking" => self.preserve_thinking.map(Value::Bool),
            "max_thinking_tokens" => self.max_thinking_tokens.map(Value::from),
            _ => None,
        }
    }

    /// `value` has been type-checked by the caller.
    fn set(&mut self, field: &str, value: &Value) {
        match field {
            "reasoning_effort" => self.reasoning_effort = value.as_str().map(str::to_string),
            "enable_thinking" => self.enable_thinking = value.as_bool(),
            "preserve_thinking" => self.preserve_thinking = value.as_bool(),
            "max_thinking_tokens" => self.max_thinking_tokens = value.as_i64(),
            _ => {}
        }
    }
}

/// A whole number >= 1, or a 400 with `message`.
fn positive(value: &Value, message: &str) -> Result<i64, ApiError> {
    whole(value)
        .filter(|&v| v >= 1)
        .ok_or_else(|| ApiError::invalid(message))
}

/// What a `reasoning_effort` value asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effort {
    /// `none`: thinking off.
    Off,
    Level(ReasoningEffort),
}

/// Maps a `reasoning_effort` value, case-insensitively: minimal and low are low, medium is
/// medium, high and xhigh are the top level, none turns thinking off.
pub fn parse_effort(value: &str) -> Result<Effort, ApiError> {
    if value.trim().eq_ignore_ascii_case("none") {
        return Ok(Effort::Off);
    }
    match value.to_lowercase().as_str() {
        "minimal" | "low" => Ok(Effort::Level(ReasoningEffort::Low)),
        "medium" => Ok(Effort::Level(ReasoningEffort::Medium)),
        "high" | "xhigh" => Ok(Effort::Level(ReasoningEffort::High)),
        _ => Err(ApiError::invalid(format!(
            "reasoning_effort '{value}' unsupported; use minimal|low|medium|high|xhigh|none"
        ))),
    }
}

/// What the chat template is told, and whether the reply is expected to open with a think block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Template {
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub preserve_thinking: Option<bool>,
    pub thinking: bool,
}

/// Applies the server defaults to the folded controls. An effort of `none` turns thinking off
/// and sends no effort.
pub fn template(reasoning: &Reasoning, defaults: &Defaults) -> Result<Template, ApiError> {
    let requested = reasoning
        .reasoning_effort
        .as_deref()
        .map(parse_effort)
        .transpose()?;
    let default = defaults
        .reasoning_effort
        .as_deref()
        .map(parse_effort)
        .transpose()?;
    let off = requested == Some(Effort::Off);
    let level = match requested.or(default) {
        Some(Effort::Level(level)) if !off => Some(level),
        _ => None,
    };
    let enable_thinking = if off {
        Some(false)
    } else {
        reasoning.enable_thinking.or(defaults.enable_thinking)
    };
    Ok(Template {
        enable_thinking,
        reasoning_effort: level,
        preserve_thinking: reasoning.preserve_thinking,
        thinking: enable_thinking != Some(false),
    })
}

/// The think-block budget: the request's or the server's, lowered so that
/// `answer_room` tokens of `max_tokens` stay for the answer. The room is
/// `--thinking-answer-room`, else `max(1024, 15% of max_tokens)`; 0 keeps no room. A budget of 0
/// means none and is never sent: `None`, as when the reply does not think.
pub fn thinking_budget(
    thinking: bool,
    requested: Option<i64>,
    max_tokens: usize,
    defaults: &Defaults,
) -> Option<usize> {
    if !thinking {
        return None;
    }
    let mut budget = requested
        .or(defaults.max_thinking_tokens.map(|b| b as i64))
        .filter(|&b| b > 0)
        .map(|b| b as usize);
    let room = defaults
        .answer_room
        .unwrap_or_else(|| (max_tokens * 15 / 100).max(1024));
    if room > 0 {
        let cap = max_tokens.saturating_sub(room).max(1);
        budget = Some(budget.map_or(cap, |b| b.min(cap)));
    }
    budget
}

/// Refuses a sampling value past its range; values are never clamped.
pub fn check_range(field: &str, value: Option<f64>, lo: f64, hi: f64) -> Result<(), ApiError> {
    match value {
        Some(v) if !(lo..=hi).contains(&v) => Err(ApiError::invalid(format!(
            "{field}={v:?} is outside {lo:?}..{hi:?}. Values past the range are NOT clamped, because a clamped request is indistinguishable, from the response, from one that was served as asked."
        ))),
        _ => Ok(()),
    }
}

/// The sampling values a request runs with: each is the request's, else the server's default.
/// A request `temperature` of 0 is greedy and takes no default; with neither a request nor a
/// default value, decoding is greedy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
}

impl Sampling {
    /// Range-checks the request's values, then fills the omitted ones from `defaults`.
    pub fn resolve(self, defaults: &Defaults) -> Result<Self, ApiError> {
        check_range("temperature", self.temperature, 0.0, 2.0)?;
        check_range("top_p", self.top_p, 0.0, 1.0)?;
        check_range("min_p", self.min_p, 0.0, 1.0)?;
        if let Some(k) = self.top_k.filter(|&k| k < 0) {
            return Err(ApiError::invalid(format!(
                "top_k={k} must be >= 0 (0 disables the filter)."
            )));
        }
        check_range(
            "presence_penalty",
            self.presence_penalty.map(f64::from),
            -2.0,
            2.0,
        )?;
        check_range(
            "frequency_penalty",
            self.frequency_penalty.map(f64::from),
            -2.0,
            2.0,
        )?;
        Ok(Self {
            temperature: self.temperature.or(defaults.temperature).or(Some(0.0)),
            top_p: self.top_p.or(defaults.top_p),
            top_k: self.top_k.or(defaults.top_k.map(|k| k as i64)),
            min_p: self.min_p.or(defaults.min_p),
            presence_penalty: self.presence_penalty.or(defaults.presence_penalty),
            frequency_penalty: self.frequency_penalty.or(defaults.frequency_penalty),
        })
    }
}

/// Checks `logit_bias`: keys are token ids below `vocab`, values are numbers,
/// at most [`LOGIT_BIAS_CAP`] entries. An empty map is no bias.
pub fn logit_bias(
    bias: Option<&Map<String, Value>>,
    vocab: Option<usize>,
) -> Result<Option<HashMap<u32, f32>>, ApiError> {
    let Some(bias) = bias.filter(|b| !b.is_empty()) else {
        return Ok(None);
    };
    if bias.len() > LOGIT_BIAS_CAP {
        return Err(ApiError::invalid(format!(
            "logit_bias has {} entries, over this server's cap of {LOGIT_BIAS_CAP}",
            bias.len()
        )));
    }
    let top = vocab.map_or(u32::MAX as i64, |v| v as i64 - 1);
    let mut out = HashMap::with_capacity(bias.len());
    for (key, value) in bias {
        let Ok(id) = key.trim().parse::<i64>() else {
            return Err(ApiError::invalid(format!(
                "logit_bias key '{key}' is not a token id"
            )));
        };
        if !(0..=top).contains(&id) {
            return Err(ApiError::invalid(format!(
                "logit_bias token id {id} is outside 0..{top}"
            )));
        }
        let bias = match value {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => s.trim().parse::<f64>().ok(),
            _ => None,
        }
        .ok_or_else(|| {
            ApiError::invalid(format!(
                "logit_bias value for token id {id} is not a number"
            ))
        })?;
        out.insert(id as u32, bias as f32);
    }
    Ok(Some(out))
}

/// Validates `tool_choice` against the request's tools. A named function
/// becomes that tool's full definition.
pub fn tool_choice(choice: Option<&Value>, tools: &[Tool]) -> Result<Option<ToolChoice>, ApiError> {
    let Some(choice) = choice.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    match choice {
        Value::String(name) => match name.as_str() {
            "auto" => Ok(Some(ToolChoice::Auto)),
            "none" => Ok(Some(ToolChoice::None)),
            "required" | "any" if tools.is_empty() => Err(ApiError::invalid(format!(
                "tool_choice '{name}' needs tools"
            ))),
            "required" | "any" => Ok(Some(ToolChoice::Required)),
            _ => Err(ApiError::invalid(format!(
                "tool_choice '{name}' unknown; use auto | none | required | {{'type': 'function', ...}}"
            ))),
        },
        Value::Object(object) if object.get("type").and_then(Value::as_str) == Some("function") => {
            let Some(name) = object
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
            else {
                return Err(ApiError::invalid("tool_choice.function.name is required"));
            };
            let tool = tools
                .iter()
                .find(|t| t.function.name == name)
                .ok_or_else(|| {
                    ApiError::invalid(format!(
                        "tool_choice names '{name}', which is not in tools"
                    ))
                })?;
            Ok(Some(ToolChoice::Tool(tool.clone())))
        }
        _ => Err(ApiError::invalid(
            "tool_choice must be a string or a {'type': 'function', ...} object",
        )),
    }
}

/// Responses' flat `{"type":"function","name":X}` becomes chat's
/// `{"type":"function","function":{"name":X}}`.
pub fn nest_tool_choice(choice: Value) -> Value {
    match choice {
        Value::Object(mut object)
            if object.get("type").and_then(Value::as_str) == Some("function")
                && !object.contains_key("function")
                && object.contains_key("name") =>
        {
            let name = object.remove("name").unwrap_or(Value::Null);
            object.insert("function".to_string(), serde_json::json!({ "name": name }));
            Value::Object(object)
        }
        other => other,
    }
}

/// What the `drafter` field asks for: `true` when the request opts out of
/// speculative decoding (`serial`). The loaded drafter's name (`SpeculativeAttachInfo::name`)
/// keeps the default path; another drafter name is unavailable; anything else is unknown.
/// Case-insensitive.
pub fn drafter(requested: Option<&str>, loaded: Option<&str>) -> Result<bool, ApiError> {
    let Some(name) = requested else {
        return Ok(false);
    };
    let key = name.to_lowercase();
    if key == "serial" {
        return Ok(true);
    }
    if Some(key.as_str()) == loaded {
        return Ok(false);
    }
    if SpeculativeAttachInfo::NAMES.contains(&key.as_str()) {
        let why = match loaded {
            Some(loaded) => {
                format!("this server runs the {loaded} drafter; use serial | {loaded}")
            }
            None => "this server runs no speculative drafter; use serial".to_string(),
        };
        return Err(ApiError::invalid(format!(
            "drafter '{key}' unavailable: {why}"
        )));
    }
    Err(ApiError::invalid(format!(
        "drafter '{name}' unknown; use serial | {}",
        SpeculativeAttachInfo::NAMES.join(" | ")
    )))
}

#[cfg(test)]
mod tests {
    use hanzo_engine::{Function, ToolType};
    use serde_json::json;

    use super::*;

    fn message(result: Result<impl std::fmt::Debug, ApiError>) -> String {
        let e = result.expect_err("must be refused");
        assert_eq!(e.status, axum::http::StatusCode::BAD_REQUEST);
        e.message
    }

    #[test]
    fn budget_names_fold_when_they_agree() {
        assert_eq!(fold_budget(&[("max_tokens", None)]).unwrap(), None);
        assert_eq!(
            fold_budget(&[
                ("max_tokens", Some(100)),
                ("max_completion_tokens", Some(100)),
                ("max_output_tokens", None),
            ])
            .unwrap(),
            Some(100)
        );
        assert_eq!(
            message(fold_budget(&[
                ("max_tokens", Some(200)),
                ("max_completion_tokens", Some(100)),
                ("max_output_tokens", None),
            ])),
            "conflicting token budgets (max_completion_tokens=100, max_tokens=200); they are the same budget under three names, send one"
        );
    }

    #[test]
    fn budget_defaults_caps_and_floors() {
        assert_eq!(resolve_budget(None, 8192, 65536).unwrap(), 8192);
        assert_eq!(resolve_budget(Some(65536), 8192, 65536).unwrap(), 65536);
        assert_eq!(resolve_budget(Some(0), 8192, 65536).unwrap(), 1);
        assert_eq!(resolve_budget(Some(-5), 8192, 65536).unwrap(), 1);
        assert!(message(resolve_budget(Some(65537), 8192, 65536))
            .starts_with("max_tokens 65537 exceeds this server's cap of 65536."));
    }

    fn fold(reasoning: Reasoning, aliases: Aliases<'_>) -> Result<Reasoning, ApiError> {
        reasoning.fold(aliases)
    }

    #[test]
    fn template_kwargs_fold_into_the_top_level() {
        let kwargs =
            json!({"enable_thinking": false, "reasoning_effort": "low", "preserve_thinking": true});
        let folded = fold(
            Reasoning::default(),
            Aliases {
                chat_template_kwargs: Some(&kwargs),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            folded,
            Reasoning {
                reasoning_effort: Some("low".into()),
                enable_thinking: Some(false),
                preserve_thinking: Some(true),
                max_thinking_tokens: None,
            }
        );
    }

    #[test]
    fn template_kwargs_refuse_unknown_keys_conflicts_and_bad_types() {
        let unknown = json!({"foo": 1, "bar": 2, "enable_thinking": true});
        assert_eq!(
            message(fold(
                Reasoning::default(),
                Aliases {
                    chat_template_kwargs: Some(&unknown),
                    ..Default::default()
                }
            )),
            "chat_template_kwargs contains unsupported key(s) 'bar', 'foo'; this server honours reasoning_effort, enable_thinking, preserve_thinking, which may also be sent as top-level fields"
        );
        let conflict = json!({"reasoning_effort": "High"});
        assert_eq!(
            message(fold(
                Reasoning {
                    reasoning_effort: Some("high".into()),
                    ..Default::default()
                },
                Aliases {
                    chat_template_kwargs: Some(&conflict),
                    ..Default::default()
                }
            )),
            "conflicting reasoning_effort: 'high' at the top level and 'High' in chat_template_kwargs; they are the same control under two names, send one"
        );
        let agree = json!({"enable_thinking": true});
        assert!(fold(
            Reasoning {
                enable_thinking: Some(true),
                ..Default::default()
            },
            Aliases {
                chat_template_kwargs: Some(&agree),
                ..Default::default()
            }
        )
        .is_ok());
        let string_bool = json!({"enable_thinking": "false"});
        assert_eq!(
            message(fold(
                Reasoning::default(),
                Aliases {
                    chat_template_kwargs: Some(&string_bool),
                    ..Default::default()
                }
            )),
            "chat_template_kwargs.enable_thinking must be true or false"
        );
        let number_effort = json!({"reasoning_effort": 3});
        assert_eq!(
            message(fold(
                Reasoning::default(),
                Aliases {
                    chat_template_kwargs: Some(&number_effort),
                    ..Default::default()
                }
            )),
            "chat_template_kwargs.reasoning_effort must be a string"
        );
        let list = json!(["enable_thinking"]);
        assert_eq!(
            message(fold(
                Reasoning::default(),
                Aliases {
                    chat_template_kwargs: Some(&list),
                    ..Default::default()
                }
            )),
            "chat_template_kwargs must be an object"
        );
    }

    #[test]
    fn budget_aliases_fold_and_conflict() {
        let five = json!(5);
        let folded = fold(
            Reasoning::default(),
            Aliases {
                thinking_budget: Some(&five),
                thinking_token_budget: Some(&five),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(folded.max_thinking_tokens, Some(5));

        let six = json!(6);
        assert_eq!(
            message(fold(
                Reasoning {
                    max_thinking_tokens: Some(5),
                    ..Default::default()
                },
                Aliases {
                    thinking_budget_tokens: Some(&six),
                    ..Default::default()
                }
            )),
            "conflicting max_thinking_tokens: 5 and 6 (as thinking_budget_tokens); they are the same control under two names, send one"
        );
        for bad in [json!(0), json!(true), json!("5"), json!(1.5)] {
            assert_eq!(
                message(fold(
                    Reasoning::default(),
                    Aliases {
                        thinking_budget: Some(&bad),
                        ..Default::default()
                    }
                )),
                "thinking_budget must be a whole number >= 1"
            );
        }
        assert_eq!(
            message(fold(
                Reasoning {
                    max_thinking_tokens: Some(-1),
                    ..Default::default()
                },
                Aliases::default()
            )),
            "max_thinking_tokens must be a whole number >= 0 (0 means no budget)"
        );
    }

    #[test]
    fn reasoning_object_folds_each_member() {
        let reasoning =
            json!({"effort": "high", "max_tokens": 300, "enabled": true, "summary": "auto"});
        let folded = fold(
            Reasoning::default(),
            Aliases {
                reasoning: Some(&reasoning),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(folded.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(folded.max_thinking_tokens, Some(300));
        assert_eq!(folded.enable_thinking, Some(true));

        let cases = [
            (
                json!("high"),
                "reasoning must be an object ({enabled, effort, max_tokens})",
            ),
            (json!({"effort": 1}), "reasoning.effort must be a string"),
            (
                json!({"max_tokens": 0}),
                "reasoning.max_tokens must be a whole number >= 1",
            ),
            (
                json!({"max_tokens": "5"}),
                "reasoning.max_tokens must be a whole number >= 1",
            ),
            (
                json!({"enabled": "yes"}),
                "reasoning.enabled must be true or false",
            ),
        ];
        for (value, expected) in cases {
            assert_eq!(
                message(fold(
                    Reasoning::default(),
                    Aliases {
                        reasoning: Some(&value),
                        ..Default::default()
                    }
                )),
                expected
            );
        }
        let low = json!({"effort": "low"});
        assert_eq!(
            message(fold(
                Reasoning {
                    reasoning_effort: Some("high".into()),
                    ..Default::default()
                },
                Aliases {
                    reasoning: Some(&low),
                    ..Default::default()
                }
            )),
            "conflicting reasoning_effort: 'high' and 'low' (as reasoning.effort); they are the same control under two names, send one"
        );
    }

    #[test]
    fn thinking_object_or_level_folds() {
        let enabled = json!({"type": "enabled", "budget_tokens": 1000});
        let folded = fold(
            Reasoning::default(),
            Aliases {
                thinking: Some(&enabled),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(folded.enable_thinking, Some(true));
        assert_eq!(folded.max_thinking_tokens, Some(1000));

        let disabled = json!({"type": "disabled"});
        let folded = fold(
            Reasoning::default(),
            Aliases {
                thinking: Some(&disabled),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(folded.enable_thinking, Some(false));

        let adaptive = json!({"type": "adaptive"});
        assert_eq!(
            fold(
                Reasoning::default(),
                Aliases {
                    thinking: Some(&adaptive),
                    ..Default::default()
                }
            )
            .unwrap(),
            Reasoning::default()
        );

        let level = json!("low");
        let folded = fold(
            Reasoning::default(),
            Aliases {
                thinking: Some(&level),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(folded.reasoning_effort.as_deref(), Some("low"));

        let cases = [
            (
                json!({"type": "on"}),
                "thinking.type must be enabled, disabled or adaptive",
            ),
            (
                json!({"type": "enabled", "budget_tokens": 0}),
                "thinking.budget_tokens must be a whole number >= 1",
            ),
            (
                json!(5),
                "thinking must be an object ({type, budget_tokens}) or a level",
            ),
        ];
        for (value, expected) in cases {
            assert_eq!(
                message(fold(
                    Reasoning::default(),
                    Aliases {
                        thinking: Some(&value),
                        ..Default::default()
                    }
                )),
                expected
            );
        }
        assert_eq!(
            message(fold(
                Reasoning {
                    enable_thinking: Some(false),
                    ..Default::default()
                },
                Aliases {
                    thinking: Some(&enabled),
                    ..Default::default()
                }
            )),
            "conflicting enable_thinking: False and True (as thinking.type); they are the same control under two names, send one"
        );
    }

    #[test]
    fn effort_maps_case_insensitively() {
        let cases = [
            ("minimal", Effort::Level(ReasoningEffort::Low)),
            ("LOW", Effort::Level(ReasoningEffort::Low)),
            ("medium", Effort::Level(ReasoningEffort::Medium)),
            ("High", Effort::Level(ReasoningEffort::High)),
            ("xhigh", Effort::Level(ReasoningEffort::High)),
            (" None ", Effort::Off),
        ];
        for (value, expected) in cases {
            assert_eq!(parse_effort(value).unwrap(), expected, "{value}");
        }
        assert_eq!(
            message(parse_effort("max")),
            "reasoning_effort 'max' unsupported; use minimal|low|medium|high|xhigh|none"
        );
        assert!(parse_effort(" low").is_err());
    }

    #[test]
    fn template_controls_apply_defaults_and_none() {
        let defaults = Defaults {
            reasoning_effort: Some("low".into()),
            ..Defaults::default()
        };
        let t = template(&Reasoning::default(), &defaults).unwrap();
        assert_eq!(t.reasoning_effort, Some(ReasoningEffort::Low));
        assert_eq!(t.enable_thinking, None);
        assert!(t.thinking);

        let off = Reasoning {
            reasoning_effort: Some("none".into()),
            enable_thinking: Some(true),
            ..Default::default()
        };
        let t = template(&off, &defaults).unwrap();
        assert_eq!(t.enable_thinking, Some(false));
        assert_eq!(t.reasoning_effort, None);
        assert!(!t.thinking);

        let no_thinking = Defaults {
            enable_thinking: Some(false),
            ..Defaults::default()
        };
        let t = template(&Reasoning::default(), &no_thinking).unwrap();
        assert_eq!(t.enable_thinking, Some(false));
        assert!(!t.thinking);
        let asked = Reasoning {
            enable_thinking: Some(true),
            preserve_thinking: Some(false),
            ..Default::default()
        };
        let t = template(&asked, &no_thinking).unwrap();
        assert_eq!(t.enable_thinking, Some(true));
        assert_eq!(t.preserve_thinking, Some(false));
        assert!(t.thinking);

        let bad = Reasoning {
            reasoning_effort: Some("ultra".into()),
            ..Default::default()
        };
        assert!(template(&bad, &defaults).is_err());
    }

    #[test]
    fn thinking_budget_leaves_answer_room() {
        let defaults = Defaults::default();
        // The budget against max_tokens, row by row.
        for (max_tokens, cap) in [
            (512, 1),
            (1024, 1),
            (2048, 1024),
            (8192, 6964),
            (16384, 13927),
            (65536, 55706),
        ] {
            assert_eq!(
                thinking_budget(true, None, max_tokens, &defaults),
                Some(cap),
                "{max_tokens}"
            );
        }
        assert_eq!(thinking_budget(true, Some(500), 8192, &defaults), Some(500));
        assert_eq!(
            thinking_budget(true, Some(8000), 8192, &defaults),
            Some(6964)
        );
        assert_eq!(thinking_budget(true, Some(0), 8192, &defaults), Some(6964));
        assert_eq!(thinking_budget(false, Some(500), 8192, &defaults), None);

        let no_room = Defaults {
            answer_room: Some(0),
            ..Defaults::default()
        };
        assert_eq!(thinking_budget(true, None, 8192, &no_room), None);
        assert_eq!(thinking_budget(true, Some(0), 8192, &no_room), None);
        assert_eq!(
            thinking_budget(true, Some(9000), 8192, &no_room),
            Some(9000)
        );

        let room = Defaults {
            answer_room: Some(100),
            max_thinking_tokens: Some(4000),
            ..Defaults::default()
        };
        assert_eq!(thinking_budget(true, None, 8192, &room), Some(4000));
        assert_eq!(thinking_budget(true, None, 1000, &room), Some(900));
    }

    #[test]
    fn sampling_ranges_and_defaults() {
        let request = Sampling {
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            frequency_penalty: None,
        };
        let defaults = Defaults {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(20),
            presence_penalty: Some(1.5),
            ..Defaults::default()
        };
        let resolved = request.resolve(&defaults).unwrap();
        assert_eq!(resolved.temperature, Some(0.7));
        assert_eq!(resolved.top_p, Some(0.9));
        assert_eq!(resolved.top_k, Some(20));
        assert_eq!(resolved.presence_penalty, Some(1.5));

        let greedy = Sampling {
            temperature: Some(0.0),
            top_k: Some(0),
            ..request
        };
        let resolved = greedy.resolve(&defaults).unwrap();
        assert_eq!(resolved.temperature, Some(0.0));
        assert_eq!(resolved.top_k, Some(0));
        assert_eq!(
            request.resolve(&Defaults::default()).unwrap().temperature,
            Some(0.0)
        );

        let hot = Sampling {
            temperature: Some(2.5),
            ..request
        };
        assert_eq!(
            message(hot.resolve(&defaults)),
            "temperature=2.5 is outside 0.0..2.0. Values past the range are NOT clamped, because a clamped request is indistinguishable, from the response, from one that was served as asked."
        );
        let wide = Sampling {
            top_p: Some(5.0),
            ..request
        };
        assert!(message(wide.resolve(&defaults)).starts_with("top_p=5.0 is outside 0.0..1.0."));
        let low = Sampling {
            min_p: Some(-0.1),
            ..request
        };
        assert!(message(low.resolve(&defaults)).starts_with("min_p=-0.1 is outside 0.0..1.0."));
        let negative = Sampling {
            top_k: Some(-1),
            ..request
        };
        assert_eq!(
            message(negative.resolve(&defaults)),
            "top_k=-1 must be >= 0 (0 disables the filter)."
        );
        let penalty = Sampling {
            frequency_penalty: Some(3.0),
            ..request
        };
        assert!(message(penalty.resolve(&defaults))
            .starts_with("frequency_penalty=3.0 is outside -2.0..2.0."));
    }

    #[test]
    fn logit_bias_checks_keys_ids_values_and_count() {
        let bias = |v: Value| v.as_object().cloned().unwrap();
        let ok = logit_bias(Some(&bias(json!({"5": 1.5, " 7": "-2"}))), Some(10))
            .unwrap()
            .unwrap();
        assert_eq!(ok[&5], 1.5);
        assert_eq!(ok[&7], -2.0);
        assert_eq!(logit_bias(Some(&Map::new()), Some(10)).unwrap(), None);
        assert_eq!(
            message(logit_bias(Some(&bias(json!({"abc": 1}))), Some(10))),
            "logit_bias key 'abc' is not a token id"
        );
        assert_eq!(
            message(logit_bias(Some(&bias(json!({"10": 1}))), Some(10))),
            "logit_bias token id 10 is outside 0..9"
        );
        assert_eq!(
            message(logit_bias(Some(&bias(json!({"-1": 1}))), Some(10))),
            "logit_bias token id -1 is outside 0..9"
        );
        assert_eq!(
            message(logit_bias(Some(&bias(json!({"3": "x"}))), Some(10))),
            "logit_bias value for token id 3 is not a number"
        );
        let many: Map<String, Value> = (0..=LOGIT_BIAS_CAP)
            .map(|i| (i.to_string(), json!(0)))
            .collect();
        assert_eq!(
            message(logit_bias(Some(&many), None)),
            "logit_bias has 20481 entries, over this server's cap of 20480"
        );
    }

    fn tool(name: &str) -> Tool {
        Tool {
            tp: ToolType::Function,
            function: Function {
                description: None,
                name: name.to_string(),
                parameters: None,
                strict: None,
            },
        }
    }

    #[test]
    fn tool_choice_is_validated_against_tools() {
        let tools = vec![tool("get_weather"), tool("search")];
        assert!(tool_choice(None, &tools).unwrap().is_none());
        assert!(matches!(
            tool_choice(Some(&json!("auto")), &tools).unwrap(),
            Some(ToolChoice::Auto)
        ));
        assert!(matches!(
            tool_choice(Some(&json!("none")), &[]).unwrap(),
            Some(ToolChoice::None)
        ));
        assert!(matches!(
            tool_choice(Some(&json!("required")), &tools).unwrap(),
            Some(ToolChoice::Required)
        ));
        match tool_choice(
            Some(&json!({"type": "function", "function": {"name": "search"}})),
            &tools,
        )
        .unwrap()
        {
            Some(ToolChoice::Tool(t)) => assert_eq!(t.function.name, "search"),
            other => panic!("expected a named tool, got {other:?}"),
        }
        let cases = [
            (json!("required"), "tool_choice 'required' needs tools", true),
            (
                json!("sometimes"),
                "tool_choice 'sometimes' unknown; use auto | none | required | {'type': 'function', ...}",
                false,
            ),
            (
                json!({"type": "function", "function": {}}),
                "tool_choice.function.name is required",
                false,
            ),
            (
                json!({"type": "function", "function": {"name": "nope"}}),
                "tool_choice names 'nope', which is not in tools",
                false,
            ),
            (
                json!({"type": "auto"}),
                "tool_choice must be a string or a {'type': 'function', ...} object",
                false,
            ),
            (
                json!(3),
                "tool_choice must be a string or a {'type': 'function', ...} object",
                false,
            ),
        ];
        for (value, expected, no_tools) in cases {
            let tools: &[Tool] = if no_tools { &[] } else { &tools };
            assert_eq!(message(tool_choice(Some(&value), tools)), expected);
        }
    }

    #[test]
    fn flat_tool_choice_is_nested() {
        assert_eq!(
            nest_tool_choice(json!({"type": "function", "name": "search"})),
            json!({"type": "function", "function": {"name": "search"}})
        );
        let nested = json!({"type": "function", "function": {"name": "search"}});
        assert_eq!(nest_tool_choice(nested.clone()), nested);
        assert_eq!(nest_tool_choice(json!("auto")), json!("auto"));
    }

    #[test]
    fn drafter_names() {
        assert!(!drafter(None, Some("mtp")).unwrap());
        assert!(drafter(Some("serial"), Some("mtp")).unwrap());
        assert!(drafter(Some("Serial"), None).unwrap());
        assert!(!drafter(Some("MTP"), Some("mtp")).unwrap());
        assert_eq!(
            message(drafter(Some("dflash"), Some("mtp"))),
            "drafter 'dflash' unavailable: this server runs the mtp drafter; use serial | mtp"
        );
        assert_eq!(
            message(drafter(Some("MTP"), None)),
            "drafter 'mtp' unavailable: this server runs no speculative drafter; use serial"
        );
        assert!(!drafter(Some("spec"), Some("spec")).unwrap());
        assert_eq!(
            message(drafter(Some("dflash2"), Some("mtp"))),
            "drafter 'dflash2' unknown; use serial | mtp | dflash | spec"
        );
    }
}
