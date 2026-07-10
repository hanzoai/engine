//! The one client for the Hanzo Engine's OpenAI-compatible `/v1` surface.
//!
//! Every network call the pipeline makes goes through here; nothing else builds
//! a URL or touches reqwest. Endpoints used:
//!   POST /v1/chat/completions   (planner, structured via response_format)
//!   POST /v1/images/generations (reference images, shot keyframes)
//!   POST /v1/audio/speech       (dialogue TTS)
//!   POST /v1/embeddings         (coherence hook, when a vision-embed model is loaded)
//!   POST /v1/videos + GET /v1/videos/{id}[/content]  (WAN, async job)

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine as _;
use serde_json::{json, Value};
use std::time::Duration;

#[derive(Clone)]
pub struct Engine {
    base: String,
    http: reqwest::Client,
}

impl Engine {
    /// `url` may include a trailing `/v1`; it is stripped so we always build `/v1/...`.
    pub fn new(url: &str) -> Result<Self> {
        let mut base = url.trim().trim_end_matches('/').to_string();
        if base.ends_with("/v1") {
            base.truncate(base.len() - 3);
        }
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?;
        Ok(Self { base, http })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    /// True once the engine answers `GET /v1/models`.
    pub async fn ready(&self) -> bool {
        self.http
            .get(self.url("/v1/models"))
            .timeout(Duration::from_secs(3))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    // --- chat -------------------------------------------------------------

    /// Chat completion returning the raw assistant string.
    pub async fn chat(&self, model: &str, system: &str, user: &str, max_tokens: usize) -> Result<String> {
        let body = json!({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        });
        self.chat_raw(body).await
    }

    /// Structured chat: constrain output to `schema` (engine llguidance JsonSchema),
    /// then parse. Falls back to extracting the first JSON object and one retry, so a
    /// weak model that ignores the constraint still yields usable JSON.
    pub async fn chat_json(
        &self,
        model: &str,
        system: &str,
        user: &str,
        schema: Value,
        max_tokens: usize,
    ) -> Result<Value> {
        let body = json!({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.4,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": schema}
            }
        });
        let text = self.chat_raw(body).await?;
        if let Some(v) = extract_json(&text) {
            return Ok(v);
        }
        // Retry once, unconstrained, asking explicitly for bare JSON.
        let retry = json!({
            "model": model,
            "messages": [
                {"role": "system", "content": format!("{system}\nRespond with ONLY a single JSON object, no prose, no code fence.")},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        });
        let text = self.chat_raw(retry).await?;
        extract_json(&text).ok_or_else(|| anyhow!("model did not return parseable JSON: {}", truncate(&text, 300)))
    }

    async fn chat_raw(&self, body: Value) -> Result<String> {
        let resp = self
            .http
            .post(self.url("/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .context("chat/completions request failed (is the engine up?)")?;
        let v: Value = ok_json(resp, "/v1/chat/completions").await?;
        v["choices"][0]["message"]["content"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| anyhow!("no choices[0].message.content in response"))
    }

    // --- images -----------------------------------------------------------

    /// Generate one image, returned as decoded bytes (requests b64_json).
    pub async fn image(&self, model: &str, prompt: &str, width: usize, height: usize) -> Result<Vec<u8>> {
        let body = json!({
            "model": model,
            "prompt": prompt,
            "n": 1,
            // Engine enum is PascalCase (Url|B64Json), not the OpenAI snake_case.
            "response_format": "B64Json",
            "width": width,
            "height": height,
        });
        let resp = self
            .http
            .post(self.url("/v1/images/generations"))
            .json(&body)
            .send()
            .await
            .context("images/generations request failed")?;
        let v: Value = ok_json(resp, "/v1/images/generations").await?;
        let b64 = v["data"][0]["b64_json"]
            .as_str()
            .ok_or_else(|| anyhow!("no data[0].b64_json in image response"))?;
        base64::engine::general_purpose::STANDARD
            .decode(b64)
            .context("decoding image b64")
    }

    // --- speech -----------------------------------------------------------

    /// Text-to-speech, returned as WAV bytes.
    pub async fn speech(&self, model: &str, input: &str) -> Result<Vec<u8>> {
        let body = json!({ "model": model, "input": input, "response_format": "wav" });
        let resp = self
            .http
            .post(self.url("/v1/audio/speech"))
            .json(&body)
            .send()
            .await
            .context("audio/speech request failed")?;
        if !resp.status().is_success() {
            let code = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("/v1/audio/speech returned {code}: {}", truncate(&text, 300));
        }
        Ok(resp.bytes().await?.to_vec())
    }

    /// Generate a music cue (ACE-Step sibling; `/v1/audio/music` pending). Returns
    /// encoded audio bytes. Errors until the endpoint lands — callers fall back to silence.
    pub async fn music(&self, model: &str, prompt: &str, duration_s: f32) -> Result<Vec<u8>> {
        let body = json!({ "model": model, "prompt": prompt, "duration": duration_s, "response_format": "wav" });
        let resp = self
            .http
            .post(self.url("/v1/audio/music"))
            .json(&body)
            .send()
            .await
            .context("audio/music request failed")?;
        if !resp.status().is_success() {
            let code = resp.status();
            bail!("/v1/audio/music returned {code} (endpoint pending)");
        }
        Ok(resp.bytes().await?.to_vec())
    }

    // --- embeddings -------------------------------------------------------

    /// Embed a single input, returning its vector. `input` may be text or, for a
    /// vision-embedding model, a data URI — the coherence hook uses whichever the
    /// loaded model accepts.
    pub async fn embed(&self, model: &str, input: &str) -> Result<Vec<f32>> {
        let body = json!({ "model": model, "input": input });
        let resp = self
            .http
            .post(self.url("/v1/embeddings"))
            .json(&body)
            .send()
            .await
            .context("embeddings request failed")?;
        let v: Value = ok_json(resp, "/v1/embeddings").await?;
        let arr = v["data"][0]["embedding"]
            .as_array()
            .ok_or_else(|| anyhow!("no data[0].embedding in response"))?;
        Ok(arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect())
    }

    // --- video (async job) ------------------------------------------------

    /// Queue a text-to-video job; returns the job id.
    pub async fn video_create(&self, prompt: &str, num_frames: usize, width: usize, height: usize, steps: usize) -> Result<String> {
        let body = json!({
            "prompt": prompt,
            "num_frames": num_frames,
            "width": width,
            "height": height,
            "steps": steps,
        });
        let resp = self
            .http
            .post(self.url("/v1/videos"))
            .json(&body)
            .send()
            .await
            .context("videos request failed")?;
        let v: Value = ok_json(resp, "/v1/videos").await?;
        v["id"].as_str().map(str::to_string).ok_or_else(|| anyhow!("no id in video job response"))
    }

    /// One poll of a video job: `(status, progress)`.
    pub async fn video_status(&self, id: &str) -> Result<(String, f32)> {
        let resp = self
            .http
            .get(self.url(&format!("/v1/videos/{id}")))
            .send()
            .await
            .context("video status request failed")?;
        let v: Value = ok_json(resp, "/v1/videos/{id}").await?;
        let status = v["status"].as_str().unwrap_or("unknown").to_string();
        let progress = v["progress"].as_f64().unwrap_or(0.0) as f32;
        Ok((status, progress))
    }

    /// Fetch the finished mp4 bytes.
    pub async fn video_content(&self, id: &str) -> Result<Vec<u8>> {
        let resp = self
            .http
            .get(self.url(&format!("/v1/videos/{id}/content")))
            .send()
            .await
            .context("video content request failed")?;
        if !resp.status().is_success() {
            bail!("/v1/videos/{id}/content returned {}", resp.status());
        }
        Ok(resp.bytes().await?.to_vec())
    }
}

async fn ok_json(resp: reqwest::Response, what: &str) -> Result<Value> {
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("{what} returned {status}: {}", truncate(&text, 400));
    }
    serde_json::from_str(&text).with_context(|| format!("{what} returned non-JSON: {}", truncate(&text, 200)))
}

/// Pull the first balanced JSON object/array out of arbitrary model text
/// (handles ```json fences, leading prose, trailing tokens). If the value was
/// truncated at a token limit, attempts to repair it by closing at the last
/// complete element.
pub fn extract_json(s: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str::<Value>(s.trim()) {
        return Some(v);
    }
    let bytes = s.as_bytes();
    let open = bytes.iter().position(|&b| b == b'{' || b == b'[')?;
    let (openc, closec) = if bytes[open] == b'{' { (b'{', b'}') } else { (b'[', b']') };
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for i in open..bytes.len() {
        let c = bytes[i];
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            continue;
        }
        match c {
            b'"' => in_str = true,
            x if x == openc => depth += 1,
            x if x == closec => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str(&s[open..=i]).ok();
                }
            }
            _ => {}
        }
    }
    repair_truncated(&s[open..])
}

/// Repair a truncated JSON value (never returned to depth 0) by cutting at the
/// last structural closer and appending the brackets needed to balance it. Yields
/// the complete prefix of a value that was cut off mid-generation.
fn repair_truncated(body: &str) -> Option<Value> {
    // Structural close positions (byte index of a '}' or ']' seen outside a string).
    let bytes = body.as_bytes();
    let mut closes = Vec::new();
    let (mut in_str, mut esc) = (false, false);
    for (i, &c) in bytes.iter().enumerate() {
        if in_str {
            match (esc, c) {
                (true, _) => esc = false,
                (false, b'\\') => esc = true,
                (false, b'"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            b'"' => in_str = true,
            b'}' | b']' => closes.push(i),
            _ => {}
        }
    }
    for &ci in closes.iter().rev() {
        let head = &body[..=ci];
        if let Some(closers) = balance(head) {
            if let Ok(v) = serde_json::from_str::<Value>(&format!("{head}{closers}")) {
                return Some(v);
            }
        }
    }
    None
}

/// Closers needed to balance `head` (string-aware). None if `head` ends inside a string.
fn balance(head: &str) -> Option<String> {
    let mut stack = Vec::new();
    let (mut in_str, mut esc) = (false, false);
    for &c in head.as_bytes() {
        if in_str {
            match (esc, c) {
                (true, _) => esc = false,
                (false, b'\\') => esc = true,
                (false, b'"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            b'"' => in_str = true,
            b'{' => stack.push(b'}'),
            b'[' => stack.push(b']'),
            b'}' | b']' => {
                stack.pop();
            }
            _ => {}
        }
    }
    if in_str {
        return None;
    }
    Some(stack.iter().rev().map(|&b| b as char).collect())
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_trailing_v1() {
        let e = Engine::new("http://h:1234/v1").unwrap();
        assert_eq!(e.url("/v1/models"), "http://h:1234/v1/models");
    }

    #[test]
    fn extract_bare_object() {
        let v = extract_json(r#"{"a":1}"#).unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn extract_from_fence_and_prose() {
        let s = "Sure! Here you go:\n```json\n{\"title\":\"X\",\"n\":2}\n```\nThanks";
        let v = extract_json(s).unwrap();
        assert_eq!(v["title"], "X");
        assert_eq!(v["n"], 2);
    }

    #[test]
    fn extract_ignores_braces_in_strings() {
        let v = extract_json(r#"garbage {"s":"a}b{c","k":3} tail"#).unwrap();
        assert_eq!(v["s"], "a}b{c");
        assert_eq!(v["k"], 3);
    }

    #[test]
    fn extract_array() {
        let v = extract_json("prefix [1,2,3] suffix").unwrap();
        assert_eq!(v[2], 3);
    }

    #[test]
    fn extract_none_on_junk() {
        assert!(extract_json("no json here").is_none());
    }

    #[test]
    fn repairs_truncated_shot_list() {
        // Cut off mid-way through the third object (as a token-limited model would).
        let s = r#"{"shots":[{"shot_type":"wide","action_prompt":"a"},{"shot_type":"close","action_prompt":"b"},{"shot_type":"med","action_pro"#;
        let v = extract_json(s).expect("must repair truncated json");
        let shots = v["shots"].as_array().unwrap();
        assert_eq!(shots.len(), 2, "keeps the two complete shots, drops the cut one");
        assert_eq!(shots[0]["shot_type"], "wide");
        assert_eq!(shots[1]["action_prompt"], "b");
    }

    #[test]
    fn repairs_truncated_after_complete_object() {
        let s = r#"{"a":1,"b":[{"x":1},{"y":2},"#;
        let v = extract_json(s).unwrap();
        assert_eq!(v["b"].as_array().unwrap().len(), 2);
    }
}
