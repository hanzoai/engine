//! Vocabulary representation, HuggingFace `tokenizer.json` loader, real-corpus
//! BPE vocabulary constructor, and compact byte-prefix [`VocabTrie`].
//!
//! The [`VocabTrie`] enables XGrammar's compile-time 3-way token classification
//! to evaluate shared BPE prefixes once and prune dead subtrees in O(1), making
//! compilation over `V = 32,768` and `V = 128,256` vocabularies fast while
//! remaining 100% bit-identical to per-token classification.

use std::collections::{HashMap, HashSet};

use crate::json::{parse_json, JsonValue};

/// A node in the compact byte-prefix trie over a [`Vocabulary`].
#[derive(Debug, Clone)]
pub struct TrieNode {
    /// Sorted child transitions `(byte, child_node_idx)`.
    pub children: Vec<(u8, u32)>,
    /// Token IDs whose exact byte sequence terminates at this node.
    pub tokens: Vec<u32>,
    /// Half-open range `[subtree_start, subtree_end)` into `VocabTrie::preorder_tokens`
    /// covering ALL non-empty tokens in this node's strict subtree (including `self.tokens`).
    pub subtree_start: u32,
    pub subtree_end: u32,
}

/// Compact byte-prefix trie built once per [`Vocabulary`] and shared across grammars.
#[derive(Debug, Clone)]
pub struct VocabTrie {
    nodes: Vec<TrieNode>,
    /// Flat DFS pre-order array of all non-empty non-EOS token IDs.
    preorder_tokens: Vec<u32>,
}

impl VocabTrie {
    fn build(tokens: &[Vec<u8>], eos_token_id: u32) -> Self {
        let mut nodes = vec![TrieNode {
            children: Vec::new(),
            tokens: Vec::new(),
            subtree_start: 0,
            subtree_end: 0,
        }];

        for (tid, bytes) in tokens.iter().enumerate() {
            let tid_u32 = tid as u32;
            if tid_u32 == eos_token_id || bytes.is_empty() {
                continue;
            }
            let mut cur = 0usize;
            for &b in bytes {
                let next_idx = match nodes[cur].children.binary_search_by_key(&b, |&(k, _)| k) {
                    Ok(pos) => nodes[cur].children[pos].1 as usize,
                    Err(insert_pos) => {
                        let new_id = nodes.len() as u32;
                        nodes.push(TrieNode {
                            children: Vec::new(),
                            tokens: Vec::new(),
                            subtree_start: 0,
                            subtree_end: 0,
                        });
                        nodes[cur].children.insert(insert_pos, (b, new_id));
                        new_id as usize
                    }
                };
                cur = next_idx;
            }
            nodes[cur].tokens.push(tid_u32);
        }

        // Compute DFS pre-order token ranges iteratively (no recursion overflow risk).
        let mut preorder_tokens = Vec::with_capacity(tokens.len());
        enum Frame {
            Enter(usize),
            Exit(usize),
        }
        let mut stack = vec![Frame::Enter(0)];
        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(u) => {
                    nodes[u].subtree_start = preorder_tokens.len() as u32;
                    for &tid in &nodes[u].tokens {
                        preorder_tokens.push(tid);
                    }
                    stack.push(Frame::Exit(u));
                    for &(_, child) in nodes[u].children.iter().rev() {
                        stack.push(Frame::Enter(child as usize));
                    }
                }
                Frame::Exit(u) => {
                    nodes[u].subtree_end = preorder_tokens.len() as u32;
                }
            }
        }

        Self {
            nodes,
            preorder_tokens,
        }
    }

    #[inline]
    pub fn node(&self, idx: u32) -> &TrieNode {
        &self.nodes[idx as usize]
    }

    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// All token IDs inside `node`'s subtree (including tokens ending at `node`).
    #[inline]
    pub fn subtree_tokens(&self, node: &TrieNode) -> &[u32] {
        &self.preorder_tokens[node.subtree_start as usize..node.subtree_end as usize]
    }

    pub fn memory_bytes(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>()
            + self.preorder_tokens.capacity() * std::mem::size_of::<u32>()
            + self.nodes.capacity() * std::mem::size_of::<TrieNode>();
        for n in &self.nodes {
            bytes += n.children.capacity() * std::mem::size_of::<(u8, u32)>();
            bytes += n.tokens.capacity() * std::mem::size_of::<u32>();
        }
        bytes
    }
}

/// Token vocabulary with byte strings, EOS token ID, stable fingerprint, and [`VocabTrie`].
#[derive(Debug, Clone)]
pub struct Vocabulary {
    tokens: Vec<Vec<u8>>,
    eos_token_id: u32,
    fingerprint: u64,
    trie: VocabTrie,
}

impl Vocabulary {
    /// Construct a [`Vocabulary`] from raw byte slices and an EOS token ID.
    pub fn new(tokens: Vec<Vec<u8>>, eos_token_id: u32) -> Self {
        assert!(
            (eos_token_id as usize) < tokens.len(),
            "eos_token_id ({eos_token_id}) out of bounds for vocab size {}",
            tokens.len()
        );
        let mut hash: u64 = 0xcbf29ce484222325;
        let mix = |h: &mut u64, v: u64| {
            *h ^= v;
            *h = h.wrapping_mul(0x100000001b3);
        };
        mix(&mut hash, tokens.len() as u64);
        mix(&mut hash, eos_token_id as u64);
        for tok in &tokens {
            mix(&mut hash, tok.len() as u64);
            for &b in tok {
                mix(&mut hash, b as u64);
            }
        }
        let trie = VocabTrie::build(&tokens, eos_token_id);
        Self {
            tokens,
            eos_token_id,
            fingerprint: hash,
            trie,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    #[inline]
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    #[inline]
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    #[inline]
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        &self.tokens[id as usize]
    }

    #[inline]
    pub fn trie(&self) -> &VocabTrie {
        &self.trie
    }

    /// Decode a sequence of token IDs into raw bytes (stopping at EOS).
    pub fn decode_bytes(&self, token_ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &tid in token_ids {
            if tid == self.eos_token_id {
                break;
            }
            if let Some(bytes) = self.tokens.get(tid as usize) {
                out.extend_from_slice(bytes);
            }
        }
        out
    }

    /// Decode a sequence of token IDs into a UTF-8 string (lossy if unconstrained broke UTF-8).
    pub fn decode_string_lossy(&self, token_ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(token_ids)).into_owned()
    }

    /// Summary statistics of the vocabulary (useful for benchmark reporting).
    pub fn stats(&self) -> VocabStats {
        let mut total_bytes = 0usize;
        let mut space_prefixed = 0usize;
        let mut quote_containing = 0usize;
        let mut multibyte_utf8 = 0usize;
        let mut single_byte = 0usize;
        for (idx, tok) in self.tokens.iter().enumerate() {
            if idx as u32 == self.eos_token_id || tok.is_empty() {
                continue;
            }
            total_bytes += tok.len();
            if tok.len() == 1 {
                single_byte += 1;
            }
            if tok[0] == b' ' {
                space_prefixed += 1;
            }
            if tok.contains(&b'"') {
                quote_containing += 1;
            }
            if tok.iter().any(|&b| b >= 0x80) && std::str::from_utf8(tok).is_ok() {
                multibyte_utf8 += 1;
            }
        }
        let non_empty = self.tokens.len().saturating_sub(1).max(1);
        VocabStats {
            vocab_size: self.tokens.len(),
            trie_nodes: self.trie.num_nodes(),
            mean_token_bytes: total_bytes as f64 / non_empty as f64,
            space_prefixed_fraction: space_prefixed as f64 / non_empty as f64,
            quote_containing_fraction: quote_containing as f64 / non_empty as f64,
            multibyte_utf8_tokens: multibyte_utf8,
            single_byte_tokens: single_byte,
        }
    }

    /// Parse a real HuggingFace `tokenizer.json` file (byte-level BPE / WordPiece / Unigram).
    pub fn from_tokenizer_json(json_str: &str) -> Result<Self, String> {
        let root = parse_json(json_str).map_err(|e| format!("tokenizer.json parse error: {e}"))?;
        let model = root
            .get("model")
            .ok_or_else(|| "missing 'model' in tokenizer.json".to_string())?;
        let vocab_obj = model
            .get("vocab")
            .and_then(JsonValue::as_object)
            .ok_or_else(|| "missing 'model.vocab' object in tokenizer.json".to_string())?;

        let mut max_id = 0usize;
        for (_, v) in vocab_obj {
            if let Some(id) = v.as_u64() {
                max_id = max_id.max(id as usize);
            }
        }

        let mut eos_id: Option<u32> = None;
        let mut added_special: HashSet<u32> = HashSet::new();
        if let Some(added) = root.get("added_tokens").and_then(JsonValue::as_array) {
            for entry in added {
                if let Some(id) = entry.get("id").and_then(JsonValue::as_u64) {
                    max_id = max_id.max(id as usize);
                    let is_special = entry
                        .get("special")
                        .and_then(JsonValue::as_bool)
                        .unwrap_or(false);
                    if is_special {
                        added_special.insert(id as u32);
                        if let Some(content) = entry.get("content").and_then(JsonValue::as_str) {
                            let lower = content.to_ascii_lowercase();
                            if lower.contains("eos")
                                || lower.contains("endoftext")
                                || lower.contains("eot_id")
                                || lower == "</s>"
                            {
                                eos_id = Some(id as u32);
                            }
                        }
                    }
                }
            }
        }

        let mut tokens = vec![Vec::new(); max_id + 1];
        let gpt2_table = gpt2_unicode_to_byte();
        for (tok_str, id_val) in vocab_obj {
            if let Some(id) = id_val.as_u64() {
                let id_u32 = id as u32;
                if added_special.contains(&id_u32) {
                    tokens[id as usize] = Vec::new();
                } else {
                    tokens[id as usize] = decode_gpt2_token(tok_str, &gpt2_table);
                }
            }
        }

        let eos = eos_id.unwrap_or_else(|| {
            let id = tokens.len() as u32;
            tokens.push(Vec::new());
            id
        });
        tokens[eos as usize] = Vec::new();
        Ok(Self::new(tokens, eos))
    }

    /// Build a realistic byte-level BPE vocabulary of exact size `target_size`
    /// (e.g., `32_768` or `128_256`) harvested from real corpus text (such as
    /// repository Rust/JSON/Markdown sources) with deterministic BPE subword fills.
    ///
    /// Guarantees:
    /// - Token `0` is EOS (`<|eos|>`).
    /// - Tokens `1..=256` cover all 256 raw single bytes `0x00..=0xFF` (byte-level BPE completeness).
    /// - Realistic proportions of space-prefixed words, JSON/code delimiters, digits,
    ///   quoted/delimiter-attached tokens, and valid multi-byte UTF-8 sequences.
    pub fn build_from_corpus(target_size: usize, corpus_texts: &[&str], seed: u64) -> Self {
        assert!(
            target_size >= 512,
            "target_size must be at least 512 for byte-level BPE"
        );

        let mut tokens: Vec<Vec<u8>> = Vec::with_capacity(target_size);
        let mut seen: HashSet<Vec<u8>> = HashSet::with_capacity(target_size * 2);

        // Token 0: EOS
        tokens.push(Vec::new());

        let push_tok = |bytes: Vec<u8>, tokens: &mut Vec<Vec<u8>>, seen: &mut HashSet<Vec<u8>>| {
            if tokens.len() < target_size && !bytes.is_empty() && seen.insert(bytes.clone()) {
                tokens.push(bytes);
            }
        };

        // 1. All 256 raw single bytes (guarantees full byte-level fallback).
        for b in 0u8..=255u8 {
            push_tok(vec![b], &mut tokens, &mut seen);
        }

        // 2. Canonical JSON, tool-call, code, escape, and UTF-8 BPE tokens.
        let structural: &[&[u8]] = &[
            b"{\"",
            b"\": ",
            b"\":\"",
            b"\":",
            b"\", \"",
            b"\",\"",
            b"\", ",
            b"\",",
            b"\"}",
            b"\"]",
            b"\": [",
            b"\": {",
            b"\": true",
            b"\": false",
            b"\": null",
            b"\": \"",
            b", \"",
            b",\n  \"",
            b",\n    \"",
            b"{\n  \"",
            b"\n}",
            b"\n]",
            b"}]",
            b"}}",
            b"]}",
            b"[]",
            b"{}",
            b"[\"",
            b"true",
            b"false",
            b"null",
            b" true",
            b" false",
            b" null",
            b"true,",
            b"false,",
            b"null,",
            b"true}",
            b"false}",
            b"null}",
            b"tool",
            b"arguments",
            b"name",
            b"status",
            b"items",
            b"children",
            b"label",
            b"confidence",
            b"rationale",
            b"\\n",
            b"\\t",
            b"\\\"",
            b"\\\\",
            b"\\u00",
            b"  ",
            b"    ",
            b"\n  ",
            b"\n    ",
            b"()",
            b"((",
            b"))",
            b" + ",
            b" - ",
            b" * ",
            b" / ",
            b" == ",
            b" != ",
            b" -> ",
            b"::",
            // Valid multi-byte UTF-8 pieces (CJK, European accented, Greek, emoji)
            "日本語".as_bytes(),
            "中文".as_bytes(),
            "한국어".as_bytes(),
            "café".as_bytes(),
            "naïve".as_bytes(),
            "résumé".as_bytes(),
            "über".as_bytes(),
            "αβγ".as_bytes(),
            "€100".as_bytes(),
            "🚀".as_bytes(),
            "✅".as_bytes(),
            // Partial multi-byte UTF-8 BPE splits (like Llama-3 / GPT-4 byte-level BPE)
            &[0xE4, 0xB8],
            &[0xE6, 0x97],
            &[0xF0, 0x9F, 0x9A],
            &[0xC3, 0xA9],
        ];
        for &s in structural {
            push_tok(s.to_vec(), &mut tokens, &mut seen);
        }

        // Common numeric tokens (0..250 or 0..1000 depending on target_size).
        let num_cap = if target_size >= 8192 { 1000u32 } else { 200u32 };
        for n in 0..num_cap {
            push_tok(n.to_string().into_bytes(), &mut tokens, &mut seen);
            if n < 100 {
                push_tok(format!(" {n}").into_bytes(), &mut tokens, &mut seen);
                push_tok(format!("{n},").into_bytes(), &mut tokens, &mut seen);
                push_tok(format!("{n}}}").into_bytes(), &mut tokens, &mut seen);
                push_tok(format!("-{n}").into_bytes(), &mut tokens, &mut seen);
                push_tok(format!("0.{n:02}").into_bytes(), &mut tokens, &mut seen);
            }
        }

        // 3. Harvest real word, space-prefixed word, subword, and punctuation-joined
        //    tokens ranked by frequency from the provided corpus texts.
        let mut word_counts: HashMap<String, u32> = HashMap::new();
        let mut bigram_counts: HashMap<String, u32> = HashMap::new();

        for text in corpus_texts {
            let mut prev_word: Option<&str> = None;
            for raw_piece in text.split(|c: char| c.is_ascii_whitespace()) {
                let cleaned = raw_piece
                    .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-');
                if cleaned.len() >= 2 && cleaned.len() <= 24 {
                    *word_counts.entry(cleaned.to_string()).or_insert(0) += 1;
                    // Also harvest subword prefixes/suffixes (real BPE fragments).
                    if cleaned.len() >= 5 {
                        let mid = cleaned.len() / 2;
                        if cleaned.is_char_boundary(mid) {
                            *word_counts.entry(cleaned[..mid].to_string()).or_insert(0) += 1;
                            *word_counts.entry(cleaned[mid..].to_string()).or_insert(0) += 1;
                        }
                    }
                    if let Some(prev) = prev_word {
                        if prev.len() + cleaned.len() <= 16 {
                            *bigram_counts
                                .entry(format!("{prev}_{cleaned}"))
                                .or_insert(0) += 1;
                        }
                    }
                    prev_word = Some(cleaned);
                } else {
                    prev_word = None;
                }
            }
        }

        let mut ranked_words: Vec<(String, u32)> = word_counts.into_iter().collect();
        ranked_words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // Insert top corpus words in raw, space-prefixed (realistic ~35% BPE ratio),
        // and JSON/code delimiter-attached variants.
        for (idx, (word, _)) in ranked_words.iter().enumerate() {
            if tokens.len() >= target_size {
                break;
            }
            push_tok(word.as_bytes().to_vec(), &mut tokens, &mut seen);
            push_tok(format!(" {word}").into_bytes(), &mut tokens, &mut seen);
            if idx < 1500 {
                push_tok(format!("{word}\"").into_bytes(), &mut tokens, &mut seen);
                push_tok(format!("{word}\",").into_bytes(), &mut tokens, &mut seen);
                push_tok(format!("{word}\":").into_bytes(), &mut tokens, &mut seen);
                push_tok(format!("\"{word}").into_bytes(), &mut tokens, &mut seen);
            }
            if idx < 600 {
                push_tok(format!("{word}\"}}").into_bytes(), &mut tokens, &mut seen);
                push_tok(format!("{word}]").into_bytes(), &mut tokens, &mut seen);
                push_tok(format!("{word}(").into_bytes(), &mut tokens, &mut seen);
                push_tok(format!("{word})").into_bytes(), &mut tokens, &mut seen);
            }
        }

        let mut ranked_bigrams: Vec<(String, u32)> = bigram_counts.into_iter().collect();
        ranked_bigrams.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for (bg, _) in &ranked_bigrams {
            if tokens.len() >= target_size {
                break;
            }
            push_tok(bg.as_bytes().to_vec(), &mut tokens, &mut seen);
            push_tok(format!(" {bg}").into_bytes(), &mut tokens, &mut seen);
        }

        // 4. Deterministic realistic BPE syllable/subword/suffix merges to reach exact target_size
        //    (matching Llama-3 128,256 / Mistral 32,768 length & space-prefix distribution).
        const PREFIXES: &[&str] = &[
            " ", " ", " ", "", "", "un", "re", "in", "de", "pre", "sub", "inter", "auto", "meta",
            "micro", "hyper", "multi", "over", "under", "trans", "non", "anti", "co", "ex", "pro",
            "sys", "net", "api", "cfg", "kv", "gpu", "cpu", "tok", "mem", "log", "err",
        ];
        const ROOTS: &[&str] = &[
            "act", "add", "agg", "alloc", "app", "arg", "arr", "ast", "async", "attn", "auth",
            "back", "base", "batch", "bind", "bit", "blk", "bool", "bound", "branch", "buf",
            "build", "byte", "cache", "call", "cap", "cast", "cell", "cfg", "chain", "chan",
            "char", "check", "chunk", "class", "clear", "cli", "clone", "close", "cloud", "code",
            "col", "comp", "conf", "conn", "const", "ctx", "cuda", "cur", "data", "db", "debug",
            "dec", "def", "del", "delta", "dep", "depth", "desc", "dev", "dict", "diff", "dim",
            "dir", "doc", "dom", "done", "draft", "drop", "dyn", "edge", "elem", "emit", "enc",
            "end", "eng", "entry", "enum", "env", "eos", "err", "eval", "event", "exec", "exit",
            "exp", "expr", "ext", "fail", "fast", "feat", "fetch", "field", "file", "fill", "filt",
            "find", "fix", "flag", "flat", "float", "flow", "flush", "fmt", "fold", "fork", "form",
            "frame", "free", "from", "full", "func", "fuse", "gate", "gen", "get", "glob", "gpu",
            "grad", "graph", "grid", "group", "guard", "half", "hash", "head", "heap", "help",
            "hex", "high", "hint", "hit", "hook", "host", "http", "id", "idx", "impl", "in", "inc",
            "index", "info", "init", "input", "inst", "int", "io", "item", "iter", "join", "json",
            "jump", "keep", "kern", "key", "kind", "kv", "label", "lang", "last", "lat", "layer",
            "lazy", "leaf", "left", "len", "level", "lex", "lib", "limit", "line", "link", "list",
            "lit", "live", "load", "loc", "lock", "log", "logit", "long", "look", "loop", "loss",
            "low", "lru", "main", "make", "map", "mark", "mask", "match", "math", "max", "mcp",
            "mean", "mem", "merge", "mesh", "meta", "metal", "metric", "mid", "min", "miss", "mix",
            "mode", "model", "mod", "move", "msg", "mul", "mut", "name", "nan", "neg", "nest",
            "net", "new", "next", "node", "none", "norm", "note", "null", "num", "obj", "off",
            "ok", "old", "one", "op", "open", "opt", "ord", "org", "out", "over", "own", "pack",
            "pad", "page", "pair", "pan", "par", "param", "parse", "part", "pass", "path", "pat",
            "pda", "peak", "peek", "perf", "perm", "pipe", "plan", "play", "plot", "plug", "plus",
            "pod", "point", "pol", "pool", "pop", "port", "pos", "post", "pow", "pref", "prep",
            "prev", "print", "priv", "prob", "proc", "prod", "prof", "prog", "proj", "prom",
            "prop", "proto", "prov", "proxy", "prune", "pub", "pull", "pure", "push", "put",
            "quant", "query", "queue", "quick", "quote", "race", "radix", "rand", "range", "rank",
            "rate", "ratio", "raw", "read", "real", "rec", "recv", "red", "ref", "reg", "regex",
            "rel", "rem", "ren", "rep", "req", "res", "reset", "resp", "rest", "ret", "retry",
            "rev", "right", "ring", "risk", "role", "roll", "root", "rot", "route", "row", "rpc",
            "rule", "run", "rust", "safe", "same", "samp", "sand", "save", "scale", "scan",
            "sched", "schema", "scope", "score", "seal", "sec", "seed", "seek", "seg", "sel",
            "self", "send", "seq", "ser", "serv", "sess", "set", "sha", "shape", "share", "shift",
            "short", "show", "shut", "side", "sig", "sign", "sim", "simd", "size", "skip", "slab",
            "slice", "slot", "slow", "small", "snap", "sock", "soft", "sort", "span", "spec",
            "speed", "spin", "split", "sql", "src", "stack", "stage", "start", "stat", "state",
            "step", "stop", "store", "str", "stream", "strict", "strip", "struct", "stub", "style",
            "sub", "sum", "swap", "sym", "sync", "sys", "tab", "table", "tag", "tail", "take",
            "tap", "targ", "task", "tco", "tcp", "team", "temp", "tens", "term", "test", "text",
            "thr", "tick", "tie", "tile", "time", "tip", "title", "tls", "tmp", "tok", "token",
            "toml", "tool", "top", "total", "trace", "track", "train", "trait", "trap", "tree",
            "trie", "trim", "trip", "true", "trunc", "try", "tuple", "turn", "type", "uint", "un",
            "undo", "uni", "union", "unit", "unix", "up", "upd", "uri", "url", "use", "user",
            "utf", "util", "uuid", "val", "valid", "var", "vec", "ver", "vert", "view", "virt",
            "vis", "vocab", "void", "vol", "vote", "wait", "wake", "walk", "wall", "warn", "warp",
            "watch", "wave", "web", "weight", "well", "when", "wide", "win", "wire", "with",
            "word", "work", "world", "wrap", "write", "ws", "xml", "xor", "yaml", "yield", "zen",
            "zero", "zip", "zone",
        ];
        const SUFFIXES: &[&str] = &[
            "", "s", "ed", "ing", "er", "or", "ly", "al", "ic", "ion", "ity", "ize", "ate", "ive",
            "ous", "ful", "less", "ness", "ment", "able", "ible", "ist", "ism", "ary", "ory",
            "ure", "age", "ant", "ent", "_id", "_ms", "_us", "_kb", "_v2", "0", "1", "2", "\"",
            "\",", "\":", "}", "]", ",", ":", ")", "(", ".", "/", "-", "_", "\n",
        ];

        // Deterministic xorshift64* PRNG seeded from `seed`.
        let mut rng = seed ^ 0x9e37_79b9_7f4a_7c15;
        let next_u64 = |state: &mut u64| -> u64 {
            let mut x = *state;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            *state = x;
            x.wrapping_mul(0x2545_f491_4f6c_dd1d)
        };

        // First deterministic structured pass over PREFIX x ROOT x SUFFIX combinations.
        let mut salt = 0u64;
        while tokens.len() < target_size {
            let r1 = next_u64(&mut rng) as usize;
            let r2 = next_u64(&mut rng) as usize;
            let r3 = next_u64(&mut rng) as usize;
            let lead_space = if (r1 >> 8) % 20 < 7 { " " } else { "" };
            let p = PREFIXES[r1 % PREFIXES.len()].trim_start();
            let root = ROOTS[r2 % ROOTS.len()];
            let s = SUFFIXES[r3 % SUFFIXES.len()];

            let candidate = if (r1 >> 16).is_multiple_of(7) {
                let root2 = ROOTS[(r2 / ROOTS.len().max(1) + salt as usize) % ROOTS.len()];
                format!("{lead_space}{p}{root}_{root2}")
            } else if (r1 >> 16).is_multiple_of(11) {
                let root2 = ROOTS[(r3 + salt as usize) % ROOTS.len()];
                format!("{lead_space}{p}{root}{root2}")
            } else if (r1 >> 16).is_multiple_of(19) {
                format!("{lead_space}{p}{root}{}", (r3 % 97) + 3)
            } else {
                format!("{lead_space}{p}{root}{s}")
            };
            push_tok(candidate.into_bytes(), &mut tokens, &mut seen);
            salt = salt.wrapping_add(1);
            if salt > (target_size as u64) * 32 {
                // Guaranteed termination fallback with deterministic unique alphanumeric pieces.
                let fallback = format!(" tok_{salt:x}");
                push_tok(fallback.into_bytes(), &mut tokens, &mut seen);
            }
        }

        Self::new(tokens, 0)
    }

    /// Convenience constructor for deterministic unit tests and quick benchmarks.
    pub fn realistic_bpe(target_size: usize) -> Self {
        const BUILTIN_CORPUS: &[&str] = &[
            r#"{"status":"ok","request_id":"req_1234","latency_ms":12.5,"cached":true,"items":[{"id":"a1","score":0.98,"tags":["rust","llm","xgrammar"]}]}"#,
            r#"{"tool":"search_code","arguments":{"query":"SequenceRecognizer","path":"hanzo-engine/src","max_results":10,"case_sensitive":false}}"#,
            r#"{"kind":"binary","value":"+","line":42,"children":[{"kind":"literal","value":"1","line":42,"children":[]}]}"#,
            "hanzo engine zen kai router tokenizer paged attention kv cache grammar mask pushdown automaton schema",
        ];
        Self::build_from_corpus(target_size, BUILTIN_CORPUS, 0x1346_2026)
    }
}

/// Summary statistics for a [`Vocabulary`].
#[derive(Debug, Clone)]
pub struct VocabStats {
    pub vocab_size: usize,
    pub trie_nodes: usize,
    pub mean_token_bytes: f64,
    pub space_prefixed_fraction: f64,
    pub quote_containing_fraction: f64,
    pub multibyte_utf8_tokens: usize,
    pub single_byte_tokens: usize,
}

/// Build the inverse of GPT-2's `bytes_to_unicode()` bijection for HuggingFace `tokenizer.json`.
fn gpt2_unicode_to_byte() -> HashMap<char, u8> {
    let mut bs: Vec<u8> = Vec::new();
    bs.extend(b'!'..=b'~');
    bs.extend(0xA1u8..=0xACu8);
    bs.extend(0xAEu8..=0xFFu8);
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;
    for b in 0u8..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut map = HashMap::with_capacity(256);
    for (b, c) in bs.into_iter().zip(cs) {
        if let Some(ch) = char::from_u32(c) {
            map.insert(ch, b);
        }
    }
    map
}

fn decode_gpt2_token(s: &str, table: &HashMap<char, u8>) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    for ch in s.chars() {
        if let Some(&b) = table.get(&ch) {
            out.push(b);
        } else {
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            out.extend_from_slice(encoded.as_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_exact_vocab_sizes_with_byte_completeness() {
        let v = Vocabulary::realistic_bpe(2048);
        assert_eq!(v.len(), 2048);
        assert_eq!(v.eos_token_id(), 0);
        // Tokens 1..=256 cover every raw byte 0x00..=0xFF.
        for b in 0u8..=255u8 {
            assert_eq!(v.token_bytes((b as u32) + 1), &[b]);
        }
        let stats = v.stats();
        assert_eq!(stats.vocab_size, 2048);
        assert!(stats.trie_nodes > 500);
        assert!(stats.space_prefixed_fraction > 0.15);
    }

    #[test]
    fn parses_minimal_hf_tokenizer_json() {
        let sample = r#"{
            "model": {
                "vocab": {
                    "hello": 0,
                    "Ġworld": 1,
                    "Ċ": 2,
                    "<|eos|>": 3
                }
            },
            "added_tokens": [
                {"id": 3, "content": "<|eos|>", "special": true}
            ]
        }"#;
        let vocab = Vocabulary::from_tokenizer_json(sample).expect("parse tokenizer.json");
        assert_eq!(vocab.len(), 4);
        assert_eq!(vocab.eos_token_id(), 3);
        assert_eq!(vocab.token_bytes(0), b"hello");
        assert_eq!(vocab.token_bytes(1), b" world");
        assert_eq!(vocab.token_bytes(2), b"\n");
        assert!(vocab.token_bytes(3).is_empty());
    }
}
