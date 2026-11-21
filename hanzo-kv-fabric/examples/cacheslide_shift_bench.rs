// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

//! CacheSlide-style position-shifted reuse demo at the fabric level
//! (HIP-1345 §2.2, Bounty #12 / engine#93).
//!
//! Simulates an 8-turn agent loop where the system and tool segments are
//! byte-identical across turns but shift in absolute prompt position as turn
//! history accumulates in front of them. Compares:
//!
//! 1. Strict prefix (parent-chained) block hashing — mirroring
//!    `paged_attention::block_hash` — which misses every block of a shifted
//!    segment.
//! 2. The position-independent [`salted_segment_hash`] + [`SegmentDirectory`]
//!    pair from this crate, which hits and exposes the position shift for the
//!    attention path to translate (fail-closed).
//!
//! Run: `cargo run --release -p hanzo-kv-fabric --example cacheslide_shift_bench`

use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Instant;

use hanzo_kv_fabric::segment::{salted_segment_hash, SegmentDirectory, SegmentMeta};

const BLOCK_SIZE: usize = 16;
const SYSTEM_PROMPT: &[u32] = &[7001; 64]; // 64-token invariant system segment
const TOOL_SCHEMA: &[u32] = &[7002; 64]; // 64-token invariant tool schema
const TURN_REPLY: &[u32] = &[7100; 48]; // fresh turn content, grows the history
const TURNS: usize = 8;

/// Chained block hash over one fixed-size block (strict prefix caching):
/// `hash(parent, tokens)` — position dependent by construction.
fn chained_block_hash(parent: u64, tokens: &[u32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    parent.hash(&mut hasher);
    tokens.hash(&mut hasher);
    hasher.finish()
}

fn chain_block_hashes(sequence: &[u32], block_size: usize) -> Vec<u64> {
    sequence
        .chunks(block_size)
        .scan(0u64, |parent, chunk| {
            let h = chained_block_hash(*parent, chunk);
            *parent = h;
            Some(h)
        })
        .collect()
}

/// Replay `sequence`'s chained block hashes against the turn-0 chain and count hits.
fn chained_prefix_hits(sequence: &[u32], published: &[u64], block_size: usize) -> usize {
    chain_block_hashes(sequence, block_size)
        .into_iter()
        .zip(published.iter().chain(std::iter::repeat(&0)))
        .filter(|(hash, published_hash)| **published_hash != 0u64 && *hash == **published_hash)
        .count()
}

fn build_turn_sequence(turn: usize) -> Vec<u32> {
    let history_len = turn * TURN_REPLY.len();
    let tail = SYSTEM_PROMPT.len() + TOOL_SCHEMA.len() + TURN_REPLY.len();
    let mut sequence = vec![0u32; history_len + tail];
    for (i, t) in sequence.iter_mut().enumerate() {
        *t = if i < history_len {
            TURN_REPLY[i % TURN_REPLY.len()]
        } else if i < history_len + SYSTEM_PROMPT.len() {
            SYSTEM_PROMPT[i - history_len]
        } else if i < history_len + SYSTEM_PROMPT.len() + TOOL_SCHEMA.len() {
            TOOL_SCHEMA[i - history_len - SYSTEM_PROMPT.len()]
        } else {
            TURN_REPLY[(i - history_len - SYSTEM_PROMPT.len() - TOOL_SCHEMA.len())
                % TURN_REPLY.len()]
        };
    }
    sequence
}

fn segment_meta(tokens: &[u32], salt: u64, offset: u64) -> SegmentMeta {
    let layer_group = 0u16;
    let mut buf = Vec::with_capacity(2 + std::mem::size_of_val(tokens));
    buf.extend_from_slice(&layer_group.to_le_bytes());
    for &t in tokens {
        buf.extend_from_slice(&t.to_le_bytes());
    }
    SegmentMeta {
        segment_hash: salted_segment_hash(tokens, layer_group, salt),
        cache_salt: salt,
        layer_group,
        num_tokens: tokens.len() as u64,
        num_computed_tokens: tokens.len() as u64,
        num_bytes: buf.len() as u64,
        sha256: hanzo_kv_fabric::compute_block_digest(&buf),
        token_offset: offset,
    }
}

fn main() {
    let salt = 42u64;
    let dir = SegmentDirectory::new();
    let published_turn0: Vec<u64>;

    let mut prefix_hits_total = 0usize;
    let mut prefix_blocks_total = 0usize;
    let mut segment_hits_total = 0usize;
    let mut segment_lookup_opportunities = 0usize;
    let lookup_start = Instant::now();

    let turn0 = build_turn_sequence(0);
    published_turn0 = chain_block_hashes(&turn0, BLOCK_SIZE);

    // Publish the invariant segments once (turn 0 layout).
    dir.publish_segment(
        "worker",
        1,
        "127.0.0.1:0".parse().unwrap(),
        &segment_meta(SYSTEM_PROMPT, salt, 0),
    )
    .unwrap();
    dir.publish_segment(
        "worker",
        1,
        "127.0.0.1:0".parse().unwrap(),
        &segment_meta(TOOL_SCHEMA, salt, SYSTEM_PROMPT.len() as u64),
    )
    .unwrap();
    let sys_hash = salted_segment_hash(SYSTEM_PROMPT, 0, salt);
    let tool_hash = salted_segment_hash(TOOL_SCHEMA, 0, salt);

    for turn in 1..TURNS {
        let sequence = build_turn_sequence(turn);
        let history_len = turn * TURN_REPLY.len();
        let system_offset = history_len as u64;
        let tool_offset = system_offset + SYSTEM_PROMPT.len() as u64;

        // Strict prefix caching: chain replay, expect misses.
        prefix_blocks_total += sequence.len().div_ceil(BLOCK_SIZE);
        prefix_hits_total += chained_prefix_hits(&sequence, &published_turn0, BLOCK_SIZE);

        // Segment directory: shifted system + tool segments should hit.
        for (hash, offset) in [(sys_hash, system_offset), (tool_hash, tool_offset)] {
            segment_lookup_opportunities += 1;
            let hits = dir.lookup_segments(salt, hash);
            if let Some(loc) = hits.first() {
                if loc.position_shift(offset).is_some() {
                    segment_hits_total += 1;
                }
            }
        }
    }

    let elapsed = lookup_start.elapsed();
    println!("CacheSlide fabric-level demo: {TURNS} turns, 64+64 token invariant segments, {BLOCK_SIZE}-token blocks");
    println!("=====================================================================");
    println!(
        "strict prefix (chained) block hits on shifted segments : {prefix_hits_total}/{prefix_blocks_total}"
    );
    println!(
        "segment-directory hits on shifted segments             : {segment_hits_total}/{segment_lookup_opportunities}"
    );
    println!("directory lookups elapsed                              : {elapsed:?}");
    println!();
    println!("Chained prefix hashing misses every block of a shifted segment;");
    println!("the salted, unchained segment hash hits and exposes the position");
    println!("shift for the attention path to translate (fail-closed).");
}
