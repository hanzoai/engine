//! `hanzo cluster` - link machines into a distributed tensor-parallel inference ring.
//!
//! Two commands wrap the static RingConfig JSON mechanism that hanzo-quant's ring backend loads
//! from the `RING_CONFIG` env var:
//!   `hanzo cluster init` - the head (rank 0): print a join token + QR, then launch the ring.
//!   `hanzo cluster join` - a worker (rank 1..N-1): decode the token, then launch as a daemon.
//!
//! Topology assumption: the ring is static. Every node binds `0.0.0.0:port` (its left neighbor
//! connects in) and dials a fixed `right_ip:right_port` (rank (r+1)%N), so the full member list
//! must be known up front. `init` enumerates it (`--bind` + `--node`) and bakes it into the token;
//! workers then need only the token. The one exception that keeps 2-node bring-up trivial: the
//! last hop always dials the head, so a token without a peer list still fully describes the
//! terminal worker (see `ring_config_for`).

use std::net::UdpSocket;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::args::ClusterAction;

/// Default start of the data-ring port block; rank r binds `base_port + r`.
pub const DEFAULT_BASE_PORT: u16 = 29500;

const TOKEN_VERSION: u8 = 1;

/// Portable ring topology, bs58(JSON)-encoded so it fits in one QR / one paste.
#[derive(Debug, Serialize, Deserialize)]
struct JoinToken {
    v: u8,
    /// rank-0 (head) reachable IP.
    master: String,
    base_port: u16,
    world_size: usize,
    model: String,
    nonce: u64,
    /// full ordered ring `[rank0..rankN-1]`, present when `init` was given every member.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    addrs: Option<Vec<String>>,
}

/// On-disk shape hanzo-quant's `RingConfig` deserializes. The field names are the wire contract;
/// `tests::ring_config_contract` locks them against the real type so drift fails the build.
#[derive(Debug, Serialize)]
struct RingConfigOut {
    master_ip: Option<String>,
    master_port: u16,
    port: u16,
    right_port: u16,
    right_ip: Option<String>,
    rank: usize,
    world_size: usize,
}

pub fn run(action: ClusterAction) -> Result<()> {
    match action {
        ClusterAction::Init {
            model,
            world_size,
            bind,
            node,
            base_port,
            print,
            engine_args,
        } => init(model, world_size, bind, node, base_port, print, engine_args),
        ClusterAction::Join {
            token,
            rank,
            print,
            engine_args,
        } => join(token, rank, print, engine_args),
    }
}

#[allow(clippy::too_many_arguments)]
fn init(
    model: String,
    world_size: usize,
    bind: Option<String>,
    node: Vec<String>,
    base_port: u16,
    print: bool,
    engine_args: Vec<String>,
) -> Result<()> {
    validate_layout(base_port, world_size)?;

    let bind = match bind {
        Some(b) => b,
        None => local_ip_towards(PROBE_HOST, PROBE_PORT).context(
            "could not detect a local IP; pass `--bind <ip>` (the head's reachable address)",
        )?,
    };

    // members[0] = head (this box); members[1..] = workers in rank order.
    let mut members = vec![bind.clone()];
    members.extend(node.iter().cloned());
    let addrs = if members.len() == world_size {
        Some(members)
    } else if node.is_empty() {
        None // head can't dial its worker yet, but the token still lets a worker join.
    } else {
        bail!(
            "--world-size {world_size} expects {} `--node <ip>` peer(s) (or none); got {}",
            world_size - 1,
            node.len()
        );
    };

    let token = JoinToken {
        v: TOKEN_VERSION,
        master: bind,
        base_port,
        world_size,
        model,
        nonce: nonce(),
        addrs,
    };
    let token_str = encode_token(&token)?;
    print_token_banner(&token_str, &token);

    if print {
        return Ok(());
    }

    // Launch the head (rank 0). Its right neighbor is another node, so this needs the full
    // topology; without it, print guidance and stop (the token above is still valid for workers).
    match ring_config_for(&token, 0) {
        Ok(cfg) => {
            print_ring_config(&cfg);
            launch(&token.model, cfg, token.nonce, &engine_args)
        }
        Err(_) => {
            println!(
                "Head node not launched: it needs every peer's address to dial the ring.\n  \
                 Re-run with `--node <ip>` once per worker, e.g.\n      \
                 hanzo cluster init -m {} -w {} --bind {} --node <worker-ip>\n  \
                 Workers can already join with the token above.",
                token.model, token.world_size, token.master
            );
            Ok(())
        }
    }
}

fn join(token: String, rank: Option<usize>, print: bool, engine_args: Vec<String>) -> Result<()> {
    let token = decode_token(&token)?;

    let rank = match rank {
        Some(r) => r,
        None => auto_rank(&token)?,
    };
    if rank == 0 {
        bail!("rank 0 is the head node; run `hanzo cluster init` there instead of `join`");
    }
    if rank >= token.world_size {
        bail!(
            "rank {rank} is out of range for world_size {} (workers are ranks 1..{})",
            token.world_size,
            token.world_size - 1
        );
    }

    let cfg = ring_config_for(&token, rank)?;
    println!(
        "Joining ring {:016x} as rank {}/{} (model {})",
        token.nonce, rank, token.world_size, token.model
    );
    print_ring_config(&cfg);

    if print {
        return Ok(());
    }
    launch(&token.model, cfg, token.nonce, &engine_args)
}

/// Compute rank `rank`'s config from the token, resolving its right neighbor's address.
fn ring_config_for(token: &JoinToken, rank: usize) -> Result<RingConfigOut> {
    let n = token.world_size;
    let right = (rank + 1) % n;
    let right_ip = match &token.addrs {
        Some(addrs) => addrs
            .get(right)
            .cloned()
            .with_context(|| format!("token address list has no entry for rank {right}"))?,
        // Last hop always dials the head, so it is resolvable even without a peer list.
        None if right == 0 => token.master.clone(),
        None => bail!(
            "the token carries no address for rank {right} (this node's right neighbor).\n  \
             Re-run `hanzo cluster init` with a `--node <ip>` for every peer so the token carries\n  \
             the full ring, or run this on the terminal worker (rank {}).",
            n - 1
        ),
    };
    Ok(RingConfigOut {
        master_ip: Some(token.master.clone()),
        master_port: master_port(token.base_port, n),
        port: token.base_port + rank as u16,
        right_port: token.base_port + right as u16,
        right_ip: Some(right_ip),
        rank,
        world_size: n,
    })
}

/// Control/replicator port on the head: one past the data-ring block `[base, base+N-1]`.
fn master_port(base_port: u16, world_size: usize) -> u16 {
    base_port + world_size as u16
}

fn validate_layout(base_port: u16, world_size: usize) -> Result<()> {
    if world_size < 2 {
        bail!("world_size must be >= 2 for a ring, got {world_size}");
    }
    if base_port as usize + world_size > u16::MAX as usize {
        bail!(
            "base_port {base_port} + world_size {world_size} overflows the max TCP port {}",
            u16::MAX
        );
    }
    Ok(())
}

fn encode_token(t: &JoinToken) -> Result<String> {
    let json = serde_json::to_vec(t).context("serializing join token")?;
    Ok(bs58::encode(json).into_string())
}

fn decode_token(s: &str) -> Result<JoinToken> {
    let bytes = bs58::decode(s.trim())
        .into_vec()
        .context("join token is not valid bs58; copy it exactly as printed by `cluster init`")?;
    let token: JoinToken = serde_json::from_slice(&bytes)
        .context("join token payload is not a valid cluster token")?;
    if token.v != TOKEN_VERSION {
        bail!(
            "unsupported join-token version {} (this build speaks v{TOKEN_VERSION})",
            token.v
        );
    }
    validate_layout(token.base_port, token.world_size)?;
    if let Some(addrs) = &token.addrs {
        if addrs.len() != token.world_size {
            bail!(
                "token address list has {} entries but world_size is {}",
                addrs.len(),
                token.world_size
            );
        }
    }
    Ok(token)
}

/// Random-ish cluster instance id. Not a secret and not a security control (the ring itself is
/// unauthenticated) - it only disambiguates one launch from another in logs and temp paths.
fn nonce() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ ((std::process::id() as u64) << 32)
}

const PROBE_HOST: &str = "1.1.1.1";
const PROBE_PORT: u16 = 53;

/// Source IP the kernel would use to reach `peer` (a UDP connect sends no packets). `None` if
/// there is no route (offline box), in which case the caller falls back to explicit `--bind`/`--rank`.
fn local_ip_towards(peer: &str, port: u16) -> Option<String> {
    let sock = UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect((peer, port)).ok()?;
    sock.local_addr().ok().map(|a| a.ip().to_string())
}

/// Pick this node's rank by matching its outbound IP against the token's peer list.
fn auto_rank(token: &JoinToken) -> Result<usize> {
    let addrs = token.addrs.as_ref().context(
        "cannot auto-pick a rank: this token has no peer list. Pass `--rank <r>` (1..N-1), or \
         re-run `cluster init` with `--node <ip>` peers so ranks can self-assign.",
    )?;
    let my_ip = local_ip_towards(
        &token.master,
        master_port(token.base_port, token.world_size),
    )
    .context("could not detect this node's IP; pass `--rank <r>`")?;
    let matches: Vec<usize> = addrs
        .iter()
        .enumerate()
        .skip(1) // rank 0 is the head, never a join target
        .filter(|(_, ip)| **ip == my_ip)
        .map(|(i, _)| i)
        .collect();
    match matches.as_slice() {
        [r] => Ok(*r),
        [] => bail!("this node's IP {my_ip} is not in the ring's peer list; pass `--rank <r>`"),
        _ => bail!("this node's IP {my_ip} matches multiple ranks; pass `--rank <r>`"),
    }
}

fn launch(model: &str, cfg: RingConfigOut, nonce: u64, engine_args: &[String]) -> Result<()> {
    let path = write_ring_config(&cfg, nonce)?;
    if !cfg!(feature = "ring") {
        println!(
            "\nNOTE: this `hanzo` binary was built without the `ring` feature, so it will not launch\n\
             distributed inference. The ring config was written to:\n    {p}\n\
             Rebuild with `--features ring` to launch automatically, or start the engine yourself:\n    \
             RING_CONFIG={p} hanzo serve -m {model}\n",
            p = path.display()
        );
        return Ok(());
    }
    exec_engine(model, &path, engine_args)
}

fn write_ring_config(cfg: &RingConfigOut, nonce: u64) -> Result<PathBuf> {
    let path = std::env::temp_dir().join(format!("hanzo-ring-{nonce:016x}-rank{}.json", cfg.rank));
    let json = serde_json::to_string_pretty(cfg).context("serializing ring config")?;
    std::fs::write(&path, json)
        .with_context(|| format!("writing ring config to {}", path.display()))?;
    Ok(path)
}

/// Replace this process with `hanzo serve -m <model> [engine_args...]`, RING_CONFIG set. The engine
/// makes rank 0 serve HTTP and every rank != 0 become a ring daemon automatically.
fn exec_engine(model: &str, ring_config: &Path, engine_args: &[String]) -> Result<()> {
    let exe = std::env::current_exe().context("resolving the current hanzo executable")?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("serve").arg("-m").arg(model).args(engine_args);
    cmd.env("RING_CONFIG", ring_config);
    println!(
        "Launching: hanzo serve -m {model} {}  (RING_CONFIG={})",
        engine_args.join(" "),
        ring_config.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        Err(anyhow::Error::new(cmd.exec()).context("failed to exec `hanzo serve`"))
    }
    #[cfg(not(unix))]
    {
        let status = cmd.status().context("spawning `hanzo serve`")?;
        std::process::exit(status.code().unwrap_or(1));
    }
}

fn render_qr(data: &str) -> Option<String> {
    use qrcode::render::unicode::Dense1x2;
    use qrcode::{EcLevel, QrCode};
    let code = QrCode::with_error_correction_level(data.as_bytes(), EcLevel::L).ok()?;
    Some(code.render::<Dense1x2>().quiet_zone(true).build())
}

fn print_token_banner(token_str: &str, token: &JoinToken) {
    let last = token.base_port + token.world_size as u16 - 1;
    println!();
    println!(
        "  hanzo cluster {:016x} | {} nodes | model {}",
        token.nonce, token.world_size, token.model
    );
    println!(
        "  head rank 0 at {} | data ports {}..{} | control port {}",
        token.master,
        token.base_port,
        last,
        master_port(token.base_port, token.world_size)
    );
    println!();
    match render_qr(token_str) {
        Some(qr) => print!("{qr}"),
        None => println!("  (token too large to render as a QR; copy the text below)"),
    }
    println!();
    println!("  JOIN TOKEN");
    println!("  {token_str}");
    println!();
    if token.addrs.is_some() {
        println!("  On each worker box, run:");
        println!("      hanzo cluster join {token_str}");
    } else {
        println!("  On each worker box, run (rank required - token has no peer list):");
        println!(
            "      hanzo cluster join {token_str} --rank <1..{}>",
            token.world_size - 1
        );
    }
    println!();
}

fn print_ring_config(cfg: &RingConfigOut) {
    println!(
        "  RingConfig rank {}: master {}:{} | bind 0.0.0.0:{} | right {}:{} | world_size {}",
        cfg.rank,
        cfg.master_ip.as_deref().unwrap_or("0.0.0.0"),
        cfg.master_port,
        cfg.port,
        cfg.right_ip.as_deref().unwrap_or("0.0.0.0"),
        cfg.right_port,
        cfg.world_size
    );
    if let Ok(json) = serde_json::to_string_pretty(cfg) {
        for line in json.lines() {
            println!("    {line}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(world_size: usize, addrs: Option<Vec<String>>) -> JoinToken {
        JoinToken {
            v: TOKEN_VERSION,
            master: "10.0.0.1".into(),
            base_port: DEFAULT_BASE_PORT,
            world_size,
            model: "Qwen/Qwen3-4B".into(),
            nonce: 7,
            addrs,
        }
    }

    // A generated config must deserialize into the real hanzo-quant RingConfig with matching
    // values. If a RingConfig field is ever renamed, this stops compiling/passing.
    #[test]
    fn ring_config_contract() {
        let t = token(
            4,
            Some(vec![
                "10.0.0.1".into(),
                "10.0.0.2".into(),
                "10.0.0.3".into(),
                "10.0.0.4".into(),
            ]),
        );
        let cfg = ring_config_for(&t, 2).unwrap();
        let json = serde_json::to_string(&cfg).unwrap();
        let rc: hanzo_quant::distributed::RingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(rc.rank, 2);
        assert_eq!(rc.world_size, 4);
        assert_eq!(rc.port, DEFAULT_BASE_PORT + 2);
        assert_eq!(rc.right_port, DEFAULT_BASE_PORT + 3);
        assert_eq!(rc.master_port, DEFAULT_BASE_PORT + 4);
        assert_eq!(rc.master_ip(), "10.0.0.1");
        assert_eq!(rc.right_ip(), "10.0.0.4"); // right of rank 2 in N=4 is rank 3 = addrs[3]
    }

    #[test]
    fn token_roundtrips() {
        let t = token(2, None);
        let s = encode_token(&t).unwrap();
        let back = decode_token(&s).unwrap();
        assert_eq!(back.master, t.master);
        assert_eq!(back.world_size, 2);
        assert_eq!(back.model, t.model);
        assert!(back.addrs.is_none());
    }

    // 2-node token with no peer list: rank 1's right is rank 0 (the head), so it is fully
    // resolvable; rank 0's right is the unknown worker, so it is not.
    #[test]
    fn last_hop_resolves_without_peer_list() {
        let t = token(2, None);
        let cfg = ring_config_for(&t, 1).unwrap();
        assert_eq!(cfg.port, DEFAULT_BASE_PORT + 1);
        assert_eq!(cfg.right_port, DEFAULT_BASE_PORT);
        assert_eq!(cfg.right_ip.as_deref(), Some("10.0.0.1"));
        assert!(ring_config_for(&t, 0).is_err());
    }

    #[test]
    fn validate_layout_accepts_any_world_size_ge_2() {
        assert!(validate_layout(DEFAULT_BASE_PORT, 1).is_err()); // ring needs >= 2
        assert!(validate_layout(u16::MAX - 1, 4).is_err()); // port block overflow
        assert!(validate_layout(DEFAULT_BASE_PORT, 2).is_ok());
        assert!(validate_layout(DEFAULT_BASE_PORT, 3).is_ok()); // non-power-of-2 now allowed
        assert!(validate_layout(DEFAULT_BASE_PORT, 4).is_ok());
    }
}
