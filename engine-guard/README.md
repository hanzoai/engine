# hanzo-engine-guard

Keeps the local Hanzo engine (`hanzo serve`) alive **safely** — it self-heals a
crashed engine but can never DoS the machine. One portable daemon; native
service install on Linux, macOS, and Windows (all user-level, no admin).

```
hanzo-engine-guard install     # register as a native service for this OS
hanzo-engine-guard status      # show engine ports, health, desktop state
hanzo-engine-guard uninstall   # remove the service
hanzo-engine-guard run         # run the daemon in the foreground (what the service runs)
```

| OS | Mechanism (user-level) |
|----|------------------------|
| Linux   | systemd `--user` service (`~/.config/systemd/user/`) + `enable-linger` |
| macOS   | launchd LaunchAgent (`~/Library/LaunchAgents/ai.hanzo.engine-guard.plist`) |
| Windows | Task Scheduler `HanzoEngineGuard` (`/SC ONLOGON`, runs `pythonw`, no console) |

Only dependency: Python 3.8+ and (recommended, cross-platform) `psutil`. Without
`psutil` it falls back to `pgrep`/`pkill` on Unix.

## What it does

1. **Circuit breaker** (always): kills any process that *is* the legacy
   `mistralrs-server` binary (matched precisely by exe/argv0/comm — never a
   substring of some innocent command), and if `hanzo serve` processes ever
   exceed `RUNAWAY_CEILING` (default 256) it kills them all. This is the
   backstop against the historical recursion fork-bomb.

2. **Keepalive** (only while a desktop/node is running): if an engine port that
   was seen healthy goes down with no live process, it respawns it by replaying
   the **exact command + working dir + environment** captured while it was
   healthy (the command alone isn't enough — the engine needs its CUDA/HF env).

## Why it cannot DoS the machine

- **Backoff** between respawns: 15s → 30s → 60s → 120s → 300s. Never a tight loop.
- **Give-up cap**: after `MAX_FAILS` (8) consecutive failed respawns of a port it
  stops and logs — a broken engine is never hammered forever.
- **Gated** on a desktop/node actually running — no zombie engines after you quit.
- **Snapshot-replay only**: it only respawns a port it has *seen healthy*; it
  never guesses a command/model.
- **Runaway ceiling**: the hard backstop kills any exponential spawn.
- The systemd unit uses `KillMode=process` (a guard restart never kills the
  engines it spawned) and sets **no** `MemoryMax` (a cap would count a
  respawned engine's multi-GB RAM and cgroup-OOM it).

## Tuning (env)

- `HANZO_ENGINE_RUNAWAY_CEILING` — kill-all threshold (default 256).

## Known limitation

The keepalive only recovers ports it has **seen healthy** (it holds the
command/cwd/env snapshot in memory, never on disk — no secrets persisted). If the
guard restarts while a port is already down, it can't recover that port until the
desktop (or a sibling engine) brings it up once. This is the safe trade-off.
