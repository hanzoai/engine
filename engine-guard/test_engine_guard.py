"""Unit tests for hanzo-engine-guard.

The daemon is a single hyphenated, extension-less script, so it cannot be
imported by name - it is loaded by path via importlib. Every test drives the
pure policy fns directly, or one mocked `_tick` cycle with all I/O monkeypatched;
no real processes, sockets, or sleeps are ever touched.
"""
import importlib.util
import os
import sys
from importlib.machinery import SourceFileLoader

import pytest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hanzo-engine-guard")
_spec = importlib.util.spec_from_file_location(
    "hanzo_engine_guard", SCRIPT,
    loader=SourceFileLoader("hanzo_engine_guard", SCRIPT))
eg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = eg            # register so coverage can resolve it by name
_spec.loader.exec_module(eg)


# ----------------------------------------------------------------- fakes
class FakeProc:
    """Stands in for a psutil process (process_iter entry / Process(pid))."""

    def __init__(self, pid, name=None, exe=None, cmdline=None,
                 cwd=None, environ=None, raises=False, kill_raises=False):
        self.info = {"pid": pid, "name": name, "exe": exe, "cmdline": cmdline}
        self._cwd = cwd
        self._environ = environ or {}
        self._raises = raises
        self._kill_raises = kill_raises
        self.killed = False

    def kill(self):
        self.killed = True
        if self._kill_raises:
            raise RuntimeError("process vanished before kill")

    def cwd(self):
        if self._raises:
            raise RuntimeError("process is gone")
        return self._cwd

    def environ(self):
        if self._raises:
            raise RuntimeError("process is gone")
        return dict(self._environ)


class FakePsutil:
    def __init__(self, procs):
        self._by_pid = {p.info["pid"]: p for p in procs}

    def process_iter(self, attrs=None):
        return list(self._by_pid.values())

    def Process(self, pid):
        return self._by_pid[pid]


class FakeCompleted:
    def __init__(self, stdout=""):
        self.stdout = stdout


# ----------------------------------------------------------------- parse_port
@pytest.mark.parametrize("cmdline,expected", [
    ("/x/release/hanzo serve -p 36900", 36900),
    ("hanzo serve --model m -p 36901 --foo", 36901),
    ("hanzo  serve  -p   42", 42),                 # tolerates extra whitespace
    ("hanzo serve", None),                          # no -p at all
    ("hanzo serve -p", None),                       # -p with no value
    ("hanzo serve -p 100 -p 200", 100),             # first -p wins
    ("hanzo serve --port 36910", None),             # only -p is parsed, not --port
])
def test_parse_port(cmdline, expected):
    assert eg.parse_port(cmdline) == expected


# ----------------------------------------------------------------- is_bomb
@pytest.mark.parametrize("name,exe,argv0,expected", [
    (None, "/opt/hanzo/bin/mistralrs-server", None, True),       # exe basename
    (None, None, "/usr/local/bin/mistralrs-server", True),       # argv0 basename
    ("mistralrs-server", None, None, True),                      # full comm
    ("mistralrs-serve", None, None, True),                       # 15-char truncated comm
    # critical false-positive guard: cmd/binary merely *mentions* the name
    ("bash", "/bin/bash", "/bin/bash", False),
    ("python3", "/usr/bin/python3", "python3", False),
    ("grep", "/usr/bin/grep", "grep", False),
    # the native engine itself is never a bomb
    ("hanzo", "/x/target/release/hanzo", "/x/target/release/hanzo", False),
    (None, None, None, False),                                   # nothing known
    ("mistralrs-serv", None, None, False),                       # 14 chars: not a match
])
def test_is_bomb(name, exe, argv0, expected):
    assert eg.is_bomb(name, exe, argv0) is expected


# ----------------------------------------------------------------- backoff_seconds
@pytest.mark.parametrize("fails,expected", [
    (0, 15), (1, 30), (2, 60), (3, 120), (4, 300),
    (5, 300), (8, 300), (100, 300),                # clamps at the last entry
])
def test_backoff_seconds(fails, expected):
    assert eg.backoff_seconds(fails) == expected


# ----------------------------------------------------------------- over_ceiling
@pytest.mark.parametrize("count,ceiling,expected", [
    (255, 256, False), (256, 256, False), (257, 256, True),   # boundary == not over
    (0, 0, False), (1, 0, True),
])
def test_over_ceiling(count, ceiling, expected):
    assert eg.over_ceiling(count, ceiling) is expected


# ----------------------------------------------------------------- keepalive_decision
def test_keepalive_healthy_regardless_of_owner():
    assert eg.keepalive_decision({}, 1000, True, True, True) == "healthy"
    assert eg.keepalive_decision({"fails": 3}, 1000, False, True, True) == "healthy"


def test_keepalive_no_owner():
    assert eg.keepalive_decision({"cmd": ["x"]}, 1e9, False, False, False) == "no-owner"


def test_keepalive_health_without_proc_is_not_healthy():
    # answers /health but no matching engine proc -> NOT 'healthy'; with an owner
    # and elapsed backoff it is respawned (this matches the daemon's behavior).
    st = {"cmd": ["x"], "fails": 0, "last": 0}
    assert eg.keepalive_decision(st, 1e9, True, True, False) == "respawn"


def test_keepalive_in_backoff_then_respawn_at_boundary():
    st = {"cmd": ["x"], "fails": 1, "last": 1000}     # backoff(1) == 30s
    assert eg.keepalive_decision(st, 1029, True, False, False) == "in-backoff"
    assert eg.keepalive_decision(st, 1030, True, False, False) == "respawn"


def test_keepalive_respawn():
    st = {"cmd": ["x"], "fails": 0, "last": 0}
    assert eg.keepalive_decision(st, 1e9, True, False, False) == "respawn"


def test_keepalive_gaveup_by_flag():
    st = {"cmd": ["x"], "gaveup": True, "fails": 2}
    assert eg.keepalive_decision(st, 1e9, True, False, False) == "gaveup"


def test_keepalive_gaveup_by_count():
    st = {"cmd": ["x"], "fails": eg.MAX_FAILS, "last": 0}
    assert eg.keepalive_decision(st, 1e9, True, False, False) == "gaveup"


def test_keepalive_no_owner_precedes_gaveup():
    # a quit desktop yields 'no-owner' even past the fail cap (never hammered)
    st = {"cmd": ["x"], "fails": eg.MAX_FAILS}
    assert eg.keepalive_decision(st, 1e9, False, False, False) == "no-owner"


# ----------------------------------------------------------------- _tick fixture
class TickEnv:
    def __init__(self):
        self.logs = []
        self.spawns = []        # (argv, cwd, env)
        self.kill_all = []      # patterns passed to _kill_matching
        self.procs = []         # what _procs_matching(ENGINE_PAT) returns
        self.desktop = False
        self.healthy_ports = set()


@pytest.fixture
def env(monkeypatch):
    e = TickEnv()
    monkeypatch.setattr(eg, "_log", lambda m: e.logs.append(m))
    monkeypatch.setattr(eg, "_kill_bomb", lambda: None)
    monkeypatch.setattr(eg, "_kill_matching", lambda pat: e.kill_all.append(pat))
    monkeypatch.setattr(eg, "_procs_matching", lambda pat: list(e.procs))
    monkeypatch.setattr(eg, "_any_running", lambda pats: e.desktop)
    monkeypatch.setattr(eg, "_health", lambda port: port in e.healthy_ports)
    monkeypatch.setattr(eg, "_proc_context", lambda pid: ("/captured/cwd", {"CAP": "1"}))
    monkeypatch.setattr(eg, "_spawn_detached",
                        lambda argv, cwd=None, env=None: e.spawns.append((argv, cwd, env)))
    return e


# ----------------------------------------------------------------- circuit breaker
def test_kill_bomb_kills_only_the_real_binary(monkeypatch):
    """The critical guard: kill the real mistralrs-server, never an innocent
    process whose command line merely CONTAINS that string."""
    bomb = FakeProc(101, name="mistralrs-serve", exe="/opt/mistralrs-server",
                    cmdline=["/opt/mistralrs-server", "-p", "1234"])
    innocent = FakeProc(202, name="bash", exe="/bin/bash",
                        cmdline=["/bin/bash", "-c", "tail -f mistralrs-server.log"])
    grep = FakeProc(303, name="grep", exe="/usr/bin/grep",
                    cmdline=["grep", "mistralrs-server", "/var/log/syslog"])
    monkeypatch.setattr(eg, "psutil", FakePsutil([bomb, innocent, grep]))
    eg._kill_bomb()
    assert bomb.killed is True
    assert innocent.killed is False
    assert grep.killed is False


def test_bomb_procs_detects_by_exe_argv0_comm(monkeypatch):
    procs = [
        FakeProc(1, exe="/x/mistralrs-server", cmdline=["/x/mistralrs-server"]),     # exe
        FakeProc(2, name="mistralrs-serve", cmdline=["python", "run"]),              # comm
        FakeProc(3, exe="/bin/sh", cmdline=["/x/mistralrs-server", "-p", "1"]),      # argv0
        FakeProc(4, name="bash", exe="/bin/bash", cmdline=["bash", "-c", "echo mistralrs-server"]),
    ]
    monkeypatch.setattr(eg, "psutil", FakePsutil(procs))
    assert {pid for pid, _ in eg._bomb_procs()} == {1, 2, 3}


def test_tick_kills_bomb_not_innocent(monkeypatch):
    """One real _tick cycle: its first act precisely kills the bomb, spares others."""
    bomb = FakeProc(101, name="mistralrs-serve", exe="/opt/mistralrs-server",
                    cmdline=["/opt/mistralrs-server"])
    innocent = FakeProc(202, name="vim", exe="/usr/bin/vim",
                        cmdline=["vim", "mistralrs-server.rs"])
    monkeypatch.setattr(eg, "psutil", FakePsutil([bomb, innocent]))
    monkeypatch.setattr(eg, "_procs_matching", lambda pat: [])      # no engines
    monkeypatch.setattr(eg, "_any_running", lambda pats: False)
    monkeypatch.setattr(eg, "_log", lambda m: None)
    eg._tick({}, set())
    assert bomb.killed is True
    assert innocent.killed is False


def test_kill_bomb_swallows_kill_errors(monkeypatch):
    """A bomb that vanishes between detection and kill must not crash the guard."""
    bomb = FakeProc(101, exe="/opt/mistralrs-server",
                    cmdline=["/opt/mistralrs-server"], kill_raises=True)
    monkeypatch.setattr(eg, "psutil", FakePsutil([bomb]))
    eg._kill_bomb()                      # must not raise
    assert bomb.killed is True           # attempted


def test_kill_matching_swallows_kill_errors(monkeypatch):
    target = FakeProc(7, cmdline=["release/hanzo", "serve"], kill_raises=True)
    monkeypatch.setattr(eg, "psutil", FakePsutil([target]))
    eg._kill_matching("release/hanzo serve")   # must not raise
    assert target.killed is True


def test_tick_runaway_kills_all(env, monkeypatch):
    monkeypatch.setattr(eg, "RUNAWAY_CEILING", 2)
    env.procs = [(10, "release/hanzo serve -p 36900"),
                 (11, "release/hanzo serve -p 36901"),
                 (12, "release/hanzo serve -p 36902")]      # 3 > ceiling 2
    env.desktop = True
    env.healthy_ports = {36900, 36901, 36902}
    state = {}
    eg._tick(state, set())
    assert env.kill_all == [eg.ENGINE_PAT]                  # kill-all path taken
    assert env.spawns == []                                 # keepalive never ran
    assert state == {}                                      # bailed before the loops


# ----------------------------------------------------------------- keepalive cycle
def test_tick_healthy_resets_state_and_captures_context(env):
    env.procs = [(500, "/x/release/hanzo serve -p 36900")]
    env.desktop = True
    env.healthy_ports = {36900}
    state = {36900: {"cmd": ["old"], "fails": 5, "gaveup": True}}
    gaveup_logged = {36900}
    eg._tick(state, gaveup_logged)
    st = state[36900]
    assert st["fails"] == 0 and st["gaveup"] is False
    assert st["cmd"] == ["/x/release/hanzo", "serve", "-p", "36900"]
    assert st["cwd"] == "/captured/cwd" and st["env"] == {"CAP": "1"}
    assert 36900 not in gaveup_logged                      # cleared on recovery
    assert env.spawns == []                                 # healthy -> no respawn


def test_tick_respawns_down_port_with_owner(env):
    snap = {"cmd": ["/x/release/hanzo", "serve", "-p", "36900"],
            "cwd": "/work/engine", "env": {"HF_HOME": "/cache"},
            "fails": 0, "last": 0.0}
    state = {36900: dict(snap)}
    env.procs = []                 # the engine process is gone
    env.desktop = True             # but a desktop/node owns it
    env.healthy_ports = set()
    eg._tick(state, set())
    assert env.spawns == [(snap["cmd"], snap["cwd"], snap["env"])]   # exact replay
    assert state[36900]["fails"] == 1                                # attempt counted
    assert state[36900]["last"] > 0                                  # backoff clock started


def test_tick_skips_state_entry_without_cmd(env):
    state = {36900: {"fails": 0}}        # malformed: never captured a command
    env.procs = []
    env.desktop = True
    eg._tick(state, set())               # the guard skips it -> no spawn, no crash
    assert env.spawns == []
    assert state[36900] == {"fails": 0}


def test_tick_no_respawn_without_owner(env):
    state = {36900: {"cmd": ["e"], "fails": 0, "last": 0}}
    env.procs = []
    env.desktop = False            # no desktop/node -> never respawn
    eg._tick(state, set())
    assert env.spawns == []
    assert state[36900]["fails"] == 0


def test_tick_respects_backoff(env):
    import time
    state = {36900: {"cmd": ["e"], "fails": 1, "last": time.time()}}   # just tried
    env.procs = []
    env.desktop = True
    eg._tick(state, set())
    assert env.spawns == []                                # still within 30s backoff
    assert state[36900]["fails"] == 1                      # unchanged


def test_tick_gaveup_sets_flag_then_logs_once(env):
    state = {36900: {"cmd": ["e"], "fails": eg.MAX_FAILS, "last": 0}}
    env.procs = []
    env.desktop = True
    gaveup_logged = set()
    eg._tick(state, gaveup_logged)                         # 1st: mark gaveup, NO log yet
    assert state[36900]["gaveup"] is True
    assert env.spawns == [] and gaveup_logged == set()
    eg._tick(state, gaveup_logged)                         # 2nd: already gaveup -> log once
    assert 36900 in gaveup_logged
    assert sum("gave up" in m for m in env.logs) == 1
    eg._tick(state, gaveup_logged)                         # 3rd: no duplicate log
    assert sum("gave up" in m for m in env.logs) == 1


def test_tick_full_respawn_to_giveup_sequence(env, monkeypatch):
    """End-to-end safety: a permanently-broken port is respawned exactly
    MAX_FAILS times (never more), each gated by backoff, then given up."""
    clock = [1_000_000.0]
    monkeypatch.setattr(eg.time, "time", lambda: clock[0])
    state = {36900: {"cmd": ["e"], "fails": 0, "last": 0.0}}
    env.procs = []
    env.desktop = True
    env.healthy_ports = set()
    for _ in range(eg.MAX_FAILS + 3):
        clock[0] += 10_000          # always skip past the 300s ceiling backoff
        eg._tick(state, set())
    assert len(env.spawns) == eg.MAX_FAILS         # respawned 8x, never a 9th
    assert state[36900]["gaveup"] is True


# ----------------------------------------------------------------- I/O layer
def test_procs_matching_psutil(monkeypatch):
    procs = [FakeProc(1, cmdline=["release/hanzo", "serve", "-p", "36900"]),
             FakeProc(2, cmdline=["vim", "notes.txt"]),
             FakeProc(3, cmdline=None)]
    monkeypatch.setattr(eg, "psutil", FakePsutil(procs))
    assert eg._procs_matching("release/hanzo serve") == [(1, "release/hanzo serve -p 36900")]


def test_procs_matching_pgrep_fallback(monkeypatch):
    monkeypatch.setattr(eg, "psutil", None)
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return FakeCompleted("123 release/hanzo serve -p 36900\n"
                             "456 pgrep -af release/hanzo serve\n")   # pgrep's own line filtered
    monkeypatch.setattr(eg.subprocess, "run", fake_run)
    assert eg._procs_matching("release/hanzo serve") == [(123, "release/hanzo serve -p 36900")]
    assert seen["argv"][0] == "pgrep"


def test_kill_matching_psutil(monkeypatch):
    target = FakeProc(7, cmdline=["release/hanzo", "serve"])
    monkeypatch.setattr(eg, "psutil", FakePsutil([target]))
    eg._kill_matching("release/hanzo serve")
    assert target.killed is True


def test_kill_matching_pgrep_fallback(monkeypatch):
    monkeypatch.setattr(eg, "psutil", None)
    seen = []
    monkeypatch.setattr(eg.subprocess, "run", lambda argv, **kw: seen.append(argv) or FakeCompleted())
    eg._kill_matching("release/hanzo serve")
    assert seen == [["pkill", "-9", "-f", "release/hanzo serve"]]


def test_bomb_procs_pgrep_fallback(monkeypatch):
    monkeypatch.setattr(eg, "psutil", None)
    monkeypatch.setattr(eg.subprocess, "run", lambda argv, **kw: FakeCompleted("111\n222\n"))
    assert eg._bomb_procs() == [(111, ""), (222, "")]


def test_kill_bomb_pgrep_fallback(monkeypatch):
    monkeypatch.setattr(eg, "psutil", None)
    seen = []
    monkeypatch.setattr(eg.subprocess, "run", lambda argv, **kw: seen.append(argv) or FakeCompleted())
    eg._kill_bomb()
    assert seen == [["pkill", "-9", "-x", eg.BOMB_PAT[:15]]]


def test_any_running(monkeypatch):
    monkeypatch.setattr(eg, "_procs_matching",
                        lambda pat: [(1, pat)] if pat == "hanzo-node" else [])
    assert eg._any_running(("zoo-desktop", "hanzo-node")) is True
    monkeypatch.setattr(eg, "_procs_matching", lambda pat: [])
    assert eg._any_running(eg.DESKTOP_PATS) is False


def test_proc_context_ok(monkeypatch):
    monkeypatch.setattr(eg, "psutil", FakePsutil([FakeProc(7, cwd="/w", environ={"A": "B"})]))
    assert eg._proc_context(7) == ("/w", {"A": "B"})


def test_proc_context_dead_proc(monkeypatch):
    monkeypatch.setattr(eg, "psutil", FakePsutil([FakeProc(7, raises=True)]))
    assert eg._proc_context(7) == (None, None)


def test_spawn_detached_unix(monkeypatch):
    cap = {}
    monkeypatch.setattr(eg, "IS_WIN", False)
    monkeypatch.setattr(eg.subprocess, "Popen",
                        lambda argv, **kw: cap.update(argv=argv, kw=kw))
    eg._spawn_detached(["e", "-p", "1"], cwd="/w", env={"X": "1"})
    assert cap["argv"] == ["e", "-p", "1"]
    assert cap["kw"]["start_new_session"] is True
    assert cap["kw"]["cwd"] == "/w" and cap["kw"]["env"] == {"X": "1"}
    assert cap["kw"]["stdout"] == eg.subprocess.DEVNULL


def test_spawn_detached_windows(monkeypatch):
    cap = {}
    monkeypatch.setattr(eg, "IS_WIN", True)
    monkeypatch.setattr(eg.subprocess, "DETACHED_PROCESS", 0x8, raising=False)
    monkeypatch.setattr(eg.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    monkeypatch.setattr(eg.subprocess, "Popen", lambda argv, **kw: cap.update(kw))
    eg._spawn_detached(["e"])
    assert cap["creationflags"] == 0x208
    assert "start_new_session" not in cap


def test_health_up(monkeypatch):
    class R:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(eg.urllib.request, "urlopen", lambda url, timeout=1: R())
    assert eg._health(36900) is True


def test_health_non_200(monkeypatch):
    class R:
        status = 503

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(eg.urllib.request, "urlopen", lambda url, timeout=1: R())
    assert eg._health(36900) is False


def test_health_down(monkeypatch):
    def boom(url, timeout=1):
        raise OSError("connection refused")
    monkeypatch.setattr(eg.urllib.request, "urlopen", boom)
    assert eg._health(36900) is False


def test_log_unix(monkeypatch):
    seen = []
    monkeypatch.setattr(eg, "IS_WIN", False)
    monkeypatch.setattr(eg.subprocess, "run", lambda argv, **kw: seen.append(argv))
    eg._log("hello")
    assert seen == [["logger", "-t", eg.APP, "hello"]]


def test_log_windows(monkeypatch, capsys):
    monkeypatch.setattr(eg, "IS_WIN", True)
    eg._log("hello")
    assert "[hanzo-engine-guard] hello" in capsys.readouterr().out


# ----------------------------------------------------------------- run loop
def test_run_ticks_then_sleeps(monkeypatch):
    ticks = []
    monkeypatch.setattr(eg, "_log", lambda m: None)
    monkeypatch.setattr(eg, "_tick", lambda s, g: ticks.append((s, g)))

    def stop(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(eg.time, "sleep", stop)
    with pytest.raises(KeyboardInterrupt):
        eg.run()
    assert len(ticks) == 1
    assert ticks[0][0] == {} and ticks[0][1] == set()      # fresh state + gaveup set


# ----------------------------------------------------------------- install dispatch
def test_self_cmd():
    cmd = eg._self_cmd()
    assert cmd[-1] == "run" and len(cmd) == 3
    assert os.path.isabs(cmd[0]) and os.path.isabs(cmd[1])


def test_dispatch_linux(monkeypatch):
    monkeypatch.setattr(eg, "IS_WIN", False)
    monkeypatch.setattr(eg.sys, "platform", "linux")
    called = []
    monkeypatch.setattr(eg, "install_linux", lambda: called.append("linux"))
    eg._dispatch("install")
    assert called == ["linux"]


def test_dispatch_macos(monkeypatch):
    monkeypatch.setattr(eg, "IS_WIN", False)
    monkeypatch.setattr(eg.sys, "platform", "darwin")
    called = []
    monkeypatch.setattr(eg, "install_macos", lambda: called.append("macos"))
    eg._dispatch("install")
    assert called == ["macos"]


def test_dispatch_windows(monkeypatch):
    monkeypatch.setattr(eg, "IS_WIN", True)
    called = []
    monkeypatch.setattr(eg, "uninstall_windows", lambda: called.append("win"))
    eg._dispatch("uninstall")
    assert called == ["win"]


def test_dispatch_unsupported(monkeypatch):
    monkeypatch.setattr(eg, "IS_WIN", False)
    monkeypatch.setattr(eg.sys, "platform", "linux")
    with pytest.raises(SystemExit):
        eg._dispatch("frobnicate")                          # no frobnicate_linux exists


# ----------------------------------------------------------------- installers
def test_install_linux_writes_unit_and_enables(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USER", "tester")
    runs = []
    monkeypatch.setattr(eg.subprocess, "run", lambda argv, **kw: runs.append(argv))
    monkeypatch.setattr(eg, "_self_cmd", lambda: ["/py", "/script", "run"])
    eg.install_linux()
    unit = tmp_path / ".config/systemd/user/hanzo-engine-guard.service"
    text = unit.read_text()
    # unit content is part of the contract - lock it down
    assert "ExecStart=/py /script run" in text
    assert "Restart=always" in text and "RestartSec=3" in text
    assert "KillMode=process" in text
    assert "OOMScoreAdjust=-700" in text
    assert "MemoryMax=" not in text          # no memory cap directive (only the comment)
    assert "WantedBy=default.target" in text
    assert ["loginctl", "enable-linger", "tester"] in runs
    assert ["systemctl", "--user", "daemon-reload"] in runs
    assert ["systemctl", "--user", "enable", "--now", "hanzo-engine-guard.service"] in runs


def test_uninstall_linux(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    unit = tmp_path / ".config/systemd/user/hanzo-engine-guard.service"
    unit.parent.mkdir(parents=True)
    unit.write_text("x")
    runs = []
    monkeypatch.setattr(eg.subprocess, "run", lambda argv, **kw: runs.append(argv))
    eg.uninstall_linux()
    assert not unit.exists()
    assert ["systemctl", "--user", "disable", "--now", "hanzo-engine-guard.service"] in runs


def test_install_macos_writes_plist(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(eg.subprocess, "run", lambda argv, **kw: None)
    monkeypatch.setattr(eg, "_self_cmd", lambda: ["/py", "/script", "run"])
    eg.install_macos()
    plist = tmp_path / "Library/LaunchAgents/ai.hanzo.engine-guard.plist"
    text = plist.read_text()
    assert "<key>Label</key><string>ai.hanzo.engine-guard</string>" in text
    assert "<string>/py</string>" in text and "<string>/script</string>" in text
    assert "<key>RunAtLoad</key><true/>" in text
    assert "<key>KeepAlive</key><true/>" in text


def test_uninstall_macos(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    plist = tmp_path / "Library/LaunchAgents/ai.hanzo.engine-guard.plist"
    plist.parent.mkdir(parents=True)
    plist.write_text("x")
    monkeypatch.setattr(eg.subprocess, "run", lambda argv, **kw: None)
    eg.uninstall_macos()
    assert not plist.exists()


def test_install_windows(monkeypatch):
    runs = []
    monkeypatch.setattr(eg.subprocess, "run", lambda argv, **kw: runs.append(argv))
    eg.install_windows()
    assert runs[0][0] == "schtasks" and runs[0][1] == "/Create"
    assert any(a[1] == "/Run" for a in runs)


def test_uninstall_windows(monkeypatch):
    runs = []
    monkeypatch.setattr(eg.subprocess, "run", lambda argv, **kw: runs.append(argv))
    eg.uninstall_windows()
    assert ["schtasks", "/End", "/TN", "HanzoEngineGuard"] in runs
    assert ["schtasks", "/Delete", "/F", "/TN", "HanzoEngineGuard"] in runs


# ----------------------------------------------------------------- status + main
def test_status_reports_ports_and_bomb_warning(monkeypatch, capsys):
    monkeypatch.setattr(eg, "_procs_matching",
                        lambda pat: [(1, "release/hanzo serve -p 36900"),
                                     (2, "release/hanzo serve -p 36901")]
                        if pat == eg.ENGINE_PAT else [])
    monkeypatch.setattr(eg, "_health", lambda port: port == 36900)
    monkeypatch.setattr(eg, "_any_running", lambda pats: True)
    monkeypatch.setattr(eg, "_bomb_procs", lambda: [(99, "mistralrs-server")])
    eg.status()
    out = capsys.readouterr().out
    assert "ports [36900, 36901]" in out
    assert ":36900 health=UP" in out
    assert ":36901 health=down" in out
    assert "desktop/node running: True" in out
    assert "WARNING 1" in out


def test_status_no_engines_no_bomb(monkeypatch, capsys):
    monkeypatch.setattr(eg, "_procs_matching", lambda pat: [])
    monkeypatch.setattr(eg, "_any_running", lambda pats: False)
    monkeypatch.setattr(eg, "_bomb_procs", lambda: [])
    eg.status()
    out = capsys.readouterr().out
    assert "0 engine proc(s); ports -" in out
    assert "WARNING" not in out


def test_main_default_is_run(monkeypatch):
    called = []
    monkeypatch.setattr(eg, "run", lambda: called.append("run"))
    monkeypatch.setattr(eg.sys, "argv", ["hanzo-engine-guard"])
    eg.main()
    assert called == ["run"]


def test_main_status(monkeypatch):
    called = []
    monkeypatch.setattr(eg, "status", lambda: called.append("status"))
    monkeypatch.setattr(eg.sys, "argv", ["hanzo-engine-guard", "status"])
    eg.main()
    assert called == ["status"]


def test_main_install_and_uninstall(monkeypatch):
    seen = []
    monkeypatch.setattr(eg, "_dispatch", lambda a: seen.append(a))
    monkeypatch.setattr(eg.sys, "argv", ["hanzo-engine-guard", "install"])
    eg.main()
    monkeypatch.setattr(eg.sys, "argv", ["hanzo-engine-guard", "uninstall"])
    eg.main()
    assert seen == ["install", "uninstall"]


def test_main_unknown_action_exits(monkeypatch):
    monkeypatch.setattr(eg.sys, "argv", ["hanzo-engine-guard", "bogus"])
    with pytest.raises(SystemExit):
        eg.main()
