#!/usr/bin/env python3
"""Unattended end-to-end driver: crawl -> corpus -> indexes -> submissions.

Chains every long competition job (official download, PubMed, crawl, Wayback,
merge, Lucene/Qdrant index builds, retrieval runs, validation, ZIP packaging)
into one resumable state machine that runs detached from any terminal, so no
human or agent has to babysit it. Every stage is idempotent or resume-safe;
killing the daemon and running ``start`` again continues where it stopped.

Usage:
    python scripts/auto_pipeline.py start [--from STAGE] [--force]  # detached
    python scripts/auto_pipeline.py run                             # foreground
    python scripts/auto_pipeline.py status [--full]
    python scripts/auto_pipeline.py stop [--all]
    python scripts/auto_pipeline.py watchdog                        # cron-safe

Design notes:
- Stage state lives in runs/auto/state/<stage>.json, logs in runs/auto/logs/.
- A stage with ``group`` belongs to a sub-chain that a gate can disable as a
  whole (e.g. the v1b Wayback rebuild); gate result ``skip`` disables the
  group, ``fail`` stops the daemon with the FAILED marker.
- Side jobs (vov.vn crawl, Wayback, slow retries) run as their own detached
  processes with pid files in runs/auto/jobs/ because they outlive or overlap
  the main chain. A daemon restart relaunches a side job that died mid-flight
  (all of them are resume-safe), once per daemon invocation.
- GPU stages inherit VIBIOMIR_GPU_MEM_FRACTION (default 0.4) which
  vibiomir.gpu applies through torch.cuda.set_per_process_memory_fraction.
- Disk guard: each stage declares the minimum free GB it needs at start; the
  daemon fails fast instead of filling the disk.
- The watchdog (cron) restarts the daemon only when it died without writing
  DONE or FAILED, i.e. after a crash or reboot, and only before the
  VIBIOMIR_AUTO_DEADLINE.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTO_ROOT = REPO_ROOT / "runs" / "auto"
STATE_DIR = AUTO_ROOT / "state"
LOG_DIR = AUTO_ROOT / "logs"
JOB_DIR = AUTO_ROOT / "jobs"
PID_FILE = AUTO_ROOT / "pipeline.pid"
DONE_FILE = AUTO_ROOT / "DONE"
FAILED_FILE = AUTO_ROOT / "FAILED"

OFFICIAL_QUERIES = "data/raw/official/query.parquet"
CORPUS_DIR = "data/processed/corpus"
PUBMED_PARTS_DIR = REPO_ROOT / "data/interim/parts/pubmed"
PUBMED_EXPECTED_PARTS = 1200  # ceil(240_000 PMIDs / 200 per efetch batch)
STORE_V1A = "artifacts/full/store_v1a.sqlite"
STORE_V1B = "artifacts/full/store_v1b.sqlite"
BM25_CONFIG = "configs/full/00_bm25_lucene.yaml"
DENSE_CONFIG = "configs/full/01_mgte_dense_auto.yaml"
BM25_INDEX_DIR = Path("artifacts/full/00_bm25_lucene/chunk_index")

# v1b (Wayback-enriched corpus) is only rebuilt when Wayback finished early
# enough to leave slack before the public deadline (31/10 23:59 local).
V1B_DEADLINE = datetime.fromisoformat(os.getenv("VIBIOMIR_V1B_DEADLINE", "2026-10-30T18:00:00"))
AUTO_DEADLINE = datetime.fromisoformat(os.getenv("VIBIOMIR_AUTO_DEADLINE", "2026-10-31T22:00:00"))


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _venv_python() -> str:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


PY = _venv_python()
VB = [PY, "-m", "vibiomir"]


def _load_dotenv() -> dict[str, str]:
    """Parse the repo .env (KEY=VALUE lines) without exporting secrets to logs."""
    env: dict[str, str] = {}
    path = REPO_ROOT / ".env"
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip("'\"")
    return env


def stage_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update(_load_dotenv())
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("VIBIOMIR_GPU_MEM_FRACTION", "0.4")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if extra:
        env.update(extra)
    return env


@dataclass
class Stage:
    name: str
    commands: list[list[str]] = field(default_factory=list)
    func: str = ""  # gate/hook name in GATES; returns "pass:...", "skip:...", "fail:..."
    group: str = ""  # sub-chain a gate skip disables as a whole
    retries: int = 0
    min_free_gb: float = 4.0
    env: dict[str, str] = field(default_factory=dict)
    record_run_info: bool = False


def _bm25_build_commands(store: str) -> list[list[str]]:
    """Export chunks to JSONL, build the Lucene index, then free the JSONL."""
    collection_dir = BM25_INDEX_DIR.parent / f"{BM25_INDEX_DIR.name}_input"
    cleanup = (
        "import shutil, sys\n"
        f"for target in ({str(collection_dir)!r}, {str(BM25_INDEX_DIR)!r}):\n"
        "    shutil.rmtree(target, ignore_errors=True)\n"
        "    print('cleaned', target)\n"
    )
    return [
        [PY, "-c", cleanup],  # remove stale index + input from any earlier build
        [PY, "scripts/build_method_indexes.py",
         "--config", BM25_CONFIG, "--store", store, "--threads", "8"],
        [PY, "-c", f"import shutil; shutil.rmtree({str(collection_dir)!r}, ignore_errors=True); "
                   f"print('cleaned', {str(collection_dir)!r})"],
    ]


def _run_paths(tag: str) -> tuple[str, str]:
    return f"runs/auto/{tag}/submission.json", f"runs/auto/{tag}/submission.zip"


def _run_stage_commands(tag: str, config: str, store: str) -> list[list[str]]:
    output, _ = _run_paths(tag)
    return [
        VB + ["indexed-run", "--config", config, "--store", store,
              "--queries", OFFICIAL_QUERIES, "--output", output],
    ]


def _package_stage_commands(tag: str) -> list[list[str]]:
    output, zip_path = _run_paths(tag)
    return [
        VB + ["package", "--submission", output, "--corpus", CORPUS_DIR,
              "--output", zip_path],
    ]


STAGES: list[Stage] = [
    Stage("prepare", [
        VB + ["ingest", "download-official"],
        VB + ["ingest", "manifest"],
    ], min_free_gb=8),
    Stage("pilot", [["bash", "scripts/ingest_all.sh", "pilot"]], min_free_gb=10),
    Stage("pilot_gate", func="gate_pilot"),
    Stage("crawl", [
        VB + ["ingest", "crawl",
              "--skip-domain", "pubmed.ncbi.nlm.nih.gov", "--skip-domain", "vov.vn"],
    ], retries=2, min_free_gb=8),
    Stage("pubmed_gate", func="gate_pubmed"),
    Stage("merge_v1a", [VB + ["ingest", "merge", "--version", "v1a"]], min_free_gb=8),
    Stage("store_v1a", [
        VB + ["index", "prepare", "--config", BM25_CONFIG,
              "--corpus", CORPUS_DIR, "--store", STORE_V1A],
    ], min_free_gb=12),
    Stage("bm25_index_v1a", _bm25_build_commands(STORE_V1A), min_free_gb=14,
          env={"JAVA_OPTS": "-Xmx6g"}),
    Stage("bm25_run_v1a", _run_stage_commands("bm25_v1a", BM25_CONFIG, STORE_V1A),
          min_free_gb=6, record_run_info=True),
    Stage("bm25_package_v1a", _package_stage_commands("bm25_v1a"), min_free_gb=5),
    Stage("dense_index_v1a", [
        VB + ["index", "build-qdrant", "--config", DENSE_CONFIG,
              "--store", STORE_V1A, "--stage", "chunk", "--retriever", "0"],
    ], retries=2, min_free_gb=5),
    Stage("dense_run_v1a", _run_stage_commands("mgte_v1a", DENSE_CONFIG, STORE_V1A),
          min_free_gb=5, record_run_info=True),
    Stage("dense_package_v1a", _package_stage_commands("mgte_v1a"), min_free_gb=5),
    # ---- v1b: Wayback-enriched rebuild, gated on the side job finishing ----
    Stage("v1b_gate", func="gate_v1b", group="v1b"),
    Stage("free_disk_v1b", func="free_disk_for_v1b", group="v1b"),
    Stage("merge_v1b", [VB + ["ingest", "merge", "--version", "v1b"]],
          group="v1b", min_free_gb=8),
    Stage("cleanup_v1a_store", func="cleanup_v1a_store", group="v1b"),
    Stage("store_v1b", [
        VB + ["index", "prepare", "--config", BM25_CONFIG,
              "--corpus", CORPUS_DIR, "--store", STORE_V1B],
    ], group="v1b", min_free_gb=12),
    Stage("bm25_index_v1b", _bm25_build_commands(STORE_V1B), group="v1b",
          min_free_gb=14, env={"JAVA_OPTS": "-Xmx6g"}),
    Stage("bm25_run_v1b", _run_stage_commands("bm25_v1b", BM25_CONFIG, STORE_V1B),
          group="v1b", min_free_gb=6, record_run_info=True),
    Stage("bm25_package_v1b", _package_stage_commands("bm25_v1b"), group="v1b", min_free_gb=5),
    # ---- optional dense rebuild on v1b (expensive; opt-in via env) ----
    Stage("dense_v1b_gate", func="gate_dense_v1b", group="densev1b"),
    Stage("dense_reindex_v1b", [
        VB + ["index", "build-qdrant", "--config", DENSE_CONFIG,
              "--store", STORE_V1B, "--stage", "chunk", "--retriever", "0", "--recreate"],
    ], group="densev1b", retries=2, min_free_gb=5),
    Stage("dense_run_v1b", _run_stage_commands("mgte_v1b", DENSE_CONFIG, STORE_V1B),
          group="densev1b", min_free_gb=5, record_run_info=True),
    Stage("dense_package_v1b", _package_stage_commands("mgte_v1b"),
          group="densev1b", min_free_gb=5),
]
STAGE_NAMES = [stage.name for stage in STAGES]

# Side jobs: long, network-bound processes that overlap the main chain.
PUBMED_RETRY_LOOP = f"""
set -u
cd "{REPO_ROOT}"
for attempt in $(seq 1 15); do
  echo "[pubmed-job $(date '+%F %T')] attempt ${{attempt}}"
  "{PY}" -m vibiomir ingest pubmed || true
  have=$(ls data/interim/parts/pubmed/part-pubmed-*.parquet 2>/dev/null | wc -l)
  echo "[pubmed-job $(date '+%F %T')] parts ${{have}}/{PUBMED_EXPECTED_PARTS}"
  if [ "${{have}}" -ge 1140 ]; then echo '[pubmed-job] coverage reached'; exit 0; fi
  sleep "${{VIBIOMIR_PUBMED_RETRY_SLEEP:-1200}}"
done
echo '[pubmed-job] giving up after 15 attempts'
"""

SIDE_JOBS: dict[str, dict] = {
    "pubmed": {
        "cmd": ["bash", "-c", PUBMED_RETRY_LOOP],
        "start_after": "prepare",
    },
    "vov": {
        "cmd": VB + ["ingest", "crawl", "--domain", "vov.vn", "--links", "",
                     "--workers", "1", "--log-file", "data/interim/logs/crawl_vov.log"],
        "start_after": "pilot_gate",
    },
    "wayback": {
        "cmd": VB + ["ingest", "wayback",
                     "--skip-domain", "pubmed.ncbi.nlm.nih.gov",
                     "--skip-domain", "vov.vn",
                     "--skip-domain", "dantri.com.vn",
                     "--skip-domain", "www.qdnd.vn",
                     "--concurrency", "8"],
        "start_after": "crawl",
    },
    "slow_retry": {
        "cmd": VB + ["ingest", "crawl", "--domain", "dantri.com.vn",
                     "--domain", "www.qdnd.vn", "--links", "",
                     "--rps", "0.5", "--max-rps", "1.0", "--workers", "1",
                     "--log-file", "data/interim/logs/crawl_slow.log"],
        "start_after": "crawl",
    },
}


# --------------------------------------------------------------------------- #
# state helpers
# --------------------------------------------------------------------------- #

def free_gb(path: Path = REPO_ROOT) -> float:
    return shutil.disk_usage(path).free / 1e9


def read_state(name: str) -> dict:
    path = STATE_DIR / f"{name}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_state(name: str, **fields) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = read_state(name)
    state.update(fields)
    state["updated_at"] = _now()
    (STATE_DIR / f"{name}.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def daemon_running() -> bool:
    return pid_alive(read_pid(PID_FILE))


def record_run_info(stage_name: str) -> None:
    """Save resolved config + git commit next to a submission (ticket E4)."""
    stage = next((candidate for candidate in STAGES if candidate.name == stage_name), None)
    if stage is None:
        return
    output = cfg = None
    for command in stage.commands:
        if "--output" in command:
            output = command[command.index("--output") + 1]
        if "--config" in command:
            cfg = command[command.index("--config") + 1]
    if not output:
        return
    run_dir = Path(output).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    info = {
        "stage": stage_name,
        "config": cfg,
        "git_commit": commit,
        "finished_at": _now(),
        "gpu_mem_fraction": os.getenv("VIBIOMIR_GPU_MEM_FRACTION", "0.4"),
    }
    (run_dir / "run_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if cfg:
        resolved = subprocess.run(
            VB + ["show-config", "--config", cfg],
            cwd=REPO_ROOT, env=stage_env(), capture_output=True, text=True,
        )
        if resolved.returncode == 0:
            (run_dir / "config_resolved.yaml").write_text(resolved.stdout, encoding="utf-8")


# --------------------------------------------------------------------------- #
# gates and hooks
# --------------------------------------------------------------------------- #

def gate_pilot() -> str:
    """Require a sane pilot success ratio before committing ~10 h of crawling."""
    db = REPO_ROOT / "data/interim/pilot/crawl_state.sqlite"
    if not db.exists():
        return f"fail:pilot state missing ({db}); did the pilot stage run?"
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT status, COUNT(*) FROM pages GROUP BY status").fetchall()
    counts = dict(rows)
    ok = counts.get("ok", 0) + counts.get("short", 0)
    attempted = sum(counts.values())
    if attempted == 0:
        return "fail:no pilot rows in crawl state"
    ratio = ok / attempted
    detail = f"pilot ok+short {ok}/{attempted} = {ratio:.0%}"
    if ratio < 0.30:
        return f"fail:{detail} - network or blocking problem; fix before full crawl"
    return f"pass:{detail}"


def _job_running(name: str) -> bool:
    return pid_alive(read_pid(JOB_DIR / f"{name}.pid"))


def _pubmed_part_count() -> int:
    if not PUBMED_PARTS_DIR.exists():
        return 0
    return sum(1 for _ in PUBMED_PARTS_DIR.glob("part-pubmed-*.parquet"))


def gate_pubmed() -> str:
    """Wait out the PubMed side job, then judge EN coverage before merging.

    Failed efetch batches write no part file, so the part count is honest
    coverage. A skip still lets the chain merge: EN docs then stay
    ``source=missing`` until a later merge picks the parts up.
    """
    wait_seconds = float(os.getenv("VIBIOMIR_PUBMED_WAIT_SECONDS", str(4 * 3600)))
    deadline = time.time() + wait_seconds
    while _job_running("pubmed") and time.time() < deadline:
        time.sleep(120)
    if _job_running("pubmed"):
        # Merge must not race a part writer; the job is resume-safe, stop it.
        pid = read_pid(JOB_DIR / "pubmed.pid")
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            time.sleep(5)
        except OSError:
            pass
    have = _pubmed_part_count()
    detail = f"pubmed parts {have}/{PUBMED_EXPECTED_PARTS}"
    if have >= int(0.95 * PUBMED_EXPECTED_PARTS):
        return f"pass:{detail}"
    return f"skip:{detail} below 95%; EN docs merge as missing for now"


def gate_v1b() -> str:
    """Rebuild on the Wayback corpus only if the job is done and time remains."""
    if _job_running("wayback"):
        return "skip:wayback job still running"
    if not (REPO_ROOT / "data/interim/wayback_state.sqlite").exists():
        return "skip:wayback never ran"
    if datetime.now() > V1B_DEADLINE:
        return f"skip:past v1b deadline {V1B_DEADLINE:%d/%m %H:%M}"
    return "pass:wayback finished before deadline"


def gate_dense_v1b() -> str:
    """Dense v1b rebuild costs ~10 GPU-hours; opt in with VIBIOMIR_DENSE_V1B=1."""
    if os.getenv("VIBIOMIR_DENSE_V1B", "0") != "1":
        return "skip:set VIBIOMIR_DENSE_V1B=1 to rebuild the dense index on v1b"
    base = gate_v1b()
    if base.startswith("pass") and datetime.now() > V1B_DEADLINE:
        return "skip:past v1b deadline"
    return base


def free_disk_for_v1b() -> str:
    """Drop the re-parsable PubMed XML archive when disk is tight for rebuilds."""
    threshold = float(os.getenv("VIBIOMIR_PUBMED_RAW_KEEP_GB", "16"))
    raw = REPO_ROOT / "data/raw/pubmed"
    have = free_gb()
    if have >= threshold or not raw.exists():
        return f"pass:{have:.1f} GB free, keeping data/raw/pubmed"
    size = sum(file.stat().st_size for file in raw.rglob("*") if file.is_file())
    shutil.rmtree(raw, ignore_errors=True)
    return (f"pass:disk tight ({have:.1f} GB free); deleted PubMed raw XML "
            f"({size / 1e9:.1f} GB); parts parquet keeps the extracted text")


def cleanup_v1a_store() -> str:
    """Drop the v1a SQLite store once v1a submissions are packaged."""
    path = REPO_ROOT / STORE_V1A
    if path.exists():
        path.unlink()
        return f"pass:removed {STORE_V1A}"
    return "pass:nothing to remove"


GATES = {
    "gate_pilot": gate_pilot,
    "gate_pubmed": gate_pubmed,
    "gate_v1b": gate_v1b,
    "gate_dense_v1b": gate_dense_v1b,
    "free_disk_for_v1b": free_disk_for_v1b,
    "cleanup_v1a_store": cleanup_v1a_store,
}


# --------------------------------------------------------------------------- #
# side jobs
# --------------------------------------------------------------------------- #

def start_job(name: str) -> None:
    spec = SIDE_JOBS[name]
    pid_file = JOB_DIR / f"{name}.pid"
    if pid_alive(read_pid(pid_file)):
        print(f"[{_now()}] side job {name} already running", flush=True)
        return
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / f"job_{name}.log").open("ab", buffering=0) as log:
        log.write(f"\n===== [{_now()}] starting side job {name} =====\n".encode("utf-8"))
        process = subprocess.Popen(
            spec["cmd"], cwd=REPO_ROOT, env=stage_env(), stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    pid_file.write_text(str(process.pid), encoding="utf-8")
    (JOB_DIR / f"{name}.started").write_text(_now(), encoding="utf-8")
    print(f"[{_now()}] side job {name} started pid={process.pid} "
          f"log={LOG_DIR / ('job_' + name + '.log')}", flush=True)


def start_ready_jobs(executed: set[str], launched: set[str]) -> None:
    """Launch (or relaunch after a crash) side jobs whose trigger just passed."""
    for name, spec in SIDE_JOBS.items():
        if spec["start_after"] not in executed or name in launched:
            continue
        launched.add(name)
        start_job(name)


def stop_jobs() -> None:
    for name in SIDE_JOBS:
        pid = read_pid(JOB_DIR / f"{name}.pid")
        if pid_alive(pid):
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                print(f"[{_now()}] stopped side job {name} (pgid of {pid})")
            except OSError as exc:
                print(f"[{_now()}] could not stop {name}: {exc}")


# --------------------------------------------------------------------------- #
# stage runner
# --------------------------------------------------------------------------- #

def run_stage(stage: Stage) -> tuple[bool, str]:
    """Run one stage. Returns (chain_may_continue, gate_result_or_empty)."""
    if stage.min_free_gb:
        have = free_gb()
        if have < stage.min_free_gb:
            write_state(stage.name, status="failed",
                        error=f"disk guard: {have:.1f} GB free < {stage.min_free_gb} GB")
            print(f"[{_now()}] {stage.name}: FAILED disk guard ({have:.1f} GB free)", flush=True)
            return False, ""

    if stage.func:
        result = GATES[stage.func]()
        kind, _, detail = result.partition(":")
        if kind == "pass":
            write_state(stage.name, status="done", gate="pass", detail=detail, error=None)
            print(f"[{_now()}] {stage.name}: gate pass - {detail}", flush=True)
            return True, "pass"
        if kind == "skip":
            write_state(stage.name, status="done", gate="skip", detail=detail, error=None)
            print(f"[{_now()}] {stage.name}: gate skip - {detail}", flush=True)
            return True, "skip"
        write_state(stage.name, status="failed", gate="fail", error=detail)
        print(f"[{_now()}] {stage.name}: gate FAIL - {detail}", flush=True)
        return False, "fail"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    attempts = stage.retries + 1
    for attempt in range(1, attempts + 1):
        write_state(stage.name, status="running", attempt=attempt, started_at=_now(), error=None)
        ok = True
        with (LOG_DIR / f"{stage.name}.log").open("ab", buffering=0) as log:
            for command in stage.commands:
                printable = " ".join(str(part) for part in command)
                header = f"\n===== [{_now()}] {stage.name} attempt {attempt}: {printable} =====\n"
                log.write(header.encode("utf-8"))
                log.flush()
                print(f"[{_now()}] {stage.name}: {printable}", flush=True)
                code = subprocess.call(
                    [str(part) for part in command], cwd=REPO_ROOT,
                    env=stage_env(stage.env), stdout=log, stderr=subprocess.STDOUT,
                )
                if code != 0:
                    log.write(f"===== [{_now()}] exit {code} =====\n".encode("utf-8"))
                    ok = False
                    break
        if ok:
            write_state(stage.name, status="done", finished_at=_now(), error=None)
            print(f"[{_now()}] {stage.name}: done (disk free {free_gb():.1f} GB)", flush=True)
            if stage.record_run_info:
                try:
                    record_run_info(stage.name)
                except Exception as exc:  # metadata must never kill the chain
                    print(f"[{_now()}] {stage.name}: run_info failed: {exc}", flush=True)
            return True, ""
        write_state(stage.name, status="failed", error=f"attempt {attempt} exited non-zero")
        if attempt < attempts:
            print(f"[{_now()}] {stage.name}: attempt {attempt} failed, retrying in 60 s",
                  flush=True)
            time.sleep(60)
    print(f"[{_now()}] {stage.name}: FAILED after {attempts} attempt(s); "
          f"see {LOG_DIR / (stage.name + '.log')}", flush=True)
    return False, ""


def run_chain(only: list[str] | None = None, start_from: str | None = None,
              force: bool = False) -> int:
    AUTO_ROOT.mkdir(parents=True, exist_ok=True)
    stages = STAGES
    if start_from:
        if start_from not in STAGE_NAMES:
            print(f"Unknown stage {start_from!r}; known: {', '.join(STAGE_NAMES)}")
            return 2
        stages = STAGES[STAGE_NAMES.index(start_from):]
    executed: set[str] = set()
    launched: set[str] = set()
    disabled_groups: set[str] = set()
    for stage in stages:
        state = read_state(stage.name)
        if state.get("status") == "done" and not force:
            # A gate that skipped on an earlier run keeps its group disabled.
            if state.get("gate") == "skip" and stage.group:
                disabled_groups.add(stage.group)
            executed.add(stage.name)
            start_ready_jobs(executed, launched)
            continue
        if only and stage.name not in only:
            executed.add(stage.name)
            continue
        if stage.group and stage.group in disabled_groups:
            write_state(stage.name, status="skipped",
                        error=f"group {stage.group!r} disabled by its gate")
            print(f"[{_now()}] {stage.name}: skipped (group {stage.group} disabled)", flush=True)
            continue
        ok, gate_result = run_stage(stage)
        if gate_result == "skip" and stage.group:
            disabled_groups.add(stage.group)
        if not ok:
            if gate_result != "fail":
                # Command stage failed after retries: stop with FAILED marker.
                FAILED_FILE.write_text(f"{stage.name} at {_now()}\n", encoding="utf-8")
                return 1
            FAILED_FILE.write_text(f"{stage.name} (gate) at {_now()}\n", encoding="utf-8")
            return 1
        executed.add(stage.name)
        start_ready_jobs(executed, launched)
    DONE_FILE.write_text(_now() + "\n", encoding="utf-8")
    print(f"[{_now()}] chain complete", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# daemon control
# --------------------------------------------------------------------------- #

def cmd_start(args: argparse.Namespace) -> int:
    if daemon_running():
        print(f"Daemon already running (pid {read_pid(PID_FILE)}); use status/stop.")
        return 0
    AUTO_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DONE_FILE.unlink(missing_ok=True)
    FAILED_FILE.unlink(missing_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "run"]
    if args.force:
        command.append("--force")
    if getattr(args, "from_stage", None):
        command += ["--from", args.from_stage]
    if args.only:
        command += ["--only", *args.only]
    if shutil.which("systemd-inhibit"):
        # Laptops must not sleep mid-crawl; hold idle/lid inhibitors for the
        # whole daemon lifetime (released automatically when it exits).
        command = [
            "systemd-inhibit", "--who=vibiomir-auto-pipeline",
            "--why=competition crawl/index jobs must finish",
            "--what=idle:sleep:handle-lid-switch", *command,
        ]
    with (LOG_DIR / "pipeline.log").open("ab", buffering=0) as log:
        log.write(f"\n===== [{_now()}] daemon start: {' '.join(command)} =====\n".encode("utf-8"))
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    PID_FILE.write_text(str(process.pid), encoding="utf-8")
    print(f"Daemon started pid={process.pid}; logs: {LOG_DIR / 'pipeline.log'}")
    print("Watch progress: python scripts/auto_pipeline.py status [--full]")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    pid = read_pid(PID_FILE)
    if pid_alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            print(f"Sent SIGTERM to daemon pgid of {pid}")
        except OSError as exc:
            print(f"Could not stop daemon: {exc}")
    else:
        print("No daemon running")
    if args.all:
        stop_jobs()
    PID_FILE.unlink(missing_ok=True)
    return 0


def cmd_watchdog(_args: argparse.Namespace) -> int:
    """Cron entry: restart the daemon when it died with work left (zero tokens)."""
    if DONE_FILE.exists() or FAILED_FILE.exists():
        return 0
    if daemon_running():
        return 0
    if datetime.now() > AUTO_DEADLINE:
        return 0
    print(f"[{_now()}] watchdog: daemon dead without DONE/FAILED; restarting")
    return cmd_start(argparse.Namespace(force=False, from_stage=None, only=[]))


def cmd_status(args: argparse.Namespace) -> int:
    print(f"auto_pipeline at {_now()} | daemon "
          f"{'running pid=' + str(read_pid(PID_FILE)) if daemon_running() else 'stopped'}"
          f" | disk free {free_gb():.1f} GB"
          f" | {'DONE' if DONE_FILE.exists() else ''}"
          f"{' FAILED: ' + FAILED_FILE.read_text().strip() if FAILED_FILE.exists() else ''}")
    width = max(len(name) for name in STAGE_NAMES)
    for stage in STAGES:
        state = read_state(stage.name)
        status = state.get("status", "pending")
        extra = state.get("detail") or state.get("error") or ""
        marker = {"done": "+", "failed": "!", "running": ">", "skipped": "-"}.get(status, " ")
        print(f" {marker} {stage.name:<{width}}  {status:<8} {extra}")
        if status == "running" and args.full:
            log = LOG_DIR / f"{stage.name}.log"
            if log.exists():
                tail = log.read_bytes()[-800:].decode("utf-8", "replace").splitlines()
                for line in tail[-4:]:
                    print(f"     | {line}")
    print("side jobs:")
    for name in SIDE_JOBS:
        pid = read_pid(JOB_DIR / f"{name}.pid")
        started = (JOB_DIR / f"{name}.started").exists()
        if pid_alive(pid):
            detail = f"running pid={pid}"
        elif started:
            detail = "finished/stopped (was started)"
        else:
            detail = "not started"
        print(f"   {name:<10} {detail}")
    if args.full:
        print("ingest status:")
        subprocess.call(VB + ["ingest", "status"], cwd=REPO_ROOT, env=stage_env())
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="run the chain as a detached daemon")
    start.add_argument("--from", dest="from_stage", help="first stage to run")
    start.add_argument("--only", nargs="*", default=[], help="restrict to these stages")
    start.add_argument("--force", action="store_true", help="rerun stages marked done")
    start.set_defaults(handler=cmd_start)

    run = sub.add_parser("run", help="run the chain in the foreground")
    run.add_argument("--from", dest="from_stage")
    run.add_argument("--only", nargs="*", default=[])
    run.add_argument("--force", action="store_true")
    run.set_defaults(handler=lambda a: run_chain(a.only, a.from_stage, a.force))

    status = sub.add_parser("status", help="show stage/job/disk summary")
    status.add_argument("--full", action="store_true")
    status.set_defaults(handler=cmd_status)

    stop = sub.add_parser("stop", help="stop the daemon (and --all side jobs)")
    stop.add_argument("--all", action="store_true")
    stop.set_defaults(handler=cmd_stop)

    watchdog = sub.add_parser("watchdog", help="restart a dead daemon; safe for cron")
    watchdog.set_defaults(handler=cmd_watchdog)

    args = parser.parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
