"""
E = 8 sweep plan, LR selection and Slurm job scripts (nanochat/pretrain_sweep.py,
scripts/pretrain_sweep.py). No training here: results are written by hand and the
job scripts run against a fake launcher.

python -m pytest tests/test_pretrain_sweep.py -v
"""

import json
import math
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from nanochat.pretrain_sweep import (
    CHECKPOINT_EVERY, FULL_TOKENS, SWEEP_TOKENS, TOTAL_BATCH_SIZE, RunSpec, main_runs, num_steps,
    parabola_vertex, render_job, select_lr, sweep_points, sweep_runs, transfer_factor, warmup_steps,
)
from scripts import pretrain_sweep as cli

REPO_ROOT = Path(__file__).resolve().parents[1]


def arg(args, flag):
    values = [a.split("=", 1)[1] for a in args if a.startswith(flag + "=")]
    assert len(values) == 1, (flag, args)
    return values[0]


# ---------------------------------------------------------------------------
# Plan

def test_step_counts_warmup_and_transfer_follow_the_protocol():
    assert num_steps(SWEEP_TOKENS) == 1907 and num_steps(FULL_TOKENS) == 11445
    assert warmup_steps(SWEEP_TOKENS) == 83
    assert warmup_steps(FULL_TOKENS) == 496 == -(-130_000_000 // TOTAL_BATCH_SIZE)  # the 130M-token warmup
    assert transfer_factor() == pytest.approx((FULL_TOKENS / SWEEP_TOKENS) ** -0.32)
    assert transfer_factor() == pytest.approx(0.5635, abs=1e-4)
    with pytest.raises(ValueError):
        num_steps(SWEEP_TOKENS + 1)


def test_sweep_runs_differ_only_in_optimizer_and_learning_rate():
    runs = sweep_runs()
    assert [(r.optimizer, r.lr) for r in runs] == [("adamw", 1e-3), ("adamw", 2e-3), ("adamw", 4e-3),
                                                   ("muon", 2e-3), ("muon", 4e-3), ("muon", 8e-3)]
    names = [r.name for r in runs]
    assert len(set(names)) == 6 and all(re.match(r"^[a-zA-Z0-9._-]+$", n) for n in names)
    assert names[0] == "e8-sweep-adamw-lr0.001-s42"
    lr_flags = ("--adamw-matrix-lr", "--muon-matrix-lr", "--optimizer-name")
    common = {tuple(a for a in r.base_train_args() if not a.startswith(lr_flags)) for r in runs}
    assert len(common) == 1, "sweep runs must share every setting except optimizer and LR"
    for run in runs:
        args = run.base_train_args()
        assert float(arg(args, f"--{run.optimizer}-matrix-lr")) == run.lr
        other = "muon" if run.optimizer == "adamw" else "adamw"
        assert not any(a.startswith(f"--{other}-matrix-lr") for a in args)
        assert arg(args, "--n-experts") == "8" and arg(args, "--depth") == "10" and arg(args, "--seed") == "42"
        assert int(arg(args, "--train-tokens")) == SWEEP_TOKENS and arg(args, "--warmup-steps") == "83"
        assert arg(args, "--save-every") == "-1"  # sweep runs keep only the final checkpoint
        micro_batch = int(arg(args, "--device-batch-size")) * 2048
        assert int(arg(args, "--eval-tokens")) % micro_batch == 0
        assert int(arg(args, "--heldout-eval-tokens")) % micro_batch == 0


def test_main_runs_use_the_selected_lrs_for_3b_tokens():
    runs = main_runs({"adamw": 1.1e-3, "muon": 2.3e-3}, seeds=(42, 43))
    assert [(r.optimizer, r.seed) for r in runs] == [("adamw", 42), ("muon", 42), ("adamw", 43), ("muon", 43)]
    assert runs[0].name == "e8-main-adamw-s42"
    args = runs[1].base_train_args()
    assert float(arg(args, "--muon-matrix-lr")) == 2.3e-3
    assert int(arg(args, "--train-tokens")) == FULL_TOKENS and arg(args, "--warmup-steps") == "496"
    assert int(arg(args, "--save-every")) == CHECKPOINT_EVERY
    # Checkpoints land where 1B- and 2B-token decay branches would start (0.8B and 1.6B tokens)
    assert 2 * CHECKPOINT_EVERY * TOTAL_BATCH_SIZE == pytest.approx(0.8e9, rel=1e-3)
    assert 4 * CHECKPOINT_EVERY * TOTAL_BATCH_SIZE == pytest.approx(1.6e9, rel=1e-3)


def test_every_planned_flag_exists_in_base_train():
    help_text = subprocess.run([sys.executable, "-m", "scripts.base_train", "--help"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=300, check=True).stdout
    flags = {a.split("=", 1)[0] for run in sweep_runs() + main_runs({"adamw": 1e-3, "muon": 2e-3})
             for a in run.base_train_args()}
    assert flags and all(flag in help_text for flag in flags), sorted(f for f in flags if f not in help_text)


# ---------------------------------------------------------------------------
# LR selection

def quadratic(lr, best_lr=2.6e-3, curvature=0.02, floor=3.0):
    return floor + curvature * (math.log2(lr) - math.log2(best_lr)) ** 2


def test_parabola_vertex_is_exact_for_any_spacing():
    for xs in ([0.0, 1.0, 2.0], [-1.0, 0.0, 2.5]):
        ys = [5 + 0.3 * (x - 0.7) ** 2 for x in xs]
        assert parabola_vertex(xs, ys) == pytest.approx(0.7, abs=1e-12)


def test_selection_takes_the_parabola_vertex_and_transfers_it():
    choice = select_lr([(lr, quadratic(lr)) for lr in (4e-3, 1e-3, 2e-3)])  # unsorted on purpose
    assert choice["status"] == "selected" and choice["best_grid_lr"] == 2e-3
    assert choice["sweep_lr"] == pytest.approx(2.6e-3, rel=1e-9)
    assert choice["full_lr"] == pytest.approx(2.6e-3 * transfer_factor(), rel=1e-9)


def test_selection_asks_for_one_more_point_at_an_edge():
    low = select_lr([(lr, quadratic(lr, best_lr=4e-4)) for lr in (1e-3, 2e-3, 4e-3)])
    assert low["status"] == "extend" and low["next_lr"] == 5e-4
    high = select_lr([(lr, quadratic(lr, best_lr=2e-2)) for lr in (2e-3, 4e-3, 8e-3)])
    assert high["status"] == "extend" and high["next_lr"] == 1.6e-2
    # A diverged top point with the best at the bottom also extends downwards
    diverged_top = select_lr([(1e-3, 3.0), (2e-3, 3.1), (4e-3, None)])
    assert diverged_top["status"] == "extend" and diverged_top["next_lr"] == 5e-4
    # After the extension the parabola uses the new point
    extended = select_lr([(lr, quadratic(lr, best_lr=9e-4)) for lr in (5e-4, 1e-3, 2e-3, 4e-3)])
    assert extended["status"] == "selected" and extended["sweep_lr"] == pytest.approx(9e-4, rel=1e-9)


def test_selection_with_a_diverged_neighbour_keeps_the_best_grid_point():
    choice = select_lr([(1e-3, 3.10), (2e-3, 3.00), (4e-3, None)])
    # 2e-3 is best and interior, but its upper neighbour diverged, so no parabola can be fitted
    assert choice["status"] == "selected" and choice["sweep_lr"] == 2e-3 and "diverged" in choice["method"]


def test_selection_rejects_bad_inputs():
    with pytest.raises(ValueError, match="diverged"):
        select_lr([(1e-3, None), (2e-3, None), (4e-3, float("nan"))])
    with pytest.raises(ValueError, match="duplicate"):
        select_lr([(1e-3, 3.0), (1e-3, 3.1), (2e-3, 3.2)])


def write_result(root, run, bpb, status="completed", lr=None):
    directory = Path(root) / run.name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps({
        "status": status, "final_val_bpb": bpb if status == "completed" else None,
        "settings": {"optimizer_name": run.optimizer, "matrix_lr": run.lr if lr is None else lr},
    }))


def test_sweep_points_reads_results_and_checks_them(tmp_path):
    runs = sweep_runs()
    for run in runs[:-1]:
        write_result(tmp_path, run, quadratic(run.lr))
    with pytest.raises(FileNotFoundError, match=runs[-1].name):
        sweep_points(runs, tmp_path)
    write_result(tmp_path, runs[-1], None, status="diverged")
    points = sweep_points(runs, tmp_path)
    assert points["muon"][-1] == (8e-3, None) and points["adamw"][1] == (2e-3, quadratic(2e-3))
    write_result(tmp_path, runs[0], 3.0, lr=5e-3)
    with pytest.raises(ValueError, match="do not match"):
        sweep_points(runs, tmp_path)


# ---------------------------------------------------------------------------
# Slurm job scripts

FAKE_LAUNCHER = """#!/bin/bash
# Records each call; "fails" for runs whose name contains FAIL_PATTERN; writes result.json otherwise.
echo "$RUN_NAME NPROC=$NPROC_PER_NODE $*" >> "$CALL_LOG"
if [[ -n "${FAIL_PATTERN:-}" && "$RUN_NAME" == *"$FAIL_PATTERN"* ]]; then exit 1; fi
mkdir -p "$NANOCHAT_SCRATCH_BASE/base_checkpoints/$RUN_NAME"
echo '{"status": "completed"}' > "$NANOCHAT_SCRATCH_BASE/base_checkpoints/$RUN_NAME/result.json"
"""


@pytest.fixture
def fake_cluster(tmp_path):
    launcher = tmp_path / "fake_launcher.sh"
    launcher.write_text(FAKE_LAUNCHER)
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    scratch = tmp_path / "scratch"
    environ = dict(os.environ, NANOCHAT_LAUNCHER=str(launcher), NANOCHAT_SCRATCH_BASE=str(scratch),
                   NANOCHAT_POOL_BASE=str(tmp_path / "pool"), CALL_LOG=str(tmp_path / "calls.log"))
    return {"tmp": tmp_path, "scratch": scratch, "environ": environ, "calls": tmp_path / "calls.log",
            "pool": tmp_path / "pool"}


def test_job_runs_each_run_skips_finished_ones_and_reports_failures(fake_cluster):
    runs = [r for r in sweep_runs() if r.optimizer == "adamw"]
    script = render_job("e8-sweep-adamw", runs, "03:00:00", repo_dir=fake_cluster["tmp"])
    for directive in ("#SBATCH -p mit_normal_gpu", "#SBATCH --gres=gpu:b200:1", "#SBATCH -t 03:00:00"):
        assert directive in script
    path = fake_cluster["tmp"] / "job.sbatch"
    path.write_text(script)
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    # The last run already finished (e.g. before a resubmission); the middle one fails.
    done = fake_cluster["scratch"] / "base_checkpoints" / runs[2].name
    done.mkdir(parents=True)
    (done / "result.json").write_text("{}")
    result = subprocess.run(["bash", str(path)], env=dict(fake_cluster["environ"], FAIL_PATTERN="lr0.002"),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert f"FAILED: {runs[1].name}" in result.stderr and f"Skipping {runs[2].name}" in result.stdout
    calls = fake_cluster["calls"].read_text().splitlines()
    assert [c.split()[0] for c in calls] == [runs[0].name, runs[1].name]
    assert calls[0] == f"{runs[0].name} NPROC=1 train " + " ".join(runs[0].base_train_args())
    # Result files were copied to pool storage (weights would not be)
    assert (fake_cluster["pool"] / "results" / "base_checkpoints" / runs[0].name / "result.json").exists()
    # Resubmitting runs only what is still missing
    fake_cluster["calls"].unlink()
    result = subprocess.run(["bash", str(path)], env=fake_cluster["environ"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert [c.split()[0] for c in fake_cluster["calls"].read_text().splitlines()] == [runs[1].name]


def test_command_line_from_sweep_jobs_to_main_jobs(tmp_path):
    out = tmp_path / "jobs"
    root = tmp_path / "checkpoints"
    assert cli.main(["jobs", "--stage", "sweep", "--out", str(out), "--repo-dir", "/repo"]) in (0, None)
    assert sorted(p.name for p in out.glob("*.sbatch")) == ["e8-sweep-adamw.sbatch", "e8-sweep-muon.sbatch"]
    assert (out / "e8-sweep-muon.sbatch").read_text().count("\nrun_one ") == 3

    # Muon's best point is at the top edge: select asks for one more point
    for run in sweep_runs():
        write_result(root, run, quadratic(run.lr, best_lr=2.5e-3 if run.optimizer == "adamw" else 1.0e-2))
    assert cli.main(["select", "--out", str(out), "--checkpoints-root", str(root)]) == 3
    assert json.loads((out / "lr_selection.json").read_text())["muon"]["next_lr"] == 1.6e-2
    with pytest.raises(SystemExit, match="no selected LR for muon"):
        cli.main(["jobs", "--stage", "main", "--selection", str(out / "lr_selection.json"), "--out", str(out)])

    # Run the extra point, then everything is selected
    extra = RunSpec("sweep", "muon", 1.6e-2)
    write_result(root, extra, quadratic(1.6e-2, best_lr=1.0e-2))
    assert cli.main(["select", "--out", str(out), "--checkpoints-root", str(root), "--extra-lr", "muon=1.6e-2"]) == 0
    selection = json.loads((out / "lr_selection.json").read_text())
    assert selection["muon"]["sweep_lr"] == pytest.approx(1.0e-2, rel=1e-9)
    assert cli.main(["jobs", "--stage", "main", "--selection", str(out / "lr_selection.json"),
                     "--out", str(out), "--seeds", "42"]) in (0, None)
    main_script = (out / "e8-main-muon-s42.sbatch").read_text()
    assert f"--muon-matrix-lr={selection['muon']['full_lr']!r}" in main_script
    assert "#SBATCH -t 04:30:00" in main_script and "--train-tokens=3000238080" in main_script


def test_finetune_jobs_cover_the_grid_and_call_the_finetune_script(tmp_path):
    from nanochat.pretrain_sweep import FINETUNE_LR_FRACTIONS, finetune_lr_grid
    from scripts import finetune_jobs
    selection = {"adamw": {"full_lr": 1e-3}, "muon": {"full_lr": 2e-3}}
    (tmp_path / "sel.json").write_text(json.dumps(selection))
    finetune_jobs.main(["--selection", str(tmp_path / "sel.json"), "--base-runs", "e8-main-adamw-s42",
                        "--dataset", "wikitext103=/data/wt", "--dataset", "codeparrot=/data/cp",
                        "--d0-file", "/data/d0.parquet", "--out", str(tmp_path / "jobs"), "--repo-dir", str(tmp_path)])
    jobs = sorted(p.name for p in (tmp_path / "jobs").glob("*.sbatch"))
    assert len(jobs) == 4 and "ft-e8-main-adamw-s42-codeparrot-muon.sbatch" in jobs
    grid = finetune_lr_grid(selection)
    assert grid["muon"] == [2e-3 * f for f in FINETUNE_LR_FRACTIONS]
    # Run one job against a fake python that records its arguments
    fake = tmp_path / "fake_python.sh"
    fake.write_text('#!/bin/bash\necho "$@" > "$CALL_LOG"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    script = tmp_path / "jobs" / "ft-e8-main-adamw-s42-wikitext103-adamw.sbatch"
    env = dict(os.environ, NANOCHAT_PYTHON=str(fake), NANOCHAT_SCRATCH_BASE=str(tmp_path / "scratch"),
               NANOCHAT_POOL_BASE=str(tmp_path / "pool"), CALL_LOG=str(tmp_path / "call.log"))
    assert subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=60).returncode == 0
    call = (tmp_path / "call.log").read_text().split()
    assert call[:2] == ["-m", "scripts.finetune"] and call[call.index("--ft-name") + 1] == "wikitext103"
    assert call[call.index("--lrs") + 1] == "adamw=" + ",".join(repr(lr) for lr in grid["adamw"])
    assert call[call.index("--seeds") + 1:] == ["1", "2", "3", "4", "5"]
