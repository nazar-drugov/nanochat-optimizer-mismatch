"""New-E pretraining orchestration without launching training."""
import hashlib
import json
import math
from pathlib import Path

import pytest

from nanochat.pretrain_sweep import (
    BasePin, CHECKPOINT_EVERY, FINETUNE_BASE_CHECKPOINTS, FINETUNE_BASE_STEP,
    FULL_TOKENS, SWEEP_TOKENS, TOTAL_BATCH_SIZE, RunSpec, base_run_experts,
    completed_base_manifest, load_base_manifest, main_runs, num_steps,
    check_transfer, render_finetune_job, resolve_base_pin, sweep_lr_grid, sweep_points,
    sweep_runs, transfer_check_runs, warmup_steps,
)
from scripts import pretrain_sweep as cli


def fake_result(root, run, val, status="completed"):
    directory = root / run.name
    directory.mkdir(parents=True, exist_ok=True)
    result = {
        "run": run.name, "status": status, "num_iterations": num_steps(run.train_tokens),
        "steps_trained": num_steps(run.train_tokens), "tokens_trained": run.train_tokens,
        "final_val_bpb": val,
        "settings": {"matrix_lr": run.lr, "optimizer_name": run.optimizer,
                     "ffn_type": "moe", "n_experts": run.n_experts, "depth": 10, "seed": run.seed,
                     "total_batch_size": TOTAL_BATCH_SIZE, "train_tokens": run.train_tokens,
                     "warmup_steps": warmup_steps(run.train_tokens), "warmdown_ratio": .2,
                     "weight_decay": .1, "max_grad_norm": 1.,
                     "balancer": "aux", "disable_routing_loss": run.n_experts == 1,
                     "balance_coef": 0. if run.n_experts == 1 else .01,
                     "z_coef": 0. if run.n_experts == 1 else .001},
    }
    (directory / "result.json").write_text(json.dumps(result))
    return directory, result


def fake_checkpoint(root, run):
    directory, _ = fake_result(root, run, 1.0)
    routing = {"disable_routing_loss": run.n_experts == 1, "balancer": "aux",
               "balance_coef": 0. if run.n_experts == 1 else .01,
               "z_coef": 0. if run.n_experts == 1 else .001}
    (directory / f"model_{FINETUNE_BASE_STEP:06d}.pt").write_bytes(b"test model weights")
    (directory / f"meta_{FINETUNE_BASE_STEP:06d}.json").write_text(json.dumps({
        "step": FINETUNE_BASE_STEP, "user_config": {
            "seed": run.seed, "optimizer_name": run.optimizer,
            f"{run.optimizer}_matrix_lr": run.lr, **routing,
        },
        "model_config": {"n_experts": run.n_experts, "ffn_type": "moe", **routing},
        "train_tokens": FULL_TOKENS, "total_batch_size": TOTAL_BATCH_SIZE,
    }))
    return directory


@pytest.mark.parametrize("experts,adamw_center,muon_center", [
    (1, .002, .004), (4, .002 * 4 ** -.25, .004 * 2 ** -.25),
    (16, .001, .004 * 2 ** -.75),
])
def test_new_sweep_grid_and_arguments(experts, adamw_center, muon_center):
    grid = sweep_lr_grid(experts)
    assert grid["adamw"] == pytest.approx([adamw_center * f for f in (.25, .5, 1, 2, 4)])
    assert grid["muon"] == pytest.approx([muon_center * f for f in (.25, .5, 1, 2, 4)])
    runs = sweep_runs(n_experts=experts)
    assert len(runs) == len({r.name for r in runs}) == 10
    for run in runs:
        args = run.base_train_args()
        assert run.name.startswith(f"e{experts}-sweep")
        assert f"--n-experts={experts}" in args
        assert ("--disable-routing-loss" in args) == (experts == 1)
        assert f"--train-tokens={SWEEP_TOKENS}" in args
        assert "--warmup-steps=83" in args


def test_transfer_center_real_checkpoint_and_separate_names():
    lrs = {"adamw": .0011, "muon": .0022}
    runs = transfer_check_runs(lrs, n_experts=16)
    assert len(runs) == 6
    assert len({r.name for r in runs}) == 6
    for run in runs:
        assert run.name.startswith("e16-xfer-")
        center = run.lr == lrs[run.optimizer]
        assert run.keeps_checkpoints == center
        assert f"--save-every={CHECKPOINT_EVERY if center else -1}" in run.base_train_args()
    assert all(r.name.startswith("e16-main-") for r in main_runs(lrs, n_experts=16))
    assert len(transfer_check_runs(lrs)) == 4  # historic E8 center is main


def test_new_result_validation_and_name_collision(tmp_path):
    runs = sweep_runs(n_experts=4)
    directory, result = fake_result(tmp_path, runs[0], 1.0)
    result["settings"]["n_experts"] = 16
    (directory / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="settings"):
        sweep_points([runs[0]], tmp_path)
    result["settings"]["n_experts"] = 4
    result["settings"]["z_coef"] = 0
    (directory / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="settings"):
        sweep_points([runs[0]], tmp_path)
    with pytest.raises(ValueError, match="pool distinct"):
        sweep_points([runs[0], sweep_runs(n_experts=16)[0]], tmp_path)
    with pytest.raises(ValueError, match="duplicate run names"):
        sweep_runs(n_experts=4, lrs={"adamw": [.001, .00100001], "muon": [.002]})
    with pytest.raises(ValueError, match="positive and finite"):
        RunSpec("sweep", "adamw", float("nan"), n_experts=1)


def test_transfer_edge_vs_unfittable():
    report = check_transfer([(.5, 1.2), (1., 1.), (2., None)], 1., 1.)
    assert report["status"] == "unfittable" and "neighbor" in report["reason"]
    report = check_transfer([(.5, 1.1), (1., 1.2), (2., 1.3)], 1., 1.)
    assert report["status"] == "edge" and report["next_factor"] == .25


def test_independent_actual_result_selection_and_reused_center_manifest(tmp_path):
    experts = 1
    root = tmp_path / "checkpoints"
    out = tmp_path / "jobs"
    args = ["--experts", str(experts), "--out", str(out)]
    root_arg = ["--checkpoints-root", str(root)]
    assert cli.main(["jobs", "--stage", "sweep", *args]) is None
    assert sorted(p.name for p in out.glob("*.sbatch")) == ["e1-sweep-adamw.sbatch", "e1-sweep-muon.sbatch"]
    script = (out / "e1-sweep-adamw.sbatch").read_text()
    assert script.count("\nrun_one ") == 5
    assert "RUN_NAME=\"$name\" bash \"$LAUNCHER\" train" in script
    assert script.count("--disable-routing-loss") == 5
    centers = {"adamw": .002, "muon": .004}
    for run in sweep_runs(n_experts=experts):
        fake_result(root, run, 1 + .04 * (math.log2(run.lr / centers[run.optimizer])) ** 2)
    assert cli.main(["select", *args, *root_arg]) == 0
    selection_file = out / "lr_selection.json"
    sel = json.loads(selection_file.read_text())
    assert all(c["n_experts"] == experts and c["seed"] == 42 for c in sel.values())
    full = {o: c["full_lr"] for o, c in sel.items()}
    selection_arg = ["--selection", str(selection_file)]
    with pytest.raises(SystemExit, match="final"):
        cli.main(["jobs", "--stage", "main", *args, *selection_arg])
    assert cli.main(["jobs", "--stage", "transfer", *args, *selection_arg]) is None
    assert len(list(out.glob("e1-xfer-*.sbatch"))) == 6
    best = {"adamw": full["adamw"] * 1.6, "muon": full["muon"] * 1.05}
    for run in transfer_check_runs(full, n_experts=experts):
        fake_result(root, run, 1.0 + .05 * math.log2(run.lr / best[run.optimizer]) ** 2)
    # AdamW optimum 1.6x is at the grid edge (x2); extend once.
    assert cli.main(["transfer-check", *args, *root_arg, *selection_arg]) == 3
    with pytest.raises(SystemExit, match="no selected"):
        cli.main(["jobs", "--stage", "main", *args, "--selection", str(out / "lr_selection_3b.json")])
    for run in transfer_check_runs(full, factors={"adamw": (.5, 2., 4.), "muon": (.5, 2.)}, n_experts=experts):
        fake_result(root, run, 1.0 + .05 * math.log2(run.lr / best[run.optimizer]) ** 2)
    # Use a center optimum below geometric midpoint so it is bracketed; first case
    # had x2 winner, now x4 provides the upper neighbor.
    assert cli.main(["transfer-check", *args, *root_arg, *selection_arg, "--extra-factor", "adamw=4"]) == 0
    final_file = out / "lr_selection_3b.json"
    final = json.loads(final_file.read_text())
    assert final["adamw"]["full_lr"] == pytest.approx(best["adamw"])
    assert final["adamw"]["full_lr_source"] == "transfer-check vertex"
    assert final["muon"]["full_lr"] == full["muon"]
    assert final["muon"]["full_lr_source"] == "transferred"
    assert cli.main(["jobs", "--stage", "main", *args, "--selection", str(final_file)]) is None
    assert [p.name for p in out.glob("e1-main-*.sbatch")] == ["e1-main-adamw-s42.sbatch"]
    main = RunSpec("main", "adamw", final["adamw"]["full_lr"], n_experts=1)
    center = RunSpec("xfer", "muon", final["muon"]["full_lr"], n_experts=1, retain_checkpoints=True)
    adamw_path = fake_checkpoint(root, main)
    with pytest.raises(FileNotFoundError, match="missing final model weights"):
        cli.main(["base-manifest", *args, *root_arg, "--selection", str(final_file)])
    muon_path = fake_checkpoint(root, center)
    assert cli.main(["base-manifest", *args, *root_arg, "--selection", str(final_file)]) == 0
    manifest = out / "base_manifest.json"
    pins = load_base_manifest(manifest)
    assert pins["e1-main-adamw-s42"].checkpoint_dir == adamw_path
    assert pins["e1-main-muon-s42"].checkpoint_dir == muon_path
    assert pins["e1-main-muon-s42"].model_sha256 == hashlib.sha256(b"test model weights").hexdigest()
    assert resolve_base_pin("e1-main-muon-s42", manifest) == pins["e1-main-muon-s42"]
    script = render_finetune_job("test", "e1-main-muon-s42", "wikitext103", "/data/wt",
                                 "/data/d0", "adamw", [.001], "1:00:00", "/repo", base_manifest=manifest)
    assert f"--base-manifest {manifest}" in script
    assert script.count(f"--base-manifest {manifest}") == 2  # training and logging
    assert f"--base-checkpoints-dir {root}" in script
    with pytest.raises(SystemExit, match="expert count mismatch"):
        cli.main(["jobs", "--stage", "main", "--experts", "4", "--selection", str(final_file)])
    with pytest.raises(ValueError, match="pinned base manifest"):
        resolve_base_pin("e1-main-muon-s42")
    assert base_run_experts("e16-main-adamw-s42") == 16
    old = "e8-main-adamw-s42"
    assert resolve_base_pin(old).model_sha256 == FINETUNE_BASE_CHECKPOINTS[old][1]


def test_manifest_rejects_wrong_identity_path_and_hash(tmp_path):
    run = "e4-main-adamw-s42"
    pin = {"checkpoint_dir": str(tmp_path / run), "step": FINETUNE_BASE_STEP,
           "optimizer": "adamw", "n_experts": 4, "seed": 42, "matrix_lr": .001,
           "model_sha256": "a" * 64}
    for field, value in (("n_experts", 16), ("step", 11444), ("model_sha256", "bad"),
                         ("checkpoint_dir", "/some/e4-xfer-muon-lr0.001-s42"),
                         ("checkpoint_dir", "/some/e4-xfer-adamw-lr0.002-s42")):
        manifest = {"version": 1, "bases": {run: {**pin, field: value}}}
        path = tmp_path / "base_manifest.json"
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError):
            load_base_manifest(path)
