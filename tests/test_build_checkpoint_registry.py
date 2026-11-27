"""Checkpoint registry tests use only tiny fake files; no cluster access required."""

import csv
import json
from pathlib import Path

from scripts import build_checkpoint_registry as r


def roots(tmp_path):
    items = [tmp_path / name for name in ("hgo_scratch", "hgo_pool", "nazar_ft", "nazar_legacy")]
    for path in items:
        path.mkdir()
    return items


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real checkpoint")
    return path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_ft(root, *, stage, base_seed=42, length=60, suffix=True, completed=True,
            model=None, recorded_model=None, sha=None, n_experts=8):
    base_name, dataset = f"e{n_experts}-main-adamw-s{base_seed}", "wikitext103"
    before = root / stage if stage else root
    directory = before / base_name / dataset / ("muon-lr0.00044-s1" + (f"-T{length}" if suffix else ""))
    cell = {"base_run": base_name, "pretrain_optimizer": "adamw", "ft_dataset": dataset,
            "ft_optimizer": "muon", "lr": .000443, "seed": 1, "n_experts": n_experts, "ft_steps": length}
    base = {"step": 11445, "user_config": {"seed": base_seed, "optimizer_name": "adamw"}}
    if model or recorded_model:
        base["model_file"] = str(model or recorded_model)
    if sha:
        base["model_sha256"] = sha
    write_json(directory / "config.json", {"cell": cell, "steps": length, "base": base,
                                        "branch": {"lengths": [60, length]} if suffix else None})
    touch(directory / "ckpt_000002.npz")
    if completed:
        write_json(directory / "result.json", dict(cell, status="completed", steps=length, steps_completed=length))
        touch(directory / f"ckpt_{length:06d}.npz")
    return directory


def indexed(rows):
    return {row["run_key"]: row for row in rows}


def test_primary_model_checkpoint_and_pool_mirror_no_metadata_reads(tmp_path, monkeypatch):
    scratch, pool, nazar_ft, nazar_legacy = roots(tmp_path)
    directory = scratch / "base_checkpoints" / "e8-main-adamw-s42"
    mirror = pool / "base_checkpoints" / directory.name
    model = touch(directory / "model_011445.pt")
    meta = touch(directory / "meta_011445.json")  # intentionally malformed: never parse it
    optimizer0 = touch(directory / "optim_011445_rank0.pt")
    optimizer2 = touch(directory / "optim_011445_rank2.pt")
    route = touch(directory / "routing_011445.pt")
    pool_meta = touch(mirror / "meta_011445.json")
    touch(mirror / "meta_001526.json")  # only pool retains this step, no weights
    write_json(directory / "result.json", {"status": "completed", "num_iterations": 11445,
                                        "steps_trained": 11445, "settings": {"matrix_lr": .00131}})
    old_read = Path.read_text

    def guard(path, *args, **kwargs):
        if path.name.startswith("meta_"):
            raise AssertionError("must not read checkpoint metadata")
        return old_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guard)
    runs, checks, warnings = r.build_registry(scratch, pool, nazar_ft, nazar_legacy)
    assert not warnings
    assert len(runs) == 1 and len(checks) == 2
    run = runs[0]
    assert run["run_key"] == "hgo:base_checkpoints/e8-main-adamw-s42"
    assert run["stage"] == "pretrain_main" and run["validity"] == "valid"
    assert run["lr"] == .00131 and run["length_steps"] == 11445 and run["base_seed"] == 42
    assert run["result_path"] == str(directory / "result.json") and run["pool_run_dir"] == str(mirror)
    early, last = checks
    assert early["step"] == 1526 and early["availability"] == "pool_only"
    assert early["has_model_weights"] == "no" and early["checkpoint_kind"] == "model_related_only"
    assert last["step"] == 11445 and last["availability"] == "scratch_and_pool"
    assert last["has_model_weights"] == "yes" and last["checkpoint_kind"] == "model_weights"
    assert last["model_path"] == str(model) and last["meta_path"] == str(meta)
    assert last["pool_meta_path"] == str(pool_meta) and last["routing_probe_path"] == str(route)
    assert json.loads(last["optimizer_paths"]) == [str(optimizer0), str(optimizer2)]


def test_finetune_base_reference_and_pool_checkpoint_paths(tmp_path):
    scratch, pool, nazar_ft, nazar_legacy = roots(tmp_path)
    model = touch(scratch / "base_checkpoints" / "e8-main-adamw-s42" / "model_011445.pt")
    cell = make_ft(scratch, stage="finetune_pilot", model=model, sha="abc123")
    mirror = pool / cell.relative_to(scratch)
    touch(mirror / "ckpt_000060.npz")
    baseline = cell.parent / "base"
    base_arrays = touch(baseline / "base_arrays.npz")
    eval_set = touch(baseline / "eval_set_D1.pt")
    write_json(baseline / "base_metrics.json", {"base_step": 11445})
    touch(mirror.parent / "base" / "base_arrays.npz")
    runs, checks, warnings = r.build_registry(scratch, pool, nazar_ft, nazar_legacy)
    assert not warnings and len(runs) == 3  # base weights, branch, shared eval (not an extra experiment)
    branch = next(x for x in runs if x["run_type"] == "finetune")
    assert branch["base_model_path"] == str(model) and branch["base_model_sha256"] == "abc123"
    assert branch["base_model_path_source"] == "recorded_existing"
    assert branch["length_steps"] == 60 and branch["lr"] == .000443  # full LR, not truncated filename
    assert branch["shared_base_eval_dir"] == str(baseline)
    bchecks = [x for x in checks if x["run_key"] == branch["run_key"]]
    assert len(bchecks) == 2
    assert all(x["checkpoint_kind"] == "finetune_eval_arrays" and x["has_model_weights"] == "no" for x in bchecks)
    assert bchecks[-1]["eval_npz_path"] == str(cell / "ckpt_000060.npz")
    assert bchecks[-1]["pool_eval_npz_path"] == str(mirror / "ckpt_000060.npz")
    shared = next(x for x in checks if x["checkpoint_kind"] == "shared_base_eval")
    assert shared["step"] == 0 and shared["base_eval_arrays_path"] == str(base_arrays)
    assert json.loads(shared["base_eval_set_paths"]) == [str(eval_set)]


def test_archive_validity_and_nazar_standalone_no_length_suffix(tmp_path):
    scratch, pool, nazar_ft, nazar_legacy = roots(tmp_path)
    short = make_ft(scratch, stage="finetune_pilot_v1_replaybug", length=60)
    long = make_ft(scratch, stage="finetune_pilot_v1_replaybug", length=240, completed=False)
    contiguous = make_ft(scratch, stage="finetune_pilot_v0_contiguous_eval", length=60)
    nazar_short = make_ft(nazar_ft, stage="", base_seed=42, length=60)
    nazar_long = make_ft(nazar_ft, stage="", base_seed=42, length=240)
    standalone = make_ft(nazar_ft, stage="", base_seed=44, length=720, suffix=False)
    smoke = touch(nazar_legacy / "chatsft_checkpoints" / "verify_chat" / "model_000001.pt")
    transfer = touch(scratch / "base_checkpoints" / "e8-xfer-adamw-lr0.000651-s42" / "model_011445.pt")
    runs, _, warnings = r.build_registry(scratch, pool, nazar_ft, nazar_legacy)
    assert not warnings
    rows = {row["run_dir"]: row for row in runs}
    assert rows[str(short)]["validity"] == "unaffected_archive_excluded"
    assert rows[str(long)]["validity"] == "invalid_replaybug" and rows[str(long)]["status"] == "incomplete"
    assert rows[str(contiguous)]["validity"] == "invalid_contiguous_eval"
    assert rows[str(nazar_short)]["validity"] == "unaffected_archive_excluded"
    assert rows[str(nazar_long)]["validity"] == "invalid_replaybug"
    assert rows[str(standalone)]["stage"] == "finetune_main_reused_nazar"
    assert rows[str(standalone)]["validity"] == "valid" and rows[str(standalone)]["length_steps"] == 720
    assert rows[str(smoke.parent)]["validity"] == "verification_only"
    assert rows[str(transfer.parent)]["stage"] == "pretrain_transfer" and rows[str(transfer.parent)]["validity"] == "valid"


def test_new_sparsity_pilot_roots_are_separate_from_legacy_archive(tmp_path):
    scratch, pool, nazar_ft, nazar_legacy = roots(tmp_path)
    dense = make_ft(scratch, stage="finetune_pilot_e1", n_experts=1, length=240)
    sparse = make_ft(nazar_legacy, stage="finetune_pilot_e16", n_experts=16, length=240)
    wrong = make_ft(scratch, stage="finetune_pilot_e4", n_experts=8, length=60)
    runs, _, warnings = r.build_registry(scratch, pool, nazar_ft, nazar_legacy)
    assert not warnings
    rows = {row["run_dir"]: row for row in runs}
    assert rows[str(dense)]["validity"] == "valid"
    assert rows[str(sparse)]["validity"] == "valid" and rows[str(sparse)]["owner"] == "ndrugov"
    assert rows[str(wrong)]["validity"] == "needs_review"


def test_pool_only_incomplete_and_aliases(tmp_path):
    scratch, pool, nazar_ft, nazar_legacy = roots(tmp_path)
    orphan = make_ft(pool, stage="finetune_main", base_seed=43, length=60)
    current = make_ft(scratch, stage="finetune_main", base_seed=44, length=60, completed=False)
    stale = make_ft(pool, stage="finetune_main", base_seed=44, length=60, completed=True)
    alias = scratch / "finetune_stage_link"
    alias.symlink_to(current.parents[2], target_is_directory=True)
    runs, checks, warnings = r.build_registry(scratch, pool, nazar_ft, nazar_legacy)
    assert not warnings and len(runs) == 2  # no stage-link alias
    rows = {row["run_dir"]: row for row in runs}
    assert rows[str(orphan)]["availability"] == "pool_only" and rows[str(orphan)]["status"] == "completed"
    assert rows[str(current)]["status"] == "incomplete" and "stale" in rows[str(current)]["note"]
    assert rows[str(current)]["pool_result_path"] == str(stale / "result.json")
    assert any(x["step"] == 60 and x["availability"] == "pool_only" for x in checks if x["run_dir"] == str(current))


def test_base_path_fallback_and_lf_csv(tmp_path):
    scratch, pool, nazar_ft, nazar_legacy = roots(tmp_path)
    model = touch(scratch / "base_checkpoints" / "e8-main-adamw-s42" / "model_011445.pt")
    cell = make_ft(scratch, stage="finetune_pilot", recorded_model="/unavailable/recorded/model.pt")
    out = tmp_path / "output"
    assert r.main(["--hgo-scratch", str(scratch), "--hgo-pool", str(pool),
                   "--nazar-finetune", str(nazar_ft), "--nazar-legacy", str(nazar_legacy),
                   "--out", str(out)]) == 0
    for name in ("runs.csv", "checkpoints.csv"):
        assert b"\r\n" not in (out / name).read_bytes()
    with (out / "runs.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    branch = next(x for x in rows if x["run_dir"] == str(cell))
    assert branch["base_model_recorded_path"] == "/unavailable/recorded/model.pt"
    assert branch["base_model_path"] == str(model)
    assert branch["base_model_path_source"] == "shared_hgo_path_inferred"


def test_missing_root_reported_as_partial(tmp_path):
    scratch, pool, nazar_ft, nazar_legacy = roots(tmp_path)
    nazar_legacy.rmdir()
    _, _, warnings = r.build_registry(scratch, pool, nazar_ft, nazar_legacy)
    assert len(warnings) == 1 and "missing or inaccessible" in warnings[0]


def test_empty_primary_directory_prevents_claiming_pool_result_is_current(tmp_path):
    scratch, pool, nazar_ft, nazar_legacy = roots(tmp_path)
    old = make_ft(pool, stage="finetune_main", base_seed=43, length=60)
    current = scratch / old.relative_to(pool)
    current.mkdir(parents=True)
    runs, _, warnings = r.build_registry(scratch, pool, nazar_ft, nazar_legacy)
    assert not warnings and len(runs) == 1
    assert runs[0]["run_dir"] == str(current) and runs[0]["status"] == "incomplete"
    assert "stale" in runs[0]["note"]
