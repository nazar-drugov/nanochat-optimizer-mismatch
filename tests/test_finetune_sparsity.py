"""Regression tests for per-E pins, selections, replay, provenance and support."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nanochat.pretrain_sweep import RunSpec, load_base_manifest
from scripts import analyze_finetune as a, finetune as f, finetune_jobs as j
from scripts import log_finetune_wandb as w, export_finetune_sparsity as x
from nanochat.gpt import GPT, GPTConfig
from nanochat.eval_set import EvalSet


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def make_pins(tmp, E=1):
    bases, selection = {}, {}
    for opt, lr in (("adamw", .002), ("muon", .02)):
        name = f"e{E}-main-{opt}-s42"
        directory = tmp / RunSpec("xfer", opt, lr, 42, E).name
        directory.mkdir(parents=True, exist_ok=True)
        payload = (name + " weights").encode()
        (directory / "model_011445.pt").write_bytes(payload)
        model = {"n_experts": E, "ffn_type": "moe", "disable_routing_loss": E == 1,
                 "balance_coef": 0 if E == 1 else .01, "z_coef": 0 if E == 1 else .001}
        user = {"seed": 42, "optimizer_name": opt, f"{opt}_matrix_lr": lr, "disable_routing_loss": E == 1}
        dump(directory / "meta_011445.json", {"step": 11445, "model_config": model, "user_config": user})
        bases[name] = dict(checkpoint_dir=str(directory), step=11445, optimizer=opt, seed=42,
                           n_experts=E, matrix_lr=lr, model_sha256=hashlib.sha256(payload).hexdigest())
        selection[opt] = dict(status="selected", n_experts=E, seed=42, full_lr=lr, full_lr_source="transferred")
    manifest, selected = tmp/"pins.json", tmp/"selection.json"
    dump(manifest, {"version": 1, "bases": bases})
    dump(selected, selection)
    return manifest, selected, bases


@pytest.mark.parametrize("E", [1, 4, 16])
def test_job_generator_manifest_and_dataset_lengths(tmp_path, E):
    manifest, selected, bases = make_pins(tmp_path, E)
    out = tmp_path/"job"
    args = ["--selection", str(selected), "--base-manifest", str(manifest), "--base-runs", *bases,
            "--dataset", "starcoderdata=/sc", "--dataset", "wikitext103=/wt", "--d0-file", "/d0",
            "--out-root", f"/scratch/finetune_pilot_e{E}", "--out", str(out)]
    j.main(args)
    jobs = list(out.glob("*.sbatch"))
    assert len(jobs) == 8
    for job in jobs:
        body = job.read_text()
        assert str(manifest) in body and "--base-manifest" in body
        assert "--seeds 1 2 3" in body
        assert "--checkpoint-retention full" in body
        assert "--soft-router-capture --soft-router-sample-size 8192 --soft-router-sample-format logits" in body
        assert "--branch-tokens 3932160 " + ("62914560" if "starcoderdata" in job.name else "47185920") in body
    out2 = tmp_path/"edge"
    j.main(args + ["--out", str(out2), "--pairs", "adamw:muon", "--pair-lr-fractions", "adamw:muon=2", "--checkpoint-retention", "weights"])
    assert len(list(out2.glob("*.sbatch"))) == 2
    assert all("--checkpoint-retention weights" in p.read_text() for p in out2.glob("*.sbatch"))
    wrong = json.loads(selected.read_text()); wrong["adamw"]["full_lr"] *= 2; dump(selected, wrong)
    with pytest.raises(SystemExit):
        j.main(args)


def test_sha_and_transfer_alias_verified_before_compute(tmp_path):
    manifest, _, bases = make_pins(tmp_path)
    name = next(iter(bases))
    args = SimpleNamespace(base_run=name, base_manifest=manifest, base_checkpoints_dir=None, base_step=None)
    directory, step, model, sha, pin = f.resolve_finetune_base(args, argparse.ArgumentParser())
    assert directory.name != name and str(directory) == bases[name]["checkpoint_dir"]
    assert sha == bases[name]["model_sha256"] and step == 11445
    model.write_bytes(b"corrupt")
    with pytest.raises(SystemExit):
        f.resolve_finetune_base(args, argparse.ArgumentParser())


def test_e1_replay_is_exact_and_margin_undefined():
    config = GPTConfig(sequence_len=8, vocab_size=32, n_layer=2, n_head=2, n_kv_head=2, n_embd=32,
                       ffn_type="moe", n_experts=1, disable_routing_loss=True)
    model0, model1 = GPT(config), GPT(config)
    model0.init_weights(seed=3); model1.init_weights(seed=4)
    inputs = torch.zeros(2, 8, dtype=torch.long)
    targets = torch.ones_like(inputs)
    ids = torch.tensor([[0]*8, [1]*8])
    eval_set = EvalSet("D0", inputs, targets, ids, 2, "/d0", "identity", packing="bestfit",
                       doc_rows=torch.tensor([0, 1]))
    base = f.evaluate_base(model0, eval_set, 1, torch.device("cpu"), soft_sample_size=3)
    result = f.evaluate_checkpoint(model1, model0, eval_set, base, 1, torch.device("cpu"), full=True)
    assert all(v is None for v in base["margin"].values())
    assert torch.equal(result["doc_sums"]["normal"], result["doc_sums"]["replay"])
    assert torch.equal(result["doc_sums"]["reverse"], base["doc_sums"])
    assert all(v["flip_rate"] == 0 and v["flip_rate_normalized"] is None for v in result["per_layer"].values())
    assert all(v["cause_scored_flips"] == 0 and v["cause_flips"] == 0 for v in result["per_layer"].values())
    assert all(v["soft_top2_margin_mean"] is None and v["soft_js_base_current_nats_mean"] == 0.
               and v["soft_tv_base_current_mean"] == 0. for v in result["per_layer"].values())


def synthetic_grid(tmp, pins, E=1, dataset="starcoderdata", horizon=60):
    root = tmp / "results"
    for pre in ("adamw", "muon"):
        base = f"e{E}-main-{pre}-s42"; pin = pins[base]
        meta = json.loads((pin.checkpoint_dir/"meta_011445.json").read_text())
        for ft in ("adamw", "muon"):
            for i, lr in enumerate((.001, .002, .003, .004, .005)):
                for seed in (1, 2, 3):
                    cell = dict(base_run=base, pretrain_optimizer=pre, ft_optimizer=ft, seed=seed, lr=lr,
                                ft_steps=horizon, ft_dataset=dataset, n_experts=E)
                    checkpoints = f.checkpoint_steps(horizon, round(.2*horizon))
                    curve = []
                    for t in checkpoints:
                        learning = (.8 - .1*abs(i-2))*t/horizon
                        forgetting = .01*t/horizon
                        curve.append(dict(step=t, tokens=t*65536, learning=learning, forgetting=forgetting,
                                          loss_d0=3+forgetting, loss_d1=4-learning,
                                          F_route=0., F_fixed=forgetting, reverse_order=0.))
                    result = dict(cell, status="completed", steps=horizon, steps_completed=horizon, curve=curve,
                                  final_learning=curve[-1]["learning"], final_forgetting=curve[-1]["forgetting"],
                                  data_fingerprint=seed, warmup_steps=8, decay_steps=round(.2*horizon))
                    ident = dict(dir=str(pin.checkpoint_dir), step=11445, model_sha256=pin.model_sha256,
                                 model_file=str(pin.checkpoint_dir/"model_011445.pt"), logical_run=base,
                                 checkpoint_run=pin.checkpoint_dir.name, n_experts=E, user_config=meta["user_config"])
                    data = dict(ft_train_files={"train_0.parquet": "sha"}, eval_sets={
                        d: dict(sha256=d, rows=1, seq_len=8, scored_tokens=8, documents=1, packing="bestfit")
                        for d in ("D0", "D1")})
                    config = dict(stage="finetune", cell=cell, base=ident, model_config=meta["model_config"], data=data,
                                  args={"total_batch_size": 65536, "device_batch_size": 32}, steps=horizon,
                                  warmup_steps=8, decay_steps=round(.2*horizon), checkpoint_steps=checkpoints,
                                  moe_layer_indices=[0])
                    path = root/base/dataset/f"{ft}-lr{lr}-s{seed}-T{horizon}"
                    dump(path/"config.json", config); dump(path/"result.json", result)
    return root


def test_selection_and_export_verify_grid_and_exact_identity(tmp_path):
    manifest, _, entries = make_pins(tmp_path)
    pins = load_base_manifest(manifest)
    root = synthetic_grid(tmp_path, pins)
    argv = ["--root", str(root), "--base-manifest", str(manifest), "--bases", *entries,
            "--dataset", "starcoderdata", "--steps", "60", "--seeds", "1", "2", "3",
            "--out", str(tmp_path/"analysis")]
    a.main(argv)
    path = tmp_path/"analysis"/"analysis.json"
    saved = json.loads(path.read_text())
    assert saved["n_experts"] == 1 and saved["status"] == "grid_brackets_optima"
    assert saved["router_correlations"] == {}
    _, endpoints, trajectories = x.export_analysis(path, pins, x.DataIdentity())
    assert len(endpoints) == 12 and {v["lr"] for v in endpoints} == {.003}
    assert all(v["base_checkpoint_run"].startswith("e1-xfer-") for v in endpoints)
    assert all(v["base_loss_d0"] == pytest.approx(3) for v in endpoints)
    assert len([p for p in trajectories if p["checkpoint_step"] == 0]) == 12
    config_path = next(root.rglob("config.json"))
    config = json.loads(config_path.read_text())
    config["base"]["model_sha256"] = "0"*64; dump(config_path, config)
    with pytest.raises(ValueError, match="SHA-256"):
        x.export_analysis(path, pins, x.DataIdentity())


def test_wandb_new_e_requires_pin_and_no_paths(tmp_path):
    manifest, _, _ = make_pins(tmp_path)
    pins = load_base_manifest(manifest)
    root = synthetic_grid(tmp_path, pins)
    path = next(root.rglob("result.json"))
    result, config = [json.loads(p.read_text()) for p in (path, path.with_name("config.json"))]
    with pytest.raises(w.RecordError, match="pinned base manifest"):
        w.make_cell(result, config, stage="finetune_pilot_e1")
    cell = w.make_cell(result, config, stage="finetune_pilot_e1", base_manifest=pins)
    assert cell.group.startswith("optimizer-mismatch-e1/")
    assert cell.metadata["base_model_sha256"] == config["base"]["model_sha256"]
    assert str(tmp_path) not in json.dumps(cell.metadata)
    from nanochat.soft_routing import METRICS, schema
    config["soft_router"] = result["soft_router"] = schema(8192)
    for row in result["curve"]:
        for d in ("d0", "d1"):
            for metric in METRICS:
                row[f"soft_{d}_{metric}_mean"] = (None if metric=="top2_margin" else
                    1. if metric in ("chosen_gate","max_probability") else 0.)
    rich = w.make_cell(result, config, stage="finetune_pilot_e1", base_manifest=pins)
    assert "soft_d0_top2_margin_mean" not in rich.points[0]
    assert rich.points[0]["soft_d1_chosen_gate_mean"] == 1.
    assert rich.summary["nonfinite_measurement_count"] == 0
    result["curve"][0]["F_route"] = .1
    with pytest.raises(w.RecordError, match="structural zero"):
        w.make_cell(result, config, stage="finetune_pilot_e1", base_manifest=pins)


def test_cross_e_support_uses_intersection_no_partial_contrast():
    rows, traces = [], []
    for E in (1, 4):
        for ds, lengths in x.HORIZONS.items():
            for i, length in enumerate(lengths):
                for pre in ("adamw", "muon"):
                    for ft in ("adamw", "muon"):
                        for seed in (1, 2, 3):
                            l = (i+1)*.2 if E == 1 else (i+1)*.3
                            p = dict(n_experts=E, dataset=ds, steps=length, pretrain_optimizer=pre, ft_optimizer=ft,
                                     base_seed=42, ft_seed=seed, lr=.01, learning=l, forgetting=.02, source="pilot_seed42",
                                     source_run=f"{E}/{ds}/{length}/{pre}/{ft}/{seed}")
                            rows.append(p)
                            traces.extend([dict(p, checkpoint_step=0, learning=0., forgetting=0.),
                                           dict(p, checkpoint_step=length)])
    _, common = x.common_support(rows, traces)
    endpoints = [r for r in common if r["estimand"]=="endpoint"]
    assert len(endpoints)==2
    assert all(r["support_min"]==pytest.approx(.3) and r["support_max"]==pytest.approx(.6) for r in endpoints)
    summaries, seeds, selections = x.explicit_target_analysis(rows, traces, common, {"endpoint": {ds: .7 for ds in x.HORIZONS}})
    assert len(summaries)==8 and not seeds and not selections
    assert all(r["status"]=="unsupported_across_e" for r in summaries)
