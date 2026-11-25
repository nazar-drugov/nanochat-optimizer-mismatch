"""Analysis tests: LR selection, threshold geometry, paired resampling, and file validation."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_finetune as a


def point(step, learning, forgetting, loss_d1=1.0):
    return {"step": step, "learning": learning, "forgetting": forgetting,
            "F_route": forgetting * .25, "F_fixed": forgetting * .75,
            "loss_d1": loss_d1}


def fake_run(base, ft, seed, lr, loss):
    return a.Run(Path("/unused"), base, ft, seed, lr, 2,
                 [point(1, .2, .1, loss+.1), point(2, .5, .2, loss)], {})


def test_matched_learning_uses_interpolation_and_minimum_feasible_forgetting():
    curve = [point(2, .2, .1), point(4, .8, .5), point(6, .9, .7)]
    m = a.match_curve(curve, .5)
    assert m["step"] == pytest.approx(3) and m["forgetting"] == pytest.approx(.3)
    assert m["F_route"] + m["F_fixed"] == pytest.approx(m["forgetting"])
    curve.append(point(8, 1.0, .15))
    m = a.match_curve(curve, .5)
    assert m["step"] == 8 and m["forgetting"] == pytest.approx(.15)
    assert a.match_curve(curve, 1.1) is None
    assert a.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)


def test_lr_selection_requires_paired_complete_grid():
    runs = [fake_run(base, ft, seed, lr, 1.0 if lr == .01 else 1.2)
            for base in a.BASES for ft in a.OPTS for seed in (1, 2) for lr in (.01, .02)]
    selected, rows = a.select_lrs(runs, [1, 2], 2)
    assert all(lr == .01 for lr in selected.values())
    assert len(rows) == 8 and all(r["edge"] for r in rows if r["selected"])
    with pytest.raises(ValueError, match="missing or duplicate seeds"):
        a.select_lrs(runs[:-1], [1, 2], 2)


def test_paired_document_bootstrap_keeps_common_samples(tmp_path):
    runs = []
    for base in a.BASES:
        for ft in a.OPTS:
            for seed in (1, 2):
                path = tmp_path / base / "data" / f"{ft}-s{seed}"
                path.mkdir(parents=True)
                baseline = path.parent / "base"
                baseline.mkdir(exist_ok=True)
                np.savez_compressed(baseline / "base_arrays.npz",
                                    d0_doc_loss=[1., 1.], d0_doc_tokens=[1., 1.],
                                    d1_doc_loss=[1., 1.], d1_doc_tokens=[1., 1.])
                extra = .1 if ft != base.split("-")[2] else 0.
                np.savez_compressed(path / "ckpt_000001.npz",
                                    d0_doc_loss=[1.+extra]*2, d0_doc_loss_replay=[1.+extra*.75]*2,
                                    d1_doc_loss=[.5]*2)
                runs.append(a.Run(path, base, ft, seed, .01, 1,
                                  [point(1, .5, extra, .5)], {}))
    ci = a.bootstrap(runs, [1, 2], .2, 30, 0)
    assert ci["accepted"] == 30
    for interval in ci["mismatch_ci"].values():
        assert interval == pytest.approx([.04, .04])


def test_analysis_reads_complete_grid_and_rejects_wrong_base(tmp_path):
    root = tmp_path / "runs"
    for base in a.BASES:
        for ft in a.OPTS:
            for lr in (.01, .02):
                for seed in (1, 2):
                    path = root / base / "data" / f"{ft}-lr{lr}-s{seed}"
                    path.mkdir(parents=True)
                    cell = {"base_run": base, "ft_optimizer": ft, "seed": seed,
                            "lr": lr, "ft_steps": 2}
                    config = {"cell": cell, "moe_layer_indices": [0],
                              "model_config": {"n_experts": 8},
                              "base": {"step": 11445, "user_config": {"seed": 42,
                                      "optimizer_name": base.split("-")[2]}},
                              "args": {"total_batch_size": 65536},
                              "data": {"eval_sets": {d: {"sha256": d, "scored_tokens": 100}
                                                     for d in ("D0", "D1")}}}
                    result = {**cell, "ft_dataset": "data", "status": "completed",
                              "curve": [point(1, .2, .1, .8),
                                        point(2, .5, .2, .5 if lr == .01 else .6)]}
                    (path / "config.json").write_text(json.dumps(config))
                    (path / "result.json").write_text(json.dumps(result))
                    baseline = path.parent / "base"
                    baseline.mkdir(exist_ok=True)
                    np.savez_compressed(baseline / "base_arrays.npz",
                                        d0_doc_loss=[1., 1.], d0_doc_tokens=[1., 1.],
                                        d1_doc_loss=[1., 1.], d1_doc_tokens=[1., 1.])
                    for step, d0, d1 in ((1, 1.1, .8), (2, 1.2, .5)):
                        np.savez_compressed(path / f"ckpt_{step:06d}.npz",
                                            d0_doc_loss=[d0]*2, d0_doc_loss_replay=[1.+(d0-1)*.75]*2,
                                            d1_doc_loss=[d1]*2)
                    rows = [{"eval_set": "D0", "layer": 0, "step": step,
                             "base_maxvio": .1, "flip_rate": step*.1, "maxvio": .1+step*.1,
                             "tv": step*.05, "churn": step*.05,
                             "flip_rate_normalized": step*.2} for step in (1, 2)]
                    (path / "metrics.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    out = tmp_path / "out"
    a.main(["--root", str(root), "--dataset", "data", "--steps", "2", "--seeds", "1", "2",
            "--min-lrs", "2", "--out", str(out)])
    summary = json.loads((out / "analysis.json").read_text())
    assert len(summary["source_runs"]) == 16
    assert summary["common_learning_support"] == [0., .5]
    assert (out / "lr_selection.png").is_file() and (out / "learning_forgetting.png").is_file()
    second_root = tmp_path / "other_runs"
    second_root.mkdir()
    shutil.move(str(root / a.BASES[1]), str(second_root / a.BASES[1]))
    a.main(["--root", str(root), "--root", str(second_root), "--dataset", "data",
            "--steps", "2", "--seeds", "1", "2", "--min-lrs", "2",
            "--ell0", ".1", "--bootstrap", "20", "--out", str(out)])
    summary = json.loads((out / "analysis.json").read_text())
    assert len(summary["matched_runs"]) == 8 and len(summary["contrasts"]) == 2
    assert summary["bootstrap"]["accepted"] == 20
    assert (out / "router_diagnostics.png").is_file() and (out / "replay_terms.png").is_file()
    assert (out / "router_vs_forgetting.png").is_file() and (out / "router_runs.csv").is_file()
    wrong = next(root.glob("e8-main-adamw-s42/data/*/config.json"))
    config = json.loads(wrong.read_text())
    config["base"]["step"] = 10
    wrong.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="wrong base checkpoint"):
        a.main(["--root", str(root), "--root", str(second_root), "--dataset", "data",
                "--steps", "2", "--seeds", "1", "2", "--min-lrs", "2", "--out", str(out)])
