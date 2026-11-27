"""Validation trajectory extraction, hierarchical SD, and eight-panel-pair export."""

import json
from pathlib import Path

import pytest

from scripts import finetune_validation_trajectories as t


def fixture(tmp_path, *, loss0=3., loss1=4.5, base_seed=42, ft_seed=1, horizon=60,
            pre="adamw", ft="muon", dataset="starcoderdata"):
    base = f"e8-main-{pre}-s{base_seed}"
    path = tmp_path / base / dataset / f"{ft}-lr0.001-s{ft_seed}-T{horizon}"
    path.mkdir(parents=True, exist_ok=True)
    cell = dict(base_run=base, ft_dataset=dataset, pretrain_optimizer=pre, ft_optimizer=ft,
                lr=.001, seed=ft_seed, ft_steps=horizon)
    curve = [dict(step=2, loss_d0=loss0+.125, loss_d1=loss1-.25, learning=.25, forgetting=.125),
             dict(step=horizon, loss_d0=loss0+.25, loss_d1=loss1-.5, learning=.5, forgetting=.25)]
    result = dict(cell, steps=horizon, status="completed", steps_completed=horizon,
                  curve=curve, final_learning=.5, final_forgetting=.25)
    config = {"cell": cell, "steps": horizon, "checkpoint_steps": [2, horizon]}
    (path / "result.json").write_text(json.dumps(result))
    (path / "config.json").write_text(json.dumps(config))
    endpoint = {"dataset": dataset, "steps": horizon, "pretrain_optimizer": pre, "ft_optimizer": ft,
                "base_seed": base_seed, "ft_seed": ft_seed, "lr": .001, "learning": .5, "forgetting": .25,
                "source": "main", "source_run": str(path)}
    return endpoint, path


def test_extract_recovers_exact_step_zero_and_raw_endpoint_nll(tmp_path):
    e, _ = fixture(tmp_path)
    rows, enriched = t.extract([e])
    assert [r["checkpoint_step"] for r in rows] == [0, 2, 60]
    assert rows[0]["loss_d0"] == 3 and rows[0]["loss_d1"] == 4.5
    assert rows[0]["learning"] == rows[0]["forgetting"] == 0
    assert enriched[0]["loss_d0"] == 3.25 and enriched[0]["loss_d1"] == 4
    assert enriched[0]["base_loss_d0"] == 3 and enriched[0]["base_loss_d1"] == 4.5


def test_extraction_rejects_endpoint_mismatch_or_inconsistent_baseline(tmp_path):
    e, path = fixture(tmp_path)
    with pytest.raises(ValueError, match="endpoint learning mismatch"):
        t.extract([dict(e, learning=.51)])
    result = json.loads((path / "result.json").read_text())
    result["curve"][0]["loss_d0"] += .25
    (path / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="inconsistent step-0 baseline"):
        t.extract([e])


def test_across_horizons_baseline_is_validated(tmp_path):
    a, _ = fixture(tmp_path, horizon=60)
    b, _ = fixture(tmp_path, horizon=240, loss0=3.1)
    with pytest.raises(ValueError, match="inconsistent step-0 baseline across curves"):
        t.extract([a, b])


def test_nazar_staging_alias_maps_to_exact_original_mirror(tmp_path):
    e, directory = fixture(tmp_path / "nazar" / "finetune_pilot", dataset="wikitext103", base_seed=43, horizon=720)
    unsuffixed = directory.with_name(directory.name.removesuffix("-T720"))
    directory.rename(unsuffixed)
    directory = unsuffixed
    source = t.HGO_SCRATCH / "analysis" / "staging_nazar_wt" / Path(*directory.parts[-3:])
    assert t.resolve_source(source, tmp_path) == directory
    with pytest.raises(ValueError, match="archived/invalid"):
        t.resolve_source(t.HGO_SCRATCH / "finetune_pilot_v1_replaybug" / Path(*directory.parts[-3:]), tmp_path)


def test_hierarchical_sd_uses_base_means_or_seed42_ft_spread():
    rows = [dict(base_seed=b, ft_seed=f, loss_d0=3+(b-42)+.125*(f-2),
                 loss_d1=4+2*(b-42)+.25*(f-2)) for b in (42, 43, 44) for f in (1, 2, 3)]
    p = t.summarize_point(rows, 60)
    assert (p.n_bases, p.loss_d0, p.sd_d0, p.loss_d1, p.sd_d1) == pytest.approx((3, 4, 1, 6, 2))
    p = t.summarize_point(rows[:3], 60)
    assert (p.n_bases, p.loss_d0, p.sd_d0, p.loss_d1, p.sd_d1) == pytest.approx((1, 3, .125, 4, .25))


def synthetic_grid():
    rows = []
    for dataset in t.DATASETS:
        horizons = (60, 240, 960, 1920, 3200) if dataset == "starcoderdata" else (60, 240, 720, 1920, 3200)
        for pre, ft in t.PAIRS:
            for horizon in horizons:
                bases = (42,) if dataset == "wikitext103" and horizon > 720 else (42, 43, 44)
                for base in bases:
                    for seed in (1, 2, 3):
                        for step in (0, 2, horizon):
                            rows.append(dict(dataset=dataset, steps=horizon, pretrain_optimizer=pre, ft_optimizer=ft,
                                             base_seed=base, ft_seed=seed, checkpoint_step=step, lr=.001,
                                             loss_d0=3+(base-42)*.01+step/10000,
                                             loss_d1=4+(base-42)*.01-step/10000))
    return rows


def test_aggregate_rejects_missing_checkpoint_or_mixed_lr():
    rows = synthetic_grid()
    with pytest.raises(ValueError, match="mixed saved checkpoint schedules"):
        t.aggregate(rows[1:])
    with pytest.raises(ValueError, match="mixed/invalid selected LR"):
        t.aggregate([dict(rows[0], lr=.002), *rows[1:]])
    with pytest.raises(ValueError, match="unexpected selected horizon/replication coverage"):
        t.aggregate([row for row in rows if row["steps"] != 3200])


def test_eight_figures_and_common_y_limits_include_errors(tmp_path):
    rows = synthetic_grid()
    series = t.aggregate(rows)
    for dataset, pre, ft in series:
        assert len(series[dataset, pre, ft]) == 5
        for s in series[dataset, pre, ft]:
            expected_bases = 1 if dataset == "wikitext103" and s.horizon > 720 else 3
            assert s.n_bases == expected_bases
            for p in s.points:
                for field in ("d0", "d1"):
                    lo, hi = t.common_limits(series, dataset, field)
                    assert lo <= getattr(p, "loss_"+field)-getattr(p, "sd_"+field)
                    assert hi >= getattr(p, "loss_"+field)+getattr(p, "sd_"+field)
    outputs = t.plot_all(rows, tmp_path)
    assert len(outputs) == len(set(outputs)) == len(list(tmp_path.glob("*.png"))) == 8
    assert all(path.stat().st_size > 10_000 for path in outputs)
