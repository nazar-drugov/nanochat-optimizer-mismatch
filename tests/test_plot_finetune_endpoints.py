"""Seed-level statistics and smoke tests for the E=8 endpoint report figure."""

import csv
from dataclasses import asdict, replace

import pytest

from scripts import plot_finetune_endpoints as p


def make_runs(dataset="starcoderdata", steps=960, pair=("adamw", "muon"), bases=(42, 43, 44)):
    return [p.Run(dataset, steps, *pair, base, ft, .001,
                  learning=base - 42 + (ft - 2) * .25,
                  forgetting=2 * (base - 42) + (ft - 2) * .75,
                  source="main" if len(bases) == 3 else "pilot_long")
            for base in bases for ft in (1, 2, 3)]


def test_endpoint_sd_is_across_base_seed_means():
    point = p.summarize(make_runs())
    assert point.n_bases == 3 and point.n_ft_seeds_per_base == 3
    assert (point.learning, point.learning_sd) == pytest.approx((1, 1))
    assert (point.forgetting, point.forgetting_sd) == pytest.approx((2, 2))


def test_single_base_and_seed_42_anchor_use_finetune_seed_sd():
    single = p.summarize(make_runs(bases=(42,)))
    anchor = p.summarize(make_runs(), base_seed_only=42)
    assert single == anchor
    assert single.n_bases == 1 and single.n_ft_seeds_per_base == 3
    assert (single.learning, single.learning_sd) == pytest.approx((0, .25))
    assert (single.forgetting, single.forgetting_sd) == pytest.approx((0, .75))


def test_selected_lr_and_complete_seeds_are_validated():
    runs = make_runs()
    with pytest.raises(ValueError, match="FT seeds"):
        p.summarize(runs[:-1])
    with pytest.raises(ValueError, match="base seeds"):
        p.summarize(runs[:6])
    with pytest.raises(ValueError, match="multiple selected LRs"):
        p.summarize([replace(runs[0], lr=.002), *runs[1:]])


def write_csv(path, runs):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=p.FIELDS)
        writer.writeheader()
        writer.writerows(asdict(r) for r in runs)


def test_duplicate_run_cannot_be_counted_twice_even_with_different_source(tmp_path):
    path = tmp_path / "duplicates.csv"
    runs = make_runs()
    write_csv(path, [*runs, replace(runs[0], source="pilot_long")])
    with pytest.raises(ValueError, match="duplicate observation"):
        p.read_runs(path)


def test_plot_includes_all_observed_lengths_and_has_seed42_anchor(tmp_path, monkeypatch):
    runs = []
    for dataset in p.DATASETS:
        lengths = p.MAIN_LENGTHS[dataset]
        for pair in p.PAIRS:
            for steps in lengths:
                runs += make_runs(dataset, steps, pair)
            if dataset == "starcoderdata":
                runs += make_runs(dataset, 1920, pair, bases=(42,))
                runs += make_runs(dataset, 3200, pair, bases=(42,))
    path = tmp_path / "endpoints.csv"
    write_csv(path, runs)
    groups = p.group_runs(p.read_runs(path))
    assert p.collect_points(groups, "starcoderdata")[0] == [60, 240, 960, 1920, 3200]
    assert p.collect_points(groups, "starcoderdata", include_1920=False)[0] == [60, 240, 960, 3200]
    assert p.collect_points(groups, "wikitext103")[0] == [60, 240, 720]

    seen = []
    original = p._draw_series

    def record(ax, points, color, marker, **kwargs):
        if kwargs.get("pilot"):
            seen.append(points)
        return original(ax, points, color, marker, **kwargs)

    monkeypatch.setattr(p, "_draw_series", record)
    target = p.plot(path, tmp_path / "endpoints.png")
    assert target.stat().st_size > 10_000
    assert len(seen) == 4
    for points in seen:
        assert [t for t, _ in points] == [960, 1920, 3200]
        assert points[0][1].n_bases == 1
        assert points[0][1].learning == 0  # not the replicated endpoint mean of 1


def test_mixed_replication_in_one_length_is_rejected():
    runs = [r for t in (60, 240, 960) for pair in p.PAIRS for r in make_runs(steps=t, pair=pair)]
    runs += [r for pair in p.PAIRS for r in make_runs(steps=3200, pair=pair, bases=(42,) if pair != p.PAIRS[0] else (42, 43, 44))]
    with pytest.raises(ValueError, match="mixed replication"):
        p.collect_points(p.group_runs(runs), "starcoderdata")


def test_pilot_only_length_must_extend_main_lengths():
    runs = [r for t in (60, 240, 960) for pair in p.PAIRS for r in make_runs(steps=t, pair=pair)]
    runs += [r for pair in p.PAIRS for r in make_runs(steps=120, pair=pair, bases=(42,))]
    with pytest.raises(ValueError, match="must extend"):
        p.collect_points(p.group_runs(runs), "starcoderdata")
