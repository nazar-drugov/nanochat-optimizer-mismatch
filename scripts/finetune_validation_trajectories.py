"""Extract and plot raw held-out NLL for the selected E=8 fine-tuning runs.

The endpoint CSV identifies each clean selected run, including its exact LR and
source path. Recover its authoritative stored curve from ORCD or the local
result/config mirrors, validate identity, endpoint, checkpoint schedule, and
baseline, then write all checkpoint rows and enriched endpoint rows.

  python -m scripts.finetune_validation_trajectories extract \
    --endpoints results/e8/finetune_endpoint_runs.csv \
    --out results/e8/finetune_validation_trajectories.csv \
    --endpoint-out results/e8/finetune_endpoint_runs.csv

  python -m scripts.finetune_validation_trajectories plot \
    --trajectories results/e8/finetune_validation_trajectories.csv \
    --out-dir results/e8/figures/finetune_validation

There are eight plots: one per dataset and optimizer pair, with D0 and D1
panels and a separate selected-LR trajectory per horizon. NLL is the recorded
token-weighted cross entropy (nats/token), not next-token top-1 accuracy.
No training or checkpoint tensor loading is involved.
"""

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from scripts.plot_finetune_endpoints import BASE_SEEDS, DATASETS, FT_SEEDS, NAMES, PAIRS
from scripts.build_checkpoint_registry import HGO_SCRATCH, NAZAR_FINETUNE


DEFAULT_MIRROR = Path(".codex/wandb-backfill")
ENDPOINT_INPUT = ("dataset", "steps", "pretrain_optimizer", "ft_optimizer", "base_seed", "ft_seed",
                  "lr", "learning", "forgetting", "source", "source_run")
NLL_FIELDS = ("loss_d0", "loss_d1", "base_loss_d0", "base_loss_d1")
ENDPOINT_FIELDS = (*ENDPOINT_INPUT, *NLL_FIELDS)
TRAJECTORY_FIELDS = ("dataset", "steps", "pretrain_optimizer", "ft_optimizer", "base_seed", "ft_seed",
                     "lr", "checkpoint_step", "loss_d0", "loss_d1", "learning", "forgetting", "source",
                     "source_run")
HORIZON_COLORS = {60: "#0072B2", 240: "#E69F00", 720: "#009E73", 960: "#009E73",
                  1920: "#AA4499", 3200: "#D55E00"}
HORIZON_MARKERS = {60: "o", 240: "s", 720: "D", 960: "D", 1920: "^", 3200: "v"}
EXPECTED_COVERAGE = {"starcoderdata": ((60, 3), (240, 3), (960, 3), (1920, 3), (3200, 3)),
                     "wikitext103": ((60, 3), (240, 3), (720, 3), (1920, 1), (3200, 1))}


def read_csv(path, required):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if missing := set(required) - set(reader.fieldnames or ()):
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_key(row):
    return (row["dataset"], int(row["steps"]), row["pretrain_optimizer"], row["ft_optimizer"],
            int(row["base_seed"]), int(row["ft_seed"]))


def same(a, b):
    return math.isclose(float(a), float(b), rel_tol=0., abs_tol=1e-10)


def resolve_source(source, mirror_root=DEFAULT_MIRROR):
    """Prefer actual recorded ORCD path; otherwise resolve the exact mirror.

    The only permitted staging alias is the documented Nazar WikiText main-grid
    staging directory. Map it to the original Nazar mirror, never to an LR match
    in a different stage or an archived seed-42 run.
    """
    source = Path(source)
    if any("replaybug" in part or "contiguous_eval" in part for part in source.parts):
        raise ValueError(f"archived/invalid result cannot be a selected trajectory: {source}")
    mirror = Path(mirror_root)
    if source.is_relative_to(HGO_SCRATCH):
        relative = source.relative_to(HGO_SCRATCH)
        parts = relative.parts
        if len(parts) == 5 and parts[:2] == ("analysis", "staging_nazar_wt"):
            candidate = mirror / "nazar" / "finetune_pilot" / Path(*parts[2:])
        elif len(parts) == 4 and parts[0] in {"finetune_pilot", "finetune_main", "finetune_pilot_long",
                                               "finetune_main_long", "finetune_pilot_long_wikitext103"}:
            candidate = mirror / "hgo" / relative
        else:
            raise ValueError(f"unrecognized selected source path {source}")
    elif source.is_relative_to(NAZAR_FINETUNE):
        candidate = mirror / "nazar" / "finetune_pilot" / source.relative_to(NAZAR_FINETUNE)
    else:
        candidate = source
    if source.is_relative_to(NAZAR_FINETUNE) or "staging_nazar_wt" in source.parts:
        base, dataset, cell = candidate.parts[-3:]
        if (not base.endswith(("-s43", "-s44")) or dataset != "wikitext103" or "-T" in cell):
            raise ValueError(f"only the documented clean standalone Nazar WikiText runs are selected: {source}")
    if (source / "result.json").is_file() and (source / "config.json").is_file():
        return source
    if not ((candidate / "result.json").is_file() and (candidate / "config.json").is_file()):
        raise FileNotFoundError(f"selected result/config absent in mirror: {candidate}")
    return candidate


def validate_record(endpoint, result, config):
    dataset, horizon, pre, ft, base_seed, ft_seed = run_key(endpoint)
    base_name = f"e8-main-{pre}-s{base_seed}"
    expected = {"ft_dataset": dataset, "base_run": base_name, "pretrain_optimizer": pre,
                "ft_optimizer": ft, "seed": ft_seed}
    cell = config["cell"]
    if any(result.get(k) != v or cell.get(k) != v for k, v in expected.items()):
        raise ValueError(f"result/config identity mismatch for {run_key(endpoint)}")
    if (float(result["lr"]) != float(endpoint["lr"]) or float(cell["lr"]) != float(endpoint["lr"]) or
            result.get("ft_steps", result["steps"]) != horizon or result["steps"] != horizon or
            cell.get("ft_steps", config["steps"]) != horizon or config["steps"] != horizon or
            result["status"] != "completed" or result["steps_completed"] != horizon):
        raise ValueError(f"wrong LR/length/incomplete selected result for {run_key(endpoint)}")
    curve = result["curve"]
    steps = [int(p["step"]) for p in curve]
    if not steps or steps != sorted(set(steps)) or steps[0] <= 0 or steps[-1] != horizon:
        raise ValueError(f"nonmonotone or incomplete checkpoint curve for {run_key(endpoint)}")
    if "checkpoint_steps" in config and list(config["checkpoint_steps"]) != steps:
        raise ValueError(f"stored curve differs from config checkpoint schedule for {run_key(endpoint)}")
    for field in ("learning", "forgetting"):
        if not (same(endpoint[field], curve[-1][field]) and same(result[f"final_{field}"], curve[-1][field])):
            raise ValueError(f"endpoint {field} mismatch for {run_key(endpoint)}")
    return curve


def extract(endpoints, mirror_root=DEFAULT_MIRROR):
    """Validate and extract every selected trajectory, with exact step-0 NLL.

    The saved loss changes use loss_d0(t)−loss_d0(0) and loss_d1(0)−loss_d1(t).
    Therefore each stored point can recover both baseline losses. Require their
    agreement at every checkpoint and across horizons/seeds for the same base.
    """
    trajectories, enriched = [], []
    seen, baseline_by_base, d0_by_base = set(), {}, {}
    for endpoint in endpoints:
        key = run_key(endpoint)
        if key in seen:
            raise ValueError(f"duplicate selected endpoint {key}")
        seen.add(key)
        directory = resolve_source(endpoint["source_run"], mirror_root)
        result = json.loads((directory / "result.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        curve = validate_record(endpoint, result, config)
        prefix = {field: endpoint[field] for field in ENDPOINT_INPUT if field not in ("learning", "forgetting")}
        baseline = None
        rows = []
        for point in curve:
            d0, d1, l, f = [float(point[field]) for field in ("loss_d0", "loss_d1", "learning", "forgetting")]
            if not all(map(math.isfinite, (d0, d1, l, f))) or d0 < 0 or d1 < 0:
                raise ValueError(f"nonfinite or negative raw NLL for {key}, step {point['step']}")
            candidate = (d0 - f, d1 + l)
            if baseline is None:
                baseline = candidate
            elif not all(same(a, b) for a, b in zip(baseline, candidate)):
                raise ValueError(f"inconsistent step-0 baseline within curve for {key}")
            rows.append({**prefix, "checkpoint_step": point["step"], "loss_d0": d0, "loss_d1": d1,
                         "learning": l, "forgetting": f})
        base_key = (key[0], key[2], key[4])
        previous = baseline_by_base.setdefault(base_key, baseline)
        if not all(same(a, b) for a, b in zip(previous, baseline)):
            raise ValueError(f"inconsistent step-0 baseline across curves for {base_key}")
        d0_key = (key[2], key[4])
        if not same(d0_by_base.setdefault(d0_key, baseline[0]), baseline[0]):
            raise ValueError(f"inconsistent D0 baseline across fine-tuning corpora for {d0_key}")
        trajectories.append({**prefix, "checkpoint_step": 0, "loss_d0": baseline[0], "loss_d1": baseline[1],
                             "learning": 0., "forgetting": 0.})
        trajectories.extend(rows)
        enriched.append({**{field: endpoint[field] for field in ENDPOINT_INPUT}, "loss_d0": rows[-1]["loss_d0"],
                         "loss_d1": rows[-1]["loss_d1"], "base_loss_d0": baseline[0], "base_loss_d1": baseline[1]})
    if not trajectories:
        raise ValueError("no selected runs")
    return trajectories, enriched


@dataclass(frozen=True)
class NllPoint:
    checkpoint_step: int
    loss_d0: float
    sd_d0: float
    loss_d1: float
    sd_d1: float
    n_bases: int


@dataclass(frozen=True)
class Series:
    horizon: int
    lr: float
    n_bases: int
    points: tuple


def summarize_point(rows, checkpoint_step):
    """Hierarchical SD: across base means, or FT seed spread conditional on seed 42."""
    by_base = defaultdict(dict)
    for row in rows:
        base, ft = int(row["base_seed"]), int(row["ft_seed"])
        if ft in by_base[base]:
            raise ValueError("duplicate checkpoint observation")
        losses = (float(row["loss_d0"]), float(row["loss_d1"]))
        if not all(math.isfinite(x) and x >= 0 for x in losses):
            raise ValueError("invalid raw checkpoint NLL")
        by_base[base][ft] = losses
    if set(by_base) not in ({42}, BASE_SEEDS) or any(set(values) != FT_SEEDS for values in by_base.values()):
        raise ValueError("incomplete or unexpected base/FT seed grid")
    if len(by_base) == 1:
        observations = list(by_base[42].values())
    else:
        observations = [tuple(mean(xs) for xs in zip(*values.values())) for values in by_base.values()]
    d0, d1 = zip(*observations)
    return NllPoint(checkpoint_step, mean(d0), stdev(d0), mean(d1), stdev(d1), len(by_base))


def aggregate(trajectories):
    """Aggregate exact stored checkpoints (no interpolation) for all selected series."""
    groups = defaultdict(list)
    for row in trajectories:
        dataset, horizon, pre, ft, _, _ = run_key(row)
        if dataset not in DATASETS or (pre, ft) not in PAIRS or horizon <= 0:
            raise ValueError("unexpected selected trajectory arm")
        groups[dataset, pre, ft, horizon].append(row)
    series_by_arm = defaultdict(list)
    for (dataset, pre, ft, horizon), rows in sorted(groups.items()):
        lr = float(rows[0]["lr"])
        if not math.isfinite(lr) or lr <= 0 or any(float(row["lr"]) != lr for row in rows):
            raise ValueError(f"mixed/invalid selected LR in {dataset, pre, ft, horizon}")
        by_run, by_step = defaultdict(list), defaultdict(list)
        for row in rows:
            step = int(row["checkpoint_step"])
            by_run[int(row["base_seed"]), int(row["ft_seed"])].append(step)
            by_step[step].append(row)
        schedule = sorted(by_step)
        if schedule[0] != 0 or schedule[-1] != horizon or any(sorted(v) != schedule for v in by_run.values()):
            raise ValueError(f"incomplete or mixed saved checkpoint schedules for {dataset, pre, ft, horizon}")
        points = tuple(summarize_point(by_step[t], t) for t in schedule)
        if len({p.n_bases for p in points}) != 1:
            raise ValueError("mixed replication within curve")
        series_by_arm[dataset, pre, ft].append(Series(horizon, lr, points[0].n_bases, points))
    expected = {(dataset, *pair) for dataset in DATASETS for pair in PAIRS}
    if set(series_by_arm) != expected:
        raise ValueError("all eight optimizer pair/dataset arms must be present")
    for dataset in DATASETS:
        coverage = [tuple((s.horizon, s.n_bases) for s in series_by_arm[(dataset, *pair)]) for pair in PAIRS]
        if len(set(coverage)) != 1:
            raise ValueError(f"mixed horizon or replication coverage across {dataset} optimizer arms")
        if coverage[0] != EXPECTED_COVERAGE[dataset]:
            raise ValueError(f"incomplete or unexpected selected horizon/replication coverage for {dataset}")
    return dict(series_by_arm)


def common_limits(series_by_arm, dataset, field):
    """One inclusive y-range per dataset/evaluation set; no clipping of spikes."""
    attr = "loss_" + field
    sd = "sd_" + field
    points = [p for key, series in series_by_arm.items() if key[0] == dataset for s in series for p in s.points]
    low = min(getattr(p, attr) - getattr(p, sd) for p in points)
    high = max(getattr(p, attr) + getattr(p, sd) for p in points)
    margin = max((high - low) * .04, .005)
    return low - margin, high + margin


def plot_arm(series_by_arm, dataset, pre, ft, out_dir):
    """One pair × dataset figure; each curve is one selected-LR horizon."""
    series = series_by_arm[dataset, pre, ft]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.7), sharex=True)
    try:
        for ax, field in zip(axes, ("d0", "d1")):
            for i, s in enumerate(series):
                pilot = s.n_bases == 1
                color = HORIZON_COLORS.get(s.horizon, f"C{i}")
                marker = HORIZON_MARKERS.get(s.horizon, "o")
                label = f"T={s.horizon:,}, LR={s.lr:.4g}" + (" (base 42 only)" if pilot else "")
                ax.errorbar([p.checkpoint_step for p in s.points], [getattr(p, "loss_" + field) for p in s.points],
                            yerr=[getattr(p, "sd_" + field) for p in s.points], color=color, marker=marker,
                            markersize=4.2, markerfacecolor="white" if pilot else color,
                            linestyle="--" if pilot else "-", linewidth=1.6, elinewidth=.9, capsize=1.8,
                            alpha=.9, label=label)
            ax.set_xscale("symlog", linthresh=8)
            mid = max(s.horizon for s in series if s.horizon <= 960)
            ticks = sorted(set([0, 8, 32, 60, 240, mid, *[s.horizon for s in series if s.horizon > 960]]))
            ax.set_xticks(ticks, labels=[f"{t/1000:g}k" if t >= 1000 else str(t) for t in ticks],
                          rotation=25, ha="right")
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlim(-.35, max(s.horizon for s in series) * 1.12)
            ax.set_ylim(*common_limits(series_by_arm, dataset, field))
            ax.grid(True, color="#e5e5e5", linewidth=.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title("D0 — FineWeb-Edu retention" if field == "d0" else f"D1 — {DATASETS[dataset]} validation",
                         fontsize=11.5, loc="left", fontweight="semibold")
            ax.set_xlabel("Optimizer steps (symmetric log; linear ≤8)")
            ax.set_ylabel("Held-out NLL (nats/token) ↓")
        fig.suptitle(f"E=8 · {DATASETS[dataset]} · {NAMES[pre]} → {NAMES[ft]}", fontsize=16,
                     y=.974, fontweight="semibold")
        fig.text(.5, .914, "Each horizon uses its own selected LR and separate end-of-run decay; lower NLL is better.",
                 ha="center", fontsize=10, color="#525252")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .9), ncol=3,
                   frameon=False, fontsize=9.2, columnspacing=1.5)
        note = "Solid: mean ± SD across 3 base-seed means (3 FT seeds per base)."
        if any(s.n_bases == 1 for s in series):
            note += " Open/dashed: seed 42 only, SD across 3 FT seeds."
        fig.text(.5, .035, note, ha="center", fontsize=9, color="#525252")
        fig.text(.5, .009, "Step 0 recovered from stored loss/change identities; no interpolation or clipping of plotted means/SD.",
                 ha="center", fontsize=9, color="#525252")
        fig.tight_layout(rect=(.025, .085, .99, .77), w_pad=1.7)
        out = Path(out_dir) / f"{dataset}_{pre}_to_{ft}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.15)
        return out
    finally:
        plt.close(fig)


def plot_all(trajectories, out_dir):
    groups = aggregate(trajectories)
    return [plot_arm(groups, dataset, pre, ft, out_dir) for dataset in DATASETS for pre, ft in PAIRS]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    e = commands.add_parser("extract", help="extract exact run trajectories and endpoint NLL")
    e.add_argument("--endpoints", required=True, type=Path)
    e.add_argument("--mirror-root", type=Path, default=DEFAULT_MIRROR)
    e.add_argument("--out", required=True, type=Path)
    e.add_argument("--endpoint-out", type=Path, help="write enriched endpoint rows (may overwrite input after reading)")
    p = commands.add_parser("plot", help="render eight figures from the tracked trajectory CSV")
    p.add_argument("--trajectories", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "extract":
        trajectories, enriched = extract(read_csv(args.endpoints, ENDPOINT_INPUT), args.mirror_root)
        # Verify the complete seed/checkpoint grid before writing either table.
        aggregate(trajectories)
        write_csv(args.out, TRAJECTORY_FIELDS, trajectories)
        print(f"wrote {len(trajectories)} checkpoint rows to {args.out}")
        if args.endpoint_out:
            write_csv(args.endpoint_out, ENDPOINT_FIELDS, enriched)
            print(f"wrote {len(enriched)} enriched endpoint rows to {args.endpoint_out}")
    else:
        for out in plot_all(read_csv(args.trajectories, TRAJECTORY_FIELDS), args.out_dir):
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
