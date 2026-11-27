"""Matched-learning analysis of clean selected-LR E=8 results.

Two distinct estimands (fixed by the protocol):

* endpoint: the horizon-ordered polyline through independently optimized,
  separately decayed final endpoints; no added origin or extrapolation.
* within-run: the checkpoint-ordered polyline for each fixed horizon and its
  selected LR, including the genuine step-zero baseline.

For either curve, F(target) = min forgetting subject to learning >= target,
over eligible vertices and all crossings of adjacent segments. Pair the
mismatched and matched F within the same pretrained base and FT seed. The
reported sample SD is across pretrained-seed means (each over three paired FT
seeds), or across FT seed pairs conditional on the single base 42. This is
descriptive seed variation, not a document bootstrap or confidence interval.

  python -m scripts.matched_learning \
    --endpoint-target starcoderdata=1.50 --endpoint-target wikitext103=0.25 \
    --within-run-target starcoderdata=1.00 --within-run-target wikitext103=0.12
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
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np

from scripts.finetune_validation_trajectories import ENDPOINT_FIELDS, TRAJECTORY_FIELDS, read_csv
from scripts.plot_finetune_endpoints import DATASETS, NAMES, PAIRS


@dataclass(frozen=True)
class Cohort:
    dataset: str
    name: str
    bases: tuple
    horizons: tuple

    @property
    def title(self):
        return f"{DATASETS[self.dataset]} · " + ("3 bases" if len(self.bases) == 3 else "base 42 only")


COHORTS = (Cohort("starcoderdata", "main", (42, 43, 44), (60, 240, 960, 1920, 3200)),
           Cohort("wikitext103", "main", (42, 43, 44), (60, 240, 720)),
           Cohort("wikitext103", "pilot_seed42", (42,), (60, 240, 720, 1920, 3200)))
FT_SEEDS = (1, 2, 3)
PRE = ("adamw", "muon")
OTHER = {"adamw": "muon", "muon": "adamw"}
DIRECTION_COLORS = {"adamw": "#0072B2", "muon": "#D55E00"}
HORIZON_COLORS = {60: "#0072B2", 240: "#E69F00", 720: "#009E73", 960: "#009E73",
                  1920: "#AA4499", 3200: "#D55E00"}


@dataclass(frozen=True)
class Vertex:
    axis: int             # horizon (endpoint) or checkpoint step (within-run)
    horizon: int
    checkpoint_step: int
    lr: float
    learning: float
    forgetting: float
    source_run: str
    source: str


@dataclass(frozen=True)
class Match:
    target: float
    learning: float       # observed L at an eligible vertex; target at a crossing
    forgetting: float
    axis: float
    left: Vertex
    right: Vertex
    weight: float         # weight on the right endpoint

    @property
    def kind(self):
        return "vertex" if self.left == self.right else "interpolated_crossing"


@dataclass(frozen=True)
class Curve:
    pretrain: str
    ft: str
    base_seed: int
    ft_seed: int
    points: tuple

    @property
    def key(self):
        return self.pretrain, self.ft, self.base_seed, self.ft_seed


@dataclass(frozen=True)
class Comparison:
    cohort: Cohort
    estimand: str
    horizon: int | None  # None for endpoint polyline across horizons
    curves: tuple

    @property
    def key(self):
        return self.cohort.dataset, self.cohort.name, self.estimand, self.horizon


def select_minimum(points, target):
    """Exact piecewise-linear restricted minimum; no extrapolation.

    For a linear segment F is linear in its parameter, so any restricted
    minimum is at an eligible vertex or a crossing L=target. Descending and
    ascending crossings are both candidates; flat plateaus are handled by
    their vertices. A threshold must lie inside the observed curve L range.
    """
    if not points or not math.isfinite(target):
        return None
    if any(not (math.isfinite(p.learning) and math.isfinite(p.forgetting)) for p in points):
        raise ValueError("nonfinite curve vertex")
    axes = [p.axis for p in points]
    if axes != sorted(set(axes)):
        raise ValueError("curve vertices must be strictly ordered in time/horizon")
    if not min(p.learning for p in points) <= target <= max(p.learning for p in points):
        return None
    candidates = [Match(target, p.learning, p.forgetting, float(p.axis), p, p, 0.)
                  for p in points if p.learning >= target]
    for a, b in zip(points, points[1:]):
        lo, hi = sorted((a.learning, b.learning))
        if lo < target < hi:
            w = (target - a.learning) / (b.learning - a.learning)
            f = (1-w)*a.forgetting + w*b.forgetting
            axis = (1-w)*a.axis + w*b.axis
            candidates.append(Match(target, target, f, axis, a, b, w))
    return min(candidates, key=lambda m: (m.forgetting, m.axis, m.left.axis, m.right.axis, m.weight))


def run_identity(row):
    return (row["dataset"], int(row["steps"]), row["pretrain_optimizer"], row["ft_optimizer"],
            int(row["base_seed"]), int(row["ft_seed"]))


def vertex(row, estimand):
    horizon = int(row["steps"])
    step = horizon if estimand == "endpoint" else int(row["checkpoint_step"])
    p = Vertex(horizon if estimand == "endpoint" else step, horizon, step, float(row["lr"]),
               float(row["learning"]), float(row["forgetting"]), row["source_run"], row["source"])
    if not all(math.isfinite(x) for x in (p.lr, p.learning, p.forgetting)) or p.lr <= 0:
        raise ValueError(f"nonfinite or invalid selected-run value for {run_identity(row)}")
    return p


def build_comparisons(endpoints, trajectories, cohorts=COHORTS):
    """Cross-check selected final endpoints against their trajectory sources."""
    end = {}
    for row in endpoints:
        key = run_identity(row)
        if key in end:
            raise ValueError(f"duplicate selected endpoint {key}")
        end[key] = vertex(row, "endpoint")
    within = defaultdict(list)
    for row in trajectories:
        key = run_identity(row)
        p = vertex(row, "within-run")
        if key not in end or (p.lr, p.source_run, p.source) != (end[key].lr, end[key].source_run, end[key].source):
            raise ValueError(f"trajectory provenance differs from selected endpoint {key}")
        within[key].append(p)
    if set(within) != set(end):
        raise ValueError("endpoint and trajectory run coverage differ")
    for key, points in within.items():
        points.sort(key=lambda p: p.axis)
        axes = [p.axis for p in points]
        if axes != sorted(set(axes)) or axes[0] != 0 or axes[-1] != key[1]:
            raise ValueError(f"missing/duplicate checkpoint or baseline for {key}")
        if (points[0].learning, points[0].forgetting) != (0., 0.):
            raise ValueError(f"step-zero loss-change baseline must be (0,0) for {key}")
        if (points[-1].learning, points[-1].forgetting) != (end[key].learning, end[key].forgetting):
            raise ValueError(f"final loss changes do not match selected endpoint {key}")

    comparisons, covered = [], set()
    for c in cohorts:
        endpoint_curves = []
        within_by_horizon = defaultdict(list)
        for pre, ft in PAIRS:
            for b in c.bases:
                for seed in FT_SEEDS:
                    keys = [(c.dataset, t, pre, ft, b, seed) for t in c.horizons]
                    try:
                        p = tuple(end[k] for k in keys)
                    except KeyError as exc:
                        raise ValueError(f"missing required selected endpoint {exc.args[0]}") from exc
                    endpoint_curves.append(Curve(pre, ft, b, seed, p))
                    for k in keys:
                        within_by_horizon[k[1]].append(Curve(pre, ft, b, seed, tuple(within[k])))
                    covered.update(keys)
        comparisons.append(Comparison(c, "endpoint", None, tuple(endpoint_curves)))
        for horizon in c.horizons:
            curves = within_by_horizon[horizon]
            schedules = {tuple(p.axis for p in curve.points) for curve in curves}
            if len(schedules) != 1:
                raise ValueError(f"inconsistent saved checkpoint schedule for {c.dataset, c.name, horizon}")
            comparisons.append(Comparison(c, "within-run", horizon, tuple(curves)))
    if covered != set(end):
        raise ValueError("selected input contains runs not covered by any declared comparison")
    return comparisons


def support(comparison):
    """Common nonnegative learning support; every selected run must reach target."""
    return (max(0., max(min(p.learning for p in c.points) for c in comparison.curves)),
            min(max(p.learning for p in c.points) for c in comparison.curves))


def selection_dict(comparison, curve, match):
    row = {"dataset": comparison.cohort.dataset, "cohort": comparison.cohort.name,
           "estimand": comparison.estimand, "horizon": comparison.horizon or "",
           "pretrain_optimizer": curve.pretrain, "ft_optimizer": curve.ft,
           "base_seed": curve.base_seed, "ft_seed": curve.ft_seed,
           "target_learning": match.target, "actual_learning": match.learning,
           "forgetting": match.forgetting, "selection_kind": match.kind,
           "effective_axis": match.axis, "weight_right": match.weight}
    for side, p in (("left", match.left), ("right", match.right)):
        row.update({f"{side}_{name}": getattr(p, name) for name in
                    ("horizon", "checkpoint_step", "lr", "learning", "forgetting", "source_run", "source")})
    return row


def paired_rows(comparison, target, selections):
    """Pair before aggregating; this preserves common base and FT seed noise."""
    rows = []
    for pre in PRE:
        for base in comparison.cohort.bases:
            for seed in FT_SEEDS:
                a, b = selections[(pre, pre, base, seed)], selections[(pre, OTHER[pre], base, seed)]
                row = {"dataset": comparison.cohort.dataset, "cohort": comparison.cohort.name,
                       "estimand": comparison.estimand, "horizon": comparison.horizon or "",
                       "pretrain_optimizer": pre, "matched_ft_optimizer": pre, "mismatched_ft_optimizer": OTHER[pre],
                       "base_seed": base, "ft_seed": seed, "target_learning": target,
                       "matched_learning": a.learning, "mismatched_learning": b.learning,
                       "matched_forgetting": a.forgetting, "mismatched_forgetting": b.forgetting,
                       "penalty": b.forgetting - a.forgetting}
                rows.append(row)
    return rows


def summarize_pairs(rows, bases):
    """Each field gets its own hierarchical SD; never subtract independent SDs."""
    by_base = defaultdict(list)
    for r in rows:
        by_base[int(r["base_seed"])].append(r)
    if set(by_base) != set(bases) or any({r["ft_seed"] for r in rs} != set(FT_SEEDS) or len(rs) != 3
                                         for rs in by_base.values()):
        raise ValueError("incomplete paired seed grid")
    summary = {"n_bases": len(bases), "n_ft_seeds_per_base": 3, "n_pairs": len(rows),
               "sd_unit": "base means" if len(bases) > 1 else "FT seeds conditional on base 42"}
    for field in ("matched_learning", "mismatched_learning", "matched_forgetting", "mismatched_forgetting", "penalty"):
        vals = ([mean(r[field] for r in rs) for rs in by_base.values()] if len(bases) > 1
                else [r[field] for r in rows])
        summary[field + "_mean"] = mean(vals)
        summary[field + "_sd"] = stdev(vals)
    for base in (42, 43, 44):
        rs = by_base.get(base)
        summary[f"base{base}_penalty_mean"] = mean(r["penalty"] for r in rs) if rs else ""
        summary[f"base{base}_penalty_ft_sd"] = stdev(r["penalty"] for r in rs) if rs else ""
    return summary


def analyze(comparison, target, *, details=False):
    """Return summaries and optionally primary seed/provenance rows; no partial sample."""
    lo, hi = support(comparison)
    prefix = {"dataset": comparison.cohort.dataset, "cohort": comparison.cohort.name,
              "estimand": comparison.estimand, "horizon": comparison.horizon or "",
              "target_learning": target, "support_min": lo, "support_max": hi}
    if not lo <= target <= hi:
        return ([{**prefix, "pretrain_optimizer": pre, "status": "unsupported_target"} for pre in PRE], [], [])
    selections = {curve.key: select_minimum(curve.points, target) for curve in comparison.curves}
    if any(m is None for m in selections.values()):
        raise ValueError("computed common support disagrees with curve matching")
    pairs = paired_rows(comparison, target, selections)
    summaries = [{**prefix, "pretrain_optimizer": pre, "status": "supported",
                  **summarize_pairs([r for r in pairs if r["pretrain_optimizer"] == pre], comparison.cohort.bases)}
                 for pre in PRE]
    if details:
        return summaries, pairs, [selection_dict(comparison, c, selections[c.key]) for c in comparison.curves]
    return summaries, [], []


def sensitivity_grid(lo, hi, size, *extra):
    if lo > hi:
        return []
    return sorted(set(float(x) for x in np.linspace(lo, hi, size)) | {v for v in extra if lo <= v <= hi})


def analyze_all(comparisons, targets, size=101):
    primary, pairs, selections, sensitivity, overlaps = [], [], [], [], []
    endpoints = {(c.cohort.dataset, c.cohort.name): c for c in comparisons if c.estimand == "endpoint"}
    for c in comparisons:
        target = targets[c.estimand][c.cohort.dataset]
        rows, p, s = analyze(c, target, details=True)
        primary.extend(rows); pairs.extend(p); selections.extend(s)
        for x in sensitivity_grid(*support(c), size, target):
            sensitivity.extend(analyze(c, x)[0])
        if c.estimand == "within-run":
            endpoint = endpoints[c.cohort.dataset, c.cohort.name]
            elo, ehi = support(endpoint)
            wlo, whi = support(c)
            lo, hi = max(elo, wlo), min(ehi, whi)
            for x in sensitivity_grid(lo, hi, size, targets["endpoint"][c.cohort.dataset], target):
                ep = analyze(endpoint, x)[0]; wr = analyze(c, x)[0]
                for e, w in zip(ep, wr):
                    overlaps.append({"dataset": c.cohort.dataset, "cohort": c.cohort.name, "horizon": c.horizon,
                                     "pretrain_optimizer": e["pretrain_optimizer"], "target_learning": x,
                                     "overlap_min": lo, "overlap_max": hi,
                                     "endpoint_penalty_mean": e["penalty_mean"], "endpoint_penalty_sd": e["penalty_sd"],
                                     "within_run_penalty_mean": w["penalty_mean"], "within_run_penalty_sd": w["penalty_sd"]})
    return primary, pairs, selections, sensitivity, overlaps


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def parse_targets(values):
    result = {}
    for spec in values:
        key, sep, raw = spec.partition("=")
        if not sep or key not in DATASETS or key in result:
            raise ValueError(f"invalid or duplicate dataset target {spec!r}")
        target = float(raw)
        if not math.isfinite(target) or target < 0:
            raise ValueError(f"invalid learning target {spec!r}")
        result[key] = target
    if set(result) != set(DATASETS):
        raise ValueError("a target for each dataset is required")
    return result


def _axis_style(ax):
    ax.axhline(0, color="#777777", linewidth=.9, linestyle=":", zorder=-2)
    ax.grid(True, alpha=.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _ribbon(ax, rows, color, label, *, field="penalty", style="-"):
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["target_learning"])
    x = np.array([r["target_learning"] for r in rows])
    m = np.array([r[field+"_mean"] for r in rows]); sd = np.array([r[field+"_sd"] for r in rows])
    ax.plot(x, m, color=color, label=label, linestyle=style, linewidth=1.6)
    ax.fill_between(x, m-sd, m+sd, color=color, alpha=.12, linewidth=0)


def _save(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white", pad_inches=.15)
    plt.close(fig)
    return path


def _direction_handles():
    return [Line2D([], [], color=DIRECTION_COLORS[p], marker="o", lw=1.7,
                   label=f"{NAMES[p]}-pretrained ({NAMES[OTHER[p]]} − {NAMES[p]} FT)") for p in PRE]


def plot_primary(primary, targets, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for col, c in enumerate(COHORTS):
        cohort_rows = [r for r in primary if r["dataset"] == c.dataset and r["cohort"] == c.name]
        ax = axes[0, col]; _axis_style(ax)
        for i, pre in enumerate(PRE):
            row = next(r for r in cohort_rows if r["estimand"] == "endpoint" and r["pretrain_optimizer"] == pre)
            if row["status"] == "supported":
                ax.errorbar(i, row["penalty_mean"], yerr=row["penalty_sd"], color=DIRECTION_COLORS[pre],
                            marker="o", capsize=4, markersize=7, linestyle="none")
            else:
                ax.text(i, .5, "unsupported", transform=ax.get_xaxis_transform(), ha="center")
        ax.set_xticks(range(2), labels=[NAMES[p] + " base" for p in PRE]); ax.set_xlim(-.6, 1.6)
        ax.set_title(f"{c.title}\nEndpoint target = {targets['endpoint'][c.dataset]:g}", fontsize=11, fontweight="semibold")
        ax = axes[1, col]; _axis_style(ax)
        for pre in PRE:
            rows = [r for r in cohort_rows if r["estimand"] == "within-run" and r["pretrain_optimizer"] == pre
                    and r["status"] == "supported"]
            rows.sort(key=lambda r: r["horizon"])
            ax.errorbar([r["horizon"] for r in rows], [r["penalty_mean"] for r in rows],
                        yerr=[r["penalty_sd"] for r in rows], color=DIRECTION_COLORS[pre], marker="o", capsize=3)
        ax.set_xscale("log"); ax.set_xticks(c.horizons, labels=[f"{t:,}" for t in c.horizons], rotation=25, ha="right")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(f"Within-run target = {targets['within-run'][c.dataset]:g}", fontsize=11, fontweight="semibold")
        ax.set_xlabel("Fine-tuning horizon (optimizer steps, log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mismatch penalty (nats/token)")
    fig.suptitle("E=8 · matched-learning mismatch penalty at prespecified targets", y=.98, fontsize=15, fontweight="semibold")
    fig.legend(handles=_direction_handles(), loc="upper center", ncol=2, bbox_to_anchor=(.5,.925), frameon=False)
    fig.text(.5,.025,"Positive = mismatch forgets more. Error bars are descriptive seed SD; the two estimands use different targets.",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.065,1,.86), w_pad=1.8, h_pad=2)
    return _save(fig, out)


def plot_endpoint_sensitivity(rows, targets, out):
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.7))
    for ax, c in zip(axes, COHORTS):
        _axis_style(ax)
        for pre in PRE:
            r = [r for r in rows if r["estimand"] == "endpoint" and r["dataset"] == c.dataset
                 and r["cohort"] == c.name and r["pretrain_optimizer"] == pre]
            _ribbon(ax, r, DIRECTION_COLORS[pre], NAMES[pre]+" base")
        ax.axvline(targets["endpoint"][c.dataset], color="#555555", linestyle="--", linewidth=.9)
        ax.set_title(c.title, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Common learning threshold (nats/token)")
    axes[0].set_ylabel("Endpoint mismatch penalty (nats/token)")
    fig.suptitle("Endpoint estimand · common-support sensitivity", y=.98, fontsize=15, fontweight="semibold")
    fig.legend(handles=_direction_handles(), loc="upper center", ncol=2, bbox_to_anchor=(.5,.915), frameon=False)
    fig.text(.5,.015,"Bands: ±1 descriptive seed SD. Vertical lines: prespecified endpoint targets; no extrapolation.",ha="center",fontsize=9.5)
    fig.tight_layout(rect=(0,.06,1,.82), w_pad=1.5)
    return _save(fig, out)


def plot_within_sensitivity(rows, targets, out):
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10))
    for row, c in enumerate(COHORTS):
        for col, pre in enumerate(PRE):
            ax = axes[row, col]; _axis_style(ax)
            for horizon in c.horizons:
                r = [r for r in rows if r["estimand"] == "within-run" and r["dataset"] == c.dataset
                     and r["cohort"] == c.name and r["pretrain_optimizer"] == pre and r["horizon"] == horizon]
                _ribbon(ax, r, HORIZON_COLORS[horizon], f"{horizon:,} steps")
            ax.axvline(targets["within-run"][c.dataset], color="#555555", linestyle="--", linewidth=.9)
            ax.set_title(f"{c.title} · {NAMES[pre]} base", fontsize=11, fontweight="semibold")
            ax.set_xlabel("Common learning threshold (nats/token)")
    handles = [Line2D([],[],color=HORIZON_COLORS[t],label=("720/960" if t==960 else f"{t:,}")+" steps")
               for t in (60,240,960,1920,3200)]
    fig.suptitle("Within-run estimand · common support at each horizon", y=.98, fontsize=15, fontweight="semibold")
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(.5,.94), frameon=False)
    fig.supylabel("Within-run mismatch penalty (nats/token)", x=.012, fontsize=12)
    fig.text(.5,.015,"Bands: ±1 descriptive seed SD. Vertical lines: prespecified within-run targets; each curve ends at its own common support.",ha="center",fontsize=9.5)
    fig.tight_layout(rect=(.02,.04,1,.90), h_pad=2)
    return _save(fig, out)


def plot_overlap(rows, out):
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10))
    for row, c in enumerate(COHORTS):
        for col, pre in enumerate(PRE):
            ax = axes[row,col]; _axis_style(ax)
            selected = [r for r in rows if r["dataset"]==c.dataset and r["cohort"]==c.name and r["pretrain_optimizer"]==pre]
            ep = {r["target_learning"]:r for r in selected}
            _ribbon(ax,list(ep.values()),"#252525","Endpoint across horizons",field="endpoint_penalty",style="--")
            for h in c.horizons:
                _ribbon(ax,[r for r in selected if r["horizon"]==h],HORIZON_COLORS[h],f"{h:,} steps",field="within_run_penalty")
            ax.set_title(f"{c.title} · {NAMES[pre]} base",fontsize=11,fontweight="semibold")
            if not selected: ax.text(.5,.5,"No common threshold",transform=ax.transAxes,ha="center")
            ax.set_xlabel("Shared learning threshold (nats/token)")
    handles = [Line2D([],[],color="#252525",ls="--",label="Endpoint")]+[Line2D([],[],color=HORIZON_COLORS[t],
               label=("720/960" if t==960 else f"{t:,}")+" steps") for t in (60,240,960,1920,3200)]
    fig.suptitle("Overlap sensitivity · endpoint and within-run estimands at the same threshold", y=.98, fontsize=15, fontweight="semibold")
    fig.legend(handles=handles,loc="upper center",ncol=6,bbox_to_anchor=(.5,.94),frameon=False)
    fig.supylabel("Mismatch penalty (nats/token)", x=.012, fontsize=12)
    fig.text(.5,.015,"Only intersection thresholds are plotted. Bands: descriptive seed SD. Different estimands are not interchangeable.",ha="center",fontsize=9.5)
    fig.tight_layout(rect=(.02,.04,1,.90),h_pad=2)
    return _save(fig,out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoints", type=Path, default=Path("results/e8/finetune_endpoint_runs.csv"))
    parser.add_argument("--trajectories", type=Path, default=Path("results/e8/finetune_validation_trajectories.csv"))
    parser.add_argument("--endpoint-target", action="append", required=True)
    parser.add_argument("--within-run-target", action="append", required=True)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--out-dir", type=Path, default=Path("results/e8/matched_learning"))
    parser.add_argument("--figure-dir", type=Path, default=Path("results/e8/figures/matched_learning"))
    args = parser.parse_args(argv)
    if args.grid_size < 2: parser.error("--grid-size must be at least 2")
    try:
        targets = {"endpoint": parse_targets(args.endpoint_target), "within-run": parse_targets(args.within_run_target)}
    except ValueError as exc:
        parser.error(str(exc))
    comparisons = build_comparisons(read_csv(args.endpoints, ENDPOINT_FIELDS), read_csv(args.trajectories, TRAJECTORY_FIELDS))
    primary, pairs, selections, sensitivity, overlaps = analyze_all(comparisons, targets, args.grid_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary",primary),("per_seed",pairs),("selections",selections),
                       ("sensitivity",sensitivity),("overlap_sensitivity",overlaps)):
        if rows:
            write_csv(args.out_dir / (name+".csv"),rows)
    supports = [{"dataset":c.cohort.dataset,"cohort":c.cohort.name,"estimand":c.estimand,"horizon":c.horizon or "",
                 "support_min":support(c)[0],"support_max":support(c)[1]} for c in comparisons]
    write_csv(args.out_dir / "support.csv", supports)
    (args.out_dir / "metadata.json").write_text(json.dumps({"targets":targets,"grid_size":args.grid_size,
        "cohorts":[{"dataset":c.dataset,"name":c.name,"bases":c.bases,"horizons":c.horizons} for c in COHORTS],
        "uncertainty":"sample SD of paired penalties across base means, or FT seeds conditional on base 42",
        "endpoint_definition":"horizon-ordered polyline, no origin; no extrapolation",
        "within_run_definition":"checkpoint-ordered polyline at selected LR including genuine step zero",
        "minimum":"min F among all eligible vertices and adjacent crossings L=target",
        "sensitivity":"uniform grid incl. bounds and prespecified targets when supported; no bootstrap"},indent=2)+"\n")
    for path in (plot_primary(primary,targets,args.figure_dir/"primary.png"),
                 plot_endpoint_sensitivity(sensitivity,targets,args.figure_dir/"endpoint_sensitivity.png"),
                 plot_within_sensitivity(sensitivity,targets,args.figure_dir/"within_run_sensitivity.png"),
                 plot_overlap(overlaps,args.figure_dir/"overlap_sensitivity.png")):
        print("wrote",path)
    print(f"Wrote {len(primary)} primary summaries, {len(pairs)} paired seeds, {len(selections)} selections, "
          f"{len(sensitivity)} sensitivity rows, {len(overlaps)} overlap rows to {args.out_dir}")


if __name__ == "__main__":
    main()
