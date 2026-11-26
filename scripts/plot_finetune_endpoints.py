"""Plot learning–forgetting endpoint curves for the selected-LR E=8 runs.

Input: CSV with one row per selected run and columns dataset, steps,
pretrain_optimizer, ft_optimizer, base_seed, ft_seed, lr, learning, forgetting,
source. Include only valid results; do not include overlapping pilot/main runs
as distinct observations. Source is provenance, not an independent replicate.

The solid curves average the three pretraining-seed means (each over three
fine-tuning seeds); error bars are sample SD across those three base means.
The open/dashed curves are exploratory long-length extensions on base seed 42;
their error bars are sample SD across its three fine-tuning seeds. A dashed
curve starts at the seed-42 mean of the last fully replicated length (drawn
separately), not at the three-base mean. Both kinds of bars are one SD, not CIs.

  python -m scripts.plot_finetune_endpoints results/e8/finetune_endpoints.csv
  python -m scripts.plot_finetune_endpoints data.csv --out figures/endpoints.png

These are final, separately decayed endpoints at distinct selected LRs—not
checkpoints of one training run or estimates of forgetting at matched learning.
"""

import argparse
import csv
from dataclasses import dataclass
from collections import defaultdict
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox


PAIRS = (("adamw", "adamw"), ("adamw", "muon"), ("muon", "adamw"), ("muon", "muon"))
NAMES = {"adamw": "AdamW", "muon": "Muon"}
DATASETS = {"starcoderdata": "StarCoderData (Python)", "wikitext103": "WikiText-103"}
MAIN_LENGTHS = {"starcoderdata": (60, 240, 960), "wikitext103": (60, 240, 720)}
BASE_SEEDS = {42, 43, 44}
FT_SEEDS = {1, 2, 3}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#AA4499")
MARKERS = ("o", "s", "D", "^")
FIELDS = ("dataset", "steps", "pretrain_optimizer", "ft_optimizer", "base_seed",
          "ft_seed", "lr", "learning", "forgetting", "source")


@dataclass(frozen=True)
class Run:
    dataset: str
    steps: int
    pretrain_optimizer: str
    ft_optimizer: str
    base_seed: int
    ft_seed: int
    lr: float
    learning: float
    forgetting: float
    source: str

    @property
    def group(self):
        return (self.dataset, self.steps, self.pretrain_optimizer, self.ft_optimizer)


@dataclass(frozen=True)
class Point:
    learning: float
    forgetting: float
    learning_sd: float
    forgetting_sd: float
    n_bases: int
    n_ft_seeds_per_base: int


def read_runs(path):
    """Parse selected runs; reject duplicates and nonfinite or malformed data."""
    runs = []
    seen = set()
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            try:
                run = Run(row["dataset"], int(row["steps"]), row["pretrain_optimizer"],
                          row["ft_optimizer"], int(row["base_seed"]), int(row["ft_seed"]),
                          float(row["lr"]), float(row["learning"]), float(row["forgetting"]),
                          row["source"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid row {line}: {exc}") from exc
            if (run.dataset not in DATASETS or
                    (run.pretrain_optimizer, run.ft_optimizer) not in PAIRS or
                    run.steps <= 0 or run.base_seed not in BASE_SEEDS or
                    run.ft_seed not in FT_SEEDS or run.lr <= 0 or not run.source or
                    not all(map(math.isfinite, (run.lr, run.learning, run.forgetting)))):
                raise ValueError(f"invalid row {line}: {run}")
            key = (*run.group, run.base_seed, run.ft_seed)
            if key in seen:
                raise ValueError(f"duplicate observation: {key}")
            seen.add(key)
            runs.append(run)
    return runs


def group_runs(runs):
    """(dataset, steps, pretrain optimizer, FT optimizer) -> selected runs."""
    groups = defaultdict(list)
    for run in runs:
        groups[run.group].append(run)
    return dict(groups)


def summarize(runs, *, base_seed_only=None):
    """Compute mean and sample SD at the correct independent seed level.

    With three bases, the observation unit is a base mean, not each FT run.
    One-base summaries describe FT-seed spread conditional on that base.
    base_seed_only=42 may be used to draw the comparable anchor inside the main
    grid for the seed-42-only long extension.
    """
    if not runs:
        raise ValueError("empty group")
    group = runs[0].group
    if any(r.group != group for r in runs):
        raise ValueError("summarize expects a single dataset/length/optimizer pair")
    if not all(math.isclose(r.lr, runs[0].lr, rel_tol=1e-8) for r in runs):
        raise ValueError(f"{group}: multiple selected LRs")
    by_base = defaultdict(list)
    for run in runs:
        by_base[run.base_seed].append(run)
    if set(by_base) not in ({42}, BASE_SEEDS):
        raise ValueError(f"{group}: unexpected base seeds {sorted(by_base)}")
    for b, base_runs in by_base.items():
        if len(base_runs) != len(FT_SEEDS) or {r.ft_seed for r in base_runs} != FT_SEEDS:
            raise ValueError(f"{group}: base seed {b} missing or duplicate FT seeds")
    if base_seed_only is not None:
        if base_seed_only not in by_base:
            raise ValueError(f"{group}: missing base seed {base_seed_only}")
        by_base = {base_seed_only: by_base[base_seed_only]}
    if len(by_base) == 1:
        single = next(iter(by_base.values()))
        learning, forgetting = ([r.learning for r in single], [r.forgetting for r in single])
    else:
        learning = [mean(r.learning for r in base_runs) for base_runs in by_base.values()]
        forgetting = [mean(r.forgetting for r in base_runs) for base_runs in by_base.values()]
    return Point(mean(learning), mean(forgetting), stdev(learning), stdev(forgetting),
                 len(by_base), len(FT_SEEDS))


def collect_points(groups, dataset, *, include_1920=True):
    """Check complete optimizer coverage and consistent replication at each length."""
    lengths = sorted({key[1] for key in groups if key[0] == dataset and
                      (include_1920 or key[1] != 1920)})
    if not set(MAIN_LENGTHS[dataset]).issubset(lengths):
        raise ValueError(f"{dataset}: missing main-grid length(s)")
    points = {}
    for steps in lengths:
        replicas = set()
        for pre, ft in PAIRS:
            key = (dataset, steps, pre, ft)
            if key not in groups:
                raise ValueError(f"missing selected runs for {key}")
            p = summarize(groups[key])
            points[key] = p
            replicas.add(p.n_bases)
        if len(replicas) != 1:
            raise ValueError(f"{dataset}, {steps}: mixed replication among optimizer pairs")
        if steps in MAIN_LENGTHS[dataset] and replicas != {3}:
            raise ValueError(f"{dataset}, {steps}: main grid must have all three base seeds")
        if replicas == {1} and steps <= max(MAIN_LENGTHS[dataset]):
            raise ValueError(f"{dataset}, {steps}: seed-42-only point must extend the original main lengths")
    return lengths, points


def _label_points(ax, points, color):
    # Distinct offsets for pairs keep densely overlapping endpoints readable.
    i = COLORS.index(color)
    dx, dy = ((-5, -8), (7, 8), (-7, 8), (6, -8))[i]
    for t, p in points:
        ax.annotate(f"{t:,}", (p.learning, p.forgetting), xytext=(dx, dy),
                    textcoords="offset points", color=color, fontsize=8,
                    ha="right" if dx < 0 else "left", va="bottom" if dy > 0 else "top",
                    zorder=6, annotation_clip=False,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": .5})


def _place_labels(ax, renderer):
    """Keep short endpoint labels inside the axes and avoid label collisions."""
    labels = [a for a in ax.texts if isinstance(a, matplotlib.text.Annotation)]
    # Reserve the data marker itself, even if its label is placed later.
    marker_boxes = []
    marker_pad = 5 * ax.figure.dpi / 72
    for label in labels:
        x, y = ax.transData.transform(label.xy)
        marker_boxes.append(Bbox.from_extents(x - marker_pad, y - marker_pad,
                                              x + marker_pad, y + marker_pad))
    placed = []
    margin = 2 * ax.figure.dpi / 72
    axes_box = ax.get_window_extent(renderer).padded(-margin)

    def overlap(a, b):
        w = max(0., min(a.x1, b.x1) - max(a.x0, b.x0))
        h = max(0., min(a.y1, b.y1) - max(a.y0, b.y0))
        return w * h

    for label in labels:
        preferred_x, preferred_y = label.get_position()
        candidates = [(preferred_x, preferred_y)]
        for offset in (8, 13, 19, 25):
            candidates.extend((sx * offset, sy * offset) for sx, sy in
                              ((1, 1), (1, -1), (-1, 1), (-1, -1)))
        candidates.sort(key=lambda v: (v[0] - preferred_x)**2 + (v[1] - preferred_y)**2)
        best = None
        for dx, dy in candidates:
            label.set_position((dx, dy))
            label.set_horizontalalignment("right" if dx < 0 else "left")
            label.set_verticalalignment("bottom" if dy > 0 else "top")
            box = label.get_window_extent(renderer).padded(margin)
            outside = box.width * box.height - overlap(box, axes_box)
            score = max(0., outside) + sum(overlap(box, other) for other in placed + marker_boxes)
            if best is None or score < best[0]:
                best = (score, dx, dy, box)
            if score <= 1e-6:
                break
        _, dx, dy, box = best
        label.set_position((dx, dy))
        label.set_horizontalalignment("right" if dx < 0 else "left")
        label.set_verticalalignment("bottom" if dy > 0 else "top")
        placed.append(box)


def _draw_series(ax, points, color, marker, *, pilot=False, skip_first_label=False):
    """All errors are one sample SD; label endpoint lengths individually."""
    if not points:
        return
    _, ps = zip(*points)
    ax.errorbar([p.learning for p in ps], [p.forgetting for p in ps],
                xerr=[p.learning_sd for p in ps], yerr=[p.forgetting_sd for p in ps],
                color=color, marker=marker, markersize=6.5, markeredgewidth=1.3,
                markerfacecolor="white" if pilot else color,
                linestyle="--" if pilot else "-", linewidth=1.7,
                elinewidth=1.1, capsize=2.6, zorder=4 if pilot else 3)
    _label_points(ax, points[1:] if skip_first_label else points, color)


def plot(input_csv, out_path, *, include_1920=True):
    groups = group_runs(read_runs(input_csv))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    has_pilot = False
    try:
        for ax, (dataset, title) in zip(axes, DATASETS.items()):
            lengths, points = collect_points(groups, dataset, include_1920=include_1920)
            for (pre, ft), color, marker in zip(PAIRS, COLORS, MARKERS):
                series = [(t, points[dataset, t, pre, ft]) for t in lengths]
                main = [(t, p) for t, p in series if p.n_bases == 3]
                pilot = [(t, p) for t, p in series if p.n_bases == 1]
                _draw_series(ax, main, color, marker)
                if pilot:
                    has_pilot = True
                    # Do not join a change in replication population as if it
                    # were an optimizer effect. Connect only seed-42 means.
                    last_main = max(t for t, _ in main if t < pilot[0][0])
                    anchor = summarize(groups[dataset, last_main, pre, ft], base_seed_only=42)
                    _draw_series(ax, [(last_main, anchor), *pilot], color, marker, pilot=True,
                                 skip_first_label=True)
            ax.set_title(title, fontsize=13, fontweight="semibold", loc="left", pad=12)
            ax.set_xlabel("Learning: decrease in D₁ loss (nats/token)")
            ax.set_ylabel("Forgetting: increase in D₀ loss (nats/token)")
            ax.grid(True, color="#e6e6e6", linewidth=.75)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.margins(x=.15, y=.14)
        handles = [Line2D([], [], color=c, marker=m, lw=1.8, markersize=7,
                          label=f"{NAMES[p]} → {NAMES[f]}")
                   for (p, f), c, m in zip(PAIRS, COLORS, MARKERS)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .895), ncol=4,
                   frameon=False, fontsize=10, columnspacing=2.1)
        fig.suptitle("E=8 MoE: fine-tuning endpoint learning–forgetting trade-off", x=.5,
                     y=.97, fontsize=16, fontweight="semibold")
        fig.text(.5, .92, "Each point is a selected-LR endpoint; labels give optimizer steps.",
                 ha="center", fontsize=10, color="#525252")
        style_handles = [Line2D([], [], color="#525252", marker="o", lw=1.5,
                                label="Solid: mean ± SD across 3 base-seed means")]
        if has_pilot:
            style_handles.append(Line2D([], [], color="#525252", marker="o", markerfacecolor="white",
                                        linestyle="--", lw=1.5,
                                        label="Open/dashed: seed 42 only, mean ± SD across 3 FT seeds"))
        fig.legend(handles=style_handles, loc="lower center", bbox_to_anchor=(.5, .03),
                   ncol=len(style_handles), frameon=False, fontsize=9.5)
        fig.text(.5, .022, "Each base mean averages 3 fine-tuning seeds. Error bars are sample SD, not confidence intervals.",
                 ha="center", fontsize=9, color="#525252")
        fig.tight_layout(rect=(.025, .11, .99, .85), w_pad=3.1)
        fig.canvas.draw()
        for ax in axes:
            _place_labels(ax, fig.canvas.get_renderer())
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=.2)
        return out
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/e8/figures/finetune_endpoints.png"))
    parser.add_argument("--omit-1920", action="store_true", help="omit 1,920-step endpoints")
    args = parser.parse_args(argv)
    print(f"wrote {plot(args.input_csv, args.out, include_1920=not args.omit_1920)}")


if __name__ == "__main__":
    main()
