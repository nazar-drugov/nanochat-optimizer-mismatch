"""Plot a fine-tuning pilot summary: final learning and forgetting against the learning rate.

One column per (pretraining → fine-tuning) optimizer pair, one line per length, mean over seeds.
The ring marks the selected LR (lowest mean final D1 loss) at each length. Points that diverged
(negative learning) are drawn at the edge of the plot and listed in the caption line.

  python -m scripts.plot_finetune_pilot results/e8/finetune_pilot_starcoderdata_summary.json
  python -m scripts.plot_finetune_pilot results/e8/*_summary.json --out results/e8/figures

Input: the summary JSON lists written for each pilot (fields pre, ft, T, lr, learn, forget, d1).
Output: <out>/<summary name without "_summary">_lr.png. No GPU needed.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIRS = [("adamw", "adamw"), ("adamw", "muon"), ("muon", "adamw"), ("muon", "muon")]
NAMES = {"adamw": "AdamW", "muon": "Muon"}
DATASETS = {"starcoderdata": "StarCoderData", "wikitext103": "WikiText-103"}
# ordinal blue ramp (light surface), short → long; extra steps if a summary has more lengths
RAMP = ["#84b9f3", "#257de0", "#0f4a96", "#519ced", "#1757ac", "#0e3d7e"]
ORDERED_RAMP = {3: ["#84b9f3", "#257de0", "#0f4a96"], 4: ["#84b9f3", "#338ced", "#1b64be", "#0e3d7e"],
                5: ["#84b9f3", "#519ced", "#257de0", "#1757ac", "#0e3d7e"]}
MARKERS = ["o", "s", "D", "^", "v", "P"]
INK, INK2, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
FORGET_CAP = 3.4  # forgetting above this (diverged runs) is drawn at the edge


def selected(cells):
    """(pre, ft, T) -> LR with the lowest mean final D1 loss."""
    best = {}
    for c in cells:
        k = (c["pre"], c["ft"], c["T"])
        if k not in best or c["d1"] < best[k]["d1"]:
            best[k] = c
    return {k: c["lr"] for k, c in best.items()}


def plot(summary_path, out_dir):
    cells = json.loads(Path(summary_path).read_text())
    stem = Path(summary_path).stem.removesuffix("_summary")
    dataset = next((name for key, name in DATASETS.items() if key in stem), stem)
    lengths = sorted({c["T"] for c in cells})
    colors = ORDERED_RAMP.get(len(lengths), RAMP[:len(lengths)])
    picks = selected(cells)
    off_scale = []
    good_learn = [c["learn"] for c in cells if c["learn"] >= 0]
    learn_lim = (max(0.0, min(good_learn) - 0.08), max(good_learn) + 0.1)

    fig, axes = plt.subplots(2, len(PAIRS), figsize=(13, 6.4), sharex="col", sharey="row")
    for col, (pre, ft) in enumerate(PAIRS):
        for row, field in enumerate(("learn", "forget")):
            ax = axes[row, col]
            for i, T in enumerate(lengths):
                pts = sorted((c for c in cells if (c["pre"], c["ft"], c["T"]) == (pre, ft, T)), key=lambda c: c["lr"])
                if not pts:
                    continue
                xs, ys = [], []
                for c in pts:
                    y = c[field]
                    diverged = c["learn"] < 0
                    if diverged:
                        off_scale.append(f"{NAMES[pre]}→{NAMES[ft]} {c['lr']:.3g} at {T} steps")
                        y = learn_lim[0] if field == "learn" else FORGET_CAP
                    xs.append(c["lr"]); ys.append(y)
                    if diverged:
                        ax.scatter([c["lr"]], [y], marker="x", s=40, color=colors[i], linewidths=2, zorder=4,
                                   clip_on=False)
                good = [(x, y) for x, y, c in zip(xs, ys, pts) if c["learn"] >= 0]
                ax.plot([x for x, _ in good], [y for _, y in good], color=colors[i], lw=2,
                        marker=MARKERS[i], ms=6, mec="white", mew=1.2, label=f"{T} steps", zorder=3)
                lr = picks.get((pre, ft, T))
                sel = next((c for c in pts if c["lr"] == lr), None)
                if sel is not None:
                    ax.scatter([sel["lr"]], [sel[field]], s=170, facecolors="none", edgecolors=INK, linewidths=1.3,
                               zorder=5)
            ax.set_xscale("log")
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
            ax.tick_params(colors=INK2, labelsize=8)
            if row == 0:
                ax.set_title(f"{NAMES[pre]} → {NAMES[ft]}", color=INK, fontsize=11)
                ax.set_ylim(*learn_lim)
            else:
                ax.set_ylim(0, FORGET_CAP)
                ax.set_xlabel("Fine-tuning matrix LR", color=INK2, fontsize=9)
    axes[0, 0].set_ylabel("Final learning (nats/token)", color=INK2, fontsize=9)
    axes[1, 0].set_ylabel("Final forgetting (nats/token)", color=INK2, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls="none", marker="o", ms=11, mfc="none", mec=INK))
    labels.append("selected LR (lowest D1 loss)")
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.995, 0.965))
    n_seeds = max(c["n"] for c in cells)
    fig.suptitle(f"{dataset} fine-tuning pilot (seed-42 bases): learning and forgetting vs LR",
                 x=0.01, ha="left", color=INK, fontsize=13)
    fig.text(0.01, 0.925, f"Mean over {n_seeds} fine-tuning seeds; pretraining → fine-tuning optimizer per column",
             color=INK2, fontsize=9)
    note = ("Diverged (learning < 0), drawn as × at the plot edge: " + "; ".join(sorted(set(off_scale)))
            if off_scale else "No run diverged.")
    fig.text(0.01, 0.01, note, color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_lr.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("summaries", type=Path, nargs="+")
    p.add_argument("--out", type=Path, default=Path("results/e8/figures"))
    args = p.parse_args(argv)
    for s in args.summaries:
        print(f"wrote {plot(s, args.out)}")


if __name__ == "__main__":
    main()
