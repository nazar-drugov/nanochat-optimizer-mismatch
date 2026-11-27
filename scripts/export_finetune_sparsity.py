"""Export selected seed-42 fine-tuning pilots and learning support across E.

Use finalized `analyze_finetune` analysis.json files for each E/dataset/length;
the script revalidates all source result/config records against checkpoint pins,
recomputes D1-only LR selection, and refuses grid-edge winners. It never chooses
a matched-learning target or computes a forgetting contrast. A partially
finished E may not be pooled into the cross-E support table.

  python -m scripts.export_finetune_sparsity --base-manifest <pins.json> \
    --analysis <path1>/analysis.json --analysis <path2>/analysis.json ... \
    --out-dir results/sparsity_finetune

By default the existing clean E8 seed-42 short-horizon records are included from
the tracked endpoint/trajectory CSVs, with their result/config source records
checked (ORCD paths or --e8-mirror-root). No new-E path is hardcoded. The
output CSVs add an explicit n_experts column; the historical E8 files are not
modified. All reported support is conditional on the single pretraining seed42.
"""

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path

from nanochat.pretrain_sweep import base_run_experts, load_base_manifest, resolve_base_pin
from scripts.analyze_finetune import EVAL_IDENTITY_FIELDS, eval_identity, load_runs, select_lrs
from scripts.finetune_validation_trajectories import (
    ENDPOINT_FIELDS, TRAJECTORY_FIELDS, read_csv, resolve_source, validate_record,
)
from scripts.matched_learning import Cohort, analyze, build_comparisons, parse_targets, support


HORIZONS = {"starcoderdata": (60, 240, 960), "wikitext103": (60, 240, 720)}
SEEDS = (1, 2, 3)
EXPERTS = {1, 4, 8, 16}
EXTRA = ("n_experts", "base_model_sha256", "base_checkpoint_run", "base_checkpoint_step")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def key(row):
    return (int(row["n_experts"]), row["dataset"], int(row["steps"]), row["pretrain_optimizer"],
            row["ft_optimizer"], int(row["base_seed"]), int(row["ft_seed"]))


class DataIdentity:
    """Cross-E data checks: identical held-out sets, training files and paired batch streams."""

    def __init__(self):
        self.eval = {}
        self.train = {}
        self.fingerprints = {}
        self.baselines = {}

    def check_baseline(self, base, dataset, d0, d1):
        for k, v in (((base, "D0"), d0), ((base, dataset, "D1"), d1)):
            check(math.isclose(self.baselines.setdefault(k, v), v, rel_tol=0, abs_tol=1e-10),
                  f"pretrained loss baseline differs for {k}")

    def check(self, dataset, horizon, seed, config, fingerprint):
        check(config.get("warmup_steps") == 8 and config.get("decay_steps") == round(.2*horizon)
              and config["args"].get("total_batch_size") == 65536
              and config["args"].get("device_batch_size") == 32,
              f"cross-E pilot schedule or batch recipe differs for {dataset}, {horizon}")
        data = config["data"]
        identity = tuple((d, tuple(data["eval_sets"][d].get(f) for f in (*EVAL_IDENTITY_FIELDS, "packing")))
                         for d in ("D0", "D1"))
        check(all(None not in v and v[-1] == "bestfit" for _, v in identity),
              f"missing/incompatible evaluation identity for {dataset}")
        check(self.eval.setdefault(dataset, identity) == identity, f"cross-E evaluation set differs for {dataset}")
        train = data.get("ft_train_files")
        check(bool(train) and self.train.setdefault(dataset, train) == train,
              f"cross-E training file identity differs for {dataset}")
        paired = (dataset, horizon, seed)
        check(fingerprint is not None and self.fingerprints.setdefault(paired, fingerprint) == fingerprint,
              f"cross-E paired data fingerprint differs for {paired}")


def export_run(E, dataset, horizon, run, pin, identity, *, source="pilot_seed42"):
    identity.check(dataset, horizon, run.seed, run.config, run.fingerprint)
    base = run.config["base"]
    prefix = dict(n_experts=E, dataset=dataset, steps=horizon, pretrain_optimizer=pin.optimizer,
                  ft_optimizer=run.ft, base_seed=pin.seed, ft_seed=run.seed, lr=run.lr,
                  source=source, source_run=str(run.path), base_model_sha256=base.get("model_sha256") or pin.model_sha256,
                  base_checkpoint_run=pin.checkpoint_dir.name, base_checkpoint_step=pin.step)
    check(run.status == "completed" and run.curve and run.curve[-1]["step"] == horizon, "selected run incomplete")
    baseline = None; rows = []
    for p in run.curve:
        d0, d1, l, f = (float(p[k]) for k in ("loss_d0", "loss_d1", "learning", "forgetting"))
        b = d0-f, d1+l
        check(all(math.isfinite(x) for x in (d0, d1, l, f)), "nonfinite selected losses")
        if baseline is None: baseline = b
        check(all(math.isclose(x, y, rel_tol=0., abs_tol=1e-10) for x, y in zip(b, baseline)),
              f"inconsistent loss-change baseline: {run.path}")
        rows.append(dict(prefix, checkpoint_step=p["step"], loss_d0=d0, loss_d1=d1, learning=l, forgetting=f))
    identity.check_baseline(run.base, dataset, *baseline)
    origin = dict(prefix, checkpoint_step=0, loss_d0=baseline[0], loss_d1=baseline[1], learning=0., forgetting=0.)
    end = {k: v for k, v in rows[-1].items() if k != "checkpoint_step"}
    end.update(base_loss_d0=baseline[0], base_loss_d1=baseline[1])
    return end, [origin, *rows]


def export_analysis(path, pins, identity):
    """Recompute the D1 LR choices and verify that the finalized analysis matches source files."""
    a = json.loads(Path(path).read_text())
    bases = tuple(a.get("bases", ()))
    Eset = {base_run_experts(b) for b in bases}
    check(len(Eset) == 1, f"analysis mixes expert counts: {path}")
    E = Eset.pop()
    check(E in EXPERTS and E != 8, f"use the tracked E8 source separately; unexpected analysis E={E}")
    expected = (f"e{E}-main-adamw-s42", f"e{E}-main-muon-s42")
    check(set(bases) == set(expected) and len(bases) == 2 and sorted(a.get("seeds", [])) == list(SEEDS),
          f"non-E8 pilot must use exactly seed42 base pair, FT seeds 1–3: {path}")
    dataset, horizon = a.get("dataset"), a.get("steps")
    check(dataset in HORIZONS and horizon in HORIZONS[dataset], f"unapproved pilot length/dataset: {path}")
    check(a.get("n_experts") == E and a.get("status") == "grid_brackets_optima" and not a.get("edge_winners")
          and not a.get("lr_selection_source"), f"analysis does not establish an independently bracketed D1 optimum: {path}")
    roots = [Path(p) for p in a.get("source_roots", [])]
    check(bool(roots), f"missing analysis source roots: {path}")
    runs = load_runs(roots, dataset, horizon, SEEDS, expected, pins)
    selected, rows = select_lrs(runs, SEEDS, 5, expected)
    check(not any(r["edge"] for r in rows if r["selected"]), f"grid-edge winner must be extended: {path}")
    saved = {(r["base_run"], r["ft_optimizer"]): r["lr"] for r in a["lr_selection"] if r["selected"]}
    check(saved == selected, f"saved D1 selection differs from current validated source runs: {path}")
    endpoints, trajectories = [], []
    for run in runs:
        if run.lr != selected[run.arm]: continue
        end, curve = export_run(E, dataset, horizon, run, resolve_base_pin(run.base, pins), identity)
        endpoints.append(end); trajectories.extend(curve)
    check(len(endpoints) == 12, f"expected 12 selected pilot endpoints for {path}")
    return (E, dataset, horizon), endpoints, trajectories


def export_e8(endpoints_path, trajectories_path, mirror, identity, pins):
    endpoints = [r for r in read_csv(endpoints_path, ENDPOINT_FIELDS)
                 if int(r["base_seed"]) == 42 and int(r["steps"]) in HORIZONS[r["dataset"]]]
    trajectories = [r for r in read_csv(trajectories_path, TRAJECTORY_FIELDS)
                    if int(r["base_seed"]) == 42 and int(r["steps"]) in HORIZONS[r["dataset"]]]
    check(len(endpoints) == 72, "E8 seed42 shortened pilot must cover 72 selected endpoints")
    provenance = {}
    by_source = defaultdict(list)
    for row in trajectories:
        by_source[row["source_run"]].append(row)
    for r in endpoints:
        path = resolve_source(r["source_run"], mirror)
        result, config = (json.loads((path / filename).read_text()) for filename in ("result.json", "config.json"))
        curve = validate_record(r, result, config)
        saved = sorted(by_source[r["source_run"]], key=lambda p: int(p["checkpoint_step"]))
        check([int(p["checkpoint_step"]) for p in saved] == [0, *[p["step"] for p in curve]],
              f"E8 tracked trajectory schedule differs from its source: {path}")
        check(all(float(t[k]) == float(p[k]) for t, p in zip(saved[1:], curve)
                  for k in ("learning", "forgetting", "loss_d0", "loss_d1")),
              f"E8 tracked trajectory losses differ from its source: {path}")
        identity.check(r["dataset"], int(r["steps"]), int(r["ft_seed"]), config, result.get("data_fingerprint"))
        pin = resolve_base_pin(f"e8-main-{r['pretrain_optimizer']}-s42", pins)
        base = config["base"]
        check(base.get("model_sha256") == pin.model_sha256 and base.get("step") == pin.step,
              f"E8 tracked source has unexpected/missing base pin: {path}")
        identity.check_baseline(f"e8-main-{r['pretrain_optimizer']}-s42", r["dataset"], float(r["base_loss_d0"]), float(r["base_loss_d1"]))
        fields = dict(n_experts=8, base_model_sha256=pin.model_sha256,
                      base_checkpoint_run=pin.checkpoint_dir.name, base_checkpoint_step=pin.step)
        provenance[r["source_run"]] = fields
        r.update(fields)
    for r in trajectories:
        r.update(provenance[r["source_run"]])
    return endpoints, trajectories


def e_comparisons(endpoints, trajectories):
    check(len({key(r) for r in endpoints}) == len(endpoints), "duplicate selected endpoint across analyses")
    e_values = sorted({int(r["n_experts"]) for r in endpoints})
    check(e_values and set(e_values) <= EXPERTS, "unexpected expert counts")
    comparisons = {}
    for E in e_values:
        end = [r for r in endpoints if int(r["n_experts"]) == E]
        traj = [r for r in trajectories if int(r["n_experts"]) == E]
        cohorts = tuple(Cohort(dataset, "pilot_seed42", (42,), horizons) for dataset, horizons in HORIZONS.items())
        comparisons[E] = build_comparisons(end, traj, cohorts)
    return comparisons


def common_support(endpoints, trajectories):
    """Return per-E and intersected common support without computing contrasts."""
    comparisons = e_comparisons(endpoints, trajectories)
    e_values = sorted(comparisons)
    rows = []
    for E, comps in comparisons.items():
        for c in comps:
            lo, hi = support(c)
            rows.append(dict(n_experts=E, dataset=c.cohort.dataset, estimand=c.estimand,
                             horizon=c.horizon or "", support_min=lo, support_max=hi, supported=lo <= hi))
    grouped = defaultdict(list)
    for r in rows: grouped[r["dataset"], r["estimand"], r["horizon"]].append(r)
    common = []
    for (dataset, estimand, horizon), values in grouped.items():
        check({r["n_experts"] for r in values} == set(e_values), "missing E in common support")
        lo, hi = max(r["support_min"] for r in values), min(r["support_max"] for r in values)
        common.append(dict(expert_counts=",".join(map(str, e_values)), dataset=dataset, estimand=estimand,
                           horizon=horizon, support_min=lo, support_max=hi, supported=lo <= hi))
    return rows, common


def explicit_target_analysis(endpoints, trajectories, common, targets):
    """Optional future caller-declared targets; never choose one from outcomes.

    No across-E group is partially analyzed: even if some E reaches an explicit
    target, all E must have common support for that comparison.
    """
    comparisons = e_comparisons(endpoints, trajectories)
    limits = {(r["dataset"], r["estimand"], r["horizon"]): r for r in common}
    summaries, seeds, selections = [], [], []
    for E, comps in comparisons.items():
        for c in comps:
            if c.estimand not in targets:
                continue
            target = targets[c.estimand][c.cohort.dataset]
            limit = limits[c.cohort.dataset, c.estimand, c.horizon or ""]
            if not limit["support_min"] <= target <= limit["support_max"]:
                for pre in ("adamw", "muon"):
                    summaries.append(dict(n_experts=E, dataset=c.cohort.dataset, cohort=c.cohort.name,
                                          estimand=c.estimand, horizon=c.horizon or "", pretrain_optimizer=pre,
                                          target_learning=target, status="unsupported_across_e",
                                          common_e_support_min=limit["support_min"], common_e_support_max=limit["support_max"]))
                continue
            sr, pr, cr = analyze(c, target, details=True)
            summaries.extend(dict(n_experts=E, common_e_support_min=limit["support_min"],
                                  common_e_support_max=limit["support_max"], **r) for r in sr)
            seeds.extend(dict(n_experts=E, **r) for r in pr)
            selections.extend(dict(n_experts=E, **r) for r in cr)
    return summaries, seeds, selections


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-manifest", type=Path, required=True)
    p.add_argument("--analysis", type=Path, action="append", default=[])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--exclude-e8", action="store_true", help="omit existing E8 tracked seed42 records for partial-pilot checks")
    p.add_argument("--e8-endpoints", type=Path, default=Path("results/e8/finetune_endpoint_runs.csv"))
    p.add_argument("--e8-trajectories", type=Path, default=Path("results/e8/finetune_validation_trajectories.csv"))
    p.add_argument("--e8-mirror-root", type=Path, default=Path(".codex/wandb-backfill"))
    p.add_argument("--endpoint-target", action="append", default=[], metavar="DATASET=TARGET",
                   help="optional explicitly approved target; absent by default (support only)")
    p.add_argument("--within-run-target", action="append", default=[], metavar="DATASET=TARGET",
                   help="optional explicitly approved target; absent by default (support only)")
    args = p.parse_args(argv)
    pins = load_base_manifest(args.base_manifest)
    identity = DataIdentity(); endpoints = []; trajectories = []; seen = set()
    if not args.exclude_e8:
        e, t = export_e8(args.e8_endpoints, args.e8_trajectories, args.e8_mirror_root, identity, pins)
        endpoints.extend(e); trajectories.extend(t)
    for path in args.analysis:
        k, e, t = export_analysis(path, pins, identity)
        check(k not in seen, f"duplicate E/dataset/horizon analysis {k}")
        seen.add(k); endpoints.extend(e); trajectories.extend(t)
    # build_comparisons fails on missing/extra lengths per E (no deceptively
    # narrow partial coverage); targets are deliberately absent at this stage.
    per_e, common = common_support(endpoints, trajectories)
    targets = {estimand: parse_targets(values) for estimand, values in
               (("endpoint", args.endpoint_target), ("within-run", args.within_run_target)) if values}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "selected_endpoints.csv", (*EXTRA, *ENDPOINT_FIELDS), sorted(endpoints, key=key))
    write_csv(args.out_dir / "selected_trajectories.csv", (*EXTRA, *TRAJECTORY_FIELDS),
              sorted(trajectories, key=lambda r: (*key(r), int(r["checkpoint_step"]))))
    write_csv(args.out_dir / "support_per_e.csv", tuple(per_e[0]), per_e)
    write_csv(args.out_dir / "common_support.csv", tuple(common[0]), common)
    if targets:
        summaries, seeds, selections = explicit_target_analysis(endpoints, trajectories, common, targets)
        for name, rows in (("matched_summary", summaries), ("matched_per_seed", seeds), ("matched_selections", selections)):
            if rows:
                fields = tuple(dict.fromkeys(k for r in rows for k in r))
                write_csv(args.out_dir / f"{name}.csv", fields, rows)
    (args.out_dir / "metadata.json").write_text(json.dumps({"expert_counts":sorted({int(r["n_experts"]) for r in endpoints}),
         "base_seed":42,"ft_seeds":SEEDS,"horizons":HORIZONS,"analysis_paths":[str(p) for p in args.analysis],
         "manifest":str(args.base_manifest),"e8_included":not args.exclude_e8,"explicit_targets":targets,
         "note":"D1-only selected LR; support only unless caller supplies explicit approved targets"}, indent=2)+"\n")
    print(f"Exported {len(endpoints)} selected endpoints; common support across E in {args.out_dir}")


if __name__ == "__main__":
    main()
