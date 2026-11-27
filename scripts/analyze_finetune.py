"""Analyze the fine-tuning LR grid and optimizer-mismatch measurements.

Run after the grid finishes, e.g.:
  python -m scripts.analyze_finetune --root <hgo-output> --root <nazar-output> \
      --dataset starcoderdata --steps 240 --seeds 1 2 3 --out <analysis-output>
Add --ell0 only after declaring a common D1-learning target. No GPU or W&B needed.

Main grid (seeds 43/44 reuse the seed-42 LRs): first analyze the seed-42 pilot at the same length,
then pass its analysis.json with --lr-selection and list every base run with --bases, e.g.
  python -m scripts.analyze_finetune --root <pilot> --root <main> --dataset starcoderdata --steps 960 \
      --seeds 1 2 3 --bases e8-main-adamw-s42 e8-main-adamw-s43 e8-main-adamw-s44 \
      e8-main-muon-s42 e8-main-muon-s43 e8-main-muon-s44 --lr-selection <pilot-analysis>/analysis.json --out <dir>
A (base, optimizer, seed, LR) cell found under two roots is an error: point --root at directories that
hold one copy of each run (e.g. a staging root of symlinked <base>/<dataset> directories).
"""
import argparse
import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nanochat.pretrain_sweep import FINETUNE_BASE_CHECKPOINTS, base_run_seed, base_run_experts, load_base_manifest, resolve_base_pin

BASES = ("e8-main-adamw-s42", "e8-main-muon-s42")
OPTS = ("adamw", "muon")
ROUTER = ("flip_rate", "maxvio", "tv", "churn", "flip_rate_normalized")
PACKINGS = ("bestfit", "contiguous")
LEGACY_PACKING = "contiguous"  # runs written before the packing field existed used contiguous rows
EVAL_IDENTITY_FIELDS = ("sha256", "scored_tokens", "rows", "seq_len", "documents")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


@dataclass
class Run:
    path: Path
    base: str
    ft: str
    seed: int
    lr: float
    steps: int
    curve: list
    config: dict
    fingerprint: int | None = None
    status: str = "completed"
    schedule: tuple = (None, None)  # (warmup_steps, decay_steps)

    @property
    def arm(self):
        return self.base, self.ft


def load_runs(roots, dataset, steps, seeds, bases=BASES, base_manifest=None):
    """Completed and diverged runs; a diverged run keeps its curve up to the divergence."""
    if isinstance(roots, Path):
        roots = [roots]
    runs, seen = [], {}
    experts = {base_run_experts(base) for base in bases}
    check(len(experts) == 1, "one fine-tuning analysis must compare bases of one E")
    if isinstance(base_manifest, (str, Path)):
        base_manifest = load_base_manifest(base_manifest)
    for base in bases:
        pin = resolve_base_pin(base, base_manifest)
        E = pin.n_experts
        strict_identity = E != 8
        strict_hash = strict_identity or (base_manifest is not None and base in base_manifest)
        for root in roots:
            folder = root / base / dataset
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*/result.json")):
                result = read(path)
                if result.get("ft_steps", result.get("steps")) != steps:
                    continue
                config_path = path.parent / "config.json"
                check(config_path.is_file(), f"missing config: {config_path}")
                config = read(config_path)
                cell = config.get("cell", {})
                status = result.get("status")
                check(status in ("completed", "diverged"), f"incomplete run: {path}")
                check(result.get("base_run") == base and result.get("ft_dataset") == dataset,
                      f"result identity mismatch: {path}")
                ft, seed, lr = result["ft_optimizer"], result["seed"], result["lr"]
                check(ft in OPTS and seed in seeds and math.isfinite(lr) and lr > 0,
                      f"unexpected optimizer, seed or LR: {path}")
                for field in ("base_run", "ft_optimizer", "seed", "lr", "ft_steps"):
                    check(cell.get(field) == result.get(field), f"config/result {field} mismatch: {path}")
                identity = config.get("base", {})
                check(identity.get("step") == pin.step and identity.get("user_config", {}).get("seed") == pin.seed,
                      f"wrong base checkpoint step or seed: {path}")
                check(identity["user_config"].get("optimizer_name") == pin.optimizer,
                      f"wrong base optimizer: {path}")
                if strict_hash or identity.get("model_sha256"):
                    check(identity.get("model_sha256") == pin.model_sha256,
                          f"wrong base checkpoint SHA-256: {path}")
                model_cfg = config.get("model_config", {})
                check(model_cfg.get("n_experts", 8 if not strict_identity else None) == E,
                      f"unexpected expert count: {path}")
                if strict_identity:
                    check(cell.get("n_experts") == result.get("n_experts") == E and
                          cell.get("pretrain_optimizer") == result.get("pretrain_optimizer") == pin.optimizer and
                          identity.get("logical_run") == base and identity.get("n_experts") == E and
                          identity.get("checkpoint_run") == pin.checkpoint_dir.name and
                          Path(identity.get("dir") or "").resolve() == pin.checkpoint_dir.resolve() and
                          Path(identity.get("model_file") or "").resolve() ==
                          (pin.checkpoint_dir / f"model_{pin.step:06d}.pt").resolve(),
                          f"wrong logical/actual checkpoint provenance: {path}")
                    check(identity["user_config"].get(f"{pin.optimizer}_matrix_lr") == pin.matrix_lr,
                          f"wrong base checkpoint matrix LR: {path}")
                    check(config.get("warmup_steps") == result.get("warmup_steps") == 8 and
                          config.get("decay_steps") == result.get("decay_steps") == round(.2 * steps) and
                          config.get("args", {}).get("total_batch_size") == 65536 and
                          config.get("args", {}).get("device_batch_size") == 32 and
                          all(config.get("data", {}).get("eval_sets", {}).get(d, {}).get("packing") == "bestfit"
                              for d in ("D0", "D1")), f"new-E pilot recipe differs: {path}")
                    if status == "completed":
                        check(result.get("data_fingerprint") is not None,
                              f"missing paired data fingerprint: {path}")
                if E == 1:
                    check(model_cfg.get("disable_routing_loss") is True and model_cfg.get("balance_coef") == 0
                          and model_cfg.get("z_coef") == 0, f"E=1 control routing loss enabled: {path}")
                curve = result.get("curve", [])
                if status == "completed":
                    check(curve and curve[-1]["step"] == steps, f"missing final checkpoint: {path}")
                    if strict_identity:
                        from scripts.finetune import checkpoint_steps
                        expected = checkpoint_steps(steps, round(.2 * steps))
                        check([p["step"] for p in curve] == config.get("checkpoint_steps") == expected,
                              f"unexpected checkpoint schedule: {path}")
                        check(result.get("final_learning") == curve[-1]["learning"] and
                              result.get("final_forgetting") == curve[-1]["forgetting"],
                              f"saved final learning/forgetting disagrees with curve: {path}")
                else:
                    check(not curve or curve[-1]["step"] <= steps, f"checkpoint past the end: {path}")
                check(all(a["step"] < b["step"] for a, b in zip(curve, curve[1:])),
                      f"nonmonotone checkpoints: {path}")
                for point in curve:
                    for field in ("learning", "forgetting", "F_route", "F_fixed", "loss_d1"):
                        check(math.isfinite(point[field]), f"nonfinite {field}: {path}")
                    check(abs(point["F_route"] + point["F_fixed"] - point["forgetting"]) < 1e-7,
                          f"replay identity fails: {path}, step {point['step']}")
                    if E == 1:
                        check(point["F_route"] == 0., f"E=1 route replay must be structural zero: {path}")
                key = base, ft, int(seed), f"{float(lr):.6g}"
                check(key not in seen, f"duplicate run: {path} and {seen.get(key)}")
                seen[key] = path
                runs.append(Run(path.parent, base, ft, int(seed), float(lr), steps, curve, config,
                                result.get("data_fingerprint"), status,
                                (result.get("warmup_steps"), result.get("decay_steps"))))
    check(runs, f"no completed runs for {dataset}, {steps} steps")
    schedules = {r.schedule for r in runs if r.schedule != (None, None)}
    check(len(schedules) <= 1, f"warmup/decay steps differ across runs: {sorted(schedules, key=str)}")
    eval_packing(runs)
    for d in ("D0", "D1"):
        identities = {tuple(eval_identity(r, d).get(field) for field in EVAL_IDENTITY_FIELDS) for r in runs}
        check(len(identities) == 1, f"{d} evaluation set differs across runs "
                                    f"(compared {', '.join(EVAL_IDENTITY_FIELDS)}, packing)")
    check(len({r.config["args"]["total_batch_size"] for r in runs}) == 1,
          "batch sizes differ across runs")
    train_sets = {json.dumps(r.config["data"].get("ft_train_files"), sort_keys=True) for r in runs}
    check(len(train_sets) == 1, "fine-tuning training data differ across runs")
    for seed in seeds:
        fingerprints = {r.fingerprint for r in runs
                        if r.seed == seed and r.fingerprint is not None and r.status == "completed"}
        check(len(fingerprints) <= 1, f"paired data order differs for seed {seed}")
    return runs


def mean_final_d1(cells):
    """Mean final D1 loss over seeds; None (worst) if any seed diverged."""
    if any(r.status != "completed" for r in cells):
        return None
    return float(np.mean([r.curve[-1]["loss_d1"] for r in cells]))


def select_lrs(runs, seeds, min_lrs, bases=BASES):
    selected, rows = {}, []
    for base in bases:
        for ft in OPTS:
            arm = base, ft
            group = [r for r in runs if r.arm == arm]
            grid = sorted({r.lr for r in group})
            check(len(grid) >= min_lrs, f"{arm}: only {len(grid)} LRs, expected {min_lrs}")
            group_rows = []
            for lr in grid:
                cells = [r for r in group if r.lr == lr]
                check(len(cells) == len(seeds) and {r.seed for r in cells} == set(seeds),
                      f"{arm}, LR {lr}: missing or duplicate seeds")
                row = {"base_run": base, "ft_optimizer": ft, "lr": lr,
                       "mean_final_d1_loss": mean_final_d1(cells),
                       "seeds": len(cells), "diverged_seeds": sum(r.status != "completed" for r in cells),
                       "selected": False, "edge": False}
                group_rows.append(row)
            finite = [x for x in group_rows if x["mean_final_d1_loss"] is not None]
            check(finite, f"{arm}: every LR diverged")
            choice = min(finite, key=lambda x: (x["mean_final_d1_loss"], x["lr"]))
            choice["selected"] = True
            choice["edge"] = choice["lr"] in (grid[0], grid[-1])
            selected[arm] = choice["lr"]
            rows.extend(group_rows)
    return selected, rows


def fixed_lrs(runs, seeds, path, dataset, steps, bases=BASES):
    """Use the LR selected for each (pretraining optimizer, fine-tuning optimizer) pair in an earlier
    analysis (the seed-42 pilot) for every base run, as the protocol prescribes for seeds 43 and 44."""
    source = read(Path(path))
    check(source.get("dataset") == dataset and source.get("steps") == steps,
          f"{path} is for {source.get('dataset')}, {source.get('steps')} steps, not {dataset}, {steps}")
    check(not source.get("lr_selection_source"), f"{path} reused LRs itself; give the analysis that selected them")
    check({base_run_experts(row["base_run"]) for row in source["lr_selection"]} ==
          {base_run_experts(base) for base in bases}, f"{path} selected LRs for a different E")
    check(all(row["base_run"].endswith("-s42") for row in source["lr_selection"]),
          f"{path} did not select on the seed-42 base runs")
    check(sorted(source.get("seeds", [])) == sorted(seeds), f"{path} used seeds {source.get('seeds')}, not {seeds}")
    for d in ("D0", "D1"):
        current = {f: eval_identity(runs[0], d).get(f) for f in EVAL_IDENTITY_FIELDS}
        check(source.get("eval_sets", {}).get(d) == current, f"{path} used a different {d} evaluation set")
    picks, source_d1 = {}, {}
    for row in source["lr_selection"]:
        if row["selected"]:
            source_d1[row["base_run"], row["ft_optimizer"]] = row["mean_final_d1_loss"]
            pair = row["base_run"].split("-")[2], row["ft_optimizer"]
            check(pair not in picks, f"{path}: more than one selected LR for {pair}")
            check(not row["edge"], f"{path}: {pair} was selected at the grid edge")
            picks[pair] = row["lr"]
    selected, rows = {}, []
    for base in bases:
        for ft in OPTS:
            pair = base.split("-")[2], ft
            check(pair in picks, f"{path} selects no LR for {pair}")
            cells = [r for r in runs if r.arm == (base, ft) and math.isclose(r.lr, picks[pair], rel_tol=1e-6)]
            check(len(cells) == len(seeds) and {r.seed for r in cells} == set(seeds),
                  f"{(base, ft)}, LR {picks[pair]}: missing or duplicate seeds")
            check(all(r.status == "completed" for r in cells), f"{(base, ft)}, LR {picks[pair]}: a seed diverged")
            if (base, ft) in source_d1:  # the seed-42 runs must be the ones the selection used
                check(math.isclose(mean_final_d1(cells), source_d1[(base, ft)], rel_tol=1e-9, abs_tol=1e-12),
                      f"{(base, ft)}: the runs found differ from those {path} selected on")
            selected[(base, ft)] = cells[0].lr
            rows.append({"base_run": base, "ft_optimizer": ft, "lr": cells[0].lr,
                         "mean_final_d1_loss": mean_final_d1(cells), "seeds": len(cells), "diverged_seeds": 0,
                         "selected": True, "edge": False})
    return selected, rows


def pooled(penalties_by_base):
    """Mean mismatch penalty over the base runs of each pretraining optimizer."""
    by_opt = {}
    for base, value in penalties_by_base.items():
        by_opt.setdefault(base.split("-")[2], []).append(value)
    return {opt: float(np.mean(values)) for opt, values in by_opt.items()}


def match_curve(curve, ell0):
    """Minimum piecewise-linear forgetting among points with learning >= ell0."""
    points = [{"step": 0, "learning": 0.0, "forgetting": 0.0,
               "F_route": 0.0, "F_fixed": 0.0}, *curve]
    candidates = []
    for i, p in enumerate(points):
        if p["learning"] >= ell0:
            candidates.append((float(p["forgetting"]), float(p["step"]), i, i, 0.0))
    for i, (a, b) in enumerate(zip(points, points[1:])):
        lo, hi = sorted((a["learning"], b["learning"]))
        if lo < ell0 < hi:
            w = (ell0 - a["learning"]) / (b["learning"] - a["learning"])
            f = a["forgetting"] + w * (b["forgetting"] - a["forgetting"])
            t = a["step"] + w * (b["step"] - a["step"])
            candidates.append((float(f), float(t), i, i + 1, float(w)))
    if not candidates:
        return None
    f, t, i, j, w = min(candidates, key=lambda x: (x[0], x[1]))
    value = lambda field: float(points[i][field] * (1-w) + points[j][field] * w)
    return {"forgetting": f, "learning": value("learning"), "target_learning": float(ell0), "step": t,
            "F_route": value("F_route"), "F_fixed": value("F_fixed"),
            "left_step": points[i]["step"], "right_step": points[j]["step"], "weight": w}

def router_at(run, match):
    by_step = {}
    with (run.path / "metrics.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["eval_set"] == "D0" and row["layer"] >= 0:
                by_step[(row["step"], row["layer"])] = row
    result = []
    for layer in run.config["moe_layer_indices"]:
        left = by_step.get((match["left_step"], layer))
        right = by_step.get((match["right_step"], layer))
        check(left is not None or right is not None,
              f"missing router metrics: {run.path}, layer {layer}")
        if left is None:  # step zero has no route changes
            left = {k: 0.0 for k in ROUTER}
            left["maxvio"] = right["base_maxvio"]
        if right is None:
            right = left
        row = {"base_run": run.base, "ft_optimizer": run.ft, "seed": run.seed,
               "lr": run.lr, "layer": layer, "step": match["step"]}
        for field in ROUTER:
            w = match["weight"]
            a, b = left.get(field), right.get(field)
            row[field] = (None if a is None else float(a)) if w == 0 else (
                (None if b is None else float(b)) if w == 1 else
                (None if a is None or b is None else float(a * (1-w) + b * w)))
        result.append(row)
    return result


def eval_identity(run, d):
    identity = run.config.get("data", {}).get("eval_sets", {}).get(d)
    check(isinstance(identity, dict), f"missing {d} evaluation-set identity in config: {run.path}")
    return identity


def run_packing(run):
    """(packing, legacy) of one run; a missing field means the original contiguous packing."""
    found, legacy = set(), False
    for d in ("D0", "D1"):
        value = eval_identity(run, d).get("packing")
        if value is None:
            value, legacy = LEGACY_PACKING, True
        check(value in PACKINGS, f"unknown evaluation packing {value!r} for {d}: {run.path}")
        found.add(value)
    check(len(found) == 1, f"D0 and D1 use different evaluation packings: {run.path}")
    return found.pop(), legacy


def eval_packing(runs):
    """The one evaluation packing shared by all runs, plus how many runs lack the field."""
    info = [(r, *run_packing(r)) for r in runs]
    packings = sorted({packing for _, packing, _ in info})
    if len(packings) > 1:
        legacy = [r for r, _, is_legacy in info if is_legacy]
        counts = ", ".join(f"{p}: {sum(1 for _, q, _ in info if q == p)} runs" for p in packings)
        if legacy:
            raise ValueError(f"runs without a packing field (legacy {LEGACY_PACKING}, e.g. {legacy[0].path}) "
                             f"cannot be combined with other packings ({counts})")
        raise ValueError(f"evaluation packing differs across runs ({counts})")
    return {"packing": packings[0], "runs_without_packing_field": sum(1 for _, _, is_legacy in info if is_legacy)}


@lru_cache(maxsize=None)
def check_eval_set_file(path, d, packing, sha256, n_docs, counts):
    """Per-document checks against the saved evaluation set, when the file was copied with the run:
    IDs inside [-1, n_docs), per-document scored counts equal the saved ones, and for best-fit sets
    documents listed in file order and never split across rows."""
    from nanochat.eval_set import EvalSet
    es = EvalSet.load(path)
    ids = es.doc_ids.numpy()
    check(es.packing == packing, f"{path}: saved set packs {es.packing}, config says {packing}")
    check(sha256 is None or es.source_sha256 == sha256, f"{path}: saved set was built from a different file")
    check(es.n_docs == n_docs, f"{path}: {es.n_docs} documents in the set, {n_docs} in the saved arrays")
    check(ids.shape == tuple(es.inputs.shape) and ids.min() >= -1 and ids.max() < n_docs,
          f"{path}: document IDs outside [-1, {n_docs})")
    scored = ids[ids >= 0]
    if packing == "bestfit":
        rows = es.doc_rows
        check(rows is not None and len(rows) == n_docs and bool((rows[1:] > rows[:-1]).all()),
              f"{path}: best-fit set needs doc_rows with one increasing file row per document")
        row_of = np.broadcast_to(np.arange(ids.shape[0])[:, None], ids.shape)[ids >= 0]
        first = np.full(n_docs, ids.shape[0]); last = np.full(n_docs, -1)
        np.minimum.at(first, scored, row_of); np.maximum.at(last, scored, row_of)
        used = last >= 0
        check(np.array_equal(first[used], last[used]), f"{path}: a best-fit document is split across rows")
    check(np.array_equal(np.bincount(scored, minlength=n_docs), np.array(counts)),
          f"{path}: per-document scored counts differ from the saved arrays")
    return True


def doc_data(run):
    path = run.path.parent / "base" / "base_arrays.npz"
    check(path.is_file(), f"missing base arrays: {path}")
    with np.load(path) as b:
        base0, count0 = b["d0_doc_loss"].astype(float), b["d0_doc_tokens"].astype(float)
        base1, count1 = b["d1_doc_loss"].astype(float), b["d1_doc_tokens"].astype(float)
    d0, replay, d1 = [], [], []
    for point in run.curve:
        ckpt = run.path / f"ckpt_{point['step']:06d}.npz"
        check(ckpt.is_file(), f"missing checkpoint arrays: {ckpt}")
        with np.load(ckpt) as z:
            d0.append(z["d0_doc_loss"].astype(float))
            replay.append(z["d0_doc_loss_replay"].astype(float))
            d1.append(z["d1_doc_loss"].astype(float))
    d0, replay, d1 = np.stack(d0), np.stack(replay), np.stack(d1)
    check(d0.shape[1] == len(count0) == len(base0) and d1.shape[1] == len(count1) == len(base1),
          f"per-document shape mismatch: {run.path}")
    check(count0.sum() > 0 and count1.sum() > 0, f"empty evaluation set: {run.path}")
    for d, counts in (("D0", count0), ("D1", count1)):
        identity = run.config.get("data", {}).get("eval_sets", {}).get(d)
        if identity is None:
            continue
        check(identity.get("documents", len(counts)) == len(counts),
              f"{d} has {len(counts)} documents in the arrays, {identity.get('documents')} in the config: {run.path}")
        check(identity.get("scored_tokens", counts.sum()) == counts.sum(),
              f"{d} scored tokens in the arrays differ from the config: {run.path}")
        file = run.path.parent / "base" / f"eval_set_{d}.pt"
        if file.is_file():
            check_eval_set_file(str(file), d, run_packing(run)[0], identity.get("sha256"), len(counts),
                                tuple(counts.astype(int)))
    l00, l01 = base0.sum()/count0.sum(), base1.sum()/count1.sum()
    for i, point in enumerate(run.curve):
        l10, l11 = d0[i].sum()/count0.sum(), d1[i].sum()/count1.sum()
        rep = replay[i].sum()/count0.sum()
        for field, value in (("forgetting", l10-l00), ("learning", l01-l11),
                             ("F_route", l10-rep), ("F_fixed", rep-l00)):
            check(abs(point[field]-value) < 1e-6,
                  f"per-document arrays disagree with {field}: {run.path}, step {point['step']}")
    return base0, count0, base1, count1, d0, replay, d1


def resampled_curve(run, data, idx0, idx1):
    base0, count0, base1, count1, d0, replay, d1 = data
    n0, n1 = count0[idx0].sum(), count1[idx1].sum()
    l00, l01 = base0[idx0].sum()/n0, base1[idx1].sum()/n1
    l10, l11 = d0[:, idx0].sum(axis=1)/n0, d1[:, idx1].sum(axis=1)/n1
    rep = replay[:, idx0].sum(axis=1)/n0
    return [{"step": p["step"], "learning": float(l01-l11[i]),
             "forgetting": float(l10[i]-l00), "F_route": float(l10[i]-rep[i]),
             "F_fixed": float(rep[i]-l00)} for i, p in enumerate(run.curve)]


def bootstrap(runs, seeds, ell0, repetitions, random_seed, bases=BASES):
    if repetitions == 0:
        return None
    data = {id(r): doc_data(r) for r in runs}
    n0 = {len(d[1]) for d in data.values()}
    n1 = {len(d[3]) for d in data.values()}
    check(len(n0) == len(n1) == 1, "document counts differ across runs")
    n0, n1 = n0.pop(), n1.pop()
    rng = np.random.default_rng(random_seed)
    lookup = {(r.base, r.ft, r.seed): r for r in runs}
    values = {(base, ft): [] for base in bases for ft in OPTS}
    penalties = {base: [] for base in bases}
    pooled_penalties = {}
    rejected = 0
    for _ in range(repetitions):
        sample_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        idx0 = rng.integers(n0, size=n0)
        idx1 = rng.integers(n1, size=n1)
        arm_f = {}
        for arm in values:
            found = []
            for seed in sample_seeds:
                run = lookup[(*arm, int(seed))]
                m = match_curve(resampled_curve(run, data[id(run)], idx0, idx1), ell0)
                if m is None:
                    break
                found.append(m["forgetting"])
            if len(found) != len(seeds):
                break
            arm_f[arm] = float(np.mean(found))
        if len(arm_f) != len(values):
            rejected += 1
            continue
        for arm, value in arm_f.items():
            values[arm].append(value)
        sample = {}
        for base in bases:
            opt = base.split("-")[2]
            other = "muon" if opt == "adamw" else "adamw"
            sample[base] = arm_f[(base, other)]-arm_f[(base, opt)]
            penalties[base].append(sample[base])
        for opt, value in pooled(sample).items():
            pooled_penalties.setdefault(opt, []).append(value)
    accepted = repetitions-rejected
    check(accepted >= max(1, math.ceil(0.8*repetitions)),
          f"too many bootstrap samples cannot reach ell0 ({rejected}/{repetitions}); lower ell0")
    ci = lambda x: [float(v) for v in np.quantile(x, [0.025, 0.975])]
    return {"repetitions": repetitions, "accepted": accepted, "rejected": rejected,
            "method": "paired seed and common D0/D1 document resampling, selected LRs fixed"
                      + (", base runs fixed (not resampled)" if len(bases) > 2 else ""),
            "arm_ci": {"/".join(k): ci(v) for k, v in values.items()},
            "mismatch_ci": {k: ci(v) for k, v in penalties.items()},
            "pooled_mismatch_ci": {k: ci(v) for k, v in pooled_penalties.items()}}


def midranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2
        i = j
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = midranks(x), midranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def write_csv(path, rows):
    if not rows:
        path.unlink(missing_ok=True)
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def arm_label(base, ft):
    _, _, pre, seed = base.split("-")
    return f"{pre[0].upper()}{seed}→{ft[0].upper()}"


def make_plots(out, runs, lr_rows, selected, matches, router_rows, router_runs, ell0, bases=BASES):
    fig, axes = plt.subplots(1, len(bases), figsize=(5.5*len(bases), 4), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, base in zip(axes, bases):
        for ft in OPTS:
            rows = sorted((x for x in lr_rows if (x["base_run"], x["ft_optimizer"]) == (base, ft)
                           and x["mean_final_d1_loss"] is not None), key=lambda x: x["lr"])
            ax.plot([x["lr"] for x in rows], [x["mean_final_d1_loss"] for x in rows], "o-", label=ft)
            chosen = next(x for x in rows if x["selected"])
            ax.scatter(chosen["lr"], chosen["mean_final_d1_loss"], s=130,
                       facecolors="none", edgecolors="black")
        ax.set_xscale("log", base=2)
        ax.set_title(base)
        ax.set_xlabel("Fine-tuning matrix LR")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Mean final D1 loss (nats/token)")
    axes[-1].legend(title="FT optimizer")
    fig.tight_layout()
    fig.savefig(out / "lr_selection.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for run in runs:
        if not math.isclose(selected[run.arm], run.lr, rel_tol=1e-6):
            continue
        label = f"{arm_label(*run.arm)}, s{run.seed}"
        ax.plot([0, *[p["learning"] for p in run.curve]],
                [0, *[p["forgetting"] for p in run.curve]], alpha=.5, label=label)
    if ell0 is not None:
        ax.axvline(ell0, color="black", linestyle="--", label="matched learning")
    ax.set(xlabel="D1 learning (nats/token)", ylabel="D0 forgetting (nats/token)")
    ax.grid(alpha=.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "learning_forgetting.png", dpi=180)
    plt.close(fig)

    if matches:
        arms = [(base, ft) for base in bases for ft in OPTS]
        labels = [arm_label(*arm) for arm in arms]
        width_in = max(7, 0.9*len(arms))
        fig, ax = plt.subplots(figsize=(width_in, 4))
        for i, base in enumerate(bases):
            for j, ft in enumerate(OPTS):
                values = [r["forgetting"] for r in matches
                          if (r["base_run"], r["ft_optimizer"]) == (base, ft)]
                ax.scatter([i*3+j]*len(values), values, alpha=.6)
                ax.plot(i*3+j, np.mean(values), "k_", markersize=18)
        ax.set_xticks([i*3+j for i in range(len(bases)) for j in range(len(OPTS))], labels)
        ax.set_ylabel(f"D0 forgetting at learning {ell0:g}")
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(out / "matched_forgetting.png", dpi=180)
        plt.close(fig)

        if router_rows:
            fig, axes = plt.subplots(1, 2, figsize=(2*width_in, 4))
            for ax, field, title in zip(axes, ("flip_rate", "maxvio"),
                                        ("D0 route flip rate", "D0 maximum load violation")):
                means = [np.mean([r[field] for r in router_rows
                                  if (r["base_run"], r["ft_optimizer"]) == arm]) for arm in arms]
                ax.bar(range(len(arms)), means)
                ax.set_xticks(range(len(arms)), labels)
                ax.set_title(title)
                ax.grid(axis="y", alpha=.25)
            fig.tight_layout()
            fig.savefig(out / "router_diagnostics.png", dpi=180)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(width_in, 4))
        width = .35
        for i, arm in enumerate(arms):
            rows = [r for r in matches if (r["base_run"], r["ft_optimizer"]) == arm]
            ax.bar(i-width/2, np.mean([r["F_route"] for r in rows]), width,
                   color="tab:blue", label="F_route" if i == 0 else None)
            ax.bar(i+width/2, np.mean([r["F_fixed"] for r in rows]), width,
                   color="tab:orange", label="F_fixed" if i == 0 else None)
        ax.set_xticks(range(len(arms)), labels)
        ax.set_ylabel("D0 loss change (nats/token)")
        ax.legend(); ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(out / "replay_terms.png", dpi=180)
        plt.close(fig)

        if router_runs:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            for ax, field in zip(axes, ("flip_rate", "maxvio")):
                for arm in arms:
                    rows = [r for r in router_runs if (r["base_run"], r["ft_optimizer"]) == arm]
                    ax.scatter([r[f"mean_{field}"] for r in rows], [r["forgetting"] for r in rows],
                               label=arm_label(*arm))
                ax.set_xlabel(f"Mean layer {field}")
                ax.grid(alpha=.25)
            axes[0].set_ylabel("D0 forgetting at matched learning")
            axes[1].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out / "router_vs_forgetting.png", dpi=180)
            plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, action="append", required=True,
                   help="fine-tuning output root (repeat for outputs from both accounts)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--steps", type=int, required=True, help="fine-tuning length in steps")
    p.add_argument("--seeds", type=int, nargs="+", required=True, help="all expected fine-tuning seeds")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ell0", type=float, default=None, help="predeclared D1-learning target in nats/token")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--min-lrs", type=int, default=5)
    p.add_argument("--bases", nargs="+", default=list(BASES),
                   help="base runs to analyze (default: the seed-42 pair)")
    p.add_argument("--base-manifest", type=Path, help="required for non-E8: verified checkpoint pin manifest")
    p.add_argument("--lr-selection", type=Path, default=None,
                   help="analysis.json of the seed-42 pilot at this dataset and length: use its selected "
                        "LR per optimizer pair for every base run instead of selecting")
    args = p.parse_args(argv)
    check(len(set(args.bases)) == len(args.bases), "--bases must be distinct pinned base runs")
    pins = load_base_manifest(args.base_manifest) if args.base_manifest is not None else None
    try:
        expert_counts = {resolve_base_pin(b, pins).n_experts for b in args.bases}
    except ValueError as exc:
        raise ValueError(f"--bases must be distinct pinned base runs: {exc}") from None
    check(len(expert_counts) == 1, "one analysis must compare bases of one E")
    E = expert_counts.pop()
    check(args.steps > 0 and args.seeds and len(set(args.seeds)) == len(args.seeds)
          and args.bootstrap >= 0 and args.min_lrs > 0, "invalid steps, seeds, bootstrap or min-lrs")
    if args.ell0 is not None:
        check(math.isfinite(args.ell0) and args.ell0 > 0, "ell0 must be positive and finite")
    bases = tuple(args.bases)
    runs = load_runs(args.root, args.dataset, args.steps, args.seeds, bases, pins)
    if args.lr_selection is not None:
        selected, lr_rows = fixed_lrs(runs, args.seeds, args.lr_selection, args.dataset, args.steps, bases)
    else:
        selected, lr_rows = select_lrs(runs, args.seeds, args.min_lrs, bases)
    chosen = [r for r in runs if math.isclose(selected[r.arm], r.lr, rel_tol=1e-6)]
    check(all(r.status == "completed" for r in chosen), "a seed diverged at a selected LR")
    support = min(max(0.0, *[p["learning"] for p in r.curve]) for r in chosen)
    matches, router_rows, contrasts = [], [], []
    if args.ell0 is not None:
        check(args.ell0 <= support, f"ell0 {args.ell0} exceeds common support {support}")
        for run in chosen:
            m = match_curve(run.curve, args.ell0)
            check(m is not None, f"ell0 not reached: {run.path}")
            matches.append({"base_run": run.base, "ft_optimizer": run.ft, "seed": run.seed,
                            "lr": run.lr, **m,
                            "route_share": m["F_route"]/m["forgetting"]
                            if E > 1 and abs(m["forgetting"]) >= 1e-6 else None})
            if E > 1:
                router_rows.extend(router_at(run, m))
        for base in bases:
            opt = base.split("-")[2]
            other = "muon" if opt == "adamw" else "adamw"
            by_arm = {(r["ft_optimizer"], r["seed"]): r["forgetting"]
                      for r in matches if r["base_run"] == base}
            diffs = [by_arm[(other, seed)]-by_arm[(opt, seed)] for seed in args.seeds]
            contrasts.append({"base_run": base, "matched_ft_optimizer": opt,
                              "mismatched_ft_optimizer": other,
                              "mean_penalty": float(np.mean(diffs)), "seed_penalties": diffs})
    router_runs = []
    for match in matches if E > 1 else []:
        rows = [r for r in router_rows if (r["base_run"], r["ft_optimizer"], r["seed"]) ==
                (match["base_run"], match["ft_optimizer"], match["seed"])]
        router_runs.append({"base_run": match["base_run"], "ft_optimizer": match["ft_optimizer"],
                            "seed": match["seed"], "forgetting": match["forgetting"],
                            **{f"mean_{field}": (float(np.mean([r[field] for r in rows]))
                                                 if rows and all(r[field] is not None for r in rows) else None)
                               for field in ROUTER}})
    correlations = {field: spearman([r[f"mean_{field}"] for r in router_runs],
                                     [r["forgetting"] for r in router_runs])
                    for field in ("flip_rate", "maxvio")} if router_runs else {}
    uncertainty = bootstrap(chosen, args.seeds, args.ell0, args.bootstrap, args.random_seed, bases) \
        if args.ell0 is not None else None
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "lr_selection.csv", lr_rows)
    write_csv(args.out / "matched_runs.csv", matches)
    write_csv(args.out / "router_layers.csv", router_rows)
    write_csv(args.out / "router_runs.csv", router_runs)
    packing = eval_packing(runs)
    summary = {"dataset": args.dataset, "steps": args.steps, "seeds": args.seeds,
               **({"n_experts": E, "base_manifest": str(args.base_manifest),
                   "router_diagnostics": "not applicable: E=1 has structural zero route change" if E == 1 else "active"}
                  if E != 8 else {}),
               "eval_packing": packing["packing"],
               "runs_without_packing_field": packing["runs_without_packing_field"],
               "eval_sets": {d: {f: eval_identity(runs[0], d).get(f) for f in EVAL_IDENTITY_FIELDS}
                             for d in ("D0", "D1")},
               "ell0": args.ell0, "common_learning_support": [0.0, support],
               "lr_selection": lr_rows, "edge_winners": [x for x in lr_rows if x["edge"]],
               "matched_runs": matches, "contrasts": contrasts,
               "pooled_contrasts": pooled({c["base_run"]: c["mean_penalty"] for c in contrasts}),
               "bootstrap": uncertainty, "bases": list(bases),
               "lr_selection_source": str(args.lr_selection) if args.lr_selection else None,
               "diverged_runs": [str(r.path) for r in runs if r.status != "completed"],
               "router_correlations": correlations,
               "source_roots": [str(root.resolve() if E != 8 else root) for root in args.root],
               "source_runs": [str(r.path.resolve() if E != 8 else r.path) for r in runs],
               "runs_without_checkpoint_hash": [str(r.path) for r in runs
                                                 if not r.config["base"].get("model_sha256")],
               "status": "fixed_lrs" if args.lr_selection else
                         "needs_lr_extension" if any(x["edge"] for x in lr_rows) else "grid_brackets_optima",
               "note": "Exploratory until ell0 and main length are fixed; bootstrap holds selected LRs fixed. "
                       "LR selection and learning use the same D1 set."}
    (args.out / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    make_plots(args.out, runs, lr_rows, selected, matches, router_rows, router_runs, args.ell0, bases)
    print(f"Analyzed {len(runs)} runs; wrote {args.out}")
    print(f"Evaluation packing: {packing['packing']}" +
          (f" ({packing['runs_without_packing_field']} runs had no packing field: legacy)"
           if packing["runs_without_packing_field"] else ""))
    print(f"Common D1-learning support: 0 to {support:.6g}")
    for row in lr_rows:
        if row["selected"]:
            print(f"{row['base_run']} → {row['ft_optimizer']}: LR {row['lr']:.8g}"+
                  (" (grid edge: extend sweep)" if row["edge"] else ""))


if __name__ == "__main__":
    main()
