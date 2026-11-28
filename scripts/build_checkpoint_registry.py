"""Read-only inventory of both authors' ORCD model and fine-tuning checkpoints.

  python -m scripts.build_checkpoint_registry --out results/checkpoint_registry

The scientific run locations are explicit rather than inferred by searching an
entire account. Scratch is primary; matching relative directories in Harry's
pool results are recorded as mirrors, including artifacts no longer on scratch.
Nazar's older base/SFT runs are verification-only. Symlink directory aliases
are not traversed. Files are enumerated by name; no weights, checkpoint meta,
routing arrays, or evaluation arrays are loaded or hashed. Only small run-level
config/result JSON records are read. A shared base evaluation is a reference
record, not an extra experimental run or a saved fine-tuned model. Fine-tuning
branches can contain actual model and optimizer files as well as evaluation
NPZs; interrupted attempts are retained as distinct archival records.
"""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys


HGO_SCRATCH = Path("/orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch")
HGO_POOL = Path("/orcd/pool/006/hgo/nanochat-optimizer-mismatch/results")
NAZAR_FINETUNE = Path("/orcd/scratch/orcd/010/ndrugov/finetune_pilot")
NAZAR_LEGACY = Path("/orcd/scratch/orcd/010/ndrugov/nanochat-optimizer-mismatch")
MAX_RUN_JSON_BYTES = 16 * 1024 * 1024
PRETRAIN_NAME = re.compile(r"e(\d+)-(sweep|main|xfer)-(adamw|muon)(?:-lr([^/]+))?-s(\d+)$")
FT_NAME = re.compile(r"(adamw|muon)-lr(.+?)-s(\d+)(?:-T(\d+))?$")
INTERRUPTED_FT_NAME = re.compile(r"(.+)\.[0-9a-f]{32}$")
MODEL_FILES = re.compile(r"(model|meta|routing)_(\d+)\.(pt|json)$")
OPTIM_FILES = re.compile(r"optim_(\d+)_rank(\d+)\.pt$")
EVAL_FILES = re.compile(r"ckpt_(\d+)\.npz$")
BASE_EVAL_NAMES = ("base_arrays.npz", "base_metrics.json", "eval_set_D0.pt", "eval_set_D1.pt")
SCIENTIFIC_FT_ROOTS = {"finetune_pilot", "finetune_main", "finetune_pilot_long",
                       "finetune_pilot_long_wikitext103", "finetune_main_long",
                       "finetune_pilot_e1", "finetune_pilot_e4", "finetune_pilot_e16"}
NEW_E_FT_ROOT = re.compile(r"finetune_pilot_e(1|4|16)$")
RUN_FIELDS = (
    "run_key", "owner", "relative_id", "stage", "run_type", "validity", "status", "run_name",
    "availability", "run_dir", "pool_run_dir", "dataset", "base_run", "base_seed", "pretrain_optimizer",
    "ft_optimizer", "ft_seed", "lr", "length_steps", "steps_completed", "n_experts",
    "config_path", "result_path", "pool_config_path", "pool_result_path", "resume_config_paths",
    "pool_resume_config_paths", "base_checkpoint_step", "base_model_path", "base_model_recorded_path",
    "base_model_sha256", "base_model_path_source", "shared_base_eval_dir", "pool_shared_base_eval_dir",
    "attempt_id", "pool_attempt_id", "pool_attempt_consistency", "note",
)
CHECKPOINT_FIELDS = (
    "run_key", "owner", "relative_id", "stage", "run_type", "validity", "status", "run_name", "step",
    "checkpoint_kind", "has_model_weights", "has_optimizer_state", "availability", "run_dir", "pool_run_dir",
    "attempt_id", "pool_attempt_id", "pool_attempt_consistency", "model_path",
    "optimizer_paths", "meta_path", "routing_probe_path", "eval_npz_path", "base_eval_dir",
    "base_eval_arrays_path", "base_eval_metrics_path", "base_eval_set_paths", "pool_model_path",
    "pool_optimizer_paths", "pool_meta_path", "pool_routing_probe_path", "pool_eval_npz_path",
    "pool_base_eval_dir", "pool_base_eval_arrays_path", "pool_base_eval_metrics_path", "pool_base_eval_set_paths",
)


@dataclass
class Location:
    owner: str
    relative_id: str
    root_stage: str
    run_type: str
    primary: Path | None = None
    pool: Path | None = None

    @property
    def key(self):
        return f"{self.owner}:{self.relative_id}"

    @property
    def directory(self):
        return self.primary or self.pool


def first(*values):
    return next((v for v in values if v is not None and v != ""), "")


def existing(path):
    return str(path) if path is not None and path.is_file() else ""


def path_list(paths):
    """JSON arrays inside CSV preserve all optimizer ranks without ambiguous separators."""
    return json.dumps([str(p) for p in paths], separators=(",", ":")) if paths else ""


def child_dirs(path):
    if path is None or not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if not p.is_symlink() and p.is_dir())


def run_files(path):
    return [] if path is None else sorted(p for p in path.iterdir() if p.is_file())


def recognize(path, kind):
    names = [p.name for p in run_files(path)]
    if kind == "shared_base_eval":
        return bool(set(names).intersection(BASE_EVAL_NAMES))
    if any(n in ("config.json", "result.json") or n.startswith("config_resume_") for n in names):
        return True
    if kind == "finetune":
        return any(EVAL_FILES.fullmatch(n) or MODEL_FILES.fullmatch(n) or OPTIM_FILES.fullmatch(n)
                   or n in ("metrics.parquet", "metrics.jsonl") for n in names)
    return any(MODEL_FILES.fullmatch(n) or OPTIM_FILES.fullmatch(n) for n in names)


def discover(container, stage, kind):
    """Yield (root relative ID, actual path, run type), never following directory symlinks."""
    if kind == "finetune":
        for base in child_dirs(container):
            if base.name == ".interrupted":
                continue
            for dataset in child_dirs(base):
                for cell in child_dirs(dataset):
                    typ = "shared_base_eval" if cell.name == "base" else "finetune"
                    if typ == "finetune" and not FT_NAME.fullmatch(cell.name):
                        continue
                    if recognize(cell, typ):
                        yield str(Path(stage) / base.name / dataset.name / cell.name), cell, typ
        # Retained interrupted attempts are not active LR-grid branches. Their
        # randomized suffix disambiguates successive attempts at the same cell.
        for base in child_dirs(container / ".interrupted"):
            for dataset in child_dirs(base):
                for cell in child_dirs(dataset):
                    match = INTERRUPTED_FT_NAME.fullmatch(cell.name)
                    if match and FT_NAME.fullmatch(match[1]) and recognize(cell, "finetune"):
                        yield str(Path(stage) / ".interrupted" / base.name / dataset.name / cell.name), cell, "finetune"
    else:
        for directory in child_dirs(container):
            if recognize(directory, kind):
                yield str(Path(stage) / directory.name), directory, kind


def inventory(hgo_scratch, hgo_pool, nazar_finetune, nazar_legacy):
    """Discover logical runs, consolidating corresponding scratch/pool directories."""
    locs = {}
    seen = set()
    warnings = []

    def scan(root, stage, kind, owner, is_pool=False):
        try:
            for relative, directory, typ in discover(root, stage, kind):
                canonical = directory.resolve()
                # The same actual directory must not be listed as two logical runs. A pool
                # root may itself alias scratch, so don't duplicate that mirror either.
                if canonical in seen:
                    continue
                seen.add(canonical)
                key = (owner, relative)
                loc = locs.setdefault(key, Location(owner, relative, stage, typ))
                if is_pool:
                    loc.pool = canonical
                    # An empty current scratch directory may be a run between setup and
                    # the first config write. Do not mislabel an old pool result as current.
                    candidate = hgo_scratch / relative
                    if (loc.primary is None and candidate.is_dir() and candidate.resolve() == candidate
                            and candidate not in seen):
                        loc.primary = candidate
                        seen.add(candidate)
                else:
                    loc.primary = canonical
        except OSError as exc:
            warnings.append(f"cannot scan {root}: {type(exc).__name__}")

    for root in (hgo_scratch, hgo_pool, nazar_finetune, nazar_legacy):
        if not root.is_dir():
            warnings.append(f"missing or inaccessible registry root: {root}")
    for root, is_pool in ((hgo_scratch, False), (hgo_pool, True)):
        if not root.is_dir():
            continue
        scan(root / "base_checkpoints", "base_checkpoints", "pretrain", "hgo", is_pool)
        scan(root / "chatsft_checkpoints", "chatsft_checkpoints", "legacy_chatsft", "hgo", is_pool)
        try:
            ft_stages = [p for p in child_dirs(root) if (p.name == "finetune" or p.name.startswith("finetune_"))
                         and p.name != "finetune_data"]
        except OSError as exc:
            ft_stages = []
            warnings.append(f"cannot scan fine-tuning stage roots at {root}: {type(exc).__name__}")
        for stage in ft_stages:
            scan(stage, stage.name, "finetune", "hgo", is_pool)
    scan(nazar_finetune, "finetune_pilot", "finetune", "ndrugov")
    # The new explicitly named sparsity pilots may be run under either account.
    # Keep the old standalone Nazar pilot/archive classification separate.
    for stage in sorted(s for s in SCIENTIFIC_FT_ROOTS if NEW_E_FT_ROOT.fullmatch(s)):
        scan(nazar_legacy / stage, stage, "finetune", "ndrugov")
    scan(nazar_legacy / "base_checkpoints", "base_checkpoints", "legacy_pretrain", "ndrugov")
    scan(nazar_legacy / "chatsft_checkpoints", "chatsft_checkpoints", "legacy_chatsft", "ndrugov")
    return sorted(locs.values(), key=lambda loc: loc.key), warnings


def run_record(path, note, field):
    if not path:
        return {}
    try:
        path = Path(path)
        if path.stat().st_size > MAX_RUN_JSON_BYTES:
            note.append(f"{field} too large to read safely; path retained")
            return {}
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("not an object")
        return data
    except (OSError, ValueError):
        note.append(f"{field} could not be read; path retained")
        return {}


def nested(obj, key):
    value = obj.get(key, {})
    return value if isinstance(value, dict) else {}


def availability(primary, pool):
    return "scratch_and_pool" if primary and pool else ("primary_only" if primary else "pool_only")


def pretrain_identity(name):
    match = PRETRAIN_NAME.fullmatch(name)
    if match:
        return {"n_experts": int(match[1]), "phase": match[2], "pretrain_optimizer": match[3],
                "lr": match[4] or "", "base_seed": int(match[5])}
    return {}


def safe_int(value):
    try:
        return int(value) if value != "" and value is not None else None
    except (TypeError, ValueError):
        return None


def classify(loc, row, config, note):
    """Validity is an experimental-use classification, not a numerical quality check."""
    if loc.run_type.startswith("legacy"):
        return loc.root_stage, "verification_only"
    if loc.run_type == "pretrain":
        phase = pretrain_identity(row["run_name"]).get("phase")
        stage = {"sweep": "pretrain_sweep", "main": "pretrain_main", "xfer": "pretrain_transfer"}.get(phase)
        if stage is None:
            note.append("not a named scientific pretraining run")
            return "pretrain_other", "verification_only"
        if row["run_name"] == "e8-xfer-adamw-lr0.000651-s42":
            note.append("superseded AdamW base; retained as valid transfer check, not a fine-tuning base")
        return stage, "valid"
    stage = loc.root_stage
    if loc.run_type == "finetune" and len(Path(loc.relative_id).parts) == 5 and Path(loc.relative_id).parts[1] == ".interrupted":
        note.append("retained partial fine-tuning attempt; excluded from scientific LR-grid analysis")
        return stage, "interrupted_archive"
    if "contiguous" in stage:
        return stage, "invalid_contiguous_eval"
    if "smoke" in stage or "verify" in stage:
        return stage, "verification_only"
    if loc.run_type == "shared_base_eval":
        return stage, "shared_reference"
    if match := NEW_E_FT_ROOT.fullmatch(stage):
        if safe_int(row["n_experts"]) != int(match[1]):
            note.append("sparsity does not match the explicit pilot root")
            return stage, "needs_review"
        return stage, "valid"
    base_seed = safe_int(row["base_seed"])
    length = safe_int(row["length_steps"])
    replay_archive = "replaybug" in stage or (loc.owner == "ndrugov" and base_seed == 42)
    if replay_archive:
        if length == 60:
            note.append("short branch unaffected by replay bug; historical archive excluded from main analysis")
            return stage, "unaffected_archive_excluded"
        if length is not None and length > 60:
            return stage, "invalid_replaybug"
        return stage, "archive_needs_review"
    if loc.owner == "ndrugov":
        if (base_seed in (43, 44) and row["dataset"] == "wikitext103" and length == 720 and
                not config.get("branch") and not re.search(r"-T\d+$", row["run_name"])):
            note.append("standalone WikiText branch reused as clean seed-43/44 main-grid cell")
            return "finetune_main_reused_nazar", "valid"
        note.append("not one of the documented standalone WikiText main-grid runs")
        return stage, "needs_review"
    if stage in SCIENTIFIC_FT_ROOTS:
        if "long" in stage and length in (60, 240, 720, 960):
            note.append("short-length branch duplicated in original pilot/main; follow protocol for analysis selection")
        return stage, "valid"
    note.append("unclassified fine-tuning stage; review before scientific reuse")
    return stage, "needs_review"


def build_run(loc, hgo_scratch):
    """Read a run-level record only; exact paths remain usable when metadata is missing."""
    note = []
    primary, pool = loc.primary, loc.pool
    config_primary = existing(primary / "config.json" if primary else None)
    result_primary = existing(primary / "result.json" if primary else None)
    config_pool = existing(pool / "config.json" if pool else None)
    result_pool = existing(pool / "result.json" if pool else None)
    config_path, result_path = ((config_primary, result_primary) if primary else (config_pool, result_pool))
    if primary and result_pool and not result_primary:
        note.append("pool result exists but primary result is absent; pool result may be stale")
    elif not primary:
        note.append("only the pool copy was found; run records and paths are from the pool")
    config = run_record(config_path, note, "config.json")
    result = run_record(result_path, note, "result.json")
    row = dict.fromkeys(RUN_FIELDS, "")
    row.update(run_key=loc.key, owner=loc.owner, relative_id=loc.relative_id, run_type=loc.run_type,
               run_name=loc.directory.name, availability=availability(primary, pool), run_dir=str(loc.directory),
               pool_run_dir=str(pool) if pool else "", config_path=config_path, result_path=result_path,
               pool_config_path=config_pool, pool_result_path=result_pool,
               resume_config_paths=path_list(sorted(primary.glob("config_resume_*.json")) if primary else []),
               pool_resume_config_paths=path_list(sorted(pool.glob("config_resume_*.json")) if pool else []))
    if loc.run_type == "finetune":
        # A new retention-aware branch records an explicit attempt identifier.
        # An older pool mirror from a superseded/partial attempt is still
        # discoverable but must not supply artifacts for the current attempt.
        pool_config_record = (run_record(config_pool, note, "pool config.json") if primary and pool else
                              config if pool else {})
        attempt = first(config.get("attempt_id"))
        pool_attempt = first(pool_config_record.get("attempt_id"))
        row.update(attempt_id=attempt, pool_attempt_id=pool_attempt)
        if primary and pool:
            if attempt and pool_attempt:
                row["pool_attempt_consistency"] = "same_attempt" if attempt == pool_attempt else "different_attempt"
            elif attempt or pool_attempt:
                row["pool_attempt_consistency"] = "unverified_attempt"
            else:
                row["pool_attempt_consistency"] = "legacy_unrecorded"
            if row["pool_attempt_consistency"] in ("different_attempt", "unverified_attempt"):
                note.append("pool artifacts may be from a different attempt; not merged into current checkpoint paths")
    if loc.run_type == "shared_base_eval":
        row["status"] = "reference_only"
    elif loc.run_type.startswith("legacy"):
        row["status"] = first(result.get("status"), "unrecorded")
        if not config_path:
            note.append("legacy metadata is not opened; run settings unavailable without a small config record")
    else:
        row["status"] = first(result.get("status"), "unknown_result" if result_path else "incomplete")

    if loc.run_type in ("finetune", "shared_base_eval"):
        parts = Path(loc.relative_id).parts
        if len(parts) == 5 and parts[1] == ".interrupted":
            _, _, base_name, dataset, archive_name = parts
            cell_name = INTERRUPTED_FT_NAME.fullmatch(archive_name)[1]
        else:
            _, base_name, dataset, cell_name = parts
        base = nested(config, "base")
        cell = nested(config, "cell")
        args = nested(config, "args")
        pre = pretrain_identity(base_name)
        ft_match = FT_NAME.fullmatch(cell_name)
        # Prefer actual run-level records; directory suffix only when they don't exist.
        row.update(dataset=first(result.get("ft_dataset"), cell.get("ft_dataset"), dataset),
                   base_run=first(result.get("base_run"), cell.get("base_run"), base_name),
                   base_seed=first(nested(base, "user_config").get("seed"), pre.get("base_seed")),
                   pretrain_optimizer=first(result.get("pretrain_optimizer"), cell.get("pretrain_optimizer"),
                                            pre.get("pretrain_optimizer")),
                   n_experts=first(result.get("n_experts"), cell.get("n_experts"), pre.get("n_experts")),
                   ft_optimizer=first(result.get("ft_optimizer"), cell.get("ft_optimizer"), ft_match[1] if ft_match else None),
                   ft_seed=first(result.get("seed"), cell.get("seed"), int(ft_match[3]) if ft_match else None),
                   lr=first(result.get("lr"), cell.get("lr"), ft_match[2] if ft_match else None),
                   length_steps=first(result.get("steps"), result.get("ft_steps"), cell.get("ft_steps"), config.get("steps"),
                                      int(ft_match[4]) if ft_match and ft_match[4] else None),
                   steps_completed=first(result.get("steps_completed")),
                   base_checkpoint_step=first(base.get("step"), args.get("base_step")),
                   base_model_sha256=first(base.get("model_sha256")))
        if loc.run_type == "finetune" and not row["length_steps"]:
            # Older unbranched record: a whole batch count can recover the planned length.
            tokens, batch = safe_int(args.get("train_tokens")), safe_int(args.get("total_batch_size"))
            if tokens is not None and batch and tokens % batch == 0:
                row["length_steps"] = tokens // batch
        base_dir = primary if loc.run_type == "shared_base_eval" else (primary.parent / "base" if primary else None)
        pool_base_dir = pool if loc.run_type == "shared_base_eval" else (pool.parent / "base" if pool else None)
        if base_dir is not None and base_dir.is_dir():
            row["shared_base_eval_dir"] = str(base_dir)
        if pool_base_dir is not None and pool_base_dir.is_dir():
            row["pool_shared_base_eval_dir"] = str(pool_base_dir)
        if not row["shared_base_eval_dir"]:
            row["shared_base_eval_dir"] = row["pool_shared_base_eval_dir"]
        if loc.run_type == "shared_base_eval":
            row["length_steps"] = 0
            baseline = run_record(existing(loc.directory / "base_metrics.json"), note, "base_metrics.json")
            row["base_checkpoint_step"] = first(baseline.get("base_step"), row["base_checkpoint_step"])

        recorded = base.get("model_file")
        step = safe_int(row["base_checkpoint_step"])
        if not recorded and base.get("dir") and step is not None:
            recorded = str(Path(base["dir"]) / f"model_{step:06d}.pt")
        row["base_model_recorded_path"] = recorded or ""
        if recorded and Path(recorded).is_file():
            row["base_model_path"] = str(Path(recorded).resolve())
            row["base_model_path_source"] = "recorded_existing"
        elif step is not None and row["base_run"]:
            fallback = hgo_scratch / "base_checkpoints" / row["base_run"] / f"model_{step:06d}.pt"
            if fallback.is_file():
                row["base_model_path"] = str(fallback.resolve())
                row["base_model_path_source"] = "shared_hgo_path_inferred"
            elif recorded:
                note.append("recorded base model path is no longer accessible")

    else:
        pre = pretrain_identity(loc.directory.name)
        args, setting = nested(config, "args"), nested(result, "settings")
        opt = first(setting.get("optimizer_name"), args.get("optimizer_name"), pre.get("pretrain_optimizer"))
        row.update(dataset="fineweb_edu" if pre else "", pretrain_optimizer=opt,
                   base_seed=first(setting.get("seed"), args.get("seed"), pre.get("base_seed")),
                   lr=first(setting.get("matrix_lr"), args.get(f"{opt}_matrix_lr"), pre.get("lr")),
                   length_steps=first(result.get("num_iterations"), config.get("num_iterations")),
                   steps_completed=first(result.get("steps_trained")),
                   n_experts=first(setting.get("n_experts"), nested(config, "model_config").get("n_experts"),
                                   pre.get("n_experts")))
        if loc.run_type == "pretrain":
            row["base_run"] = loc.directory.name

    row["stage"], row["validity"] = classify(loc, row, config, note)
    row["note"] = "; ".join(note)
    return row


def checkpoint_artifacts(directory, kind):
    """step -> artifact names/path(s). File names only, no checkpoint contents."""
    results = {}
    if directory is None:
        return results
    if kind == "shared_base_eval":
        files = [directory / name for name in BASE_EVAL_NAMES]
        if any(p.is_file() for p in files):
            results[0] = {
                "base_eval_dir": str(directory), "base_eval_arrays_path": existing(files[0]),
                "base_eval_metrics_path": existing(files[1]),
                "base_eval_set_paths": path_list([p for p in files[2:] if p.is_file()]),
            }
        return results
    optimizers = {}
    for p in run_files(directory):
        if kind == "finetune" and (match := EVAL_FILES.fullmatch(p.name)):
            results.setdefault(int(match[1]), {})["eval_npz_path"] = str(p)
        elif match := MODEL_FILES.fullmatch(p.name):
            field = {"model": "model_path", "meta": "meta_path", "routing": "routing_probe_path"}[match[1]]
            if match[3] == ("json" if match[1] == "meta" else "pt"):
                results.setdefault(int(match[2]), {})[field] = str(p)
        elif match := OPTIM_FILES.fullmatch(p.name):
            optimizers.setdefault(int(match[1]), []).append((int(match[2]), p))
    for step, ranks in optimizers.items():
        results.setdefault(step, {})["optimizer_paths"] = path_list([p for _, p in sorted(ranks)])
    return results


def build_checkpoints(loc, run):
    primary, pool = checkpoint_artifacts(loc.primary, loc.run_type), checkpoint_artifacts(loc.pool, loc.run_type)
    rows = []
    for step in sorted(set(primary) | set(pool)):
        main, mirror = primary.get(step, {}), pool.get(step, {})
        matching_mirror = {} if run["pool_attempt_consistency"] in ("different_attempt", "unverified_attempt") else mirror
        row = dict.fromkeys(CHECKPOINT_FIELDS, "")
        row.update({key: run[key] for key in RUN_FIELDS if key in CHECKPOINT_FIELDS})
        row.update(step=step, availability=availability(bool(main), bool(mirror)))
        row.update({key: first(main.get(key), matching_mirror.get(key)) for key in set(main) | set(matching_mirror)})
        row.update({"pool_" + key: value for key, value in mirror.items()})
        has_weights = bool(main.get("model_path") or matching_mirror.get("model_path"))
        has_optimizer = bool(main.get("optimizer_paths") or matching_mirror.get("optimizer_paths"))
        has_eval = bool(main.get("eval_npz_path") or matching_mirror.get("eval_npz_path"))
        row["has_model_weights"] = "yes" if has_weights else "no"
        row["has_optimizer_state"] = "yes" if has_optimizer else "no"
        row["checkpoint_kind"] = ("shared_base_eval" if loc.run_type == "shared_base_eval" else
                                  ("finetune_model_weights_and_eval" if has_weights and has_eval else
                                   "finetune_model_weights" if has_weights else
                                   "finetune_eval_arrays" if has_eval else
                                   "finetune_related_only") if loc.run_type == "finetune" else
                                  "model_weights" if has_weights else "model_related_only")
        if loc.run_type == "finetune":
            if not main and mirror and not matching_mirror:
                row["checkpoint_kind"] = "stale_pool_artifacts"
            row["base_eval_dir"] = run["shared_base_eval_dir"]
            row["pool_base_eval_dir"] = run["pool_shared_base_eval_dir"]
        rows.append(row)
    return rows


def build_registry(hgo_scratch=HGO_SCRATCH, hgo_pool=HGO_POOL, nazar_finetune=NAZAR_FINETUNE,
                   nazar_legacy=NAZAR_LEGACY):
    roots = [Path(root).resolve() for root in (hgo_scratch, hgo_pool, nazar_finetune, nazar_legacy)]
    locs, warnings = inventory(*roots)
    runs, checkpoints = [], []
    for loc in locs:
        try:
            run = build_run(loc, roots[0])
            runs.append(run)
            checkpoints.extend(build_checkpoints(loc, run))
        except OSError as exc:
            warnings.append(f"could not fully scan {loc.directory}: {type(exc).__name__}")
    return runs, checkpoints, warnings


def write_csv(path, fields, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hgo-scratch", type=Path, default=HGO_SCRATCH)
    parser.add_argument("--hgo-pool", type=Path, default=HGO_POOL)
    parser.add_argument("--nazar-finetune", type=Path, default=NAZAR_FINETUNE)
    parser.add_argument("--nazar-legacy", type=Path, default=NAZAR_LEGACY)
    parser.add_argument("--out", type=Path, default=Path("results/checkpoint_registry"))
    args = parser.parse_args(argv)
    runs, checkpoints, warnings = build_registry(args.hgo_scratch, args.hgo_pool, args.nazar_finetune, args.nazar_legacy)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "runs.csv", RUN_FIELDS, runs)
    write_csv(args.out / "checkpoints.csv", CHECKPOINT_FIELDS, checkpoints)
    for warning in warnings:
        print("WARNING: " + warning, file=sys.stderr)
    owner_counts = Counter(row["owner"] for row in runs)
    type_counts = Counter(row["run_type"] for row in runs)
    print(f"Inventory timestamp (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"{'PARTIAL registry' if warnings else 'Registry'}: {len(runs)} records, {len(checkpoints)} step records; {args.out}")
    print("Records by owner: " + ", ".join(f"{k}={v}" for k, v in sorted(owner_counts.items())))
    print("Records by type: " + ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())))
    return int(bool(warnings))


if __name__ == "__main__":
    raise SystemExit(main())
