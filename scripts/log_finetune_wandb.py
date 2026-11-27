"""Backfill W&B runs from authoritative fine-tuning branch results (no GPU needed).

Use the same explicit stage/source for a stage on the cluster and its mirrors.
Use a distinct source (and --invalid for contaminated archives) for separate
experiments; a different curve presented under the same identity is an error.

  python -m scripts.log_finetune_wandb --root /path/to/results --stage pilot --dry-run
  python -m scripts.log_finetune_wandb --root /path/to/results --stage pilot --query-existing

Online mode resumes a deterministic W&B run ID and skips already-committed
checkpoints. The atomic, project-scoped local ledger only marks completed
uploads after finish. --query-existing checks completion markers via a bulk
project listing; --verify-online additionally checks exact remote history steps
after upload (slower). Offline preparation is explicit, separately recorded,
and never mistaken for an online upload; offline W&B itself cannot resume
interrupted local histories, so sync a completed offline history only once.
Do not separately sync that offline history after directly uploading the same
cell online (the stable run ID is intentionally identical).
Only whitelisted scientific metadata and scalar curve entries are uploaded.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

from filelock import FileLock


DEFAULT_ENTITY = "hgo-mit"
DEFAULT_PROJECT = "nanochat"
DEFAULT_LEDGER = Path.home() / ".cache" / "nanochat" / "finetune_wandb_ledger.json"
OPTIMIZERS = {"adamw", "muon"}
METRICS = ("tokens", "loss_d0", "loss_d1", "learning", "forgetting", "F_route",
           "F_fixed", "reverse_order", "train_loss")
REQUIRED_METRICS = {"learning", "forgetting", "loss_d1"}
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\Z")
_BASE = re.compile(r"e(\d+)-main-(adamw|muon)-s(\d+)\Z")


class RecordError(ValueError):
    """An authoritative result/config pair cannot be safely logged."""


class UploadError(RuntimeError):
    """A nonsecret diagnostic for a W&B failure or identity conflict."""


class HistoryNotReady(UploadError):
    """A completion marker exists, but the public history may still be ingesting."""


def label(value):
    if not isinstance(value, str) or not _LABEL.fullmatch(value):
        raise RecordError("stage/source/dataset/entity/project labels must be simple identifiers")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def stable_run_id(stage, source, dataset, base_run, ft_optimizer, lr, seed, steps, invalid=False):
    """Independent of local path, project or account and exact for the resolved LR."""
    identity = ["nanochat-finetune-v1", stage, source, dataset, base_run, ft_optimizer,
                float(lr).hex(), int(seed), int(steps), bool(invalid)]
    return "ft-" + digest(identity)[:32]


def integer(value, field, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise RecordError(f"invalid {field}")
    return value


def number(value, field, *, required=True, allow_nonfinite=False):
    if value is None and not required:
        return None
    if type(value) not in (int, float):
        raise RecordError(f"invalid {field}")
    if not math.isfinite(value):
        if allow_nonfinite:
            return None  # preserve absence without emitting NaN as a reliable measurement
        raise RecordError(f"nonfinite {field}")
    return value


@dataclass(frozen=True)
class Cell:
    run_id: str
    name: str
    group: str
    tags: tuple
    metadata: dict
    points: tuple
    summary: dict
    source_digest: str


def make_cell(result, config, *, stage, source=None, invalid=False, base_manifest=None):
    """Validate saved branch identity/curve, construct a strictly whitelisted payload."""
    stage, source = label(stage), label(source if source is not None else stage)
    if result.get("status") not in ("completed", "diverged"):
        raise RecordError("result must be completed or diverged")
    status = result["status"]
    completed = status == "completed"
    base = result.get("base_run")
    match = _BASE.fullmatch(base) if isinstance(base, str) else None
    if not match:
        raise RecordError("invalid base run identity")
    experts, pre, pre_seed = int(match[1]), match[2], int(match[3])
    if experts not in (1, 4, 8, 16):
        raise RecordError("unsupported expert count")
    if result.get("pretrain_optimizer") != pre or result.get("n_experts", experts) != experts:
        raise RecordError("base optimizer/expert count mismatch")
    dataset = label(result.get("ft_dataset"))
    ft = result.get("ft_optimizer")
    if ft not in OPTIMIZERS:
        raise RecordError("invalid fine-tuning optimizer")
    lr = float(number(result.get("lr"), "lr"))
    if lr <= 0:
        raise RecordError("nonpositive lr")
    seed, steps = integer(result.get("seed"), "seed"), integer(result.get("steps"), "steps", minimum=1)
    if result.get("ft_steps", steps) != steps:
        raise RecordError("result length mismatch")
    cell = config.get("cell", {})
    identities = {"base_run": base, "ft_dataset": dataset, "ft_optimizer": ft, "lr": lr, "seed": seed}
    if any(cell.get(k) != v for k, v in identities.items()):
        raise RecordError("config/result identity mismatch")
    if cell.get("ft_steps", steps) != steps or config.get("steps", steps) != steps:
        raise RecordError("config/result length mismatch")
    if cell.get("pretrain_optimizer", pre) != pre or cell.get("n_experts", experts) != experts:
        raise RecordError("config/result model identity mismatch")
    pin = None
    if experts != 8:
        from nanochat.pretrain_sweep import resolve_base_pin
        try:
            pin = resolve_base_pin(base, base_manifest)
        except ValueError as exc:
            raise RecordError(str(exc)) from None
        identity = config.get("base", {})
        user = identity.get("user_config", {})
        if (user.get("seed") != pin.seed or user.get("optimizer_name") != pin.optimizer or
                user.get(f"{pin.optimizer}_matrix_lr") != pin.matrix_lr):
            raise RecordError("base metadata disagrees with verified manifest")
        model = config.get("model_config", {})
        if experts == 1 and (model.get("disable_routing_loss") is not True or
                             model.get("balance_coef") != 0 or model.get("z_coef") != 0):
            raise RecordError("E1 routing-loss policy is not disabled")
        if (identity.get("model_sha256") != pin.model_sha256 or identity.get("step") != pin.step or
                identity.get("logical_run") != base or identity.get("checkpoint_run") != pin.checkpoint_dir.name or
                identity.get("n_experts") != experts or result.get("n_experts") != experts or
                cell.get("n_experts") != experts or config.get("model_config", {}).get("n_experts") != experts or
                Path(identity.get("dir") or "").resolve() != pin.checkpoint_dir.resolve() or
                Path(identity.get("model_file") or "").resolve() !=
                (pin.checkpoint_dir / f"model_{pin.step:06d}.pt").resolve()):
            raise RecordError("saved logical/actual base checkpoint identity disagrees with verified manifest")
    if config.get("stage", "finetune") != "finetune":
        raise RecordError("config is not fine-tuning")
    done = integer(result.get("steps_completed"), "steps_completed")
    if done > steps or (completed and done != steps):
        raise RecordError("invalid completed-step count")

    points = []
    nonfinite = 0
    last = -1
    curve = result.get("curve")
    if not isinstance(curve, list):
        raise RecordError("missing result curve")
    for row in curve:
        if not isinstance(row, dict):
            raise RecordError("malformed curve point")
        step = integer(row.get("step"), "curve step")
        if step <= last or step > done:
            raise RecordError("nonmonotone or out-of-range checkpoint steps")
        last = step
        if not REQUIRED_METRICS.issubset(row):
            raise RecordError("missing aggregate curve measurements")
        point = {"step": step}
        for key in METRICS:
            if key in row:
                v = number(row[key], key, required=completed, allow_nonfinite=not completed)
                if v is None:
                    nonfinite += 1
                point[key] = v
        if experts == 1 and (point.get("F_route") != 0. or
                not math.isclose(point["F_fixed"], point["forgetting"], rel_tol=0, abs_tol=1e-7)):
            raise RecordError("E1 replay identity must be structural zero")
        points.append(point)
    if completed and (not points or last != steps):
        raise RecordError("missing final checkpoint")

    meta = {"stage": stage, "source": source, "dataset": dataset, "base_run": base,
            "pretrain_optimizer": pre, "pretrain_seed": pre_seed, "ft_optimizer": ft,
            "ft_seed": seed, "lr": lr, "ft_steps": steps, "n_experts": experts,
            "invalid": bool(invalid), "archive": bool(invalid), "result_status": status}
    for key in ("warmup_steps", "decay_steps"):
        value = result.get(key, config.get(key))
        if value is not None:
            meta[key] = integer(value, key)
    if (batch := config.get("args", {}).get("total_batch_size")) is not None:
        meta["batch_tokens"] = integer(batch, "batch_tokens", minimum=1)
    if (base_step := config.get("base", {}).get("step")) is not None:
        meta["base_checkpoint_step"] = integer(base_step, "base_checkpoint_step")
    if pin is not None:
        # Preserve existing E8 run IDs and source digests exactly. New E runs
        # record actual checkpoint identity (no filesystem paths in W&B).
        meta.update(base_checkpoint_run=pin.checkpoint_dir.name, base_model_sha256=pin.model_sha256,
                    base_matrix_lr=pin.matrix_lr)
    summary = {"result_status": status, "invalid": bool(invalid), "steps_completed": done,
               "checkpoint_count": len(points), "nonfinite_measurement_count": nonfinite}
    for key in ("learning", "forgetting"):
        final = points[-1].get(key) if points else None
        recorded = number(result.get(f"final_{key}"), f"final_{key}", required=False,
                          allow_nonfinite=not completed)
        if recorded is not None and (final is None or not math.isclose(recorded, final, abs_tol=1e-12)):
            raise RecordError(f"final_{key} does not match saved curve")
        if final is not None:
            summary[f"final_{key}"] = final
    run_id = stable_run_id(stage, source, dataset, base, ft, lr, seed, steps, invalid)
    source_digest = digest([meta, points, summary])
    meta["backfill_digest"] = source_digest
    stage_source = stage if source == stage else f"{stage}/{source}"
    name = f"{stage_source}/{dataset}/{pre}-s{pre_seed}->{ft}-s{seed}/lr{lr:.8g}/T{steps}"
    if invalid:
        name = f"INVALID/{name}"
    group = f"optimizer-mismatch-e{experts}/finetune/{dataset}/{stage}"
    tags = ("optimizer-mismatch", f"E{experts}", "finetune", f"stage:{stage}", f"source:{source}", f"dataset:{dataset}",
            f"pretrain:{pre}", f"ft:{ft}", f"pair:{pre}-{ft}", f"base_seed:{pre_seed}",
            f"ft_seed:{seed}", f"lr:{lr:.8g}", f"length:{steps}", status,
            "invalid-archive" if invalid else "valid")
    return Cell(run_id, name, group, tags, meta, tuple(points), summary, source_digest)


def load_cell(path, *, stage, source=None, invalid=False, base_manifest=None):
    path = Path(path)
    with path.open() as f:
        result = json.load(f)
    with (path.parent / "config.json").open() as f:
        config = json.load(f)
    cell = make_cell(result, config, stage=stage, source=source, invalid=invalid, base_manifest=base_manifest)
    base, dataset, run_label = path.parts[-4:-1]
    if (base != cell.metadata["base_run"] or dataset != cell.metadata["dataset"]
            or not run_label.startswith(cell.metadata["ft_optimizer"] + "-")):
        raise RecordError("saved branch identity does not match its result directory")
    return cell


class Ledger:
    """Atomic per-project completion ledger, with a separate lock per run ID."""

    def __init__(self, path, entity, project):
        self.path = Path(path)
        self.key_prefix = f"{entity}/{project}/"
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = FileLock(str(self.path) + ".lock")

    def _read(self):
        if not self.path.exists():
            return {}
        with self.path.open() as f:
            records = json.load(f)
        if not isinstance(records, dict):
            raise UploadError("invalid local W&B ledger")
        return records

    def get(self, run_id):
        with self.lock:
            return self._read().get(self.key_prefix + run_id)

    def put(self, cell, state):
        with self.lock:
            records = self._read()
            records[self.key_prefix + cell.run_id] = {"digest": cell.source_digest, "state": state}
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as out:
                tmp = Path(out.name)
                json.dump(records, out, sort_keys=True, indent=2)
                out.write("\n")
                out.flush()
                os.fsync(out.fileno())
            try:
                os.replace(tmp, self.path)
            finally:
                tmp.unlink(missing_ok=True)

    def run_lock(self, run_id):
        return FileLock(str(self.path) + "." + digest(self.key_prefix + run_id)[:24] + ".lock")


def remote_complete(run, cell, *, check_history=False):
    """Check digest marker; never accept an unrelated run or silently replace history."""
    if run is None:
        return False
    remote_digest = run.config.get("backfill_digest")
    if remote_digest is not None and remote_digest != cell.source_digest:
        raise UploadError("remote source digest differs; assign a distinct --source for a distinct experiment")
    done = (remote_digest == cell.source_digest and run.summary.get("backfill_complete") is True
            and run.summary.get("backfill_digest") == cell.source_digest)
    if done and check_history:
        # The real API can return no rows when filtering on W&B's internal _step.
        # The explicit scientific step metric is written alongside every checkpoint.
        actual = [int(row["step"]) for row in run.scan_history(keys=["step"])]
        expected = [row["step"] for row in cell.points]
        if actual != expected:
            raise HistoryNotReady("remote checkpoint history is incomplete or inconsistent; no upload will be marked verified")
    return done


def query_existing(wb, entity, project, stage, source):
    """One paginated query for completed markers for the requested stage/source."""
    api = wb.Api()
    return {run.id: run for run in api.runs(f"{entity}/{project}",
                                           filters={"config.stage": stage, "config.source": source})}


def verify_uploaded(wb, cell, entity, project, *, attempts=6, delay=3):
    for attempt in range(attempts):
        try:
            run = wb.Api().run(f"{entity}/{project}/{cell.run_id}")
            if remote_complete(run, cell, check_history=True):
                return
        except HistoryNotReady:
            pass  # summary and history can arrive at different times on the server
        except UploadError:
            raise
        except Exception:
            # Public-API reads may briefly lag a successful upload; retry without printing
            # arbitrary SDK exception text (it could include sensitive URLs/parameters).
            pass
        if attempt + 1 < attempts:
            time.sleep(delay)
    raise UploadError("W&B online verification failed: completion marker/history not confirmed")


def upload_cell(wb, cell, ledger, entity, project, *, mode="online", existing=None, verify_online=False):
    """Return uploaded/offline_prepared/skipped; ledger never marks failed attempts."""
    with ledger.run_lock(cell.run_id):
        saved = ledger.get(cell.run_id)
        if saved and saved.get("digest") != cell.source_digest:
            raise UploadError("local source digest differs; assign a distinct --source for a distinct experiment")
        if existing is not None and remote_complete(existing, cell):
            if verify_online:
                verify_uploaded(wb, cell, entity, project)
            ledger.put(cell, "online")
            return "skipped"
        if saved and saved.get("state") == "online" and not verify_online:
            return "skipped"
        if mode == "offline" and saved and saved.get("state") == "offline_prepared":
            return "skipped"
        # Verify a local online completion if explicitly requested (not by trusting the ledger alone).
        if saved and saved.get("state") == "online" and verify_online:
            verify_uploaded(wb, cell, entity, project)
            return "skipped"

        run = None
        operation = "init"
        try:
            settings = wb.Settings(console="off", quiet=True, disable_git=True, disable_code=True,
                                   save_code=False, x_disable_meta=True, x_disable_stats=True)
            run = wb.init(entity=entity, project=project, id=cell.run_id, name=cell.name,
                          group=cell.group, tags=list(cell.tags), job_type="finetune", mode=mode,
                          resume="allow" if mode == "online" else None, settings=settings)
            if run is None or (mode == "online" and run.offline):
                raise UploadError("requested online mode did not initialize an online W&B run")
            if run.id != cell.run_id:
                raise UploadError("W&B did not use the requested stable run ID")
            operation = "config/history"
            persisted = run.config.get("backfill_digest")
            if persisted is not None and persisted != cell.source_digest:
                raise UploadError("remote source digest differs; assign a distinct --source")
            start = int(run.step)
            if start > 0 and persisted is None:
                raise UploadError("existing W&B history has no matching source digest; not overwriting it")
            run.config.update(cell.metadata, allow_val_change=False)
            for row in cell.points:
                # With explicit checkpoint steps and commit=True, W&B advances its resume
                # cursor; replaying all earlier steps is unnecessary and would be rejected.
                if row["step"] >= start:
                    run.log(row, step=row["step"], commit=True)
            run.summary.update({**cell.summary, "backfill_digest": cell.source_digest,
                                "backfill_complete": True})
            operation = "finish"
            run.finish(exit_code=0 if cell.summary["result_status"] == "completed" else 1)
            run = None
            if mode == "online" and verify_online:
                operation = "verify"
                verify_uploaded(wb, cell, entity, project)
            ledger.put(cell, "online" if mode == "online" else "offline_prepared")
            return "uploaded" if mode == "online" else "offline_prepared"
        except UploadError:
            raise
        except Exception as exc:
            raise UploadError(f"W&B {operation} failed ({type(exc).__name__}); retry with the same stage/source") from None
        finally:
            if run is not None:
                try:
                    run.finish(exit_code=1)
                except Exception:
                    pass


def result_paths(roots, datasets=None, bases=None, ft_optimizers=None):
    """Scan only branch results at the documented root/base/dataset/cell layout."""
    for root in roots:
        for path in sorted(Path(root).glob("*/*/*/result.json")):
            base, dataset, run_label = path.parts[-4:-1]
            if datasets and dataset not in datasets or bases and base not in bases:
                continue
            if ft_optimizers and not any(run_label.startswith(f"{ft}-") for ft in ft_optimizers):
                continue
            yield path


def main(argv=None, *, wb=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--source", "--source-label", help="defaults to --stage; distinct replication/archive label")
    parser.add_argument("--base-manifest", type=Path, help="verify non-E8 saved base checkpoint pins before upload")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--base-run", action="append")
    parser.add_argument("--ft-optimizer", action="append", choices=sorted(OPTIMIZERS))
    parser.add_argument("--invalid", action="store_true", help="mark all supplied branches as invalid/archive")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--query-existing", action="store_true", help="bulk query stage/source for completed IDs")
    parser.add_argument("--verify-online", action="store_true", help="verify uploaded checkpoint steps through API")
    parser.add_argument("--dry-run", action="store_true", help="validate/print run IDs without W&B or ledger writes")
    args = parser.parse_args(argv)
    if args.base_manifest is not None:
        from nanochat.pretrain_sweep import load_base_manifest
        pins = load_base_manifest(args.base_manifest)
    else:
        pins = None
    try:
        args.entity, args.project = label(args.entity), label(args.project)
        args.stage = label(args.stage)
        args.source = label(args.source if args.source is not None else args.stage)
    except RecordError as exc:
        parser.error(str(exc))
    if args.mode == "offline" and (args.query_existing or args.verify_online):
        parser.error("remote querying/verification requires --mode online")
    for root in args.root:
        if not root.is_dir():
            parser.error(f"missing result root: {root}")
    counts = {"uploaded": 0, "offline_prepared": 0, "skipped": 0, "planned": 0, "failed": 0}
    existing = {}
    if not args.dry_run:
        if wb is None:
            import wandb as wb
        ledger = Ledger(args.ledger, args.entity, args.project)
        if args.query_existing:
            try:
                existing = query_existing(wb, args.entity, args.project, args.stage, args.source)
            except Exception as exc:
                # Stable IDs and local ledger still permit online upload; the missing optional
                # query is visible, not a silent fallback to offline mode.
                print(f"W&B bulk existing-run query failed ({type(exc).__name__}); using stable IDs and ledger",
                      file=sys.stderr)
    seen = {}
    for path in result_paths(args.root, args.dataset, args.base_run, args.ft_optimizer):
        try:
            cell = load_cell(path, stage=args.stage, source=args.source, invalid=args.invalid, base_manifest=pins)
            if cell.run_id in seen:
                if seen[cell.run_id] != cell.source_digest:
                    raise RecordError("different branch results share an identity; use distinct --source labels")
                counts["skipped"] += 1
                continue
            if args.dry_run:
                outcome = "planned"
            else:
                outcome = upload_cell(wb, cell, ledger, args.entity, args.project, mode=args.mode,
                                      existing=existing.get(cell.run_id), verify_online=args.verify_online)
            seen[cell.run_id] = cell.source_digest
            print(f"{outcome}: {cell.run_id} {cell.name}", flush=True)
            counts[outcome] += 1
        except (RecordError, UploadError) as exc:
            counts["failed"] += 1
            print(f"FAILED {path}: {exc}", file=sys.stderr, flush=True)
        except Exception as exc:
            counts["failed"] += 1
            print(f"FAILED {path}: {type(exc).__name__}", file=sys.stderr, flush=True)
    print("W&B fine-tuning backfill: " + ", ".join(f"{key}={value}" for key, value in counts.items()), flush=True)
    if args.mode == "offline" and not args.dry_run:
        print("Offline histories are prepared locally but NOT verified/synced online.", flush=True)
    if not any(counts.values()):
        print("No saved branches matched the requested filters.", file=sys.stderr)
        return 1
    return int(counts["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
