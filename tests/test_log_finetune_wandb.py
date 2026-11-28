"""W&B backfill tests: no credentials, network or GPU required."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import log_finetune_wandb as b


def records(*, steps=60, seed=1, status="completed"):
    identity = {"base_run": "e8-main-adamw-s42", "pretrain_optimizer": "adamw",
                "ft_dataset": "wikitext103", "ft_optimizer": "muon", "lr": .000443,
                "seed": seed, "ft_steps": steps, "n_experts": 8}
    curve = [dict(step=step, tokens=step*65536, learning=.01*step, forgetting=.003*step,
                  loss_d0=3., loss_d1=4., F_route=.001*step, F_fixed=.002*step,
                  reverse_order=.01, train_loss=4., raw_documents=[1, 2, 3])
             for step in ([10, 20, steps] if status == "completed" else [10])]
    result = dict(identity, steps=steps, steps_completed=steps if status == "completed" else 12,
                  status=status, curve=curve, final_learning=curve[-1]["learning"],
                  final_forgetting=curve[-1]["forgetting"], warmup_steps=8, decay_steps=12,
                  cumulative_expert_counts=[[[1, 2, 3]]], arbitrary_path="/secret/result")
    config = dict(stage="finetune", cell=identity, steps=steps,
                  args={"total_batch_size": 65536, "api_key": "dummy-secret", "ft_data_dir": "/private/data"},
                  base={"step": 11445, "checkpoint_path": "/secret/checkpoint"})
    return result, config


def cell(**kwargs):
    result, config = records()
    return b.make_cell(result, config, stage="ft_pilot", **kwargs)


def write_record(root, *, seed=1):
    result, config = records(seed=seed)
    path = root / "e8-main-adamw-s42" / "wikitext103" / f"muon-lr0.000443-s{seed}"
    path.mkdir(parents=True)
    (path / "result.json").write_text(json.dumps(result))
    (path / "config.json").write_text(json.dumps(config))
    return path


class FakeConfig(dict):
    def update(self, values, allow_val_change=False):
        assert allow_val_change is False
        for k, v in values.items():
            if k in self:
                assert self[k] == v
        super().update(values)


class FakeRun:
    def __init__(self, id, mode="online", history=None):
        self.id = id
        self.offline = mode == "offline"
        self.config = FakeConfig()
        self.summary = {}
        self.history = history or []
        self.step = self.history[-1]["_step"] + 1 if self.history else 0
        self.exit_codes = []
        self.fail_after = None

    def log(self, row, *, step, commit):
        assert commit and step >= self.step
        if self.fail_after is not None and len(self.history) >= self.fail_after:
            raise RuntimeError("dummy SDK secret must not appear in error")
        self.history.append(dict(row, _step=step))
        self.step = step + 1

    def finish(self, *, exit_code):
        self.exit_codes.append(exit_code)

    def scan_history(self, *, keys):
        assert keys == ["step"]
        return [{"step": row["step"]} for row in self.history]


class FakeWandb:
    def __init__(self):
        self.runs = {}
        self.inits = []

    def Settings(self, **kwargs):
        assert kwargs["disable_git"] and kwargs["disable_code"] and kwargs["x_disable_meta"]
        return kwargs

    def init(self, **kwargs):
        self.inits.append(kwargs)
        key = kwargs["id"]
        if kwargs["mode"] == "online":
            assert kwargs["resume"] == "allow"
            if key not in self.runs or self.runs[key].offline:
                self.runs[key] = FakeRun(key)
            return self.runs[key]
        assert kwargs["resume"] is None
        self.runs[key] = FakeRun(key, "offline")
        return self.runs[key]

    def Api(self):
        wb = self

        class Api:
            def run(self, path):
                return wb.runs[path.split("/")[-1]]

            def runs(self, path, *, filters):
                return [run for run in wb.runs.values() if all(run.config.get(k.removeprefix("config.")) == v
                                                               for k, v in filters.items())]
        return Api()


def test_stable_id_distinguishes_stage_source_validity_and_exact_lr():
    a = cell()
    assert a.run_id == cell().run_id
    assert a.run_id == cell(source="ft_pilot").run_id
    assert a.run_id != cell(source="replicated").run_id
    assert a.run_id != cell(invalid=True).run_id
    result, config = records()
    other_stage = b.make_cell(result, config, stage="ft_main")
    assert other_stage.run_id != a.run_id
    result["lr"] = config["cell"]["lr"] = .00044300001
    assert b.make_cell(result, config, stage="ft_pilot").run_id != a.run_id
    assert cell(invalid=True).metadata["invalid"] and "invalid-archive" in cell(invalid=True).tags


def test_payload_only_whitelisted_aggregate_metadata():
    a = cell()
    s = json.dumps([a.metadata, a.points, a.summary])
    for forbidden in ("dummy-secret", "/private", "/secret", "raw_documents", "cumulative_expert_counts"):
        assert forbidden not in s
    assert a.metadata["pretrain_seed"] == 42 and a.metadata["ft_seed"] == 1
    assert a.metadata["ft_steps"] == 60 and a.metadata["batch_tokens"] == 65536
    assert a.metadata["base_checkpoint_step"] == 11445
    assert a.group == "optimizer-mismatch-e8/finetune/wikitext103/ft_pilot"
    assert {"E8", "optimizer-mismatch"}.issubset(a.tags)
    assert a.name.startswith("ft_pilot/wikitext103/")
    assert {row["step"] for row in a.points} == {10, 20, 60}


def test_config_result_mismatch_or_incomplete_curve_rejected():
    result, config = records()
    result["seed"] = 2
    with pytest.raises(b.RecordError, match="identity mismatch"):
        b.make_cell(result, config, stage="ft_pilot")
    result, config = records()
    result["curve"] = result["curve"][:-1]
    with pytest.raises(b.RecordError, match="final checkpoint"):
        b.make_cell(result, config, stage="ft_pilot")


def test_online_idempotence_project_ledger_and_bulk_skip(tmp_path):
    wb, a = FakeWandb(), cell()
    ledger = b.Ledger(tmp_path / "ledger.json", "entity", "project")
    assert b.upload_cell(wb, a, ledger, "entity", "project", verify_online=True) == "uploaded"
    assert ledger.get(a.run_id)["state"] == "online"
    assert b.upload_cell(wb, a, ledger, "entity", "project") == "skipped"
    fresh = b.Ledger(tmp_path / "fresh.json", "entity", "project")
    existing = b.query_existing(wb, "entity", "project", "ft_pilot", "ft_pilot")
    assert b.upload_cell(wb, a, fresh, "entity", "project", existing=existing[a.run_id]) == "skipped"
    assert len(wb.inits) == 1 and len(wb.runs[a.run_id].history) == 3
    other = b.Ledger(tmp_path / "ledger.json", "entity", "another_project")
    assert other.get(a.run_id) is None


def test_resume_partial_history_after_failure_without_duplicate_or_loss(tmp_path):
    wb, a = FakeWandb(), cell()
    ledger = b.Ledger(tmp_path / "ledger.json", "entity", "project")
    damaged = wb.runs[a.run_id] = FakeRun(a.run_id)
    damaged.fail_after = 1
    with pytest.raises(b.UploadError, match="W&B config/history failed") as exc:
        b.upload_cell(wb, a, ledger, "entity", "project")
    assert "dummy SDK secret" not in str(exc.value)
    assert ledger.get(a.run_id) is None
    assert [row["_step"] for row in damaged.history] == [10]
    damaged.fail_after = None
    assert b.upload_cell(wb, a, ledger, "entity", "project", verify_online=True) == "uploaded"
    assert [row["_step"] for row in damaged.history] == [10, 20, 60]


def test_remote_history_ingestion_can_lag_completion_marker(tmp_path, monkeypatch):
    wb, a = FakeWandb(), cell()
    ledger = b.Ledger(tmp_path / "ledger.json", "entity", "project")
    b.upload_cell(wb, a, ledger, "entity", "project")
    run = wb.runs[a.run_id]
    original = run.scan_history
    seen = []

    def delayed(*, keys):
        seen.append(keys)
        return original(keys=keys)[:1] if len(seen) == 1 else original(keys=keys)

    monkeypatch.setattr(run, "scan_history", delayed)
    b.verify_uploaded(wb, a, "entity", "project", attempts=2, delay=0)
    assert seen == [["step"], ["step"]]


def test_different_saved_curve_with_same_identity_cannot_be_silently_replaced(tmp_path):
    wb, a = FakeWandb(), cell()
    ledger = b.Ledger(tmp_path / "ledger.json", "entity", "project")
    b.upload_cell(wb, a, ledger, "entity", "project")
    result, config = records()
    result["curve"][0]["learning"] += .001
    changed = b.make_cell(result, config, stage="ft_pilot")
    assert changed.run_id == a.run_id and changed.source_digest != a.source_digest
    with pytest.raises(b.UploadError, match="source digest differs"):
        b.upload_cell(wb, changed, ledger, "entity", "project")
    fresh = b.Ledger(tmp_path / "fresh.json", "entity", "project")
    with pytest.raises(b.UploadError, match="remote source digest differs"):
        b.upload_cell(wb, changed, fresh, "entity", "project")


def test_offline_ledger_is_not_synced_online(tmp_path):
    wb, a = FakeWandb(), cell()
    ledger = b.Ledger(tmp_path / "ledger.json", "entity", "project")
    assert b.upload_cell(wb, a, ledger, "entity", "project", mode="offline") == "offline_prepared"
    assert ledger.get(a.run_id)["state"] == "offline_prepared"
    assert b.upload_cell(wb, a, ledger, "entity", "project", mode="offline") == "skipped"
    assert b.upload_cell(wb, a, ledger, "entity", "project", mode="online") == "uploaded"
    assert ledger.get(a.run_id)["state"] == "online"


def test_divergence_is_logged_with_failed_status_and_incomplete_curve(tmp_path):
    wb = FakeWandb()
    ledger = b.Ledger(tmp_path / "ledger.json", "entity", "project")
    result, config = records(status="diverged")
    a = b.make_cell(result, config, stage="ft_pilot")
    assert b.upload_cell(wb, a, ledger, "entity", "project") == "uploaded"
    run = wb.runs[a.run_id]
    assert run.summary["result_status"] == "diverged" and run.exit_codes == [1]
    assert len(run.history) == 1 and ledger.get(a.run_id)["state"] == "online"


def test_cli_dry_run_filters_mirrors_and_no_ledger_or_network(tmp_path, capsys):
    root, mirror = tmp_path / "root", tmp_path / "mirror"
    for r in (root, mirror):
        write_record(r)
        write_record(r, seed=2)
    wb = FakeWandb()
    target = tmp_path / "ledger.json"
    assert b.main(["--root", str(root), "--root", str(mirror), "--stage", "pilot",
                   "--ledger", str(target), "--dataset", "wikitext103",
                   "--ft-optimizer", "muon", "--dry-run"], wb=wb) == 0
    assert not target.exists() and not wb.inits
    assert "planned=2" in capsys.readouterr().out


def test_cli_continues_independent_cells_on_failure(tmp_path, capsys):
    root = tmp_path / "root"
    bad = write_record(root)
    write_record(root, seed=2)
    (bad / "config.json").unlink()
    wb = FakeWandb()
    assert b.main(["--root", str(root), "--stage", "pilot", "--ledger", str(tmp_path / "ledger.json")],
                  wb=wb) == 1
    out, err = capsys.readouterr()
    assert "uploaded=1" in out and "failed=1" in out and "FAILED" in err
    assert len(wb.inits) == 1


def test_soft_router_scalars_whitelisted_without_raw_arrays():
    from nanochat.soft_routing import METRICS, schema
    result, config = records()
    result["soft_router"] = config["soft_router"] = schema(8192, "logits")
    for row in result["curve"]:
        for d in ("d0", "d1"):
            for metric in METRICS:
                row[f"soft_{d}_{metric}_mean"] = .5
        row["soft_sample_logits"] = [[1234.]]
    cell = b.make_cell(result, config, stage="ft_pilot")
    assert cell.metadata["soft_router_sample_format"] == "logits"
    assert cell.metadata["soft_router_sample_size"] == 8192
    assert len(b.SOFT_SCALARS) == 2*len(METRICS)
    assert set(b.SOFT_SCALARS) <= set(cell.points[0])
    assert "soft_sample_logits" not in cell.points[0]
    del result["curve"][0]["soft_d0_entropy_nats_mean"]
    with pytest.raises(b.RecordError, match="missing soft router"):
        b.make_cell(result, config, stage="ft_pilot")


def test_legacy_digest_unchanged_when_soft_evidence_absent():
    result, config = records()
    old = b.make_cell(result, config, stage="ft_pilot")
    result["soft_router"] = config["soft_router"] = None
    assert b.make_cell(result, config, stage="ft_pilot").source_digest == old.source_digest
