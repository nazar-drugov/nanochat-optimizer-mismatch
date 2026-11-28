"""The optional collaboration kit must not block authorized independent work."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "collab_cli", Path(__file__).resolve().parents[1] / "collab/bin/collab.py"
)
collab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collab)


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    def fake_git(args, root, **kwargs):
        return SimpleNamespace(stdout={
            "status": " M tracked.py",
            "rev-parse": "abcdef0123456789",
            "branch": "",
        }[args[0]])

    monkeypatch.setattr(collab, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(collab, "fetch", lambda root: None)
    monkeypatch.setattr(collab, "git", fake_git)
    monkeypatch.setattr(collab, "check_decisions", lambda root, ids: ["not ratified"])
    monkeypatch.setattr(collab, "me", lambda: "harry")
    monkeypatch.setattr(collab, "read_file", lambda root, path: None)
    records = {}
    monkeypatch.setattr(
        collab, "commit_on_remote", lambda root, build, message: records.update(build(root))
    )
    return tmp_path, records


@pytest.mark.parametrize("config", [{}, {"decisions": []}, {"decisions": ["D-001"]}])
def test_check_run_is_advisory(fake_repo, capsys, config):
    root, _ = fake_repo
    path = root / "run.json"
    path.write_text(json.dumps(config))
    collab.main(["check-run", str(path)])
    output = capsys.readouterr().out
    assert "advisory" in output
    assert "check-run OK" in output


@pytest.mark.parametrize("flag", ["--require-ratification", "--require-pushed"])
def test_strict_checks_are_opt_in(fake_repo, flag):
    root, _ = fake_repo
    path = root / "run.json"
    path.write_text(json.dumps({"decisions": ["D-001"]}))
    with pytest.raises(SystemExit) as exc:
        collab.main(["check-run", str(path), flag])
    assert exc.value.code == 1


@pytest.mark.parametrize("config", [{}, {"decisions": ["D-001"]}])
def test_enqueue_without_ratification(fake_repo, config):
    root, records = fake_repo
    path = root / "run.json"
    path.write_text(json.dumps(config))
    args = ["enqueue", "--run-id", "R-001", "--config", str(path), "--gpu-hours", "1"]
    collab.main(args)
    record, _ = collab.parse_md(records["collab/queue/R-001.md"])
    assert record["owner"] == "harry"
    assert record["decisions"] == config.get("decisions", [])
    with pytest.raises(SystemExit) as exc:
        collab.main(args + ["--require-ratification"])
    assert exc.value.code == 2


def test_owner_claim_and_duplicate_protection(fake_repo, monkeypatch):
    _, records = fake_repo
    path = "collab/queue/R-001.md"
    records[path] = collab.dump_md({"owner": "harry", "status": "queued"}, "")
    monkeypatch.setattr(collab, "read_file", lambda root, path: records.get(path))
    collab.main(["claim", "R-001"])
    meta, _ = collab.parse_md(records[path])
    assert meta["claimed_by"] == "harry"
    with pytest.raises(SystemExit) as exc:
        collab.main(["claim", "R-001"])
    assert exc.value.code == 2


def test_advisory_review_does_not_require_remote(fake_repo, monkeypatch, capsys):
    root, _ = fake_repo
    path = root / "run.json"
    path.write_text(json.dumps({"decisions": ["D-001"]}))

    def unavailable(root):
        raise collab.CollabError("remote unavailable")

    monkeypatch.setattr(collab, "fetch", unavailable)
    collab.main(["check-run", str(path)])
    assert "could not retrieve current decision state" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        collab.main(["check-run", str(path), "--require-pushed"])
    assert exc.value.code == 1
