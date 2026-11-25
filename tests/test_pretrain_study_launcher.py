"""
Tests for the data checks in runs/pretrain_study.sh `train`.

They use tiny fake shards and a fake manifest, so they are fast and need no GPU.
`train` is stopped before it would start torchrun: the run name already has a
config.json, which the launcher rejects only AFTER the data checks pass.
"""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "runs" / "pretrain_study.sh"
RUN_NAME = "launcher-test"
REACHED_RUN_NAME_CHECK = f"Run {RUN_NAME} already has a config.json"

pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / ".venv" / "bin" / "torchrun").exists() or shutil.which("bash") is None,
    reason="the launcher needs this checkout's .venv and bash",
)


def sha256_line(path):
    return f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"


@pytest.fixture
def setup(tmp_path):
    scratch = tmp_path / "scratch"
    data = scratch / "base_data_fineweb_edu_100b"
    heldout = scratch / "heldout" / "fineweb-edu-100b"
    manifest = tmp_path / "manifest"
    for directory in (data, heldout, manifest, scratch / "base_checkpoints" / RUN_NAME):
        directory.mkdir(parents=True)
    (scratch / "base_checkpoints" / RUN_NAME / "config.json").write_text("{}")
    shards = []
    for name in ("shard_00000", "shard_00001", "shard_00002", "shard_01822"):  # last one is the validation shard
        shard = data / f"{name}.parquet"
        shard.write_bytes(f"contents of {name}".encode())
        shards.append(shard)
    (manifest / "shards.sha256").write_text("".join(sha256_line(shard) for shard in shards))
    for name in ("d0.parquet", "d0_report.json"):
        (heldout / name).write_bytes(f"contents of {name}".encode())
    (manifest / "heldout.sha256").write_text("".join(sha256_line(heldout / name) for name in ("d0.parquet", "d0_report.json")))
    environ = dict(os.environ, NANOCHAT_SCRATCH_BASE=str(scratch), NANOCHAT_POOL_BASE=str(tmp_path / "pool"),
                   NANOCHAT_DATA_MANIFEST_DIR=str(manifest), RUN_NAME=RUN_NAME, NPROC_PER_NODE="1")
    return {"data": data, "manifest": manifest, "environ": environ}


def launch(setup):
    return subprocess.run(["bash", str(LAUNCHER), "train", "--train-tokens=1"], cwd=REPO_ROOT, env=setup["environ"],
                          capture_output=True, text=True, timeout=120)


def test_matching_shards_pass_the_data_checks(setup):
    result = launch(setup)
    assert "Checking training shard fingerprints" in result.stderr
    assert REACHED_RUN_NAME_CHECK in result.stderr
    assert "differs from the frozen data manifest" not in result.stderr


def test_a_data_directory_that_is_a_symbolic_link_passes_the_data_checks(setup, tmp_path):
    real = tmp_path / "real_data"
    setup["data"].rename(real)
    setup["data"].symlink_to(real, target_is_directory=True)
    result = launch(setup)
    assert REACHED_RUN_NAME_CHECK in result.stderr
    assert "do not match the data manifest" not in result.stderr


def test_a_shard_edited_without_changing_its_size_is_rejected(setup):
    shard = setup["data"] / "shard_00001.parquet"
    size = shard.stat().st_size
    shard.write_bytes(b"X" * size)  # same size, different bytes: only a hash check can see this
    result = launch(setup)
    assert result.returncode == 1
    assert "shard_00001.parquet: FAILED" in result.stderr
    assert "A training shard differs from the frozen data manifest" in result.stderr
    assert REACHED_RUN_NAME_CHECK not in result.stderr


def test_a_missing_shard_is_rejected(setup):
    (setup["data"] / "shard_00001.parquet").unlink()
    result = launch(setup)
    assert result.returncode == 1
    assert "do not match the data manifest" in result.stderr
    assert "< shard_00001.parquet" in result.stderr
    assert REACHED_RUN_NAME_CHECK not in result.stderr


def test_an_extra_shard_is_rejected(setup):
    (setup["data"] / "shard_00003.parquet").write_bytes(b"not in the manifest")
    result = launch(setup)
    assert result.returncode == 1
    assert "do not match the data manifest" in result.stderr
    assert "> shard_00003.parquet" in result.stderr
    assert REACHED_RUN_NAME_CHECK not in result.stderr


def test_dry_run_does_not_check_the_data(setup):
    (setup["data"] / "shard_00001.parquet").unlink()
    result = subprocess.run(["bash", str(LAUNCHER), "dry-run", "--train-tokens=1"], cwd=REPO_ROOT, env=setup["environ"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and "Command:" in result.stdout


def test_wandb_is_online_with_a_key_and_offline_without(setup, tmp_path):
    # Without a key, online mode would stop the run at wandb.init.
    environ = {k: v for k, v in setup["environ"].items() if k not in ("WANDB_MODE", "WANDB_API_KEY")}
    netrc = tmp_path / "netrc"
    netrc.write_text("machine api.wandb.ai\n  login user\n  password secret-key-123\n")
    other = tmp_path / "netrc_other"
    other.write_text("machine example.com\n  login user\n  password x\n")
    cases = (({"NETRC": str(tmp_path / "missing")}, "offline"),
             ({"NETRC": str(other)}, "offline"),
             ({"NETRC": str(netrc)}, "online"),
             ({"NETRC": str(netrc), "WANDB_MODE": "offline"}, "offline"),
             ({"NETRC": str(tmp_path / "missing"), "WANDB_API_KEY": "k"}, "online"))
    for extra, expected in cases:
        result = subprocess.run(["bash", str(LAUNCHER), "dry-run", "--train-tokens=1"], cwd=REPO_ROOT,
                                env={**environ, **extra}, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0 and f"WANDB_MODE={expected}\n" in result.stdout, extra
        assert "secret-key-123" not in result.stdout + result.stderr


def test_validation_uses_the_validation_shard_and_d0_is_only_logged(setup):
    result = subprocess.run(["bash", str(LAUNCHER), "dry-run", "--train-tokens=1"], cwd=REPO_ROOT, env=setup["environ"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0
    command = result.stdout.split("Command:", 1)[1]
    heldout = setup["data"].parent / "heldout" / "fineweb-edu-100b"
    assert f"--heldout-data-dir={heldout}" in command
    assert "--val-data-dir" not in command  # base_train then validates on the last shard, shard_01822
    assert "--eval-tokens=10485760" in command
    # torchrun must not parse base_train's options ("--run=..." is ambiguous with --run-path)
    assert " -m scripts.base_train -- " in command and command.index(" -- ") < command.index("--run=")


def test_torchrun_passes_every_base_train_option_through(tmp_path):
    """The exact torchrun invocation of the launcher reaches base_train's own argument parser."""
    torchrun = REPO_ROOT / ".venv" / "bin" / "torchrun"
    result = subprocess.run([str(torchrun), "--standalone", "--nproc_per_node=1", "-m", "scripts.base_train", "--",
                             "--run=launcher-probe", f"--data-dir={tmp_path}", "--train-tokens=1"],
                            cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert "ambiguous option" not in result.stderr
    assert "--train-tokens must be divisible by --total-batch-size" in result.stderr  # base_train parsed it


@pytest.mark.parametrize("flag", ["--heldout-data-dir=/elsewhere", "--val-data-dir=/elsewhere"])
def test_callers_cannot_redirect_validation_or_heldout(setup, flag):
    result = subprocess.run(["bash", str(LAUNCHER), "dry-run", "--train-tokens=1", flag], cwd=REPO_ROOT,
                            env=setup["environ"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 2 and "is set by this launcher" in result.stderr
