"""
End-to-end smoke tests of scripts/base_train.py on CPU: tiny synthetic shards,
the GPT-2 tokenizer, a 2-layer top-1 MoE, 8 optimizer steps (~30 s per run).

Checks the things a real pretraining run relies on:
- AdamW and Muon cells consume the identical batch sequence (data fingerprints),
- routers use the run's optimizer, the run record and checkpoint metadata are complete,
- expert counts add up, gradient norms / MaxVio / bias are logged,
- resuming from a checkpoint reproduces the uninterrupted run.

python -m pytest tests/test_base_train_smoke.py -v      (marked slow)
"""

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

pytestmark = pytest.mark.slow
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_TOKENS, BATCH = 2048, 256  # 8 steps


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    try:
        from nanochat.tokenizer import get_tokenizer
        get_tokenizer("gpt2")
    except Exception as exc:
        pytest.skip(f"GPT-2 tokenizer unavailable: {exc}")
    base = tmp_path_factory.mktemp("nanochat_base")
    data = base / "data"
    data.mkdir()
    rng = random.Random(0)
    words = "the of and to in is was for on with as by at from that this which are be or".split() + [f"w{i}" for i in range(300)]
    for shard in range(3):  # 2 train shards + val shard (last)
        docs = [" ".join(rng.choice(words) for _ in range(rng.randint(20, 200))) for _ in range(400)]
        pq.write_table(pa.table({"text": docs}), data / f"shard_{shard:05d}.parquet", row_group_size=50)
    environ = dict(os.environ, NANOCHAT_BASE_DIR=str(base), WANDB_MODE="disabled",
                   TIKTOKEN_CACHE_DIR=os.environ["TIKTOKEN_CACHE_DIR"])
    return {"base": base, "data": data, "environ": environ}


def run_base_train(env, run, *extra):
    cmd = [sys.executable, "-m", "scripts.base_train", "--device-type", "cpu", "--data-dir", str(env["data"]),
           "--depth", "2", "--aspect-ratio", "32", "--head-dim", "32", "--max-seq-len", "64",
           "--device-batch-size", "2", "--total-batch-size", str(BATCH), "--train-tokens", str(TRAIN_TOKENS),
           "--warmup-steps", "1", "--warmdown-ratio", "0.25", "--eval-every", "4", "--eval-tokens", "256",
           "--core-metric-every", "-1", "--sample-every", "-1", "--save-every", "4",
           "--ffn-type", "moe", "--n-experts", "4", "--run", run, "--train-stats-every", "1",
           "--train-stats-jsonl", str(env["base"] / f"{run}.jsonl"), *extra]
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env["environ"], capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, f"base_train failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    return env["base"] / "base_checkpoints" / run


def meta(ckpt, step):
    return json.loads((ckpt / f"meta_{step:06d}.json").read_text())


def test_adamw_and_muon_cells_see_identical_data_and_route_routers_with_their_optimizer(env):
    adamw = run_base_train(env, "smoke-adamw", "--optimizer-name", "adamw", "--balancer", "aux")
    muon = run_base_train(env, "smoke-muon", "--optimizer-name", "muon", "--balancer", "aux")
    for step in (4, 8):
        assert meta(adamw, step)["data_fingerprints"] == meta(muon, step)["data_fingerprints"]
    assert meta(adamw, 4)["data_fingerprints"] != meta(adamw, 8)["data_fingerprints"]

    for ckpt, kind in ((adamw, "adamw"), (muon, "muon")):
        record = json.loads((ckpt / "config.json").read_text())
        roles = {(g["role"], g["kind"]) for g in record["optimizer"]["groups"]}
        assert ("router", kind) in roles and ("matrix", kind) in roles
        assert ("embedding", "adamw") in roles and ("unembedding", "adamw") in roles
        assert record["optimizer"]["max_grad_norm"] == 1.0
        assert record["data"]["tokenizer_name"] == "gpt2" and record["git"]["commit"]
        m = meta(ckpt, 8)
        assert m["tokenizer"] == "gpt2" and m["data_dir"] == str(env["data"])
        counts = torch.tensor(m["cumulative_expert_counts"])
        assert counts.shape == (2, 4) and (counts.sum(dim=1) == TRAIN_TOKENS).all()
        assert (ckpt / "routing_000008.pt").exists()

    records = [json.loads(line) for line in (env["base"] / "smoke-muon.jsonl").read_text().splitlines()]
    assert len(records) == TRAIN_TOKENS // BATCH
    last = records[-1]
    assert last["train/grad_norm"] > 0 and 0 < last["train/grad_clip_coef"] <= 1
    assert "train/moe/layer_1/max_vio" in last and "train/ce_loss" in last and "train/balance_loss" in last
    assert not any(key.endswith("_bias") or key.endswith("bias_spread") for key in last)  # aux balancer


def test_resume_reproduces_the_uninterrupted_run(env):
    ckpt = run_base_train(env, "smoke-resume", "--optimizer-name", "muon", "--balancer", "bias")
    full_meta = meta(ckpt, 8)
    full_state = torch.load(ckpt / "model_000008.pt", map_location="cpu")
    # Simulate a job that died after the step-4 checkpoint
    for name in ("model_000008.pt", "meta_000008.json", "optim_000008_rank0.pt", "routing_000008.pt"):
        (ckpt / name).unlink()
    run_base_train(env, "smoke-resume", "--optimizer-name", "muon", "--balancer", "bias", "--resume-from-step", "4")
    resumed_meta = meta(ckpt, 8)
    resumed_state = torch.load(ckpt / "model_000008.pt", map_location="cpu")

    assert resumed_meta["data_fingerprints"] == full_meta["data_fingerprints"]
    assert resumed_meta["cumulative_expert_counts"] == full_meta["cumulative_expert_counts"]
    assert (ckpt / "config_resume_000004.json").exists()
    bias_keys = [k for k in full_state if k.endswith("expert_bias")]
    assert bias_keys and any(full_state[k].abs().sum() > 0 for k in bias_keys)
    for key in full_state:
        torch.testing.assert_close(resumed_state[key], full_state[key], rtol=0, atol=0, msg=f"{key} differs after resume")


def test_schedule_that_does_not_fit_the_run_is_rejected(env):
    cmd = [sys.executable, "-m", "scripts.base_train", "--device-type", "cpu", "--data-dir", str(env["data"]),
           "--total-batch-size", str(BATCH), "--train-tokens", str(TRAIN_TOKENS)]  # default 130M-token warmup
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env["environ"], capture_output=True, text=True, timeout=300)
    assert result.returncode != 0 and "exceeds the run" in result.stderr
