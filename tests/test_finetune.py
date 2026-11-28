"""
Fine-tuning infrastructure (EXPERIMENT_SPEC.md §2, §6, §7, §8): dataset preparation, fixed
evaluation sets, routing/forgetting metrics, and an end-to-end CPU run of scripts/finetune.py.

python -m pytest tests/test_finetune.py -v
"""

import json
import math
import os
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from nanochat.checkpoint_manager import save_checkpoint
from nanochat.dataset import list_parquet_files
from nanochat.eval_set import EvalSet, build_eval_set, per_doc_sums
from nanochat.finetune_data import (
    assemble_heldout, detokenize_wikitext, hf_request, sample_heldout, wikitext_articles, write_manifest, write_shards,
)
from nanochat.finetune_metrics import (
    expert_relative_change, flip_cause_counts, load_shares, maxvio, route_comparison, select_experts, separation,
    top2_margin,
)
from nanochat.gpt import GPT, GPTConfig


class TinyTokenizer:
    def get_bos_token_id(self):
        return 0

    def encode(self, texts, prepend=None, num_threads=None):
        return [[prepend] + [ord(c) % 50 + 1 for c in text] for text in texts]


# ---------------------------------------------------------------------------
# Data preparation

def test_wikitext_is_grouped_into_articles_and_detokenized():
    lines = ["", " = Valkyria = \n", "", " = = Gameplay = = \n", " It sold 1 @,@ 000 copies , in 2011 @-@ 12 . \n",
             " = Second article = \n", " The ( big ) \" quoted \" cat 's tail : 3 @.@ 5 cm ! \n"]
    articles = wikitext_articles(lines)
    assert len(articles) == 2 and articles[0].startswith("= Valkyria =")
    text = detokenize_wikitext(articles[0])
    assert "1,000 copies, in 2011-12." in text and "== Gameplay ==" in text
    assert detokenize_wikitext(articles[1]).endswith('The (big) "quoted" cat\'s tail: 3.5 cm!')


def test_shards_put_validation_last_and_manifest_hashes_them(tmp_path):
    write_shards([f"doc {i}" for i in range(25)], tmp_path, "train", docs_per_shard=10)
    write_shards(["held out"], tmp_path, "val")
    files = [p.split("/")[-1] for p in list_parquet_files(str(tmp_path))]
    assert files == ["train_00000.parquet", "train_00001.parquet", "train_00002.parquet", "val_00000.parquet"]
    info = write_manifest(tmp_path, {"dataset": "x"})
    assert info["files"]["train_00002.parquet"]["rows"] == 5 and len(info["files"]["val_00000.parquet"]["sha256"]) == 64
    assert json.loads((tmp_path / "manifest.json").read_text())["dataset"] == "x"


# ---------------------------------------------------------------------------
# Evaluation sets

def test_eval_set_is_fixed_and_tracks_documents(tmp_path):
    docs = ["abc", "defgh", "ij", "klmnopq", "rs"] * 4
    path = tmp_path / "d.parquet"
    pq.write_table(pa.table({"text": docs}), path, row_group_size=3)
    a = build_eval_set("D0", path, TinyTokenizer(), seq_len=8, max_tokens=40, packing="contiguous")
    b = build_eval_set("D0", path, TinyTokenizer(), seq_len=8, max_tokens=40, packing="contiguous")
    assert a.inputs.shape == (5, 8) and torch.equal(a.inputs, b.inputs) and torch.equal(a.doc_ids, b.doc_ids)
    assert torch.equal(a.inputs.view(-1)[1:], a.targets.view(-1)[:-1])  # one continuous stream
    assert ((a.targets == 0) == (a.doc_ids == -1)).all()  # BOS targets are exactly the unscored ones
    assert a.doc_ids.view(-1)[:4].tolist() == [0, 0, 0, -1]  # "abc" then the next document's BOS
    sums, counts = per_doc_sums(torch.ones(5, 8), a.doc_ids, a.n_docs)
    assert counts[0] == 3 and counts[1] == 5 and sums.sum() == a.n_scored
    a.save(tmp_path / "set.pt")
    assert torch.equal(EvalSet.load(tmp_path / "set.pt").doc_ids, a.doc_ids)


# ---------------------------------------------------------------------------
# Metrics

def test_route_comparison_matches_hand_computation():
    r0 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    r1 = torch.tensor([0, 1, 1, 0, 2, 2, 3, 0])  # 3 flips; loads become (3, 2, 2, 1)/8
    s = route_comparison(r0, r1, 4)
    assert s["flip_rate"] == pytest.approx(3 / 8)
    assert s["tv"] == pytest.approx(0.5 * (abs(3 - 2) + 0 + 0 + abs(1 - 2)) / 8)
    assert s["churn"] == pytest.approx(s["flip_rate"] - s["tv"]) and s["tv"] <= s["flip_rate"]
    assert s["chance_flip_rate"] == pytest.approx(1 - sum(a * b for a, b in zip([.25] * 4, [3 / 8, 2 / 8, 2 / 8, 1 / 8])))
    assert s["maxvio"] == pytest.approx(4 * 3 / 8 - 1) and s["base_maxvio"] == pytest.approx(0)
    assert route_comparison(r0, r0, 4)["flip_rate"] == 0


def test_load_separation_selection_and_margin():
    shares = load_shares(torch.tensor([0, 0, 0, 1]), 3)
    assert shares.tolist() == [0.75, 0.25, 0.0] and maxvio(shares) == pytest.approx(1.25)
    assert separation(shares, shares) == pytest.approx(1.0)
    assert separation(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == 0
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    assert select_experts(logits).item() == 0
    assert select_experts(logits, torch.tensor([0.0, 0.5, 0.0])).item() == 1  # bias acts on probabilities
    p = torch.softmax(logits, -1)[0]
    assert top2_margin(logits).item() == pytest.approx((p[0] - p[1]).item())


def test_flip_cause_classes():
    # Router of 2 experts on 1-d inputs: expert = 0 if w.h > 0 else 1 (logits [w h, 0]).
    w0, w_pos, w_neg = torch.tensor([[1.0], [0.0]]), torch.tensor([[1.0], [0.0]]), torch.tensor([[-1.0], [0.0]])
    h0 = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    r0 = torch.zeros(4, dtype=torch.long)
    # token 0: input flips sign, weights unchanged -> input-driven
    c = flip_cause_counts(h0[:1], torch.tensor([[-1.0]]), w0, w_pos, r0[:1], torch.ones(1, dtype=torch.long))
    assert c == {"flips": 1, "input_driven": 1, "weight_driven": 0, "either": 0, "interaction": 0}
    # weights flip, inputs unchanged -> weight-driven
    c = flip_cause_counts(h0[:1], h0[:1], w0, w_neg, r0[:1], torch.ones(1, dtype=torch.long))
    assert c["weight_driven"] == 1
    # both flip: each alone suffices -> either
    c = flip_cause_counts(h0[:1], torch.tensor([[-1.0]]), w0, w_neg, r0[:1], torch.ones(1, dtype=torch.long))
    assert c["either"] == 1
    # no flip -> nothing counted
    assert flip_cause_counts(h0, h0, w0, w0, r0, r0)["flips"] == 0


def test_flip_cause_scored_masks_all_classes_without_changing_legacy_counts():
    # Old router picks by first input coordinate, current by second.
    w0 = torch.tensor([[1., 0.], [0., 0.]])
    w1 = torch.tensor([[0., 1.], [0., 0.]])
    h0 = torch.tensor([[1.,1.], [1.,-1.], [1.,-1.], [1.,1.], [1.,1.]])
    h1 = torch.tensor([[-1.,-1.], [1.,-1.], [-1.,-1.], [1.,-1.], [1.,1.]])
    r0, r1 = select_experts(h0 @ w0.T), select_experts(h1 @ w1.T)
    mask = torch.tensor([True, False, True, False, True])
    legacy = flip_cause_counts(h0, h1, w0, w1, r0, r1)
    scored = flip_cause_counts(h0, h1, w0, w1, r0, r1, scored_mask=mask)
    assert legacy == {"flips":4, "input_driven":1, "weight_driven":1, "either":1, "interaction":1}
    assert {k: scored[k] for k in legacy} == legacy
    assert {k: scored[f"scored_{k}"] for k in legacy} == {
        "flips":2, "input_driven":1, "weight_driven":0, "either":1, "interaction":0}
    assert scored["scored_flips"] == int((r0[mask] != r1[mask]).sum())
    with pytest.raises(ValueError, match="boolean mask"):
        flip_cause_counts(h0, h1, w0, w1, r0, r1, scored_mask=mask.long())


def test_expert_relative_change():
    config = GPTConfig(sequence_len=16, vocab_size=32, n_layer=2, n_head=2, n_kv_head=2, n_embd=16,
                       ffn_type="moe", n_experts=3)
    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.transformer.h[1].mlp.experts[2].up.weight.mul_(2.0)
    rho = expert_relative_change(model, base, 3)
    assert all(r == 0 for r in rho[0]) and rho[1][0] == 0 and rho[1][2] > 0
    w = {p: base[f"transformer.h.1.mlp.experts.2.{p}.weight"] for p in ("gate", "up", "down")}
    expected = w["up"].norm() / torch.sqrt(sum(t.square().sum() for t in w.values()))
    assert rho[1][2] == pytest.approx(float(expected), rel=1e-5)


# ---------------------------------------------------------------------------
# End to end on CPU

@pytest.fixture(scope="module")
def tiny_setup(tmp_path_factory):
    try:
        from nanochat.tokenizer import get_tokenizer
        tokenizer = get_tokenizer("gpt2")
    except Exception as exc:
        pytest.skip(f"GPT-2 tokenizer unavailable: {exc}")
    root = tmp_path_factory.mktemp("ft")
    rng = random.Random(0)
    words = "the of and to in is was for on with as by at from that this".split() + [f"w{i}" for i in range(200)]
    def docs(n, vocab):
        return [" ".join(rng.choice(vocab) for _ in range(rng.randint(30, 120))) for _ in range(n)]
    ft = root / "ft_data"
    write_shards(docs(300, words[:60]), ft, "train", docs_per_shard=150)
    write_shards(docs(80, words[:60]), ft, "val")
    d0 = root / "d0.parquet"
    pq.write_table(pa.table({"text": docs(80, words[40:])}), d0)
    config = GPTConfig(sequence_len=64, vocab_size=tokenizer.get_vocab_size(), n_layer=2, n_head=2, n_kv_head=2,
                       n_embd=32, ffn_type="moe", n_experts=4)
    model = GPT(config)
    model.init_weights(seed=3)
    ckpt = root / "base_checkpoints" / "tiny-base"
    save_checkpoint(str(ckpt), 10, model.state_dict(), None,
                    {"step": 10, "val_bpb": 3.0, "model_config": config.__dict__, "max_seq_len": 64, "tokenizer": "gpt2",
                     "user_config": {"optimizer_name": "adamw", "seed": 3}})
    return {"root": root, "ft": ft, "d0": d0}


def run_finetune(setup, *extra):
    from scripts import finetune
    finetune.main(["--base-run", "tiny-base", "--base-checkpoints-dir", str(setup["root"] / "base_checkpoints"),
                   "--base-step", "10",
                   "--ft-data-dir", str(setup["ft"]), "--ft-name", "tiny", "--d0-file", str(setup["d0"]),
                   "--out-root", str(setup["root"] / "out"), "--device-type", "cpu",
                   "--eval-tokens", "512", "--eval-batch-size", "4", "--train-tokens", "1024",
                   "--total-batch-size", "128", "--device-batch-size", "2", "--checkpoint-retention", "none", *extra])
    return setup["root"] / "out" / "tiny-base" / "tiny"


def test_finetune_end_to_end(tiny_setup):
    out = run_finetune(tiny_setup, "--lrs", "adamw=3e-3", "muon=3e-3", "--seeds", "1", "2",
                       "--checkpoint-spacing", "linear", "--n-checkpoints", "2")
    base = json.loads((out / "base" / "base_metrics.json").read_text())
    assert set(base["layers"]) == {"0", "1"} and 0 <= base["layers"]["0"]["separation"] <= 1
    results = {p.parent.name: json.loads(p.read_text()) for p in out.glob("*/result.json")}
    assert sorted(results) == ["adamw-lr0.003-s1", "adamw-lr0.003-s2", "muon-lr0.003-s1", "muon-lr0.003-s2"]
    for name, r in results.items():
        assert r["status"] == "completed" and r["steps"] == 8 and [c["step"] for c in r["curve"]] == [4, 8]
        c = r["curve"][-1]
        assert c["forgetting"] == pytest.approx(c["F_route"] + c["F_fixed"], abs=1e-9)  # exact decomposition
        assert math.isfinite(c["learning"]) and c["learning"] > 0  # fine-tuning lowers D1 loss
        assert np.asarray(r["cumulative_expert_counts"]).sum() == 2 * 1024  # 2 MoE layers x every token
    # Paired data order: same seed -> same batches under both optimizers; different seeds differ
    assert results["adamw-lr0.003-s1"]["data_fingerprint"] == results["muon-lr0.003-s1"]["data_fingerprint"]
    assert results["adamw-lr0.003-s1"]["data_fingerprint"] != results["adamw-lr0.003-s2"]["data_fingerprint"]

    run = out / "muon-lr0.003-s1"
    config = json.loads((run / "config.json").read_text())
    assert config["cell"]["ft_optimizer"] == "muon" and config["cell"]["pretrain_optimizer"] == "adamw"
    assert {g["role"]: g["kind"] for g in config["optimizer"]["groups"]}["router"] == "muon"
    assert all(g["weight_decay"] == 0 for g in config["optimizer"]["groups"] if g["role"] in ("matrix", "router"))
    assert len(config["data"]["eval_sets"]["D0"]["sha256"]) == 64 and config["data"]["eval_sets"]["D1"]["documents"] > 0
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * 2 * (1 + 2)  # checkpoints x eval sets x (model + 2 layers)
    layer_row = next(r for r in rows if r["eval_set"] == "D0" and r["layer"] == 1 and r["step"] == 8)
    assert {"flip_rate", "tv", "churn", "chance_flip_rate", "rho", "ft_tokens_per_expert", "cause_flips"} <= set(layer_row)
    assert layer_row["cause_flips"] == sum(layer_row[f"cause_{k}"] for k in
                                           ("input_driven", "weight_driven", "either", "interaction"))
    for row in rows:
        if row["eval_set"] == "D0" and row["layer"] >= 0:
            assert row["cause_scored_flips"] == round(row["flip_rate"] * config["data"]["eval_sets"]["D0"]["scored_tokens"])
            assert row["cause_scored_flips"] == sum(row[f"cause_scored_{k}"] for k in
                ("input_driven", "weight_driven", "either", "interaction"))
            assert row["cause_scored_flips"] <= row["cause_flips"]
    assert pq.read_table(run / "metrics.parquet").num_rows == len(rows)
    arrays = np.load(run / "ckpt_000008.npz")
    assert arrays["d0_routes"].dtype == np.uint8 and arrays["d0_routes"].shape[0] == 2
    from nanochat.soft_routing import METRICS
    with np.load(out/"base"/"base_arrays.npz") as base_arr:
        for d in ("d0", "d1"):
            assert arrays[f"{d}_soft_sample_logits"].dtype == np.float32
            assert arrays[f"{d}_soft_sample_logits"].shape[0] == 2
            assert arrays[f"{d}_soft_sample_probs"].size == 0
            assert np.array_equal(arrays[f"{d}_soft_doc_counts"], base_arr[f"{d}_doc_tokens"])
            assert arrays[f"{d}_soft_sample_positions_sha256"] == base_arr[f"{d}_soft_sample_positions_sha256"]
            assert arrays[f"{d}_soft_layer_sum"].shape == (2, len(METRICS))
            assert f"soft_{d}_entropy_nats_mean" in results["muon-lr0.003-s1"]["curve"][-1]
    assert config["soft_router"]["sample_format"] == "logits"
    assert {"soft_entropy_nats_mean", "soft_js_base_current_nats_mean", "soft_raw_z_loss_mean"} <= set(layer_row)
    assert "soft_entropy_nats_mean" in pq.read_table(run/"metrics.parquet").schema.names
    assert "cause_scored_flips" in pq.read_table(run/"metrics.parquet").schema.names

    # A resubmitted job skips finished cells
    before = (run / "result.json").stat().st_mtime_ns
    run_finetune(tiny_setup, "--lrs", "muon=3e-3", "--seeds", "1", "--checkpoint-spacing", "linear",
                 "--n-checkpoints", "2")
    assert (run / "result.json").stat().st_mtime_ns == before
    with pytest.raises(RuntimeError, match="router sample/schema differs"):
        run_finetune(tiny_setup, "--lrs", "muon=3e-3", "--seeds", "1", "--checkpoint-spacing", "linear",
                     "--n-checkpoints", "2", "--soft-router-sample-size", "0")
    assert (run / "result.json").stat().st_mtime_ns == before


def test_self_replay_identity_and_unchanged_model_has_no_forgetting(tiny_setup):
    """Spec §8 test 2: replaying a model's own routes reproduces its loss; and the base model
    measured against itself has zero forgetting, zero routing terms and zero flips."""
    from nanochat.checkpoint_manager import build_model
    from scripts.finetune import evaluate_base, evaluate_checkpoint
    from nanochat.tokenizer import get_tokenizer
    device = torch.device("cpu")
    model, tok, _ = build_model(str(tiny_setup["root"] / "base_checkpoints" / "tiny-base"), 10, device, "eval")
    d0 = build_eval_set("D0", tiny_setup["d0"], get_tokenizer("gpt2"), 64, 512)
    base = evaluate_base(model, d0, 4, device, soft_sample_size=16)
    same = evaluate_checkpoint(model, model, d0, base, 4, device, full=True)
    for key in ("normal", "replay", "reverse"):
        torch.testing.assert_close(same["doc_sums"][key], base["doc_sums"], rtol=1e-6, atol=1e-6)
    assert all(stats["flip_rate"] == 0 and stats["cause_flips"] == 0 for stats in same["per_layer"].values())
    for stats in same["per_layer"].values():
        assert stats["soft_js_base_current_nats_mean"] == 0.
        assert stats["soft_tv_base_current_mean"] == 0.
    np.testing.assert_array_equal(base["soft"].sample_logits, same["soft"].sample_logits)


def test_starcoder_heldout_sample_is_seeded_and_shuffled():
    docs = [f"doc{i}" for i in range(100)]
    a, b = sample_heldout(docs, 30, seed=0), sample_heldout(docs, 30, seed=0)
    assert a == b and len(set(a)) == 30 and a != sorted(a, key=lambda d: int(d[3:]))
    assert sample_heldout(docs, 30, seed=1) != a
    with pytest.raises(ValueError):
        sample_heldout(docs, 101, seed=0)


def test_hf_token_is_not_forwarded_on_redirect():
    req = hf_request("https://huggingface.co/datasets/x/resolve/main/f.parquet", token="secret")
    assert req.get_header("Authorization") == "Bearer secret"
    assert "Authorization" not in req.headers  # unredirected: dropped when urllib follows a redirect
    assert hf_request("https://huggingface.co/x").get_header("Authorization") is None


def test_log_spaced_checkpoints():
    from scripts.finetune import checkpoint_steps
    assert checkpoint_steps(240, 48) == [2, 4, 8, 16, 32, 64, 128, 192, 216, 240]
    assert checkpoint_steps(60, 12) == [2, 4, 8, 16, 32, 48, 54, 60]
    assert checkpoint_steps(960, 192) == [2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 864, 960]
    assert checkpoint_steps(8, 2, "linear", 2) == [4, 8]


def test_replay_loader_replays_the_branch_batches_from_a_buffer_reusing_loader():
    """The real loader overwrites one buffer per call; the trunk must still get the branch's batches."""
    from scripts.finetune import ReplayLoader

    def loader():
        buf = torch.zeros(2, 3, dtype=torch.long)
        for i in range(1, 100):
            buf.fill_(i)
            yield buf[0], buf[1], {}

    replay = ReplayLoader(loader())
    assert [int(next(replay)[0][0]) for _ in range(3)] == [1, 2, 3]
    replay.record()
    assert [int(next(replay)[0][0]) for _ in range(3)] == [4, 5, 6]
    replay.rewind()
    assert [int(next(replay)[0][0]) for _ in range(5)] == [4, 5, 6, 7, 8]


def test_decay_branches_equal_separate_runs(tiny_setup):
    """A length run as a decay branch of a longer trunk gives the same data, curve and final
    measurements as a separate run of that length (same seed, same warmup steps)."""
    common = ("--lrs", "muon=3e-3", "--seeds", "1")
    branched = run_finetune(tiny_setup, *common, "--branch-tokens", "512", "2048", "--checkpoint-retention", "full")  # 4, 8 (main), 16 steps
    results = {p.parent.name: json.loads(p.read_text()) for p in branched.glob("muon-lr0.003-s1-T*/result.json")}
    assert sorted(results) == ["muon-lr0.003-s1-T16", "muon-lr0.003-s1-T4", "muon-lr0.003-s1-T8"]
    assert all(r["status"] == "completed" and r["warmup_steps"] == 1 for r in results.values())
    assert [c["step"] for c in results["muon-lr0.003-s1-T16"]["curve"]] == [2, 4, 8, 13, 14, 16]
    assert [c["step"] for c in results["muon-lr0.003-s1-T4"]["curve"]] == [2, 3, 4]
    assert results["muon-lr0.003-s1-T4"]["curve"][0] == results["muon-lr0.003-s1-T16"]["curve"][0]  # shared step 2
    # Actual state files are hard-linked only for the identical shared prefix,
    # never for two schedules that have already diverged.
    from nanochat.checkpoint_manager import load_checkpoint, build_model
    dirs = {n: branched / f"muon-lr0.003-s1-T{n}" for n in (4, 8, 16)}
    for n, d in dirs.items():
        r = results[d.name]
        assert r["checkpoint_retention"] == "full"
        for p in r["curve"]:
            step = p["step"]
            for name in (f"model_{step:06d}.pt", f"optim_{step:06d}_rank0.pt", f"meta_{step:06d}.json"):
                assert (d / name).stat().st_size > 0
            provenance = json.loads((d / f"meta_{step:06d}.json").read_text())
            assert provenance["cell"]["ft_steps"] == n and provenance["step"] == step
            assert provenance["tokens"] == step*128 and not provenance["exact_resume_supported"]
            assert provenance["optimizer_saved"] is True and provenance["data_fingerprint"] is not None
            with np.load(d / f"ckpt_{step:06d}.npz") as arrays:
                assert arrays["d1_routes"].dtype == np.uint8 and arrays["d1_routes"].shape[0] == 2
    for name in ("model_000002.pt", "optim_000002_rank0.pt", "ckpt_000002.npz"):
        assert all(os.path.samefile(dirs[4]/name, dirs[n]/name) for n in (8, 16))
    for name in ("model_000004.pt", "optim_000004_rank0.pt", "ckpt_000004.npz"):
        assert not os.path.samefile(dirs[4]/name, dirs[16]/name)
    assert not os.path.samefile(dirs[4]/"meta_000002.json", dirs[16]/"meta_000002.json")
    m4, o4, meta4 = load_checkpoint(dirs[4], 4, torch.device("cpu"), load_optimizer=True)
    m8, _, _ = load_checkpoint(dirs[8], 8, torch.device("cpu"), load_optimizer=True)
    m16, o16, _ = load_checkpoint(dirs[16], 8, torch.device("cpu"), load_optimizer=True)
    assert not os.path.samefile(dirs[8]/"model_000008.pt", dirs[16]/"model_000008.pt")
    assert any(not torch.equal(m8[k], m16[k]) for k in m8)
    restored, tok, _ = build_model(dirs[4], 4, torch.device("cpu"), "eval")
    assert all(torch.equal(restored.state_dict()[k], v) for k, v in m4.items())
    restored_optimizer = restored.setup_optimizer(optimizer_name="muon")
    restored_optimizer.load_state_dict(o4)
    assert restored_optimizer.state_dict()["param_groups"] == o4["param_groups"]
    assert o4["state"] and o16["state"]
    from scripts.finetune import evaluate_base
    d0 = build_eval_set("D0", tiny_setup["d0"], tok, 64, 512)
    replay = evaluate_base(restored, d0, 4, torch.device("cpu"))
    with np.load(dirs[4] / "ckpt_000004.npz") as arr:
        np.testing.assert_allclose(replay["doc_sums"].numpy(), arr["d0_doc_loss"], rtol=1e-6, atol=1e-6)
    rows = [json.loads(line) for line in (branched / "muon-lr0.003-s1-T4" / "metrics.jsonl").read_text().splitlines()]
    assert {r["run"] for r in rows} == {"muon-lr0.003-s1-T4"} and {r["ft_steps"] for r in rows} == {4}
    for tokens, branch in ((512, "T4"), (1024, "T8"), (2048, "T16")):  # T16 is the trunk
        from scripts import finetune
        out = tiny_setup["root"] / f"separate_{tokens}"
        finetune.main(["--base-run", "tiny-base", "--base-checkpoints-dir", str(tiny_setup["root"] / "base_checkpoints"),
                           "--base-step", "10",
                       "--ft-data-dir", str(tiny_setup["ft"]), "--ft-name", "tiny", "--d0-file", str(tiny_setup["d0"]),
                       "--out-root", str(out), "--device-type", "cpu", "--eval-tokens", "512",
                       "--eval-batch-size", "4", "--train-tokens", str(tokens), "--total-batch-size", "128",
                       "--device-batch-size", "2", "--checkpoint-retention", "none", "--no-soft-router-capture", *common])
        single = json.loads((out / "tiny-base" / "tiny" / "muon-lr0.003-s1" / "result.json").read_text())
        branch_result = results[f"muon-lr0.003-s1-{branch}"]
        assert single["data_fingerprint"] == branch_result["data_fingerprint"]
        assert [c["step"] for c in single["curve"]] == [c["step"] for c in branch_result["curve"]]
        for a, b in zip(single["curve"], branch_result["curve"]):
            for key in ("learning", "forgetting", "F_route", "train_loss"):
                assert a[key] == pytest.approx(b[key], rel=1e-5, abs=1e-7), (branch, a["step"], key)
        assert single["cumulative_expert_counts"] == branch_result["cumulative_expert_counts"]
    # Resubmission keeps all completed siblings. Simulate a preemption after
    # measured states were persisted but before a result marker was published.
    original_attempt = results[dirs[4].name]["attempt_id"]
    assert all(r["attempt_id"] == original_attempt for r in results.values())
    prev_mtime = (dirs[4]/"result.json").stat().st_mtime_ns
    (dirs[8]/"result.json").rename(dirs[8]/"result.unpublished.json")
    run_finetune(tiny_setup, *common, "--branch-tokens", "512", "2048", "--checkpoint-retention", "full")
    assert (dirs[4]/"result.json").stat().st_mtime_ns == prev_mtime
    new = json.loads((dirs[8]/"result.json").read_text())
    assert new["attempt_id"] != original_attempt
    assert new["curve"] == results[dirs[8].name]["curve"]
    archived = list((tiny_setup["root"]/"out"/".interrupted"/"tiny-base"/"tiny").glob(dirs[8].name+".*"))
    assert len(archived)==1 and (archived[0]/"result.unpublished.json").is_file()
    assert os.path.samefile(archived[0]/"model_000002.pt", dirs[4]/"model_000002.pt")
    assert not os.path.samefile(dirs[8]/"model_000002.pt", dirs[4]/"model_000002.pt")
    assert json.loads((dirs[8]/"meta_000002.json").read_text())["shared_state_horizons"] == [8]
    # Missing state or conflicting retention is diagnosed without overwriting.
    missing = dirs[4]/"optim_000004_rank0.pt"
    tmp = dirs[4]/"optim.unavailable.pt"
    missing.rename(tmp)
    with pytest.raises(RuntimeError, match="missing or incomplete retained checkpoint artifact"):
        run_finetune(tiny_setup, *common, "--branch-tokens", "512", "2048", "--checkpoint-retention", "full")
    tmp.rename(missing)
    with pytest.raises(RuntimeError, match="checkpoint retention is"):
        run_finetune(tiny_setup, *common, "--branch-tokens", "512", "2048", "--checkpoint-retention", "weights")
    assert (dirs[4]/"result.json").stat().st_mtime_ns == prev_mtime


@pytest.mark.parametrize("policy,expected", [
    ("none", (False, False, False, False)), ("weights", (True, False, True, False)),
    ("weights+final_optimizer", (True, False, True, True)), ("full", (True, True, True, True))])
def test_retention_policy(policy, expected):
    from scripts.finetune import retention_files
    assert (*retention_files(policy, 2, 4), *retention_files(policy, 4, 4)) == expected


def test_wikitext_d1_takes_all_official_articles_and_tops_up_from_training():
    official = [f"v{i} " * 10 for i in range(3)]              # 30 "tokens"
    train = [f"t{i} " * 10 for i in range(20)]
    count = lambda text: len(text.split())
    enough = lambda lengths: sum(lengths) >= 75
    d1, rest, moved, total = assemble_heldout(official, train, count, enough, seed=0, step=1)
    assert set(official) <= set(d1) and moved == 5 and total == 80 and len(d1) == 8
    assert len(rest) == 15 and not set(rest) & set(d1)        # moved articles leave training
    assert rest == [t for t in train if t not in d1]          # training keeps its order
    assert assemble_heldout(official, train, count, enough, seed=0, step=1)[0] == d1
    with pytest.raises(ValueError):
        assemble_heldout(official, train, count, lambda lengths: False, seed=0)


def test_bestfit_eval_set_reproduces_the_pretraining_loader_rows(tmp_path):
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    from nanochat.eval_set import bestfit_capacity
    rng = random.Random(0)
    docs = ["".join(rng.choice("abcdefghij") for _ in range(rng.randint(1, 40))) for _ in range(300)]
    (tmp_path / "d").mkdir()
    path = tmp_path / "d" / "val.parquet"
    pq.write_table(pa.table({"text": docs}), path, row_group_size=50)
    tok, seq_len, rows = TinyTokenizer(), 16, 40
    es = build_eval_set("D0", path, tok, seq_len, rows * seq_len, buffer_size=20)
    loader = tokenizing_distributed_data_loader_bos_bestfit(tok, 8, seq_len, split="val", device="cpu",
                                                            data_dir=str(tmp_path / "d"), buffer_size=20)
    batches = [tuple(t.clone() for t in next(loader)) for _ in range(rows // 8)]  # the loader reuses its buffers
    ref_x, ref_y = map(torch.cat, zip(*batches))
    assert torch.equal(es.inputs, ref_x) and torch.equal(es.targets, ref_y)
    assert (es.inputs[:, 0] == 0).all()                      # every row starts with a BOS
    assert ((es.targets == 0) == (es.doc_ids == -1)).all()   # BOS targets are exactly the unscored ones
    # each scored target is the right token of its document, counted from the document's BOS
    encoded = tok.encode(docs, prepend=0)
    for r in range(rows):
        start = None
        for t in range(seq_len):
            if es.inputs[r, t] == 0:
                start = t
            d = int(es.doc_ids[r, t])
            if d >= 0:
                assert int(es.targets[r, t]) == encoded[int(es.doc_rows[d])][t + 1 - start]
    assert es.n_docs == int(es.doc_rows.numel()) and es.packing == "bestfit"
    # capacity from lengths alone matches what the builder can do
    capacity = bestfit_capacity([len(e) for e in encoded], seq_len + 1, buffer_size=20, row_group_size=50)
    assert build_eval_set("D0", path, tok, seq_len, capacity * seq_len, buffer_size=20).inputs.size(0) == capacity
    with pytest.raises(ValueError):
        build_eval_set("D0", path, tok, seq_len, (capacity + 1) * seq_len, buffer_size=20)
    es.save(tmp_path / "set.pt")
    assert torch.equal(EvalSet.load(tmp_path / "set.pt").doc_rows, es.doc_rows)


# ---------------------------------------------------------------------------
# Review checks (2026-11-25): bias balancer, metrics.parquet, concurrent writes, cell naming

def _bias_update_after_one_step(finetune_module):
    from types import SimpleNamespace
    torch.manual_seed(0)
    config = GPTConfig(sequence_len=16, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, ffn_type="moe",
                       n_experts=4, balancer="bias", bias_update_rate=0.01)
    model = GPT(config)
    model.init_weights(seed=1)
    model.train()
    optimizer = model.setup_optimizer(optimizer_name="adamw", adamw_matrix_lr=0.0, embedding_lr=0.0, unembedding_lr=0.0)
    batches = [(torch.randint(1, 64, (2, 16)), torch.randint(1, 64, (2, 16))) for _ in range(2)]
    with torch.no_grad():  # the step's counts, as the micro-batches route them with the starting bias
        counts = sum(torch.stack([s["expert_counts"] for s in model(x, y, return_metrics=True)[2]]) for x, y in batches)
    before = model.expert_bias_state()
    state = {"step": 0, "counts": torch.zeros(2, 4, dtype=torch.long), "fingerprint": torch.zeros((), dtype=torch.long),
             "fp_weights": None}
    ctx = {"model": model, "compiled": model, "warmup": 0, "accum": 2}
    finetune_module.train_steps(SimpleNamespace(balancer_mode="active"), ctx, optimizer, iter(batches), state, 1, 1,
                                SimpleNamespace(steps=1, decay=0), lambda step, loss: None)
    return model.expert_bias_state() - before, counts, state["counts"]


def test_bias_balancer_updates_once_per_optimizer_step_from_summed_counts():
    """As in pretraining (base_train.py): with gradient accumulation the bias moves once per step,
    by u * sign(mean - load) of the counts summed over the micro-batches."""
    from scripts import finetune
    delta, counts, recorded = _bias_update_after_one_step(finetune)
    expected = 0.01 * torch.sign(counts.float().mean(dim=-1, keepdim=True) - counts.float())
    torch.testing.assert_close(delta, expected)
    assert torch.equal(recorded, counts)


def test_metrics_parquet_keeps_every_column(tiny_setup):
    out = run_finetune(tiny_setup, "--lrs", "adamw=2e-3", "--seeds", "3", "--checkpoint-spacing", "linear",
                       "--n-checkpoints", "2")
    run = out / "adamw-lr0.002-s3"
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    table = pq.read_table(run / "metrics.parquet")
    assert set(table.column_names) == {k for row in rows for k in row}  # incl. flip_rate, rho, D1 loss, ...
    back = [{k: v for k, v in row.items() if v is not None} for row in table.to_pylist()]
    assert back == rows


def _write_json_repeatedly(path, n):
    from nanochat.train_config import write_json_atomic
    for _ in range(n):
        write_json_atomic(path, {"x": list(range(20000))})


def test_write_json_atomic_survives_concurrent_writers(tmp_path):
    """Several fine-tuning jobs of one base model write the same base/ files at the same time."""
    import multiprocessing as mp
    path = tmp_path / "base" / "base_metrics.json"
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_write_json_repeatedly, args=(path, 150)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert [p.exitcode for p in procs] == [0, 0]
    assert len(json.loads(path.read_text())["x"]) == 20000
    assert [p.name for p in path.parent.iterdir()] == ["base_metrics.json"]  # no temporary files left


def test_cells_with_colliding_directory_names_are_rejected(tiny_setup):
    with pytest.raises(SystemExit):  # 1.0e-3 and 1.0004e-3 would share the directory adamw-lr0.001-s1
        run_finetune(tiny_setup, "--lrs", "adamw=1e-3,1.0004e-3", "--seeds", "1")
    out = run_finetune(tiny_setup, "--lrs", "adamw=1e-3", "--seeds", "4", "--train-tokens", "256")
    with pytest.raises(RuntimeError, match="result.json has lr"):  # another job's cell, same directory name
        run_finetune(tiny_setup, "--lrs", "adamw=1.0004e-3", "--seeds", "4", "--train-tokens", "256")
    assert json.loads((out / "adamw-lr0.001-s4" / "result.json").read_text())["lr"] == 1e-3


@pytest.mark.parametrize("policy,seed,intermediate_optim,final_optim", [
    ("none", 7, False, False),
    ("weights", 8, False, False),
    ("weights+final_optimizer", 9, False, True),
])
def test_checkpoint_modes_save_exact_requested_artifacts(tiny_setup, policy, seed, intermediate_optim, final_optim):
    out = run_finetune(tiny_setup, "--lrs", "adamw=3e-3", "--seeds", str(seed), "--train-tokens", "512",
                       "--checkpoint-retention", policy)
    path = out / f"adamw-lr0.003-s{seed}"
    result = json.loads((path/"result.json").read_text())
    assert result["checkpoint_retention"] == policy and result["status"] == "completed"
    for p in result["curve"]:
        step = p["step"]
        keep_weights = policy != "none"
        keep_optim = final_optim if step==4 else intermediate_optim
        assert (path/f"model_{step:06d}.pt").exists() == keep_weights
        assert (path/f"optim_{step:06d}_rank0.pt").exists() == keep_optim
        assert (path/f"meta_{step:06d}.json").exists() == keep_weights
        assert (path/f"ckpt_{step:06d}.npz").exists()
        if keep_weights:
            meta = json.loads((path/f"meta_{step:06d}.json").read_text())
            assert meta["optimizer_saved"] == keep_optim
            assert "model" in meta["artifacts"] and ("optim" in meta["artifacts"]) == keep_optim
