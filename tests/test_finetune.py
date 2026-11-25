"""
Fine-tuning infrastructure (EXPERIMENT_SPEC.md §2, §6, §7, §8): dataset preparation, fixed
evaluation sets, routing/forgetting metrics, and an end-to-end CPU run of scripts/finetune.py.

python -m pytest tests/test_finetune.py -v
"""

import json
import math
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from nanochat.checkpoint_manager import save_checkpoint
from nanochat.dataset import list_parquet_files
from nanochat.eval_set import EvalSet, build_eval_set, per_doc_sums
from nanochat.finetune_data import detokenize_wikitext, wikitext_articles, write_manifest, write_shards
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
    a = build_eval_set("D0", path, TinyTokenizer(), seq_len=8, max_tokens=40)
    b = build_eval_set("D0", path, TinyTokenizer(), seq_len=8, max_tokens=40)
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
                   "--ft-data-dir", str(setup["ft"]), "--ft-name", "tiny", "--d0-file", str(setup["d0"]),
                   "--out-root", str(setup["root"] / "out"), "--device-type", "cpu",
                   "--eval-tokens", "512", "--eval-batch-size", "4", "--train-tokens", "1024",
                   "--total-batch-size", "128", "--device-batch-size", "2", "--n-checkpoints", "2", *extra])
    return setup["root"] / "out" / "tiny-base" / "tiny"


def test_finetune_end_to_end(tiny_setup):
    out = run_finetune(tiny_setup, "--lrs", "adamw=3e-3", "muon=3e-3", "--seeds", "1", "2")
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
    assert pq.read_table(run / "metrics.parquet").num_rows == len(rows)
    arrays = np.load(run / "ckpt_000008.npz")
    assert arrays["d0_routes"].dtype == np.uint8 and arrays["d0_routes"].shape[0] == 2

    # A resubmitted job skips finished cells
    before = (run / "result.json").stat().st_mtime_ns
    run_finetune(tiny_setup, "--lrs", "muon=3e-3", "--seeds", "1")
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
    base = evaluate_base(model, d0, 4, device)
    same = evaluate_checkpoint(model, model, d0, base, 4, device, full=True)
    for key in ("normal", "replay", "reverse"):
        torch.testing.assert_close(same["doc_sums"][key], base["doc_sums"], rtol=1e-6, atol=1e-6)
    assert all(stats["flip_rate"] == 0 and stats["cause_flips"] == 0 for stats in same["per_layer"].values())
