"""Analysis tests: LR selection, threshold geometry, paired resampling, and file validation."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import analyze_finetune as a


def point(step, learning, forgetting, loss_d1=1.0):
    return {"step": step, "learning": learning, "forgetting": forgetting,
            "F_route": forgetting * .25, "F_fixed": forgetting * .75,
            "loss_d1": loss_d1}


def fake_run(base, ft, seed, lr, loss):
    return a.Run(Path("/unused"), base, ft, seed, lr, 2,
                 [point(1, .2, .1, loss+.1), point(2, .5, .2, loss)], {})


def test_matched_learning_uses_interpolation_and_minimum_feasible_forgetting():
    curve = [point(2, .2, .1), point(4, .8, .5), point(6, .9, .7)]
    m = a.match_curve(curve, .5)
    assert m["step"] == pytest.approx(3) and m["forgetting"] == pytest.approx(.3)
    assert m["F_route"] + m["F_fixed"] == pytest.approx(m["forgetting"])
    curve.append(point(8, 1.0, .15))
    m = a.match_curve(curve, .5)
    assert m["step"] == 8 and m["forgetting"] == pytest.approx(.15)
    assert a.match_curve(curve, 1.1) is None
    assert a.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)


def test_lr_selection_requires_paired_complete_grid():
    runs = [fake_run(base, ft, seed, lr, 1.0 if lr == .01 else 1.2)
            for base in a.BASES for ft in a.OPTS for seed in (1, 2) for lr in (.01, .02)]
    selected, rows = a.select_lrs(runs, [1, 2], 2)
    assert all(lr == .01 for lr in selected.values())
    assert len(rows) == 8 and all(r["edge"] for r in rows if r["selected"])
    with pytest.raises(ValueError, match="missing or duplicate seeds"):
        a.select_lrs(runs[:-1], [1, 2], 2)


def test_paired_document_bootstrap_keeps_common_samples(tmp_path):
    runs = []
    for base in a.BASES:
        for ft in a.OPTS:
            for seed in (1, 2):
                path = tmp_path / base / "data" / f"{ft}-s{seed}"
                path.mkdir(parents=True)
                baseline = path.parent / "base"
                baseline.mkdir(exist_ok=True)
                np.savez_compressed(baseline / "base_arrays.npz",
                                    d0_doc_loss=[1., 1.], d0_doc_tokens=[1., 1.],
                                    d1_doc_loss=[1., 1.], d1_doc_tokens=[1., 1.])
                extra = .1 if ft != base.split("-")[2] else 0.
                np.savez_compressed(path / "ckpt_000001.npz",
                                    d0_doc_loss=[1.+extra]*2, d0_doc_loss_replay=[1.+extra*.75]*2,
                                    d1_doc_loss=[.5]*2)
                runs.append(a.Run(path, base, ft, seed, .01, 1,
                                  [point(1, .5, extra, .5)], {}))
    ci = a.bootstrap(runs, [1, 2], .2, 30, 0)
    assert ci["accepted"] == 30
    for interval in ci["mismatch_ci"].values():
        assert interval == pytest.approx([.04, .04])


def eval_identity(d, packing, n_docs, **changes):
    identity = {"file": f"/elsewhere/{d}.parquet", "sha256": d, "rows": 1, "seq_len": n_docs,
                "scored_tokens": n_docs, "documents": n_docs, **changes}
    if packing is not None:
        identity["packing"] = packing
    return identity


def small_eval_set(d, packing, n_docs):
    """One row holding one scored token from each of n_docs documents, IDs compact in file order."""
    from nanochat.eval_set import EvalSet
    inputs = torch.zeros(1, n_docs, dtype=torch.long)
    targets = torch.ones(1, n_docs, dtype=torch.long)
    return EvalSet(d, inputs, targets, torch.arange(n_docs)[None], n_docs, f"/elsewhere/{d}.parquet", d,
                   packing=packing or "contiguous",
                   doc_rows=torch.arange(n_docs) if packing == "bestfit" else None)


def write_grid(root, packing="bestfit", n_docs=2, with_eval_sets=False, bases=a.BASES, **identity_changes):
    """A complete synthetic grid; packing None writes the legacy identity without the field."""
    for base in bases:
        for ft in a.OPTS:
            for lr in (.01, .02):
                for seed in (1, 2):
                    path = root / base / "data" / f"{ft}-lr{lr}-s{seed}"
                    path.mkdir(parents=True)
                    cell = {"base_run": base, "ft_optimizer": ft, "seed": seed,
                            "lr": lr, "ft_steps": 2}
                    config = {"cell": cell, "moe_layer_indices": [0],
                              "model_config": {"n_experts": 8},
                              "base": {"step": 11445, "user_config": {"seed": 42,
                                      "optimizer_name": base.split("-")[2]}},
                              "args": {"total_batch_size": 65536},
                              "data": {"eval_sets": {d: eval_identity(d, packing, n_docs, **identity_changes)
                                                     for d in ("D0", "D1")}}}
                    result = {**cell, "ft_dataset": "data", "status": "completed",
                              "curve": [point(1, .2, .1, .8),
                                        point(2, .5, .2, .5 if lr == .01 else .6)]}
                    (path / "config.json").write_text(json.dumps(config))
                    (path / "result.json").write_text(json.dumps(result))
                    baseline = path.parent / "base"
                    baseline.mkdir(exist_ok=True)
                    np.savez_compressed(baseline / "base_arrays.npz",
                                        d0_doc_loss=[1.]*n_docs, d0_doc_tokens=[1.]*n_docs,
                                        d1_doc_loss=[1.]*n_docs, d1_doc_tokens=[1.]*n_docs)
                    if with_eval_sets:
                        for d in ("D0", "D1"):
                            small_eval_set(d, packing, n_docs).save(baseline / f"eval_set_{d}.pt")
                    for step, d0, d1 in ((1, 1.1, .8), (2, 1.2, .5)):
                        np.savez_compressed(path / f"ckpt_{step:06d}.npz",
                                            d0_doc_loss=[d0]*n_docs, d0_doc_loss_replay=[1.+(d0-1)*.75]*n_docs,
                                            d1_doc_loss=[d1]*n_docs)
                    rows = [{"eval_set": "D0", "layer": 0, "step": step,
                             "base_maxvio": .1, "flip_rate": step*.1, "maxvio": .1+step*.1,
                             "tv": step*.05, "churn": step*.05,
                             "flip_rate_normalized": step*.2} for step in (1, 2)]
                    (path / "metrics.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    return root


def run_main(root_args, tmp_path, *extra):
    argv = [x for root in root_args for x in ("--root", str(root))]
    return a.main([*argv, "--dataset", "data", "--steps", "2", "--seeds", "1", "2", "--min-lrs", "2",
                   "--out", str(tmp_path / "out"), *extra])


def test_analysis_reads_complete_grid_and_rejects_wrong_base(tmp_path, capsys):
    root = write_grid(tmp_path / "runs")
    out = tmp_path / "out"
    a.main(["--root", str(root), "--dataset", "data", "--steps", "2", "--seeds", "1", "2",
            "--min-lrs", "2", "--out", str(out)])
    summary = json.loads((out / "analysis.json").read_text())
    assert len(summary["source_runs"]) == 16
    assert summary["common_learning_support"] == [0., .5]
    assert (out / "lr_selection.png").is_file() and (out / "learning_forgetting.png").is_file()
    second_root = tmp_path / "other_runs"
    second_root.mkdir()
    shutil.move(str(root / a.BASES[1]), str(second_root / a.BASES[1]))
    a.main(["--root", str(root), "--root", str(second_root), "--dataset", "data",
            "--steps", "2", "--seeds", "1", "2", "--min-lrs", "2",
            "--ell0", ".1", "--bootstrap", "20", "--out", str(out)])
    summary = json.loads((out / "analysis.json").read_text())
    assert len(summary["matched_runs"]) == 8 and len(summary["contrasts"]) == 2
    assert summary["bootstrap"]["accepted"] == 20
    assert (out / "router_diagnostics.png").is_file() and (out / "replay_terms.png").is_file()
    assert (out / "router_vs_forgetting.png").is_file() and (out / "router_runs.csv").is_file()
    wrong = next(root.glob("e8-main-adamw-s42/data/*/config.json"))
    config = json.loads(wrong.read_text())
    config["base"]["step"] = 10
    wrong.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="wrong base checkpoint"):
        a.main(["--root", str(root), "--root", str(second_root), "--dataset", "data",
                "--steps", "2", "--seeds", "1", "2", "--min-lrs", "2", "--out", str(out)])


def test_analysis_reports_bestfit_packing_and_checks_the_saved_sets(tmp_path, capsys):
    root = write_grid(tmp_path / "runs", n_docs=3, with_eval_sets=True)
    run_main([root], tmp_path, "--ell0", ".1", "--bootstrap", "10")
    summary = json.loads((tmp_path / "out" / "analysis.json").read_text())
    assert summary["eval_packing"] == "bestfit" and summary["runs_without_packing_field"] == 0
    assert summary["eval_sets"]["D0"]["documents"] == 3 and summary["bootstrap"]["accepted"] == 10
    assert "Evaluation packing: bestfit" in capsys.readouterr().out
    saved = next(root.glob(f"{a.BASES[0]}/data/base/eval_set_D1.pt"))
    es = torch.load(saved, weights_only=False)

    def rewrite(**changes):
        torch.save({**es, **changes}, saved)
        a.check_eval_set_file.cache_clear()

    ids = es["doc_ids"].clone()
    ids[0, 1] = 0                                   # counts no longer match the saved arrays
    rewrite(doc_ids=ids)
    with pytest.raises(ValueError, match="scored counts differ"):
        run_main([root], tmp_path, "--ell0", ".1", "--bootstrap", "10")
    ids = es["doc_ids"].clone()
    ids[0, 2] = 3                                   # an ID outside the compact range
    rewrite(doc_ids=ids)
    with pytest.raises(ValueError, match="outside"):
        run_main([root], tmp_path, "--ell0", ".1", "--bootstrap", "10")
    rewrite(doc_ids=es["doc_ids"], doc_rows=None)
    with pytest.raises(ValueError, match="doc_rows"):
        run_main([root], tmp_path, "--ell0", ".1", "--bootstrap", "10")
    rewrite(doc_rows=es["doc_rows"], packing="contiguous")
    with pytest.raises(ValueError, match="config says bestfit"):
        run_main([root], tmp_path, "--ell0", ".1", "--bootstrap", "10")
    rewrite(packing="bestfit", n_docs=4)
    with pytest.raises(ValueError, match="documents in the set"):
        run_main([root], tmp_path, "--ell0", ".1", "--bootstrap", "10")


def test_bestfit_set_with_a_document_split_across_rows_is_rejected(tmp_path):
    root = write_grid(tmp_path / "runs", n_docs=2, with_eval_sets=True)
    saved = next(root.glob(f"{a.BASES[0]}/data/base/eval_set_D0.pt"))
    es = torch.load(saved, weights_only=False)
    two_rows = {**es, "inputs": es["inputs"].repeat(2, 1), "targets": es["targets"].repeat(2, 1),
                "doc_ids": torch.tensor([[0, 1], [1, 1]])}
    torch.save(two_rows, saved)
    with pytest.raises(ValueError, match="split across rows"):
        run_main([root], tmp_path, "--ell0", ".1", "--bootstrap", "5")


def test_legacy_runs_without_packing_field_are_contiguous_but_never_mix_with_bestfit(tmp_path, capsys):
    run_main([write_grid(tmp_path / "legacy", packing=None)], tmp_path)
    summary = json.loads((tmp_path / "out" / "analysis.json").read_text())
    assert summary["eval_packing"] == "contiguous" and summary["runs_without_packing_field"] == 16
    assert "legacy" in capsys.readouterr().out
    first, second = a.BASES
    legacy = write_grid(tmp_path / "legacy_half", packing=None, bases=(second,))
    explicit = write_grid(tmp_path / "explicit_half", packing="contiguous", bases=(first,))
    run_main([explicit, legacy], tmp_path)           # same data; the field was only added later
    assert json.loads((tmp_path / "out" / "analysis.json").read_text())["runs_without_packing_field"] == 8
    bestfit = write_grid(tmp_path / "bestfit_half", packing="bestfit", bases=(first,))
    for pair in ([bestfit, legacy], [legacy, bestfit]):
        with pytest.raises(ValueError, match="without a packing field.*cannot be combined"):
            run_main(pair, tmp_path)
    with pytest.raises(ValueError, match="evaluation packing differs across runs"):
        run_main([bestfit, write_grid(tmp_path / "contiguous_half", packing="contiguous", bases=(second,))],
                 tmp_path)


def test_unknown_packing_and_different_set_identity_fail_clearly(tmp_path):
    first, second = a.BASES
    with pytest.raises(ValueError, match="unknown evaluation packing"):
        run_main([write_grid(tmp_path / "bad", packing="sideways")], tmp_path)
    left = write_grid(tmp_path / "left", bases=(first,))
    for change in ({"rows": 5}, {"documents": 9}, {"seq_len": 7}, {"scored_tokens": 8}, {"sha256": "other"}):
        right = write_grid(tmp_path / f"right_{next(iter(change))}", bases=(second,), **change)
        with pytest.raises(ValueError, match="evaluation set differs across runs"):
            run_main([left, right], tmp_path)


def test_real_bestfit_and_contiguous_sets_pass_the_per_document_checks(tmp_path):
    import random
    import pyarrow as pa
    import pyarrow.parquet as pq
    from nanochat.eval_set import build_eval_set, per_doc_sums

    class Tokenizer:
        def get_bos_token_id(self):
            return 0

        def encode(self, texts, prepend=None, num_threads=None):
            return [[prepend] + [ord(c) % 50 + 1 for c in text] for text in texts]

    rng = random.Random(1)
    docs = ["".join(rng.choice("abcdefghij") for _ in range(rng.randint(1, 30))) for _ in range(120)]
    path = tmp_path / "d.parquet"
    pq.write_table(pa.table({"text": docs}), path, row_group_size=40)
    for packing in ("bestfit", "contiguous"):
        es = build_eval_set("D0", path, Tokenizer(), 16, 16 * 20, packing=packing, buffer_size=20)
        saved = tmp_path / f"{packing}.pt"
        es.save(saved)
        counts = per_doc_sums(torch.zeros(es.inputs.shape), es.doc_ids, es.n_docs)[1].long().numpy()
        assert a.check_eval_set_file(str(saved), "D0", packing, es.source_sha256, es.n_docs, tuple(counts))
        assert es.doc_ids.max() == es.n_docs - 1 and counts.sum() == es.n_scored
        if packing == "bestfit":
            assert es.doc_rows is not None and es.doc_rows.numel() == es.n_docs
