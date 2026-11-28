"""Compact soft evidence: no GPU, cluster, training data or network required."""
import math
import numpy as np
import pytest
import torch

from nanochat.eval_set import EvalSet
from nanochat.soft_routing import METRICS, SoftRoutingCapture, scored_sample_positions, token_statistics


def evaluation(name="D0"):
    ids = torch.tensor([[0, 0, -1, 1], [1, 2, 2, -1], [2, -1, 1, 0]])
    tokens = torch.arange(ids.numel()).reshape(ids.shape)
    return EvalSet(name, tokens, tokens, ids, 4, "/mock", f"sha-{name}")


def capture(eval_set, logits, choices, sample=5, fmt="logits", base=None, batch=2):
    c = SoftRoutingCapture(eval_set, [0, 2], logits.shape[-1], sample, fmt, base)
    for start, x, _ in eval_set.batches(batch):
        rows = slice(start, start+x.shape[0])
        entries = [{"layer": l, "logits": logits[i, rows], "choices": choices[i, rows]} for i,l in enumerate((0,2))]
        c.add_batch(start, eval_set.doc_ids[rows], entries)
    return c.finish()


def independent_stats(logits, choices, old):
    p = torch.softmax(logits.double(), dim=-1)
    q = torch.softmax(old.double(), dim=-1)
    lp, lq = torch.log_softmax(logits.double(), -1), torch.log_softmax(old.double(), -1)
    mix = (p+q)/2
    js = .5 * (torch.special.xlogy(p, p/mix).sum(-1) + torch.special.xlogy(q, q/mix).sum(-1))
    top = p.topk(2, -1).values
    return torch.stack((-(p*lp).sum(-1), top[...,0]-top[...,1], p.gather(-1, choices[...,None]).squeeze(-1),
                        p.max(-1).values, js, .5*(p-q).abs().sum(-1),
                        torch.logsumexp(logits.double(), -1).square()), -1), p


def test_exact_scored_layer_document_sums_and_aligned_samples():
    ev = evaluation()
    generator = torch.Generator().manual_seed(71)
    old = torch.randn(2, 3, 4, 4, generator=generator)
    new = torch.randn(2, 3, 4, 4, generator=generator)*2.
    # Unscored positions have striking logits; their exclusion matters.
    new[:, ev.doc_ids<0] = 50
    # Actual selected experts needn't be top probability (bias selection).
    choices = torch.randint(0,4, (2,3,4), generator=generator)
    base = capture(ev, old, choices, batch=1)
    current = capture(ev, new, choices, base=base, batch=2)
    arrays = current.arrays("d0")
    baseline = base.arrays("d0", include_positions=True)
    assert current.log_probs is None and base.log_probs[0].shape == (ev.n_scored, 4)
    assert arrays["d0_soft_doc_sum"].dtype == np.float64
    assert arrays["d0_soft_sample_logits"].dtype == np.float32
    assert arrays["d0_soft_sample_probs"].shape == (2,0,4)
    positions = baseline["d0_soft_sample_positions"]
    assert len(positions)==5 and np.all(np.diff(positions)>0)
    assert np.all(ev.doc_ids.flatten().numpy()[positions]>=0)
    np.testing.assert_array_equal(baseline["d0_soft_sample_doc_ids"], ev.doc_ids.flatten()[positions].numpy())
    np.testing.assert_array_equal(arrays["d0_soft_sample_logits"], new.reshape(2,-1,4)[:,positions].numpy())
    np.testing.assert_array_equal(baseline["d0_soft_sample_logits"], old.reshape(2,-1,4)[:,positions].numpy())
    assert arrays["d0_soft_sample_positions_sha256"] == baseline["d0_soft_sample_positions_sha256"]
    assert "d0_soft_sample_positions" not in arrays
    values, probs = independent_stats(new, choices, old)
    counts = []
    for d in range(ev.n_docs):
        mask = ev.doc_ids == d
        counts.append(int(mask.sum()))
        expect = values[:,mask]
        np.testing.assert_allclose(arrays["d0_soft_doc_sum"][:,d], expect.sum(1).numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(arrays["d0_soft_doc_sumsq"][:,d], expect.square().sum(1).numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(arrays["d0_soft_doc_prob_sum"][:,d], probs[:,mask].sum(1).numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(arrays["d0_soft_doc_counts"], counts)
    mask = ev.doc_ids >= 0
    np.testing.assert_allclose(arrays["d0_soft_layer_sum"], values[:,mask].sum(1).numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(arrays["d0_soft_layer_sumsq"], values[:,mask].square().sum(1).numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(arrays["d0_soft_prob_sum"], probs[:,mask].sum(1).numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(arrays["d0_soft_hist_count"].sum(-1), np.full((2,len(METRICS)), ev.n_scored))
    assert current.across_layer_means("d0")["soft_d0_entropy_nats_mean"] == pytest.approx(float(values[:,mask,0].mean()), rel=1e-5)
    same = capture(ev, old, choices, base=base, batch=3)
    assert same.across_layer_means("d0")["soft_d0_js_base_current_nats_mean"] == 0.
    assert same.across_layer_means("d0")["soft_d0_tv_base_current_mean"] == 0.


def test_e1_structural_metrics_and_raw_z_loss():
    ev = evaluation("D1")
    old = torch.arange(24,dtype=torch.float32).reshape(2,3,4,1)
    new = old+1
    choices = torch.zeros((2,3,4), dtype=torch.long)
    base = capture(ev, old, choices)
    current = capture(ev, new, choices, base=base)
    arr = current.arrays("d1")
    assert base.log_probs is None and current.log_probs is None
    assert arr["d1_soft_sample_logits"].shape == (2,5,1)
    np.testing.assert_array_equal(arr["d1_soft_layer_count"][:,1], 0)
    np.testing.assert_array_equal(arr["d1_soft_layer_sum"][:,:6], np.tile(np.array([0.,0.,ev.n_scored,ev.n_scored,0.,0.]), (2,1)))
    np.testing.assert_array_equal(arr["d1_soft_layer_sum"][:,6], new.square()[:,ev.doc_ids>=0,0].sum(1).numpy())
    assert current.layer_means(0)["soft_top2_margin_mean"] is None
    assert current.layer_means(0)["soft_mean_expert_probs"] == [1.]
    np.testing.assert_array_equal(arr["d1_soft_hist_count"][:,1], 0)
    assert arr["d1_soft_hist_count"][:,6,-1].sum() > 0  # z-loss overflow
    probonly = capture(ev, old, choices, fmt="probabilities")
    assert probonly.positions.size==0 and probonly.sample_probs.shape==(2,0,1)


def test_zero_sample_and_sample_formats_and_alignment():
    ev = evaluation()
    logits = torch.randn(2,3,4,3, generator=torch.Generator().manual_seed(2))
    choices = logits.argmax(-1)
    rng = torch.random.get_rng_state().clone()
    base = capture(ev, logits, choices, sample=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert base.sample_logits.shape==(2,0,3) and base.sample_probs.shape==(2,0,3)
    assert scored_sample_positions(ev.doc_ids, 100).size == ev.n_scored
    both = capture(ev, logits, choices, sample=3, fmt="both")
    np.testing.assert_array_equal(both.sample_probs, torch.softmax(logits,-1).reshape(2,-1,3)[:,both.positions].numpy())
    altered = evaluation(); altered.doc_ids = altered.doc_ids.roll(1, dims=1)
    with pytest.raises(ValueError, match="alignment"):
        capture(altered, logits, choices, sample=0, base=base)
    with pytest.raises(ValueError, match="alignment"):
        capture(ev, logits, choices, sample=1, base=base)


def test_stable_extreme_router_distribution():
    old = torch.tensor([[1000., -1000.]])
    cur = torch.tensor([[-1000., 1000.]])
    log_old = torch.log_softmax(old,-1)
    _,_,v = token_statistics(cur, torch.tensor([0]), log_old)
    assert torch.isfinite(v).all()
    assert v[0,4].item() == pytest.approx(math.log(2),rel=1e-6)
    assert v[0,5].item() == 1.


def test_shared_reference_pin_does_not_accept_different_packing_or_sample(tmp_path):
    from scripts.finetune import pin_soft_router_schema
    from nanochat.soft_routing import schema
    pin_soft_router_schema(tmp_path, schema(8192), {"eval_hash":"a","base_hash":"b"})
    pin_soft_router_schema(tmp_path, schema(8192), {"eval_hash":"a","base_hash":"b"})
    with pytest.raises(RuntimeError,match="router sample/schema differs"):
        pin_soft_router_schema(tmp_path, schema(8192), {"eval_hash":"c","base_hash":"b"})
    with pytest.raises(RuntimeError,match="router sample/schema differs"):
        pin_soft_router_schema(tmp_path, schema(0), {"eval_hash":"a","base_hash":"b"})
