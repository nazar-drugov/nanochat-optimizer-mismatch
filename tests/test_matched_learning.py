"""Matched-learning geometry, support, provenance and paired uncertainty."""

import pytest

from scripts import matched_learning as m


def vertex(axis, learning, forgetting):
    return m.Vertex(axis, axis, axis, .001, learning, forgetting, f"run_{axis}", "test")


def test_exact_crossings_both_directions_and_no_sort_by_learning():
    points = (vertex(60, .2, .1), vertex(240, .8, .8), vertex(720, .4, .1), vertex(1920, .9, .9))
    # Descending segment 240→720: at L=.5, F=.275; lower than the first
    # ascending crossing. The final ascending segment crosses at F=.26 (winner).
    v = m.select_minimum(points, .5)
    assert v.kind == "interpolated_crossing" and v.learning == .5
    assert v.left.axis == 720 and v.right.axis == 1920
    assert v.weight == pytest.approx(.2) and v.forgetting == pytest.approx(.26)
    # Remove the last segment and the descending crossing is preferred.
    v = m.select_minimum(points[:3], .5)
    assert (v.left.axis, v.right.axis) == (240,720)
    assert v.weight == pytest.approx(.75) and v.forgetting == pytest.approx(.275)


def test_eligible_vertex_may_overshoot_target_and_preserves_actual_learning():
    points = (vertex(60, .2, .1), vertex(240, .7, .4), vertex(960, .8, -.1))
    v = m.select_minimum(points, .5)
    assert v.kind == "vertex" and v.learning == .8 and v.target == .5
    assert v.forgetting == -.1 and v.axis == 960 and v.weight == 0


def test_endpoint_support_no_artificial_origin_or_extrapolation():
    points = (vertex(60, .2, .1), vertex(240, .8, .2))
    assert m.select_minimum(points, .1) is None
    assert m.select_minimum(points, .9) is None
    # Within-run data really include step 0; then low thresholds are reachable.
    v = m.select_minimum((vertex(0, 0., 0.), *points), .1)
    assert (v.left.axis, v.right.axis, v.forgetting) == pytest.approx((0, 60, .05))


def test_plateau_and_deterministic_ties():
    points = (vertex(0, 0., 0.), vertex(60, .5, .2), vertex(240, .5, .1), vertex(720, .8, .1))
    v = m.select_minimum(points, .5)
    assert (v.kind, v.axis, v.learning, v.forgetting) == ("vertex", 240, .5, .1)
    with pytest.raises(ValueError, match="strictly ordered"):
        m.select_minimum((points[1],points[0]), .2)


def synthetic_inputs(cohort):
    endpoints, trajectories = [], []
    for pre, ft in m.PAIRS:
        for base in cohort.bases:
            for seed in m.FT_SEEDS:
                offset = (base-42)*.1 + (seed-2)*.01
                for i,h in enumerate(cohort.horizons):
                    row = {"dataset":cohort.dataset,"steps":h,"pretrain_optimizer":pre,"ft_optimizer":ft,
                           "base_seed":base,"ft_seed":seed,"lr":.001/(i+1),"source":"main",
                           "source_run":f"{pre}-{ft}-{base}-{seed}-{h}","learning":.5+i*.2,
                           "forgetting":.2+i*.1+(offset if ft!=pre else 0)}
                    endpoints.append(row)
                    trajectories.extend([dict(row,checkpoint_step=0,learning=0.,forgetting=0.),
                                         dict(row,checkpoint_step=h//2,learning=.2,forgetting=.08),
                                         dict(row,checkpoint_step=h)])
    return endpoints, trajectories


def test_cohort_provenance_and_pairing_before_hierarchical_sd():
    c = m.Cohort("starcoderdata","main",(42,43,44),(60,240))
    endpoints, trajectories = synthetic_inputs(c)
    comps = m.build_comparisons(endpoints,trajectories,(c,))
    endpoint = comps[0]
    assert m.support(endpoint)==pytest.approx((.5,.7))
    summary, pairs, selections = m.analyze(endpoint,.5,details=True)
    assert len(pairs)==18 and len(selections)==36
    # At .5 the first endpoint is optimal; mismatch-minus-match is the
    # shared-base offset, with mean .1 and base-mean sample SD .1.
    for row in summary:
        assert row["penalty_mean"]==pytest.approx(.1)
        assert row["penalty_sd"]==pytest.approx(.1)
        assert row["base42_penalty_ft_sd"]==pytest.approx(.01)
        assert row["n_pairs"]==9
    sel=selections[0]
    assert sel["left_horizon"]==sel["right_horizon"]==60
    assert sel["left_lr"]==.001 and sel["left_source_run"]=="adamw-adamw-42-1-60"
    # Common-support rejection leaves no partial seed contrast rows.
    summaries,pairs,selections=m.analyze(endpoint,.4,details=True)
    assert all(r["status"]=="unsupported_target" for r in summaries) and not pairs and not selections
    # A fixed-horizon trajectory includes step zero and supports .4.
    assert comps[1].estimand=="within-run" and m.support(comps[1])==pytest.approx((0.,.5))
    assert m.analyze(comps[1],.4)[0][0]["status"]=="supported"


def test_conditional_seed42_sd_and_source_mismatch_guard():
    c=m.Cohort("wikitext103","pilot_seed42",(42,),(60,240))
    endpoints,trajectories=synthetic_inputs(c)
    comps=m.build_comparisons(endpoints,trajectories,(c,))
    summary,_,_=m.analyze(comps[0],.5,details=True)
    assert summary[0]["penalty_sd"]==pytest.approx(.01)
    assert summary[0]["n_bases"]==1 and summary[0]["n_pairs"]==3
    corrupt=[dict(trajectories[0],source_run="wrong"),*trajectories[1:]]
    with pytest.raises(ValueError,match="provenance differs"):
        m.build_comparisons(endpoints,corrupt,(c,))


def test_sensitivity_overlap_only_shared_support():
    c=m.Cohort("starcoderdata","main",(42,43,44),(60,240))
    endpoints,trajectories=synthetic_inputs(c)
    comps=m.build_comparisons(endpoints,trajectories,(c,))
    _,_,_,sensitivity,overlaps=m.analyze_all(comps,{"endpoint":{"starcoderdata":.6},"within-run":{"starcoderdata":.3}},size=5)
    assert sensitivity
    assert all(r["overlap_min"]<=r["target_learning"]<=r["overlap_max"] for r in overlaps)
    # The 60-step trajectory only overlaps at exactly its maximum/endpoint minimum.
    assert {r["target_learning"] for r in overlaps if r["horizon"]==60}=={.5}
    assert {r["target_learning"] for r in overlaps if r["horizon"]==240}.issuperset({.5,.6,.7})
