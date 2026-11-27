# Fine-tuning replay bug: magnitude relative to seed noise

2026-11-26 · Diagnostic comparison (old measurements are invalid at affected lengths)

## Design of the comparison

The old replay loader stored references to a reused tensor buffer. In a 240-step or longer run, steps 49–60 replayed batch 60 repeatedly, and in the longest runs steps 193–240 replayed batch 240 repeatedly. The 60-step branches themselves were unaffected; their rerun is a useful negative control.

To isolate the bug from LR reselection, pair original and clean results at the **same** dataset, base seed 42, pretraining/fine-tuning optimizer, length, exact LR, and fine-tuning seed (1–3). There are 414 matched individual branch runs (234 StarCoderData, 180 WikiText), yielding 138 three-seed groups. Each group's correction is clean mean minus old mean. Recompute sample SDs directly from the three seeds in each version, and define pooled seed SD as `sqrt((SD_old² + SD_clean²)/2)`. This is a descriptive scale comparison, not a significance test or an independent-sample standard error.

## Full common LR grid

In nats/token; medians across common optimizer-pair/LR groups, not just selected LRs. The final columns count groups in which the magnitude of the mean shift exceeds one pooled seed SD.

| Dataset | Steps | Groups | Median abs. Δ learning | Median pooled SD (L) | Learning shift > SD | Median abs. Δ forgetting | Median pooled SD (F) | Forgetting shift > SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| StarCoderData | 60 (control) | 26 | 0.00034 | 0.00398 | 0/26 | 0.00037 | 0.00607 | 0/26 |
| StarCoderData | 240 | 26 | 0.00950 | 0.00154 | 25/26 | 0.01238 | 0.00402 | 24/26 |
| StarCoderData | 960 | 26 | 0.01128 | 0.00086 | 25/26 | 0.02351 | 0.00733 | 25/26 |
| WikiText-103 | 60 (control) | 20 | 0.00004 | 0.00162 | 0/20 | 0.00006 | 0.00110 | 1/20 |
| WikiText-103 | 240 | 20 | 0.00809 | 0.00057 | 20/20 | 0.00492 | 0.00083 | 17/20 |
| WikiText-103 | 720 | 20 | 0.01992 | 0.00057 | 20/20 | 0.01201 | 0.00117 | 20/20 |

**90/92 affected LR groups** have a learning shift larger than pooled within-cell seed SD, and **86/92** have a forgetting shift larger than seed SD. In the unaffected control, only 1/92 metric-by-group comparisons exceeds one SD (0 learning groups and 1 forgetting group).

## Endpoint impact at the original winning LR

Values are old → corrected means; parentheses show signed Δ divided by pooled within-cell seed SD. All rows hold LR fixed, including WikiText Muon → AdamW at 240 steps, where the corrected LR winner changed. Every one of the **48 paired seed comparisons** across these 16 affected groups shows more learning and less forgetting in the corrected run.

| Dataset | Pair | Steps | LR | Learning: old → clean | ΔL / SD | Forgetting: old → clean | ΔF / SD |
|---|---|---:|---:|---:|---:|---:|---:|
| SC | AdamW → AdamW | 240 | 1.31e-3 | 1.5109 → 1.5210 | +3.8× | 0.7183 → 0.6851 | −8.0× |
| SC | AdamW → Muon | 240 | 1.77e-3 | 1.5280 → 1.5426 | +13.7× | 0.8818 → 0.8414 | −2.8× |
| SC | Muon → AdamW | 240 | 6.55e-4 | 1.4361 → 1.4468 | +3.8× | 0.8604 → 0.8169 | −2.9× |
| SC | Muon → Muon | 240 | 8.87e-4 | 1.5017 → 1.5155 | +18.0× | 0.6680 → 0.6506 | −1.9× |
| SC | AdamW → AdamW | 960 | 6.55e-4 | 1.6895 → 1.6993 | +10.0× | 0.7188 → 0.6918 | −2.5× |
| SC | AdamW → Muon | 960 | 8.87e-4 | 1.7102 → 1.7240 | +13.7× | 0.8020 → 0.7722 | −3.5× |
| SC | Muon → AdamW | 960 | 6.55e-4 | 1.6313 → 1.6414 | +13.5× | 1.2060 → 1.1698 | −2.4× |
| SC | Muon → Muon | 960 | 8.87e-4 | 1.6843 → 1.6981 | +33.5× | 1.1446 → 1.1076 | −3.0× |
| WT | AdamW → AdamW | 240 | 6.55e-4 | 0.2568 → 0.2707 | +9.6× | 0.2503 → 0.2295 | −15.3× |
| WT | AdamW → Muon | 240 | 8.87e-4 | 0.2668 → 0.2808 | +11.0× | 0.2619 → 0.2563 | −2.5× |
| WT | Muon → AdamW | 240 | 1.64e-4 | 0.2190 → 0.2267 | +13.1× | 0.1621 → 0.1544 | −15.1× |
| WT | Muon → Muon | 240 | 4.44e-4 | 0.2594 → 0.2715 | +13.4× | 0.2114 → 0.2056 | −5.2× |
| WT | AdamW → AdamW | 720 | 6.55e-4 | 0.3666 → 0.3877 | +29.8× | 0.3632 → 0.3418 | −10.2× |
| WT | AdamW → Muon | 720 | 8.87e-4 | 0.3782 → 0.4051 | +28.3× | 0.3811 → 0.3664 | −4.8× |
| WT | Muon → AdamW | 720 | 3.28e-4 | 0.3177 → 0.3366 | +29.4× | 0.3298 → 0.3155 | −4.8× |
| WT | Muon → Muon | 720 | 4.44e-4 | 0.3644 → 0.3911 | +25.8× | 0.3148 → 0.3043 | −2.8× |

## Transient damage after each corrupted replay window

Original longest-horizon selected LR held fixed (SC 960, WT 720). Entries show clean minus old means; SD ratios in parentheses. The nonzero but mostly sub-SD differences at step 32 precede both affected windows; steps 64 and 256 follow the first and second windows respectively.

| Dataset | Optimizer pair | Δ learning at 32 | Δ learning at 64 | Δ learning at 256 | Δ forgetting at 64 | Δ forgetting at 256 |
|---|---|---:|---:|---:|---:|---:|
| SC | AdamW → AdamW | +0.0029 | +0.1792 (20.2×) | +0.1268 (18.8×) | −0.0684 (9.4×) | −0.1501 (8.8×) |
| SC | AdamW → Muon | −0.0014 | +0.1268 (18.3×) | +0.4108 (44.5×) | −0.0499 (3.0×) | −0.4137 (12.8×) |
| SC | Muon → AdamW | +0.0178 | +0.2013 (17.6×) | +0.1536 (21.2×) | −0.4002 (8.8×) | −0.3511 (5.6×) |
| SC | Muon → Muon | −0.0011 | +0.2313 (43.5×) | +0.5308 (20.1×) | −0.1897 (6.6×) | −1.1897 (28.9×) |
| WT | AdamW → AdamW | ≈0 | +0.2751 (23.3×) | +0.2170 (66.3×) | −0.1614 (22.9×) | −0.1811 (36.5×) |
| WT | AdamW → Muon | ≈0 | +0.1970 (36.5×) | +0.7558 (103.7×) | −0.1367 (20.4×) | −0.6302 (56.0×) |
| WT | Muon → AdamW | −0.0008 | +0.2601 (14.4×) | +0.2622 (81.9×) | −0.2197 (14.1×) | −0.2527 (24.7×) |
| WT | Muon → Muon | +0.0001 | +0.1560 (23.0×) | +1.0700 (25.2×) | −0.1013 (43.1×) | −0.9345 (53.6×) |

The bug's mid-run damage far exceeded its endpoint damage. For example, Muon → Muon on WikiText at step 256 goes from learning −0.7990 and forgetting 1.1664 in the old run to learning +0.2710 and forgetting 0.2318 in the corrected run. The affected model temporarily becomes worse even on the fine-tuning distribution. Recovery by the final checkpoint concealed most of this disturbance, though endpoint bias remains many seed SDs. The second replay window hit Muon fine-tuning particularly hard at these selected LRs.

## Conclusion

The replay bug was a large, systematic numerical bias well beyond fine-tuning-seed noise. It could have distorted mismatch and routing analyses even though LR winners changed in only two of 24 dataset/pair/length comparisons (one of those two changed selections was itself on the unaffected 60-step branch). The clean rerun was necessary, especially for any time-parametrized or matched-learning analysis. The current clean results supersede all affected old branches.

Sources: original best-fit archived seed-42 result records on hgo and Nazar's scratch (`finetune_pilot_v1_replaybug/` and `ndrugov/finetune_pilot/`), and clean hgo `finetune_pilot/`; checked using local gitignored result/config mirrors. The clean summaries and old WikiText summary are linked from [the earlier pilot report](finetune_pilot_report.md), and the old StarCoderData summary is preserved at Git revision `ed76421`. This report deliberately uses equal sample-SD conventions computed from individual runs rather than trusting archived summary conventions.
