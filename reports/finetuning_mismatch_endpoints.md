# E=8 optimizer mismatch: endpoint learning–forgetting curves

**2026-11-27 · StarCoderData long main grid complete; WikiText long pilot remains seed-42-only.**

## Question and scope

The first question in `docs/report.tex` is whether switching optimizers between pretraining and fine-tuning increases forgetting at the same learning on an MoE. This report describes the endpoint evidence to date, before the prespecified matched-learning analysis and correlated/causal routing analysis. Positive learning is the drop in held-out fine-tuning loss (D1); positive forgetting is the rise in held-out pretraining loss (D0), both in nats per token.

The main E=8 grid has three pretrained seeds (42, 43, 44) for each optimizer, each fine-tuned with seeds 1–3. Each optimizer pair/dataset/length uses the LR minimizing mean final D1 on the seed-42 pilot, then reused on the other pretrained bases. The extended pilots initially used pretrained seed 42 only, again three fine-tuning seeds. StarCoderData's 960-step LR is also optimal at 1,920 and 3,200 steps and has now been replicated on the four seed-43/44 bases at those horizons. WikiText long-horizon optimal LRs instead fall sharply (see below), and its long extension remains seed-42-only. Results are from the clean, post-replay-fix pilots and the validated standalone WikiText main runs; contaminated results are excluded.

## Endpoint learning–forgetting curves

![Learning–forgetting endpoint curves with seed-level standard deviation error bars](../results/e8/figures/finetune_endpoints.png)

![Learning and forgetting separately against training length, with seed-level standard deviation error bars](../results/e8/figures/finetune_length_trends.png)

The second figure separates the two coordinates against log training length, making the nonmonotonic WikiText endpoints visible when the trade-off points cluster.

- Solid curves: means of the three pretraining-seed means; each of those means uses three fine-tuning seeds (9 runs per point). Horizontal/vertical bars are ±1 sample SD **across the three base means**; they are not confidence intervals.
- Open markers and dashed curves: long pilot means for pretraining seed 42 only, ±1 sample SD across fine-tuning seeds 1–3. The dashed line starts at the seed-42 mean of the last fully replicated length, rather than the three-base mean. These intervals describe different sources of variability and are not comparable as estimates of population uncertainty.
- Labels are training steps. These connect endpoints of separately decayed schedules at separately selected LRs, not checkpoints along a single run. They illustrate length dependence but are not the formal matched-learning curves of the paper.

## What the data suggest

- **StarCoderData, AdamW-pretrained:** at 60 steps AdamW → AdamW and AdamW → Muon achieve nearly identical mean learning (1.2835 vs 1.2863), but switching to Muon forgets much more (0.8093 vs 0.3121). At 240, 960, 1,920 and 3,200, the switching arm still forgets more and learns slightly more, so exact matched-learning inference requires the saved checkpoint curves. At 3,200 the fully replicated means are 1.8379 / 0.9841 (matched) versus 1.8643 / 1.1058 (mismatched).
- **StarCoderData, Muon-pretrained:** at 240 and 960 steps Muon → AdamW has both lower mean learning and higher mean forgetting than Muon → Muon. At 960 the endpoints are 1.6203/1.1662 versus 1.6795/1.0961. At 3,200 (all three base seeds), the two mean forgetting endpoints are close (1.4379 vs 1.4283), although the switching arm learns less (1.7516 vs 1.7979). This is suggestive, not evidence that the mismatch penalty disappears at a fixed learning level.
- **WikiText, AdamW-pretrained:** the 60-step points again have close learning (0.1622 vs 0.1639) and greater forgetting under a switch to Muon (0.1318 vs 0.0871). At 240 and 720 the mismatched arm has more learning and more forgetting, requiring a matched-learning comparison.
- **WikiText, Muon-pretrained:** the 240- and 720-step mismatched endpoints have lower learning and higher forgetting than the matched endpoints. At 720 steps these are 0.3365/0.3155 versus 0.3915/0.3053. The 60-step endpoint alone does not settle the mismatch effect.
- **The most convincing provisional pattern** is direction- and horizon-dependent extra forgetting: the evidence is consistent with a mismatch cost in both corpora, especially early AdamW → Muon and the intermediate/long Muon → AdamW endpoints. No statistical significance or causal router attribution is claimed here. The protocol calls for choosing ℓ0 solely from common learning support before computing mismatch contrasts, then reporting the entire supported range as a robustness check.

### What the long WikiText extension establishes

The original 720-step peak LR does not transfer to multi-epoch WikiText. All four 3,200-step runs at the original 720-step winners dramatically overfit or deteriorate in held-out learning. Selecting a separate LR at each longer horizon restores much of the learning and drastically reduces forgetting. The 1,920-step optimum is ¼× the 720-step LR for three pairs and ½× for Muon → AdamW; the 3,200-step optimum is ⅛× for three pairs and 1/16× for AdamW → Muon. In contrast, StarCoderData's 960-step winners stay optimal out to 3,200.

At 1,920, both pretraining optimizers still show the provisional pattern of additional forgetting under mismatch: for the AdamW base, learning is almost identical (0.4006 vs 0.4012), with 0.3018 vs 0.3441 forgetting. For the Muon base the mismatch endpoint learns less (0.3763 vs 0.3843) and forgets more (0.3621 vs 0.2895). The 3,200-step mismatched endpoints instead have both slightly lower learning and lower forgetting than their matched counterparts. This crossing/turnback is **not** a conclusion that mismatch vanishes at a fixed learning target. Every arm has lower final held-out learning at 3,200 than at 1,920 after LR tuning, and the final selections differ; the saved trajectory and matched-learning analysis are essential here. The longer WikiText runs see ~4.28 effective epochs versus ~0.97 for StarCoderData; repetition and overfitting qualify cross-corpus interpretation. The longer WikiText lengths use a single pretrained initialization and do not establish between-base robustness; the StarCoderData long main grid is now fully replicated.

## Raw endpoint data

The two raw NLL columns are the actual held-out losses on D0 and D1 (lower is better), not next-token accuracy. The per-seed raw tables and endpoint CSV now include both values alongside the loss changes. See [all eight raw validation trajectory figures and step-0 losses](finetune_validation_trajectories.md).

Tables show the observed means and the same standard deviations as the graph, not matching or extrapolating learning levels. The expandable tables provide every individual fine-tuning seed's observed learning/forgetting and absolute D0/D1 NLL; the [full precision endpoint CSV](../results/e8/finetune_endpoint_runs.csv) includes each source run's path. All values are nats/token except LR and training steps.

### StarCoderData (Python)

| Pretrain → fine-tune | Steps | Pretrain seeds | LR | Learning (mean ± SD) | Forgetting (mean ± SD) | D0 raw NLL (mean ± SD) | D1 raw NLL (mean ± SD) |
|---|---:|:---:|---:|---:|---:|---:|---:|
| AdamW → AdamW | 60 | 42, 43, 44 | 0.00131 | 1.2835 ± 0.0278 | 0.3121 ± 0.0115 | 3.2555 ± 0.0059 | 1.5834 ± 0.0108 |
| AdamW → AdamW | 240 | 42, 43, 44 | 0.00131 | 1.5374 ± 0.0205 | 0.6822 ± 0.0123 | 3.6256 ± 0.0049 | 1.3295 ± 0.0037 |
| AdamW → AdamW | 960 | 42, 43, 44 | 0.000655 | 1.7131 ± 0.0188 | 0.7035 ± 0.0176 | 3.6469 ± 0.0159 | 1.1538 ± 0.0048 |
| AdamW → AdamW | 1,920 | 42, 43, 44 | 0.000655 | 1.7881 ± 0.0195 | 0.8502 ± 0.0257 | 3.7936 ± 0.0223 | 1.0789 ± 0.0045 |
| AdamW → AdamW | 3,200 | 42, 43, 44 | 0.000655 | 1.8379 ± 0.0205 | 0.9841 ± 0.0235 | 3.9274 ± 0.0213 | 1.0290 ± 0.0041 |
| AdamW → Muon | 60 | 42, 43, 44 | 0.00355 | 1.2863 ± 0.0221 | 0.8093 ± 0.0307 | 3.7527 ± 0.0222 | 1.5807 ± 0.0019 |
| AdamW → Muon | 240 | 42, 43, 44 | 0.00177 | 1.5568 ± 0.0197 | 0.8539 ± 0.0259 | 3.7973 ± 0.0220 | 1.3101 ± 0.0039 |
| AdamW → Muon | 960 | 42, 43, 44 | 0.000887 | 1.7385 ± 0.0193 | 0.7845 ± 0.0219 | 3.7279 ± 0.0188 | 1.1284 ± 0.0043 |
| AdamW → Muon | 1,920 | 42, 43, 44 | 0.000887 | 1.8146 ± 0.0203 | 0.9497 ± 0.0149 | 3.8931 ± 0.0123 | 1.0523 ± 0.0034 |
| AdamW → Muon | 3,200 | 42, 43, 44 | 0.000887 | 1.8643 ± 0.0203 | 1.1058 ± 0.0201 | 4.0492 ± 0.0203 | 1.0026 ± 0.0033 |
| Muon → AdamW | 60 | 42, 43, 44 | 0.000328 | 1.1401 ± 0.0364 | 0.3118 ± 0.0100 | 3.2132 ± 0.0105 | 1.6502 ± 0.0048 |
| Muon → AdamW | 240 | 42, 43, 44 | 0.000655 | 1.4247 ± 0.0400 | 0.8195 ± 0.0112 | 3.7208 ± 0.0125 | 1.3656 ± 0.0068 |
| Muon → AdamW | 960 | 42, 43, 44 | 0.000655 | 1.6203 ± 0.0385 | 1.1662 ± 0.0140 | 4.0675 ± 0.0132 | 1.1700 ± 0.0055 |
| Muon → AdamW | 1,920 | 42, 43, 44 | 0.000655 | 1.6999 ± 0.0385 | 1.3140 ± 0.0334 | 4.2154 ± 0.0339 | 1.0904 ± 0.0045 |
| Muon → AdamW | 3,200 | 42, 43, 44 | 0.000655 | 1.7516 ± 0.0394 | 1.4379 ± 0.0137 | 4.3392 ± 0.0154 | 1.0387 ± 0.0050 |
| Muon → Muon | 60 | 42, 43, 44 | 0.00177 | 1.2428 ± 0.0360 | 0.6530 ± 0.0193 | 3.5543 ± 0.0170 | 1.5475 ± 0.0020 |
| Muon → Muon | 240 | 42, 43, 44 | 0.000887 | 1.4968 ± 0.0354 | 0.6571 ± 0.0146 | 3.5585 ± 0.0125 | 1.2935 ± 0.0032 |
| Muon → Muon | 960 | 42, 43, 44 | 0.000887 | 1.6795 ± 0.0351 | 1.0961 ± 0.0294 | 3.9975 ± 0.0287 | 1.1108 ± 0.0032 |
| Muon → Muon | 1,920 | 42, 43, 44 | 0.000887 | 1.7514 ± 0.0355 | 1.2891 ± 0.0277 | 4.1904 ± 0.0271 | 1.0389 ± 0.0026 |
| Muon → Muon | 3,200 | 42, 43, 44 | 0.000887 | 1.7979 ± 0.0354 | 1.4283 ± 0.0327 | 4.3296 ± 0.0332 | 0.9924 ± 0.0020 |

<details>
<summary>Raw learning / forgetting and D0 / D1 NLL for each seed</summary>

Each cell lists **learning / forgetting; D0 raw NLL / D1 raw NLL** for fine-tuning seeds 1, 2, 3, respectively (all in nats/token). Values rounded to four decimals; the CSV preserves full precision.

| Pair | Steps | Base seed 42 | Base seed 43 | Base seed 44 |
|---|---:|---|---|---|
| AdamW → AdamW | 60 | 1.2628 / 0.3014; 3.2400 / 1.5868<br>1.2389 / 0.3276; 3.2662 / 1.6107<br>1.2639 / 0.3414; 3.2801 / 1.5856 | 1.2822 / 0.2977; 3.2360 / 1.5751<br>1.2869 / 0.3286; 3.2669 / 1.5704<br>1.2847 / 0.3113; 3.2496 / 1.5726 | 1.3109 / 0.2922; 3.2454 / 1.5829<br>1.3083 / 0.3039; 3.2571 / 1.5855<br>1.3132 / 0.3050; 3.2582 / 1.5806 |
| AdamW → AdamW | 240 | 1.5219 / 0.6828; 3.6214 / 1.3276<br>1.5173 / 0.6855; 3.6241 / 1.3323<br>1.5237 / 0.6871; 3.6257 / 1.3258 | 1.5303 / 0.6977; 3.6361 / 1.3271<br>1.5306 / 0.6822; 3.6205 / 1.3268<br>1.5322 / 0.6984; 3.6367 / 1.3252 | 1.5580 / 0.6779; 3.6311 / 1.3358<br>1.5623 / 0.6540; 3.6072 / 1.3315<br>1.5607 / 0.6741; 3.6273 / 1.3331 |
| AdamW → AdamW | 960 | 1.6999 / 0.6850; 3.6236 / 1.1497<br>1.6985 / 0.7072; 3.6458 / 1.1510<br>1.6995 / 0.6832; 3.6219 / 1.1501 | 1.7058 / 0.7224; 3.6607 / 1.1516<br>1.7053 / 0.7276; 3.6660 / 1.1521<br>1.7055 / 0.7213; 3.6596 / 1.1519 | 1.7353 / 0.6931; 3.6463 / 1.1586<br>1.7338 / 0.6972; 3.6504 / 1.1600<br>1.7347 / 0.6947; 3.6479 / 1.1592 |
| AdamW → AdamW | 1,920 | 1.7752 / 0.8391; 3.7777 / 1.0743<br>1.7738 / 0.8391; 3.7777 / 1.0757<br>1.7760 / 0.8300; 3.7687 / 1.0735 | 1.7803 / 0.8805; 3.8188 / 1.0771<br>1.7780 / 0.8869; 3.8252 / 1.0794<br>1.7779 / 0.8722; 3.8106 / 1.0795 | 1.8110 / 0.8296; 3.7829 / 1.0828<br>1.8096 / 0.8391; 3.7923 / 1.0842<br>1.8107 / 0.8356; 3.7888 / 1.0831 |
| AdamW → AdamW | 3,200 | 1.8250 / 0.9720; 3.9106 / 1.0245<br>1.8253 / 0.9652; 3.9038 / 1.0242<br>1.8251 / 0.9672; 3.9058 / 1.0244 | 1.8276 / 1.0145; 3.9529 / 1.0298<br>1.8270 / 1.0039; 3.9422 / 1.0303<br>1.8264 / 1.0146; 3.9529 / 1.0310 | 1.8614 / 0.9714; 3.9246 / 1.0325<br>1.8618 / 0.9722; 3.9254 / 1.0320<br>1.8614 / 0.9755; 3.9287 / 1.0325 |
| AdamW → Muon | 60 | 1.2677 / 0.8171; 3.7557 / 1.5818<br>1.2708 / 0.8384; 3.7770 / 1.5788<br>1.2743 / 0.8250; 3.7636 / 1.5753 | 1.2719 / 0.7815; 3.7198 / 1.5855<br>1.2752 / 0.8535; 3.7918 / 1.5822<br>1.2816 / 0.8466; 3.7849 / 1.5757 | 1.3099 / 0.7433; 3.6965 / 1.5839<br>1.3102 / 0.7686; 3.7218 / 1.5836<br>1.3146 / 0.8096; 3.7628 / 1.5792 |
| AdamW → Muon | 240 | 1.5417 / 0.8515; 3.7902 / 1.3079<br>1.5421 / 0.8323; 3.7710 / 1.3074<br>1.5440 / 0.8403; 3.7789 / 1.3056 | 1.5462 / 0.8988; 3.8372 / 1.3112<br>1.5502 / 0.8629; 3.8013 / 1.3072<br>1.5492 / 0.8894; 3.8277 / 1.3081 | 1.5798 / 0.8585; 3.8117 / 1.3141<br>1.5799 / 0.8167; 3.7699 / 1.3139<br>1.5785 / 0.8349; 3.7881 / 1.3154 |
| AdamW → Muon | 960 | 1.7245 / 0.7692; 3.7078 / 1.1250<br>1.7228 / 0.7781; 3.7168 / 1.1267<br>1.7247 / 0.7694; 3.7081 / 1.1248 | 1.7314 / 0.8088; 3.7471 / 1.1259<br>1.7305 / 0.8124; 3.7507 / 1.1269<br>1.7312 / 0.8081; 3.7464 / 1.1262 | 1.7612 / 0.7645; 3.7177 / 1.1326<br>1.7592 / 0.7793; 3.7325 / 1.1346<br>1.7609 / 0.7706; 3.7238 / 1.1329 |
| AdamW → Muon | 1,920 | 1.8005 / 0.9415; 3.8802 / 1.0491<br>1.7992 / 0.9415; 3.8802 / 1.0504<br>1.7997 / 0.9425; 3.8812 / 1.0499 | 1.8075 / 0.9723; 3.9106 / 1.0498<br>1.8053 / 0.9680; 3.9063 / 1.0521<br>1.8064 / 0.9602; 3.8985 / 1.0510 | 1.8377 / 0.9407; 3.8939 / 1.0561<br>1.8374 / 0.9372; 3.8904 / 1.0565<br>1.8380 / 0.9431; 3.8963 / 1.0558 |
| AdamW → Muon | 3,200 | 1.8499 / 1.0851; 4.0237 / 0.9997<br>1.8500 / 1.0876; 4.0263 / 0.9995<br>1.8497 / 1.0912; 4.0299 / 0.9999 | 1.8559 / 1.1334; 4.0717 / 1.0015<br>1.8555 / 1.1235; 4.0619 / 1.0019<br>1.8550 / 1.1258; 4.0641 / 1.0024 | 1.8875 / 1.1059; 4.0592 / 1.0064<br>1.8880 / 1.1029; 4.0561 / 1.0058<br>1.8872 / 1.0970; 4.0502 / 1.0066 |
| Muon → AdamW | 60 | 1.1519 / 0.2938; 3.1975 / 1.6535<br>1.1481 / 0.3355; 3.2393 / 1.6573<br>1.1525 / 0.3295; 3.2332 / 1.6529 | 1.0954 / 0.2907; 3.1924 / 1.6553<br>1.0976 / 0.3001; 3.2017 / 1.6530<br>1.1055 / 0.3111; 3.2127 / 1.6452 | 1.1715 / 0.3150; 3.2136 / 1.6433<br>1.1733 / 0.3131; 3.2117 / 1.6415<br>1.1646 / 0.3179; 3.2165 / 1.6502 |
| Muon → AdamW | 240 | 1.4448 / 0.8233; 3.7271 / 1.3606<br>1.4497 / 0.8069; 3.7106 / 1.3557<br>1.4460 / 0.8205; 3.7243 / 1.3594 | 1.3787 / 0.8228; 3.7244 / 1.3720<br>1.3821 / 0.8180; 3.7197 / 1.3686<br>1.3748 / 0.8544; 3.7560 / 1.3759 | 1.4471 / 0.8162; 3.7148 / 1.3677<br>1.4520 / 0.7996; 3.6982 / 1.3628<br>1.4473 / 0.8134; 3.7120 / 1.3675 |
| Muon → AdamW | 960 | 1.6414 / 1.1606; 4.0643 / 1.1640<br>1.6407 / 1.1827; 4.0864 / 1.1647<br>1.6420 / 1.1662; 4.0699 / 1.1634 | 1.5756 / 1.1480; 4.0497 / 1.1751<br>1.5759 / 1.1759; 4.0775 / 1.1748<br>1.5761 / 1.1284; 4.0301 / 1.1746 | 1.6439 / 1.1680; 4.0666 / 1.1709<br>1.6434 / 1.1947; 4.0933 / 1.1714<br>1.6433 / 1.1713; 4.0700 / 1.1715 |
| Muon → AdamW | 1,920 | 1.7194 / 1.3435; 4.2472 / 1.0860<br>1.7186 / 1.3379; 4.2416 / 1.0867<br>1.7198 / 1.3410; 4.2447 / 1.0856 | 1.6565 / 1.2813; 4.1830 / 1.0942<br>1.6545 / 1.2824; 4.1841 / 1.0962<br>1.6555 / 1.2658; 4.1675 / 1.0952 | 1.7255 / 1.3247; 4.2233 / 1.0893<br>1.7241 / 1.3215; 4.2201 / 1.0907<br>1.7253 / 1.3280; 4.2267 / 1.0895 |
| Muon → AdamW | 3,200 | 1.7693 / 1.4519; 4.3556 / 1.0360<br>1.7707 / 1.4470; 4.3507 / 1.0347<br>1.7708 / 1.4603; 4.3640 / 1.0346 | 1.7073 / 1.4122; 4.3139 / 1.0434<br>1.7039 / 1.4509; 4.3525 / 1.0468<br>1.7076 / 1.4169; 4.3185 / 1.0431 | 1.7783 / 1.4314; 4.3300 / 1.0366<br>1.7784 / 1.4308; 4.3294 / 1.0364<br>1.7779 / 1.4395; 4.3381 / 1.0369 |
| Muon → Muon | 60 | 1.2601 / 0.6204; 3.5241 / 1.5453<br>1.2573 / 0.6543; 3.5580 / 1.5481<br>1.2624 / 0.6486; 3.5523 / 1.5430 | 1.1978 / 0.6427; 3.5444 / 1.5529<br>1.2028 / 0.6182; 3.5199 / 1.5479<br>1.2035 / 0.6670; 3.5686 / 1.5472 | 1.2657 / 0.6886; 3.5873 / 1.5491<br>1.2602 / 0.6814; 3.5801 / 1.5546<br>1.2750 / 0.6558; 3.5545 / 1.5398 |
| Muon → Muon | 240 | 1.5154 / 0.6555; 3.5592 / 1.2900<br>1.5148 / 0.6454; 3.5491 / 1.2905<br>1.5162 / 0.6511; 3.5548 / 1.2892 | 1.4555 / 0.6569; 3.5586 / 1.2952<br>1.4569 / 0.6390; 3.5407 / 1.2938<br>1.4553 / 0.6446; 3.5463 / 1.2954 | 1.5185 / 0.6786; 3.5772 / 1.2963<br>1.5202 / 0.6605; 3.5592 / 1.2946<br>1.5181 / 0.6825; 3.5812 / 1.2967 |
| Muon → Muon | 960 | 1.6985 / 1.0963; 4.0000 / 1.1069<br>1.6977 / 1.1220; 4.0257 / 1.1077<br>1.6982 / 1.1045; 4.0083 / 1.1072 | 1.6400 / 1.0562; 3.9579 / 1.1107<br>1.6379 / 1.0762; 3.9778 / 1.1128<br>1.6390 / 1.0559; 3.9575 / 1.1117 | 1.7014 / 1.1094; 4.0080 / 1.1134<br>1.7011 / 1.1316; 4.0302 / 1.1138<br>1.7016 / 1.1130; 4.0116 / 1.1132 |
| Muon → Muon | 1,920 | 1.7699 / 1.2964; 4.2001 / 1.0355<br>1.7685 / 1.3015; 4.2052 / 1.0369<br>1.7700 / 1.2999; 4.2036 / 1.0354 | 1.7115 / 1.2581; 4.1597 / 1.0392<br>1.7093 / 1.2536; 4.1553 / 1.0414<br>1.7105 / 1.2613; 4.1629 / 1.0402 | 1.7749 / 1.3091; 4.2077 / 1.0399<br>1.7732 / 1.3110; 4.2096 / 1.0416<br>1.7746 / 1.3105; 4.2092 / 1.0402 |
| Muon → Muon | 3,200 | 1.8147 / 1.4552; 4.3589 / 0.9907<br>1.8153 / 1.4550; 4.3588 / 0.9901<br>1.8159 / 1.4542; 4.3579 / 0.9895 | 1.7575 / 1.3954; 4.2971 / 0.9932<br>1.7572 / 1.3907; 4.2924 / 0.9935<br>1.7566 / 1.3892; 4.2909 / 0.9941 | 1.8208 / 1.4347; 4.3333 / 0.9940<br>1.8217 / 1.4357; 4.3343 / 0.9931<br>1.8212 / 1.4447; 4.3433 / 0.9936 |

</details>

### WikiText-103

| Pretrain → fine-tune | Steps | Pretrain seeds | LR | Learning (mean ± SD) | Forgetting (mean ± SD) | D0 raw NLL (mean ± SD) | D1 raw NLL (mean ± SD) |
|---|---:|:---:|---:|---:|---:|---:|---:|
| AdamW → AdamW | 60 | 42, 43, 44 | 0.000328 | 0.1622 ± 0.0006 | 0.0871 ± 0.0030 | 3.0305 ± 0.0109 | 3.0491 ± 0.0090 |
| AdamW → AdamW | 240 | 42, 43, 44 | 0.000655 | 0.2715 ± 0.0007 | 0.2343 ± 0.0044 | 3.1777 ± 0.0120 | 2.9399 ± 0.0086 |
| AdamW → AdamW | 720 | 42, 43, 44 | 0.000655 | 0.3889 ± 0.0013 | 0.3412 ± 0.0133 | 3.2846 ± 0.0069 | 2.8225 ± 0.0077 |
| AdamW → AdamW | 1,920 | 42 | 0.000164 | 0.4006 ± 0.0021 | 0.3018 ± 0.0011 | 3.2405 ± 0.0011 | 2.8051 ± 0.0021 |
| AdamW → AdamW | 3,200 | 42 | 8.19e-05 | 0.3901 ± 0.0017 | 0.3207 ± 0.0007 | 3.2594 ± 0.0007 | 2.8155 ± 0.0017 |
| AdamW → Muon | 60 | 42, 43, 44 | 0.000887 | 0.1639 ± 0.0008 | 0.1318 ± 0.0030 | 3.0752 ± 0.0109 | 3.0475 ± 0.0088 |
| AdamW → Muon | 240 | 42, 43, 44 | 0.000887 | 0.2816 ± 0.0007 | 0.2614 ± 0.0054 | 3.2048 ± 0.0135 | 2.9298 ± 0.0087 |
| AdamW → Muon | 720 | 42, 43, 44 | 0.000887 | 0.4062 ± 0.0011 | 0.3738 ± 0.0127 | 3.3172 ± 0.0111 | 2.8052 ± 0.0079 |
| AdamW → Muon | 1,920 | 42 | 0.000222 | 0.4012 ± 0.0032 | 0.3441 ± 0.0028 | 3.2827 ± 0.0028 | 2.8044 ± 0.0032 |
| AdamW → Muon | 3,200 | 42 | 5.55e-05 | 0.3833 ± 0.0008 | 0.2789 ± 0.0003 | 3.2176 ± 0.0003 | 2.8224 ± 0.0008 |
| Muon → AdamW | 60 | 42, 43, 44 | 0.000164 | 0.1350 ± 0.0008 | 0.0994 ± 0.0011 | 3.0007 ± 0.0031 | 3.0255 ± 0.0009 |
| Muon → AdamW | 240 | 42, 43, 44 | 0.000328 | 0.2279 ± 0.0006 | 0.2237 ± 0.0017 | 3.1250 ± 0.0029 | 2.9326 ± 0.0010 |
| Muon → AdamW | 720 | 42, 43, 44 | 0.000328 | 0.3365 ± 0.0011 | 0.3155 ± 0.0035 | 3.2168 ± 0.0054 | 2.8240 ± 0.0020 |
| Muon → AdamW | 1,920 | 42 | 0.000164 | 0.3763 ± 0.0027 | 0.3621 ± 0.0041 | 3.2658 ± 0.0041 | 2.7848 ± 0.0027 |
| Muon → AdamW | 3,200 | 42 | 4.1e-05 | 0.3672 ± 0.0014 | 0.2940 ± 0.0005 | 3.1977 ± 0.0005 | 2.7939 ± 0.0014 |
| Muon → Muon | 60 | 42, 43, 44 | 0.000444 | 0.1558 ± 0.0020 | 0.1134 ± 0.0012 | 3.0148 ± 0.0018 | 3.0047 ± 0.0013 |
| Muon → Muon | 240 | 42, 43, 44 | 0.000444 | 0.2720 ± 0.0007 | 0.2086 ± 0.0026 | 3.1099 ± 0.0013 | 2.8885 ± 0.0009 |
| Muon → Muon | 720 | 42, 43, 44 | 0.000444 | 0.3915 ± 0.0005 | 0.3053 ± 0.0008 | 3.2066 ± 0.0018 | 2.7690 ± 0.0014 |
| Muon → Muon | 1,920 | 42 | 0.000111 | 0.3843 ± 0.0014 | 0.2895 ± 0.0016 | 3.1932 ± 0.0016 | 2.7768 ± 0.0014 |
| Muon → Muon | 3,200 | 42 | 5.55e-05 | 0.3732 ± 0.0019 | 0.3151 ± 0.0004 | 3.2189 ± 0.0004 | 2.7880 ± 0.0019 |

<details>
<summary>Raw learning / forgetting and D0 / D1 NLL for each seed</summary>

Each cell lists **learning / forgetting; D0 raw NLL / D1 raw NLL** for fine-tuning seeds 1, 2, 3, respectively (all in nats/token). Values rounded to four decimals; the CSV preserves full precision.

| Pair | Steps | Base seed 42 | Base seed 43 | Base seed 44 |
|---|---:|---|---|---|
| AdamW → AdamW | 60 | 0.1627 / 0.0822; 3.0209 / 3.0430<br>0.1598 / 0.0839; 3.0225 / 3.0458<br>0.1625 / 0.0853; 3.0239 / 3.0431 | 0.1641 / 0.0862; 3.0245 / 3.0428<br>0.1608 / 0.0880; 3.0263 / 3.0461<br>0.1638 / 0.0894; 3.0278 / 3.0431 | 0.1638 / 0.0881; 3.0413 / 3.0578<br>0.1594 / 0.0888; 3.0420 / 3.0622<br>0.1632 / 0.0923; 3.0455 / 3.0584 |
| AdamW → AdamW | 240 | 0.2709 / 0.2283; 3.1669 / 2.9348<br>0.2719 / 0.2308; 3.1694 / 2.9337<br>0.2693 / 0.2294; 3.1680 / 2.9363 | 0.2730 / 0.2355; 3.1738 / 2.9338<br>0.2720 / 0.2367; 3.1750 / 2.9349<br>0.2710 / 0.2339; 3.1723 / 2.9359 | 0.2721 / 0.2354; 3.1886 / 2.9496<br>0.2721 / 0.2400; 3.1932 / 2.9495<br>0.2712 / 0.2385; 3.1917 / 2.9504 |
| AdamW → AdamW | 720 | 0.3882 / 0.3399; 3.2786 / 2.8174<br>0.3872 / 0.3420; 3.2807 / 2.8184<br>0.3877 / 0.3434; 3.2821 / 2.8180 | 0.3886 / 0.3530; 3.2913 / 2.8182<br>0.3883 / 0.3581; 3.2964 / 2.8186<br>0.3892 / 0.3517; 3.2900 / 2.8177 | 0.3905 / 0.3283; 3.2815 / 2.8312<br>0.3900 / 0.3265; 3.2797 / 2.8316<br>0.3901 / 0.3281; 3.2814 / 2.8316 |
| AdamW → AdamW | 1,920 | 0.4003 / 0.3016; 3.2403 / 2.8053<br>0.3986 / 0.3030; 3.2416 / 2.8070<br>0.4028 / 0.3009; 3.2395 / 2.8028 | — | — |
| AdamW → AdamW | 3,200 | 0.3920 / 0.3199; 3.2586 / 2.8136<br>0.3888 / 0.3213; 3.2599 / 2.8169<br>0.3896 / 0.3209; 3.2596 / 2.8160 | — | — |
| AdamW → Muon | 60 | 0.1646 / 0.1287; 3.0674 / 3.0410<br>0.1612 / 0.1322; 3.0709 / 3.0444<br>0.1632 / 0.1370; 3.0757 / 3.0425 | 0.1666 / 0.1275; 3.0658 / 3.0402<br>0.1617 / 0.1295; 3.0678 / 3.0452<br>0.1654 / 0.1286; 3.0670 / 3.0415 | 0.1658 / 0.1317; 3.0849 / 3.0559<br>0.1617 / 0.1337; 3.0869 / 3.0600<br>0.1647 / 0.1376; 3.0908 / 3.0570 |
| AdamW → Muon | 240 | 0.2814 / 0.2566; 3.1953 / 2.9243<br>0.2813 / 0.2577; 3.1963 / 2.9243<br>0.2797 / 0.2547; 3.1934 / 2.9260 | 0.2832 / 0.2643; 3.2026 / 2.9237<br>0.2825 / 0.2598; 3.1981 / 2.9243<br>0.2808 / 0.2583; 3.1966 / 2.9261 | 0.2825 / 0.2679; 3.2211 / 2.9392<br>0.2817 / 0.2680; 3.2212 / 2.9399<br>0.2811 / 0.2652; 3.2184 / 2.9405 |
| AdamW → Muon | 720 | 0.4057 / 0.3680; 3.3066 / 2.7999<br>0.4042 / 0.3634; 3.3020 / 2.8014<br>0.4054 / 0.3678; 3.3065 / 2.8002 | 0.4059 / 0.3894; 3.3277 / 2.8010<br>0.4060 / 0.3881; 3.3264 / 2.8008<br>0.4065 / 0.3877; 3.3260 / 2.8004 | 0.4073 / 0.3704; 3.3236 / 2.8144<br>0.4069 / 0.3607; 3.3139 / 2.8147<br>0.4077 / 0.3684; 3.3216 / 2.8139 |
| AdamW → Muon | 1,920 | 0.3986 / 0.3434; 3.2820 / 2.8070<br>0.4002 / 0.3472; 3.2858 / 2.8054<br>0.4048 / 0.3416; 3.2803 / 2.8009 | — | — |
| AdamW → Muon | 3,200 | 0.3841 / 0.2787; 3.2173 / 2.8215<br>0.3825 / 0.2792; 3.2178 / 2.8231<br>0.3832 / 0.2790; 3.2177 / 2.8224 | — | — |
| Muon → AdamW | 60 | 0.1358 / 0.0969; 3.0006 / 3.0253<br>0.1325 / 0.0985; 3.0022 / 3.0286<br>0.1355 / 0.1017; 3.0054 / 3.0256 | 0.1371 / 0.1009; 3.0025 / 3.0239<br>0.1333 / 0.1005; 3.0022 / 3.0278<br>0.1376 / 0.1005; 3.0022 / 3.0234 | 0.1363 / 0.0987; 2.9973 / 3.0231<br>0.1321 / 0.0976; 2.9962 / 3.0273<br>0.1352 / 0.0994; 2.9980 / 3.0242 |
| Muon → AdamW | 240 | 0.2281 / 0.2208; 3.1245 / 2.9331<br>0.2270 / 0.2232; 3.1269 / 2.9341<br>0.2274 / 0.2228; 3.1265 / 2.9337 | 0.2297 / 0.2254; 3.1271 / 2.9313<br>0.2282 / 0.2282; 3.1298 / 2.9329<br>0.2281 / 0.2232; 3.1249 / 2.9329 | 0.2279 / 0.2262; 3.1248 / 2.9315<br>0.2275 / 0.2223; 3.1210 / 2.9319<br>0.2276 / 0.2209; 3.1196 / 2.9318 |
| Muon → AdamW | 720 | 0.3365 / 0.3119; 3.2156 / 2.8246<br>0.3361 / 0.3197; 3.2234 / 2.8250<br>0.3370 / 0.3148; 3.2186 / 2.8241 | 0.3355 / 0.3190; 3.2206 / 2.8256<br>0.3349 / 0.3208; 3.2225 / 2.8261<br>0.3357 / 0.3170; 3.2187 / 2.8253 | 0.3374 / 0.3121; 3.2107 / 2.8219<br>0.3376 / 0.3100; 3.2087 / 2.8218<br>0.3379 / 0.3140; 3.2126 / 2.8215 |
| Muon → AdamW | 1,920 | 0.3753 / 0.3607; 3.2644 / 2.7859<br>0.3743 / 0.3668; 3.2705 / 2.7869<br>0.3794 / 0.3589; 3.2626 / 2.7818 | — | — |
| Muon → AdamW | 3,200 | 0.3688 / 0.2935; 3.1972 / 2.7923<br>0.3660 / 0.2941; 3.1979 / 2.7951<br>0.3668 / 0.2943; 3.1981 / 2.7944 | — | — |
| Muon → Muon | 60 | 0.1578 / 0.1118; 3.0155 / 3.0034<br>0.1528 / 0.1133; 3.0170 / 3.0084<br>0.1566 / 0.1139; 3.0177 / 3.0045 | 0.1590 / 0.1120; 3.0137 / 3.0020<br>0.1551 / 0.1126; 3.0143 / 3.0060<br>0.1593 / 0.1128; 3.0144 / 3.0017 | 0.1551 / 0.1143; 3.0129 / 3.0042<br>0.1518 / 0.1150; 3.0136 / 3.0076<br>0.1544 / 0.1150; 3.0137 / 3.0049 |
| Muon → Muon | 240 | 0.2724 / 0.2059; 3.1096 / 2.8887<br>0.2715 / 0.2065; 3.1103 / 2.8896<br>0.2707 / 0.2043; 3.1080 / 2.8904 | 0.2736 / 0.2103; 3.1120 / 2.8874<br>0.2730 / 0.2110; 3.1127 / 2.8881<br>0.2719 / 0.2081; 3.1098 / 2.8891 | 0.2717 / 0.2120; 3.1106 / 2.8876<br>0.2719 / 0.2104; 3.1090 / 2.8875<br>0.2712 / 0.2088; 3.1074 / 2.8882 |
| Muon → Muon | 720 | 0.3910 / 0.3012; 3.2049 / 2.7702<br>0.3904 / 0.3082; 3.2119 / 2.7707<br>0.3918 / 0.3036; 3.2073 / 2.7693 | 0.3913 / 0.3048; 3.2064 / 2.7697<br>0.3915 / 0.3052; 3.2068 / 2.7696<br>0.3918 / 0.3066; 3.2083 / 2.7693 | 0.3923 / 0.3064; 3.2051 / 2.7671<br>0.3916 / 0.3055; 3.2041 / 2.7678<br>0.3921 / 0.3059; 3.2045 / 2.7673 |
| Muon → Muon | 1,920 | 0.3832 / 0.2898; 3.1935 / 2.7779<br>0.3840 / 0.2910; 3.1947 / 2.7772<br>0.3859 / 0.2878; 3.1915 / 2.7753 | — | — |
| Muon → Muon | 3,200 | 0.3753 / 0.3153; 3.2190 / 2.7859<br>0.3717 / 0.3155; 3.2192 / 2.7894<br>0.3725 / 0.3147; 3.2184 / 2.7886 | — | — |

</details>


## Long StarCoderData main extension: completed

The 48 missing long-horizon branches on the seed-43/44 bases, and 72 short reproducibility branches, completed. At 1,920 and 3,200 the six-base analyses agree with the seed-42 LR selections. The common learning support across all selected runs is 1.65454 and 1.70389 nats/token respectively; the formal ℓ0 remains undecided. See [the long main-grid milestone report](starcoderdata_long_main_report.md) for checks and the full long-horizon interpretation.

## Long WikiText LR extension: completed

All 315 branch results (105 LR/pair/length cells × three fine-tuning seeds) finished successfully in 13 B200 jobs (13.02 GPU-hours including evaluation/logging). Initial grid ×¼, ×½, ×1 the 720-step LR; extended down to ×⅛ and ×1/16 for all pairs and to ×1/32 for AdamW → Muon to bracket the 3,200-step optimum. Every 1,920/3,200-step winner is interior. The 144 shared-LR short-branch comparisons (48 each at 60/240/720) reproduce the original clean pilot: every data fingerprint, D0/D1 evaluation identity and base checkpoint hash matches, with maximum endpoint deviations 0.00134 learning and 0.00337 forgetting. Within the entire long grid, data fingerprints agree across optimizer pairs and LRs for every (length, FT seed). The independent `analyze_finetune` runs at 1,920 and 3,200 validate 63 results apiece and reproduce the chosen LRs. All 315 branch tracking records were verified present online in W&B.

| Pair | Original WT 720 LR | Selected 1,920 LR | Selected 3,200 LR | 3,200 L/F at original 720 LR | 3,200 L/F at selected LR |
|---|---:|---:|---:|---:|---:|
| AdamW → AdamW | 6.55e-4 | 1.64e-4 (¼×) | 8.19e-5 (⅛×) | 0.0916 / 1.2206 | 0.3901 / 0.3207 |
| AdamW → Muon | 8.87e-4 | 2.22e-4 (¼×) | 5.55e-5 (1/16×) | −0.1246 / 1.6357 | 0.3833 / 0.2789 |
| Muon → AdamW | 3.28e-4 | 1.64e-4 (½×) | 4.10e-5 (⅛×) | 0.1824 / 0.9630 | 0.3672 / 0.2940 |
| Muon → Muon | 4.44e-4 | 1.11e-4 (¼×) | 5.55e-5 (⅛×) | −0.2433 / 1.6821 | 0.3732 / 0.3151 |

The seed-42-only common learning support (maximum threshold every selected trajectory reaches, not merely its endpoint learning) is 0.374261 at 1,920 and 0.366772 at 3,200. For the fully replicated conditions, see HANDOFF.md. WikiText remains seed-42-only at the two long lengths; the user subsequently authorized the StarCoderData-only long main grid.

The complete LR grid, not just winners: [long WikiText summary](../results/e8/finetune_pilot_long_wikitext103_summary.json). [W&B project](https://wandb.ai/hgo-mit/nanochat) has checkpoint histories organized under the long-WikiText fine-tuning group. The next scientific decision remains main length and ℓ0, chosen from learning support before computing the mismatch contrast; follow this with router correlation and causal replay analyses.

**Reproduce graphs:** `python -m scripts.plot_finetune_endpoints results/e8/finetune_endpoint_runs.csv --out results/e8/figures/finetune_endpoints.png --length-trends-out results/e8/figures/finetune_length_trends.png`. Data sources: eight clean ORCD `analysis/<dataset>-<steps>-main/analysis.json` selection manifests and selected long WikiText pilot result files; exactly 288 fully replicated endpoints (36 per dataset/length) and 24 long WikiText pilot endpoints (12 per length) are in the CSV.
