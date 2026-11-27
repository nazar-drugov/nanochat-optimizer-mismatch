# E=1/4/16 initial experiment — implementation and launch

**2026-11-27, 15:34 EST · All coordinators running; E=1 scientific LR sweep running, E=4 sweep queued, E=16 verification queued.**

## Agreed scientific scope

Extend the E=8 optimizer-mismatch study to E=1 (dense SwiGLU control), E=4 and E=16 at fixed active size and 3B pretraining tokens. Start with seed 42, independent local log-LR fits for each architecture/optimizer, separate 3B transfer checks, and fine-tuning seeds 1–3 on the original three horizons (SC 60/240/960 and WT 60/240/720 optimizer steps). Select LRs only from the prescribed validation distribution, not forgetting. Longer horizons, additional pretrained seeds and new cross-E matched-learning threshold decisions are not part of this launch.

## Implemented and verified

The immutable training checkout on ORCD is `/home/hgo/nanochat-mismatch-sparsity-909d98b` at `909d98b6b13e13686cd88558e10e930d632e3139`.

- E=1 `--disable-routing-loss` resolves auxiliary and z coefficients to zero and persists the setting in model, checkpoint and result metadata. Multi-step clipped AdamW/Muon tests establish equivalent dense SwiGLU updates; the router remains inert. One-expert undefined diagnostics are null, actual route replay effects are exactly zero.
- Pretraining generator uses distinct per-E five-point grids and separate local LR fits / 3B checks. Full transfer-center checkpoints are reused only when the checked LR is retained, with an explicit logical base → actual path and SHA-256 manifest; no redundant base training or fabricated paths.
- Fine-tuning generator supports an E-aware pin manifest, independent dataset/pair LR extensions, approved lengths/data fingerprints, and verified W&B stage and base provenance. Exporter emits selected raw endpoints, trajectories and learning-only common support, not automatic contrasts.
- The coordinator submits one-step B200 training/evaluation smoke checks, verifies their finite results and resolved settings, then advances the reviewed sweep, transfer, final-base and FT pilot commands. Factor-of-two LR extensions are limited to actual edge winners. A failed or unfittable outcome blocks that architecture instead of silently changing the design. Durable job ledgers, submission intents and a lock prevent duplicate submission on coordinator restart.

Validation: 178 combined legacy/new regression tests passed, plus eight new fine-tuning sparsity tests (186 distinct tests). Independent subagents dry-ran 48 pretraining scripts, mock-executed 27 generated FT scripts, and ran a tiny real SHA-pinned E=1 dense-control fine-tuning fixture with both optimizers, including decay branches and exact null replay. All code-review blockers resolved. E=16 full-size FT memory and actual GPU execution are still to be verified.

## Launch and ownership

All jobs are submitted under `hgo`. E=1 and E=4 at normal priority; E=16 submitted afterward with `--nice=1000` after checking there were no visible Nazar E16 jobs or documented run dirs/claims. This is not a new permission to start redundant jobs under `ndrugov`.

| E | GPU smoke AdamW | GPU smoke Muon | CPU workflow coordinator | GPU nice |
|---:|---:|---:|---:|---:|
| 1 | 30735417 | 30735419 | 30735438 | 0 |
| 4 | 30735420 | 30735421 | 30735439 | 0 |
| 16 | 30735426 | 30735428 | 30735440 | 1000 |

At the initial verification check, all nine jobs were **PENDING (Priority)** with no reliable start time. At 15:27 EST, E=1 AdamW smoke job 30735417 began running on B200 `node5601-c1`; data fingerprints/validation paths and online [W&B run ufg62dyj](https://wandb.ai/hgo-mit/nanochat/runs/ufg62dyj) appeared in the training log. No scientific sweep results, base checkpoints or FT results exist yet. The smoke jobs are verification-only, never LR-selection points. The remaining GPU jobs will be submitted by the reviewed coordinators after the data-dependent selection gates; they are not already-running training jobs. Pretraining is normal-partition only, since the current launcher refuses blind restart of partial runs; FT supports preemptable/requeue. CPU coordinators are on `mit_preemptable` (48 h, requeue) and can resume the same ledger; re-submit the coordinator after an actual timeout if still needed, not a competing copy.

### First GPU verification results and actual sweep submission

The four E=1/4 one-step jobs completed `0:0` with online W&B runs verified by API, not just local logs:

| E | Optimizer | Final validation bpb | D0 bpb | Balance / z coefficient | W&B run |
|---:|---|---:|---:|---:|---|
| 1 | AdamW | 3.20001 | 3.14216 | 0 / 0 | [ufg62dyj](https://wandb.ai/hgo-mit/nanochat/runs/ufg62dyj) |
| 1 | Muon | 3.17316 | 3.11570 | 0 / 0 | [tpyxd8c9](https://wandb.ai/hgo-mit/nanochat/runs/tpyxd8c9) |
| 4 | AdamW | 3.18729 | 3.12987 | .01 / .001 | [06z62hd0](https://wandb.ai/hgo-mit/nanochat/runs/06z62hd0) |
| 4 | Muon | 3.17567 | 3.11833 | .01 / .001 | [wmmcfzn6](https://wandb.ai/hgo-mit/nanochat/runs/wmmcfzn6) |

These verify end-to-end launch and finite losses; they are not scientific optimizer comparisons. Both coordinators automatically advanced to the initial five-point-per-optimizer 0.5B sweep (ten runs per E). E1 sweep jobs **30735929 / 30735930** running on B200s; first cells [AdamW](https://wandb.ai/hgo-mit/nanochat/runs/nereyxnf) / [Muon](https://wandb.ai/hgo-mit/nanochat/runs/1oe7miru) are logging. E4 sweep jobs **30736231 / 30736232** pending. E16 initial smoke remains queued behind the E1/4 GPU allocation (QOS limit). All three CPU coordinators are running and have no blocked status. Snapshot registry refreshed at 15:33:57 (1,946 runs, 19,795 step records), including four verification-only model checkpoints and two active sweep cells.

## Durable records and monitoring

For each `E` in `{1,4,16}`:

- Workflow: `/home/hgo/jobs/sparsity-20261127/e<E>/state.json`, `*/jobs.json`, `coordinator-<jobid>.out`.
- Selections and actual base pin: `/home/hgo/jobs/sparsity-20261127/e<E>/selection/{lr_selection.json,transfer_check.json,lr_selection_3b.json,base_manifest.json}` (created only when the gates succeed).
- Pretraining results: `/orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/base_checkpoints/` (specific directories verified as they appear).
- FT pilot output: `/orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/finetune_pilot_e<E>/`.
- Per-horizon D1-only analysis: `/home/hgo/jobs/sparsity-20261127/e<E>/analysis/`.
- W&B groups `optimizer-mismatch-e<E>/pretrain/{verification,sweep,transfer,main}` and `optimizer-mismatch-e<E>/finetune/<dataset>/finetune_pilot_e<E>`; credentials available, online history verification pending actual execution.

The general registry now recognizes the new explicit pilot roots; actual file entries are refreshed at run milestones. No paths are claimed to contain new scientific base checkpoint weights before they exist. The next scientific milestone is completed, bracketed 0.5B sweep selection per E; report it and the subsequent 3B validations before cross-E inference.
