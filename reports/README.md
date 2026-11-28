# Reports

Markdown copies of the study's Claude artifacts, so they live with the code. The online originals may be
newer; these are snapshots taken 2026-11-26 (each file's header names its source and revision).

| File | Source | Content | State |
|---|---|---|---|
| `pretraining_summary.md` | https://claude.ai/artifact/TP3GZWWwJHK2EqVu3GHdyk | E=8 pretraining recipe, 0.5B LR sweep, 3B transfer check, the six base models | Current |
| `finetune_pilot_report.md` | https://claude.ai/artifact/2uN17JfULvGvtcTFzYUgKx | Seed-42 fine-tuning pilots (StarCoderData, WikiText): design, decisions, LR selections, observations | Current to 2026-11-26 06:30; main grids and long pilot finished later (see `HANDOFF.md` §6) |
| `finetuning_mismatch_endpoints.md` | Local report | Learning–forgetting endpoint curves with seed SDs, raw results, and long-pilot status | Complete; StarCoderData all horizons replicated, WikiText long pilot only |
| `finetune_validation_trajectories.md` | Local report | Eight D0/D1 raw validation NLL trajectory figures, baseline and endpoint tables, selected LRs and seed SDs | Complete; all five horizons per dataset |
| `matched_learning_analysis.md` | Local report | Headline 2×2 minimum-forgetting F at endpoint and within-run learning thresholds, paired contrasts, seed SDs and sensitivity | Complete for E=8; document bootstrap and replay analysis remain open |
| `sparsity_extension_plan.md` | Local report | E=1 dense control and E=4/16 extension design, LR tuning, implementation blockers, compute estimates, decisions | Initial seed-42, three-horizon plan approved; see launch report |
| `finetune_checkpoint_storage_plan.md` | Local report | Actual ORCD quota and measured checkpoint sizes, hard-link deduplication, retention scope and registry | Storage decision needed for full state retention |
| `finetune_rich_router_capture.md` | Local report | Approved compact all-token router summaries, raw sampled logits, correct scored flip attribution, tests and GPU canary | Implemented and deployed; GPU canaries pending |
| `sparsity_launch_report.md` | Local report | Reviewed implementation, pinned cluster checkout, E=1/4/16 launch IDs, workflow ledgers and pending gates | Submitted 2026-11-27; awaiting Slurm scheduling |
| `sparsity_e1_pretraining_lr_sweep.md` | Local report | Dense E1 seed-42 0.5B independent LR selection and transfer jobs | Complete; 3B checks queued |
| `sparsity_e4_pretraining_lr_sweep.md` | Local report | E4 seed-42 0.5B independent LR selection and transfer jobs | Complete; 3B checks queued |
| `sparsity_pretraining_progress.md` | Local report | Cross-E validation losses, perplexities, LR fits and initial 3B transfer checks | Snapshot 2026-11-27 23:19 EST |
| `finetune_replay_bug_impact.md` | Local report | Fixed-LR paired before/after comparison of the replay bug relative to seed spread | Complete |
| `starcoderdata_long_main_report.md` | Local report | Replicated StarCoderData long horizons, checks and endpoint interpretation | Complete |
| `tracker.md` | https://claude.ai/artifact/4tbXf1M5nneqzSYW4NEJYo | Implementation tracker: stages, plan, implemented items, action items | Stale since 2026-11-24 (kept for its resolved decisions) |

Charts are PNGs written by the repo's generators into their own output folders and linked from here, not
stored in `reports/`: `scripts/plot_finetune_pilot.py` → `results/e8/figures/` (rerun it after a summary JSON
changes). The online docs' interactive charts are not exported.

For the current state of the study, read `HANDOFF.md` and `LOG_CHANGES.md`. If an online report is
updated, re-export it here.
