# Reports

Markdown copies of the study's Claude artifacts, so they live with the code. The online originals may be
newer; these are snapshots taken 2026-11-26 (each file's header names its source and revision).

| File | Source | Content | State |
|---|---|---|---|
| `pretraining_summary.md` | https://claude.ai/artifact/TP3GZWWwJHK2EqVu3GHdyk | E=8 pretraining recipe, 0.5B LR sweep, 3B transfer check, the six base models | Current |
| `finetune_pilot_report.md` | https://claude.ai/artifact/2uN17JfULvGvtcTFzYUgKx | Seed-42 fine-tuning pilots (StarCoderData, WikiText): design, decisions, LR selections, observations | Current to 2026-11-26 06:30; main grids and long pilot finished later (see `HANDOFF.md` §6) |
| `finetuning_mismatch_endpoints.md` | Local report | Learning–forgetting endpoint curves with seed SDs, raw results, and long-pilot status | Complete; StarCoderData all horizons replicated, WikiText long pilot only |
| `finetune_validation_trajectories.md` | Local report | Eight D0/D1 raw validation NLL trajectory figures, baseline and endpoint tables, selected LRs and seed SDs | Complete; all five horizons per dataset |
| `finetune_replay_bug_impact.md` | Local report | Fixed-LR paired before/after comparison of the replay bug relative to seed spread | Complete |
| `starcoderdata_long_main_report.md` | Local report | Replicated StarCoderData long horizons, checks and endpoint interpretation | Complete |
| `tracker.md` | https://claude.ai/artifact/4tbXf1M5nneqzSYW4NEJYo | Implementation tracker: stages, plan, implemented items, action items | Stale since 2026-11-24 (kept for its resolved decisions) |

Charts are PNGs written by the repo's generators into their own output folders and linked from here, not
stored in `reports/`: `scripts/plot_finetune_pilot.py` → `results/e8/figures/` (rerun it after a summary JSON
changes). The online docs' interactive charts are not exported.

For the current state of the study, read `HANDOFF.md` and `LOG_CHANGES.md`. If an online report is
updated, re-export it here.
