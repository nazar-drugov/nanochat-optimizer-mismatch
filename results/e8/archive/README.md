# Historical fine-tuning pilot data (not for scientific analysis)

`finetune_pilot_wikitext103_pre_replay_fix_summary.json` was reconstructed on 2026-11-26 from Nazar’s original seed-42 best-fit WikiText pilot results at `/orcd/scratch/orcd/010/ndrugov/finetune_pilot/e8-main-*-s42/wikitext103/`. All 60 LR/length/pair cells contained completed fine-tuning seeds 1–3; the old and new runs share D0/D1 identities.

The 240- and 720-step historical results are invalid because of the decay-branch replay bug (fixed in `d330586`). The 60-step results were unaffected. Do not use this archive for mismatch analysis. The corresponding pre-fix StarCoderData summary is retained in Git at `ed76421:results/e8/finetune_pilot_starcoderdata_summary.json` (78 cells). The historical files are used only to compare the old and corrected pilots.
