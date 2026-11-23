# Next changes

Updated 2026-11-23 for commit `0ed4cd4`. The immediate goal is to validate and
launch the pretraining comparison. `PRETRAINING_PROTOCOL.md` contains the current
settings and learning-rate sweep. Fine-tuning and the optimizer-switch analysis
follow after the pretraining path is reliable.

## 1. Settle the protocol before spending GPU time

- Confirm the settings marked "Defaults that still need team confirmation" in
  `PRETRAINING_PROTOCOL.md`: balancer, matrix weight decay, router learning rate,
  AdamW epsilon, model size, token budget, and seed count. Record the chosen
  values and the reason for each in that file. Keep the decided data, tokenizer,
  batch size, schedule, and clipping settings fixed across comparable cells.
- Fix the reference to `EXPERIMENT_SPEC.md` in `PRETRAINING_PROTOCOL.md`: that
  file is absent from this checkout. Restore the specification if it is the
  source of the experiment requirements, or replace the reference with a
  checked-in description of the study and its acceptance criteria.
- D0 is selected for learning-rate validation. The study launcher passes its
  directory to `base_train.py`, which records the validation file SHA-256 and
  rejects changed validation data on resume. Run a short two-GPU test to verify
  the cluster reads D0. Keep the 1,048,576-token evaluation count and final
  bits-per-byte selection rule fixed across every comparable cell.

## 2. Prove the new pretraining path on the cluster

- Run the full test suite and the two-GPU CUDA Muon parity test on an allocated
  GPU node. CPU tests passed during development, but the CUDA-only tests and
  NCCL path were not verified in `0ed4cd4`.
- Run short, two-GPU training jobs for both AdamW and plain Muon with the
  intended MoE architecture, tokenizer, data path, and precision. Check that
  the jobs compile, finish optimizer steps, and produce finite losses and
  usable throughput and peak-memory measurements. Repeat for the E=1 control
  if its model configuration differs from the MoE jobs.
- Save a checkpoint, resume each kind of job, and compare data fingerprints,
  checkpoint metadata, routing snapshots, and the resumed training trace.
  Confirm that matched cells use the same examples in the same order.
- Review each run's `config.json`: Git commit and dirty state, model and active
  parameter counts, optimizer groups including the router, schedule, seed,
  data/tokenizer identity, and hardware must match the intended cell. Resolve
  any mismatch before the learning-rate sweep.


## 4. Make the learning-rate sweep reproducible

- Implement or document one launch manifest for the protocol's eight
  (optimizer, expert-count) cells and five learning rates per cell. Record the
  exact command, seed, dataset, resolved configuration, code commit, output
  directory, and run status for each job. Keep the token and batch budgets
  identical across optimizer comparisons.
- Select each cell's peak matrix/router learning rate using the same final
  validation bits-per-byte. Extend a grid whose optimum is at an edge and use
  the protocol's seed-noise check when neighboring points are close. Save the
  full sweep results and selection rule, not only the winning runs.
- Transfer the selected rates to the full budget and perform the specified
  E=4 transfer check and batch-size robustness runs. If the transfer check
  fails, use the protocol's longer-run fallback before launching all full
  runs.

## 5. Run the final pretraining comparisons and report uncertainty

- Use paired seeds and identical token order for AdamW and Muon within each
  architecture. The protocol currently proposes three seeds for E=8 and one
  elsewhere; decide whether the final claims require more seeds before
  launching expensive jobs.
- Report validation curves and final bits-per-byte at fixed token budgets,
  paired differences, variation across seeds, tokens per second, wall-clock
  time, and peak memory. Include routing balance and inactive-expert
  diagnostics. The code already estimates active parameters and FLOPs; use
  those estimates alongside token counts when comparing dense and MoE costs.

## 6. Complete the optimizer-switch study after pretraining

- Update `chat_sft.py` and `base_eval.py` for checkpoints using the GPT-2
  tokenizer and the current optimizer, router, and model configuration. Add
  end-to-end tests that load a pretraining checkpoint and run a small SFT and
  evaluation cycle.
- Start every SFT cell with fresh optimizer state. Compare matched and
  mismatched AdamW/Muon pretraining-to-SFT pairs on the same initial checkpoint,
  post-training data, tuning budget, and paired seeds. Record whether optimizer
  state was reset and measure the switch penalty and forgetting. Treat dense
  models as a secondary architecture comparison.
- Tune the post-training optimizers fairly, with equal search budgets, before
  interpreting an optimizer mismatch effect. Keep the pretraining protocol
  and its selected hyperparameters recorded separately from SFT choices.
