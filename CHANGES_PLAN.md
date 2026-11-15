# Next changes

1. **Run the plain-Muon and AdamW parity tests on two GPUs.**

   `muon` now means the original Muon algorithm: Nesterov momentum, fixed-coefficient
   Newton-Schulz orthogonalization, and the original aspect-ratio learning-rate scaling.
   It does not use Polar Express, MuonEq, Muon+, NorMuon, or cautious weight decay.

   Single-rank and two-rank parity tests compare `DistributedOptimizer` with
   `torch.optim.Muon`. The two-rank CUDA test still needs to be run on a machine with
   at least two GPUs.

2. **Measure MoE active compute separately from the fixed token budget.**

   Pretraining now requires an explicit token budget and batch size; it no longer scales the budget, learning rates, or weight decay from parameter or FLOP estimates. The separate Chinchilla-style planner counts total parameters, including all experts, to suggest 20× and 40× token budgets.

   Top-1 routing executes one expert per token. Add active-parameter and active-FLOP estimates for reporting and cost comparisons. The total-parameter planner is a heuristic, not an exact compute-optimal MoE rule. Use the same selected rounded token budget for matched AdamW and Muon runs on the same MoE architecture.

3. **Finish the post-training optimizer-switch path.**

   `chat_sft.py` still uses the old `matrix_lr` interface and does not fully expose the new optimizer choices and metadata.

   Your core experiment should eventually contain:

   | Architecture | Pretraining | Post-training |
   |---|---|---|
   | Dense | AdamW | AdamW |
   | Dense | AdamW | Muon |
   | Dense | Muon | AdamW |
   | Dense | Muon | Muon |
   | MoE | AdamW | AdamW |
   | MoE | AdamW | Muon |
   | MoE | Muon | AdamW |
   | MoE | Muon | Muon |

   Start every post-training cell with fresh optimizer state. Otherwise matched cells can carry momentum/history while mismatched cells cannot, confounding optimizer mismatch with state reset.

## Experimental rigor changes

4. **Use equal hyperparameter-search budgets.**

   Do not require AdamW and Muon to use the same numerical learning rate—that is not a fair comparison. Give every architecture × optimizer × stage combination the same tuning budget.

   At minimum tune:

   - matrix learning rate,
   - router learning rate,
   - weight decay,
   - warmup,
   - gradient-clipping norm.

   Later sensitivity experiments can include betas, epsilon, Muon momentum, and Newton–Schulz steps. The papers specifically show that conclusions can change after learning-rate tuning.

5. **Add configurable gradient clipping.**

   It is currently missing. Both papers treat clipping as an optimizer-relevant hyperparameter. Log the pre-clipping gradient norm as well as whether clipping occurred.

6. **Run paired seeds and report variation.**

   Pretraining now exposes `--seed`, uses it for global CPU/CUDA randomness and explicit model initialization, saves it in checkpoints, and validates it on resume. Verify that paired AdamW/Muon runs see the same examples in the same order, including after resume.

   Use at least three paired seeds for final experimental cells and report means, standard deviations or confidence intervals, and paired differences.

7. **Measure the optimizer-switch effect.**

   The primary comparison holds the MoE architecture and token budget fixed, then compares matched and mismatched pretraining/SFT optimizer pairs. Report the switch penalty with paired seeds. Dense experiments can provide a secondary architecture interaction analysis.

8. **Add MoE-specific diagnostics.**

   Without an auxiliary balancing loss, routing collapse is an important possible confounder. Log per layer:

   - fraction of tokens assigned to each expert,
   - routing entropy,
   - largest expert share,
   - inactive/dead experts,
   - router gradient and update norms,
   - expert gradient and update norms.

   You do not need to add an auxiliary loss to the primary experiment. Keep the routing design fixed and consider balancing losses as a later ablation.

9. **Evaluate more than final loss.**

   Record:

   - final validation loss at a fixed token budget,
   - tokens needed to reach specified loss thresholds,
   - downstream evaluation,
   - forgetting after post-training,
   - wall-clock time, tokens/second, and peak memory.

   Compare by both tokens and active FLOPs because MoE routing and Muon have different computational overheads.

10. **Improve reproducibility metadata.**

   Each checkpoint or run record should include the complete model and optimizer configuration, all parameter-group settings, seed, data/tokenizer identity, precision, schedules, Git commit, hardware, and whether optimizer state was reset.

Before an expensive sweep, run the two-GPU optimizer parity tests, finish `chat_sft.py` optimizer-switch wiring, and add MoE routing diagnostics. Keep the current Muon momentum schedule until it is reviewed separately.
