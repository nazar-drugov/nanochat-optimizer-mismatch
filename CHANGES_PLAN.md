# Next changes

1. **Make AdamW distributed-safe.**

   Nanochat does not use DDP; `DistributedOptimizer` performs its own distributed gradient synchronization. Returning ordinary `torch.optim.AdamW` means AdamW training is correct on one GPU but may train independent models on multiple GPUs.

   You need either:

   - a separate distributed AdamW implementation, or
   - a generic distributed optimizer capable of running only AdamW groups.

   This is the highest-priority correctness issue.

2. **Define exactly what “Muon” means.**

   The current `DistributedOptimizer` is not plain reference Muon. It includes Polar Express, MuonEq, Muon+, NorMuon, and cautious weight decay.

   For interpretable results, either:

   - implement a reference/plain Muon option matching the algorithm used by the papers, or
   - call the current optimizer something like `nanochat_muon` and describe every modification.

   Otherwise, a result attributed to “Muon” could actually be caused by one of those additions.

3. **Fix MoE compute and budget accounting.**

   Current parameter/FLOP accounting counts all experts, even though top-1 routing executes only one expert per token. This affects:

   - reported FLOPs,
   - automatically selected token budget,
   - batch-size scaling,
   - learning-rate and weight-decay scaling.

   Report both total parameters and active parameters, but match dense and MoE primarily by fixed training tokens and active FLOPs. Otherwise the MoE may receive a substantially larger training budget.

4. **Finish the post-training optimizer-switch path.**

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

5. **Use equal hyperparameter-search budgets.**

   Do not require AdamW and Muon to use the same numerical learning rate—that is not a fair comparison. Give every architecture × optimizer × stage combination the same tuning budget.

   At minimum tune:

   - matrix learning rate,
   - router learning rate,
   - weight decay,
   - warmup,
   - gradient-clipping norm.

   Later sensitivity experiments can include betas, epsilon, Muon momentum, and Newton–Schulz steps. The papers specifically show that conclusions can change after learning-rate tuning.

6. **Add configurable gradient clipping.**

   It is currently missing. Both papers treat clipping as an optimizer-relevant hyperparameter. Log the pre-clipping gradient norm as well as whether clipping occurred.

7. **Add seeds and statistical reporting.**

   Expose a seed argument and use it consistently for initialization and data order. Use paired seeds so corresponding dense/MoE and AdamW/Muon runs see the same examples in the same order.

   Use at least three seeds for final experimental cells and report means, standard deviations or confidence intervals, and paired differences.

8. **Measure the actual interaction.**

   The important question is not simply whether MoE or Muon has a lower loss. Estimate whether the penalty from switching optimizers differs by architecture.

   Conceptually, compare:

   `MoE mismatch penalty − dense mismatch penalty`

   Ideally model architecture, pretraining optimizer, post-training optimizer, and their interactions rather than comparing isolated runs.

9. **Add MoE-specific diagnostics.**

   Without an auxiliary balancing loss, routing collapse is an important possible confounder. Log per layer:

   - fraction of tokens assigned to each expert,
   - routing entropy,
   - largest expert share,
   - inactive/dead experts,
   - router gradient and update norms,
   - expert gradient and update norms.

   You do not need to add an auxiliary loss to the primary experiment. Keep the routing design fixed and consider balancing losses as a later ablation.

10. **Evaluate more than final loss.**

   Record:

   - final validation loss at a fixed token budget,
   - tokens needed to reach specified loss thresholds,
   - downstream evaluation,
   - forgetting after post-training,
   - wall-clock time, tokens/second, and peak memory.

   Compare by both tokens and active FLOPs because MoE routing and Muon have different computational overheads.

11. **Improve reproducibility metadata.**

   Each checkpoint or run record should include the complete model and optimizer configuration, all parameter-group settings, seed, data/tokenizer identity, precision, schedules, Git commit, hardware, and whether optimizer state was reset.

I would fix the distributed AdamW path, reference-Muon definition, active-expert accounting, and `chat_sft.py` wiring before starting any expensive sweep. Those four can otherwise invalidate the experiment rather than merely reduce its statistical power.
