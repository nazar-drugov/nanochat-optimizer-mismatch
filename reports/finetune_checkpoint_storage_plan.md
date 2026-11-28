# Fine-tuning checkpoint retention: storage estimate and inventory design

**2026-11-27 · Planning evidence, not a retention-scope authorization.** Harry
requests preserving actual weights and optimizer state at measured fine-tuning
checkpoints in addition to the evaluation/routing records. This report quantifies
the practical constraint on carrying out that instruction at the current scale.
No cluster storage allocation, migration, deletion, pruning, or training change
was performed for this calculation.

## Actual available storage and measured checkpoint sizes

Read-only check via the existing ORCD SSH connection:

```sh
ssh -n -o BatchMode=yes -o ConnectTimeout=5 \
  -o ControlPath="$HOME/.ssh/cm_socket/orcd-%C" hgo@orcd-login.mit.edu \
  'cat /home/hgo/orcd/.quota'
```

The quota report, last updated **Fri Nov 27 21:59:55 EST 2026**, showed:

| Space | Used (reported GB) | Limit (reported GB) | Percent used | Remaining (reported GB) |
|---|---:|---:|---:|---:|
| HOME | 11.2 | 200.0 | 5.62% | 188.8 |
| SCRATCH | 414.1 | 1,024.0 | 40.44% | **609.9** |
| POOL | 48.0 | 1,024.0 | 4.69% | 976.0 |

The report is a **personal quota**, not the shared filesystem's free capacity.
See [STORAGE_PATHS.md](../STORAGE_PATHS.md). The remaining scratch quota must
also cover in-flight pretraining, existing E8 evidence, logs, and future data.
Do not treat it as entirely available for fine-tuning state. No larger approved
project allocation is documented; Nazar's separate personal quota is not
automatically available to Harry. We have not inspected Nazar's private quota.

Read-only `stat --printf="%n %s B\n"` on the actual E8 seed-42 final base
checkpoint files in Harry's documented scratch root measured:

| Existing file (step 11,445) | Bytes |
|---|---:|
| AdamW `model_011445.pt` | 1,438,681,447 |
| AdamW `optim_011445_rank0.pt` | 2,877,293,065 |
| Muon `model_011445.pt` | 1,438,681,447 |
| Muon `optim_011445_rank0.pt` | 1,631,728,551 |

These are useful proxies because fine-tuning uses the same full model and
optimizer group layout (fresh state, one B200, same tensor dtypes), though
actual serialization sizes should be checked on the first retained checkpoint.
Estimates below use decimal GB/TB (bytes divided by 1e9/1e12); quota display
units are reported GB, so budget conservatively. Metadata and evaluation arrays
are **additional**, not included in binary-state estimates.

## How many physical states?

The current short pilot design is 60/240/960 steps for StarCoderData and
60/240/720 for WikiText. For one fixed base, fine-tuning optimizer, LR and
seed, the three branches have 8, 10 and 12 measured checkpoints on the
log-spaced evaluation schedule. Their shared-prefix states are identical;
hard-linking the weight, optimizer and eval arrays across branch directories
reduces 30 logical checkpoint references to **18 unique physical states**.

Each corpus has the same count. Each E has 2 pretraining optimizers × 2
fine-tuning optimizers × 2 corpora × 5 initial LRs × 3 fine-tuning seeds =
**120 trunks**, 3,600 logical step references, or 2,160 unique state writes with
hard links. There are 60 trunks for each fine-tuning optimizer. Every extra
LR for one pretrain/FT optimizer pair and one corpus adds 3 trunks / 54 unique
states. The later five-horizon E8 long design would have 57 logical checkpoint
references but 26 unique states per trunk. This reduction relies on identical
shared prefixes; distinct LR, seed, optimizer, corpus and post-branch decay
states must not be hard-linked.

Parameter counts are derived from the fixed-width top-1 model
(`GPT.parameter_counts()`; formula in [sparsity extension plan](sparsity_extension_plan.md)).
All model parameters except the input embedding are FP32; the 50,304×640
embedding is BF16 in the existing training recipe. Estimated model bytes =
4 × total parameters − 2 × 50,304 × 640. AdamW's two moment buffers total
approximately twice model bytes. Muon has one momentum buffer for transformer
matrices and routers, while the input embedding and output head still have
two AdamW moments; actual E8 optimizer bytes exceed model bytes by about
193 MB. This reproduces the measured E8 sizes to within 0.02%.

| E | Model GB | AdamW optimizer GB | Muon optimizer GB | One AdamW full state GB | One Muon full state GB |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.406 | 0.812 | 0.599 | 1.219 | 1.005 |
| 4 | 0.849 | 1.697 | 1.042 | 2.546 | 1.890 |
| 8 | 1.439 | 2.877 | 1.632 | 4.316 | 3.070 |
| 16 | 2.618 | 5.237 | 2.811 | 7.855 | 5.430 |

## Scope alternatives

For the initial five-rate grid at each new E (before edge extensions):

| E | Every measured state, weights + optimizer | Weights only, every point | Weights at every point + optimizer at branch finals only |
|---:|---:|---:|---:|
| 1 | 2.402 TB | 0.877 TB | 1.131 TB |
| 4 | 4.791 TB | 1.833 TB | 2.326 TB |
| 16 | 14.348 TB | 5.656 TB | 7.104 TB |
| **Sum** | **21.541 TB** | **8.366 TB** | **10.561 TB** |

Harry also suggested full-state retention every 60 steps. This is distinct
from the existing measured checkpoints (for a 60-step schedule: 2, 4, 8, 16,
32, 48, 54, 60; for 240: 2, 4, 8, 16, 32, 64, 128, 192, 216, 240; longer
curves continue powers of two then include decay start, midpoint, endpoint).
If weights/optimizer are instead saved at multiples of 60 plus every decayed
endpoint, a three-length trunk has 18 unique states on SC (16 longest + 2
shorter finals) and 14 on WT (12 longest + 2 shorter finals). This is 1,920
unique states per E across the two datasets, ~19.15 TB full state in total.
Saving both schedules would be larger. Combining the *available* personal
scratch and pool quotas gives about 1.586 TB, before further pretraining and
arrays, so even the proposed regular interval cannot fit all trials. The
exact save cadence and actual storage allocation/capture subset require
agreement before FT starts.

Any edge extension for one pair and corpus adds 54 states: approximately
66/54 GB at E1 AdamW/Muon FT, 137/102 GB at E4, and 424/293 GB at E16,
respectively. This is not the cost of replica pretraining seeds or long horizons.

A separate possibility is to retain **only the selected LRs** at every measured
checkpoint. Initial sweep metrics are inexpensive to keep, but they do not
contain model weights, so the only way to obtain full selected states after
an evaluation-only pilot is to retrain the selected cells under exactly the
same protocol. This does not reconstruct the original GPU run bit-for-bit.
If each horizon of a pair uses the same selected LR, 24 trunks across the two
datasets have 432 unique states per E; if all three horizons use different
selected LRs, there are up to 720 measured states (only the selected horizon
for each run, not automatically retaining unused branches).

| E | Selected LRs, all measured weights + optimizer (432–720 states) | Selected weights only, all points | Selected final states only, weights + optimizer (72 states) |
|---:|---:|---:|---:|
| 1 | 0.480–0.801 TB | 0.175–0.292 TB | 0.080 TB |
| 4 | 0.958–1.597 TB | 0.367–0.611 TB | 0.160 TB |
| 16 | 2.870–4.783 TB | 1.131–1.885 TB | 0.478 TB |

The selected-only full retention path still cannot fit on the current
personal scratch quotas for E4/E16. Final-only states do not retain the
intermediate checkpoints Harry requested; neither weights-only nor
evaluation-only is an acceptable silent substitution for full state.
A selected-LR replay of 24–72 trunks is roughly 20–60% of the initial
120-trunk grid's training work if rerun via the same trunk mechanism
(actual overhead depends on individual selected horizons), in addition
to the original pilot. Exact GPU-hour costs require actual new-E pilot
timings.

Checkpoint I/O is also a compute-time cost. Pure writes of 21.54 TB take
about six aggregate hours at 1 GB/s sustained or 30 at 0.2 GB/s; these are
illustrative bandwidth assumptions, not ORCD benchmark measurements.
Device transfer, serialization, metadata, I/O contention and filesystem
limits can add to this. The E16 AdamW state alone is ~7.9 GB per unique
checkpoint. Metadata/routing arrays and interrupted attempts require
additional space, so any storage expansion must provide substantial headroom.
At the nominal 1,048,576 scored tokens, 10-layer uint8 route-ID arrays for both
D0 and D1 would occupy about 21 MB of raw route data per unique checkpoint,
about 45 GB per E across the initial 2,160 states before compression. The NPZ
compression ratio varies with actual assignments (E1 all zeros compresses
strongly), and document-level metrics add other data. This is why the existing
evaluation-only evidence is much smaller than full tensor checkpoints, but
pool usage should still be monitored.

## Retention, provenance, and pool mirroring

The retained branch checkpoint naming follows existing pretraining naming:
`model_<step>.pt`, `optim_<step>_rank0.pt`, `meta_<step>.json`, plus
`ckpt_<step>.npz` evaluation arrays (actual D0/D1 routes, losses and flips).
Branch-specific metadata remains separate; identical saved tensors/arrays
can be hard-linked for shared checkpoints within the same attempt. The
logical branch record still identifies its horizon, exact LR, base model
digest and data seed. A saved model/optimizer pair supports later evaluation
and optimizer inspection; it is not necessarily a byte-exact training
resume unless the data-loader position and RNG have also been retained.

The checkpoint registry now inventories actual FT weights, optimizer files,
metadata and eval arrays independently. It records missing components as
missing, and lists retained incomplete attempts in their separate
`.interrupted/<base>/<dataset>/<cell>.<uuid>` archives, flagged as excluded
from scientific analysis. Where a pool mirror's recorded attempt ID differs
from the active scratch attempt, its actual paths are listed in separate pool
columns rather than treated as the current checkpoint. The inventory counts
logical step records, not physical hard-linked storage usage.

The generated FT persistence filter is intentionally restricted to metadata,
JSON/JSONL, logs, parquet, NPZ evaluation arrays, and `eval_set_*.pt` inputs.
It excludes model and optimizer files. The prior `*.pt` wildcard would have
copied full states indiscriminately into the personal 1 TB pool; this is not
an authorized storage strategy. The pool is a small-results mirror, not a
backup for retained binary states. The pretraining persistence path already
excludes model/optimizer tensors. Do not delete existing pool files or move
old evidence as part of this change.

## Decision required

The storage constraint is material. Full retention for all initial short
grid checkpoints in E1/4/16 needs over 21 TB (plus extensions and headroom).
Even selected-only retention needs a multi-TB allocation and repeated
training if those model states were not captured in the existing pilot.
A larger project storage allocation or an explicitly agreed smaller scope
is necessary. Until Harry settles scope and placement, do not silently
drop checkpoint state, enable an infeasible full-write policy, or launch
new training merely to recreate missing artifacts.
