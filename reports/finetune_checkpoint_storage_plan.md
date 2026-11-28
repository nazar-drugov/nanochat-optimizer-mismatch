# Fine-tuning checkpoint retention: storage estimate and inventory design

**2026-11-27 · Retention scope agreed for initial pilots; selected-state archive
capacity still open.** Harry requests preserving actual weights and optimizer
state at measured fine-tuning checkpoints in addition to evaluation/routing
records when practical. The initial E1/4/16 pilots will instead retain richer
router evidence without binary model/optimizer states, because full state
retention exceeds the currently allocated storage. Selected-LR actual-state
reruns will require a separate storage decision.
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

## Agreed initial-pilot router evidence: cost estimate

The decision for the initial LR pilots is **no binary weight or optimizer
snapshots**, more informative all-token router summaries and a fixed raw
sample at every existing measured checkpoint, including step zero. This
preserves evidence before selecting an LR without spending tens of terabytes.
The separate proposed 60-step save cadence is deferred; do not silently
change the log-spaced scientific measurement schedule.

The actual held-out identities from clean E8 seed-42 configurations are:

| Evaluation set | Packed tensor shape | Scored tokens | Scored documents |
|---|---|---:|---:|
| D0 FineWeb-Edu | 512 × 2048 | 1,047,614 | 1,474 |
| StarCoderData D1 | 512 × 2048 | 1,047,642 | 1,446 |
| WikiText-103 D1 | 512 × 2048 | 1,048,461 | 627 |

Thus one checkpoint has 2,920 document records (SC) or 2,101 (WT) across the
two evaluation populations. The fixed sample is at most 8,192 **scored**
token positions per D0/D1, reused across checkpoints and aligned with the
base evaluation. It is not an independent random sample at each checkpoint.

The bounded raw FP32 sample uses one or both of logits and softmax
probabilities for every sampled token, expert and layer on each set.
Storing logits alone costs 2 sets × 8,192 × 10 layers × E × 4 bytes =
655,360 E bytes per checkpoint; storing both doubles the sample term
(except E1 softmax probabilities are structural ones and need no array).
Logits retain information about additive shifts and z-loss that raw
probabilities alone do not. Full raw route IDs (uint8) on all packed positions add
20,971,520 bytes per checkpoint. This includes D1 routes, which the old FT
NPZs omitted.

The implemented exact all-scored-token statistics use
per-document/layer FP64 sums and sums of squares for seven scalar diagnostics
(entropy, top-two margin, chosen gate, maximum probability, JS, TV and raw z-loss)
and one FP64 soft-expert probability sum per document/layer/expert. The
per-document term is therefore 10 × (D0 documents + D1 documents) ×
8 × (14 + E) bytes. Layer counts/totals, 32-bin histograms and edges, sample indices
stored once in the base record, and metadata are small relative to the
sample matrices; existing document loss/flip arrays also add about
0.21–0.28 MB per state. Undefined E1 metrics remain explicitly null;
trivial structural-zero/one arrays compress well and need not dominate
actual storage. This estimate includes the known aggregate arrays and will
be checked against the first real compressed checkpoint file.

| E | Sample logits only (MB/state) | All-token summary, SC / WT (MB/state) | Total raw state (logits only), SC / WT (MB/state) | Initial grid, logits only (GB) | Initial grid, both raw matrices (GB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.66 | 3.57 / 2.58 | 25.48 / 24.42 | 53.9 | 53.9 |
| 4 | 2.62 | 4.27 / 3.09 | 28.15 / 26.89 | 59.4 | 65.1 |
| 16 | 10.49 | 7.08 / 5.10 | 38.81 / 36.77 | 81.6 | 104.3 |
| **Total** | | | | **195.0** | **223.3** |

These are 1,080 unique states per corpus per E (2,160 per E), not 3,600
logical branch links. The all-token summaries and sampled logits alone add
about 58 GB total over full hard routes and existing
document-level arrays. Each one-pair/corpus LR extension adds 54 unique
states, at most about 1.4 GB E1, 1.6 GB E4, or 2.1 GB E16 with logits alone
(about 2.7 GB E16 if both matrices are retained).
Actual NPZs are compressed and can be far smaller: for example the
historical E8 SC 60-step AdamW→AdamW seed-1 endpoint at the selected LR
1.311e−3 is 4,221,841 bytes on scratch (only D0 route IDs). Raw FP32 data
likely compress less than uint8 routes. The base snapshots and shared
packed evaluation sets are additional (order 1 GB over this initial pilot).

For an independent serialization check only (no model training or real data),
we created temporary, shape-matched NPZ fixtures with uniformly random uint8
routes, high-entropy seeded random float32/float64 soft arrays, and the E1
structural all-zero route IDs. NumPy's compressed NPZ sizes were:

| E | Compressed SC / WT state, logits-only (MB) | Initial grid extrapolation, logits-only (GB) | Both matrices (GB) |
|---:|---:|---:|---:|
| 1 | 4.23 / 3.24 | 8.1 | 8.1 |
| 4 | 12.83 / 11.64 | 26.4 | 31.5 |
| 16 | 28.57 / 26.64 | 59.6 | 79.9 |
| **Total** | | **94.1** | **119.5** |

The fixtures deliberately do not reproduce actual metric distributions and
are not a storage upper bound or a cluster benchmark. They illustrate that
the compression assumptions can plausibly reduce total pilot storage to
roughly half the raw sizing; real arrays must be checked during rollout.
Local compression of these fixtures took about 0.1–3 seconds per unique
state, not including any router computation or cluster I/O.

**Capacity assessment:** the personal pool has 976 GB free in the last
quota report, enough even for the conservative 224 GB evaluation mirror.
Scratch has only 610 GB of reported free space; unlike pool, it also
retains pretraining binary states. A conservative remaining pretraining
budget from the last report could be approximately 20 GB (E1 extra
transfer/final), 80 GB (E4 transfer/final), and up to 319 GB (E16 smoke,
0.5B sweep, 3B checks and two displaced-vertex final models), before
additional LR edges. Some E1/4 work after the quota timestamp is already
complete, so this is a planning upper estimate, not a new usage reading.
The uncompressed combined worst case could exceed scratch by tens of
GB; actual compressed arrays and reuse of checked transfer-center bases
should lower requirements. With the synthetic logits-only compressed
estimate, the combined 420 GB remaining pretraining allowance + 94 GB
pilot arrays would leave about 96 GB headroom before extra LR trials and
other overhead; this is only a planning scenario. Do not declare the entire pipeline guaranteed
to fit a stale quota. Measure compressed NPZ bytes after a representative
cell, refresh quota at each workflow phase and before E16 FT and large LR
extensions, and settle extra space if needed. No silent evidence deletion.

Pure writes of 195–223 GB take about 3.3–18.6 aggregate minutes at illustrative
1–0.2 GB/s sustained; mirroring duplicates the writes. The 58 GB increment
for logits and exact soft summaries would take approximately 1–4.8 minutes of pure writes. Actual computation
of all-token diagnostics, device transfers, compression and I/O
contention may dominate; these are not measured ORCD performance results.
Caching base soft log-probabilities for both full scored sets in FP32 takes
at most ~1.34 GB of CPU RAM at E16 (omitted in E1). The implementation batches
the exact calculations against this transient CPU cache instead of persisting
full million-token soft tensors at every checkpoint.
At E16 two sets × 1.05M positions × 10 layers × 16 experts is ~335 million
probability values per measured state; timing needs validation on the
actual device and cannot be inferred from disk bytes alone. For E≥4, the
current batched reduction copies the base log-probabilities to the device
and the current probabilities plus seven scalar values back to CPU for
FP64 reductions. At E16 that is on the order of 3.3 GB of transient
device/host transfer per measured state (not stored), or ~7 TB over the
initial grid. Short batches avoid a persistent GPU tensor of that size,
but CPU reduction/compression and synchronization warrant timing checks
on the first real jobs. E1 uses an explicit structural shortcut.

The added evidence is stored in the existing `ckpt_*.npz` and
`base_arrays.npz` files, so the established NPZ inventory and scratch/pool
mirror recognize it without a new binary-state schema. Preserve branch/
base provenance and sample indices; do not upload raw router arrays to W&B
merely for tracking.

## Later actual-state retention remains a storage decision

The storage constraint is material. Full retention for all initial short
grid checkpoints in E1/4/16 needs over 21 TB (plus extensions and headroom),
so the agreed initial LR pilots record richer evaluation evidence without
binary state. Selected-LR actual-state reruns remain the scientific plan
for deeper follow-up. They require a verified multi-TB allocation and a
separate agreed cadence before launch, because state cannot be recovered
from the original evaluation-only runs. Do not silently substitute
lossy/partial retention, delete historical evidence or launch reruns before
storage is settled.
