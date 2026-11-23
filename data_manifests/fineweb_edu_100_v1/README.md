# FineWeb-Edu 100-shard data snapshot

This manifest describes the 100 training shards `shard_00000.parquet` through
`shard_00099.parquet`, the validation shard `shard_01822.parquet`, and the D0
files built from them on 2026-11-23. D0 contains 1,881 documents and 2,000,142
GPT-2 tokens; its report records a near-duplicate scan of all 100 training
shards. The selected tokenizer is GPT-2 through `tiktoken` (50,257 tokens).

- Canonical shards: `/home/ndrugov/orcd/pool/nanochat-optimizer-mismatch/base_data_fineweb_edu_100b/`
- Staged training shards: `/home/ndrugov/orcd/scratch/public/nanochat-optimizer-mismatch/base_data_fineweb_edu_100b/`
- D0 and report: `/home/ndrugov/orcd/scratch/public/nanochat-optimizer-mismatch/heldout/fineweb-edu-100b/`
- `shards.sha256`: SHA-256 of each staged shard, using relative filenames.
- `shard_sizes.tsv`: filename and size in bytes for each staged shard.
- `heldout.sha256`: SHA-256 of D0 and its construction report, using relative filenames.

Verify the bytes in scratch before starting or resuming a run:

```bash
MANIFEST="$HOME/nanochat-optimizer-mismatch/data_manifests/fineweb_edu_100_v1"
DATA="$HOME/orcd/scratch/public/nanochat-optimizer-mismatch/base_data_fineweb_edu_100b"
D0="$HOME/orcd/scratch/public/nanochat-optimizer-mismatch/heldout/fineweb-edu-100b"
(cd "$DATA" && sha256sum -c "$MANIFEST/shards.sha256")
(cd "$D0" && sha256sum -c "$MANIFEST/heldout.sha256")
(cd "$DATA" && diff -u <(awk '{print $2}' "$MANIFEST/shards.sha256") <(find . -maxdepth 1 -type f -name 'shard_*.parquet' -printf '%f\n' | LC_ALL=C sort))
```

The last command detects added or removed shards; SHA-256 checks detect changed
bytes. To verify the pool copy, replace `DATA` with the canonical shard path
and repeat the first and third commands. Hashing the full set reads about 9.5 GB
per copy. These checks detect changes; the manifest does not make files
physically immutable. If any shard changes or more are added, build a new D0
against that exact set and create a new manifest version.

`runs/pretrain_study.sh train` makes these same checks automatically before
every start and every resume: the Parquet files in the scratch data folder must
match `shards.sha256` exactly, every shard's SHA-256 must match, and D0 and its
report must match `heldout.sha256`. Hashing all shards took about 26 seconds
on a CPU node from a cold start. Any difference stops the run before it begins.
The launcher then passes D0 to `base_train.py` for validation. On resume,
`base_train.py` also compares the shard names, order and sizes saved in the
checkpoint with the files it finds. You only need the commands above to check
the pool copy or to check by hand.
