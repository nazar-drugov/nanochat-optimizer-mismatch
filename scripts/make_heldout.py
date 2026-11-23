"""
Build the held-out pretraining set D0 (spec: "Before any training: set aside a held-out
D0 of about 2M tokens. Deduplicate it against the training sample with MinHash, and
record the document IDs.")

Procedure
1. Candidates come from the dataset's validation shard (the last shard; the pretraining
   loaders never train on it). Rows are visited in a seeded random order
   (np.random.default_rng(seed).permutation) and taken until the candidates hold
   num_tokens * (1 + oversample) tokens, so that ~num_tokens remain after dedup.
   Tokens are counted with the nanochat tokenizer (<base_dir>/tokenizer, no BOS) when it
   exists; otherwise (or with --token-count chars) as ceil(len(text) / 4) characters, a
   rough English-text estimate. The method used is recorded in the report.
2. Near-duplicate detection (nanochat/dedup.py, numpy only):
   - shingles: word 5-grams after lowercasing and replacing non-alphanumeric runs by spaces
   - MinHash: 128 multiply-shift hash functions (seed fixed by --minhash-seed)
   - LSH: 16 bands x 8 rows. The S-curve midpoint is (1/16)^(1/8) = 0.71, so pairs at the
     0.8 threshold become candidates with probability 1-(1-0.8^8)^16 = 0.947 (0.9999 at 0.9);
     every LSH candidate pair is then verified: estimated Jaccard (fraction of equal
     signature entries) >= 0.8 means near-duplicate.
   - exact prefilter: a training doc A can only reach Jaccard 0.8 with a held-out doc if
     >= 80% of A's distinct shingles occur among all held-out shingles; docs below 70%
     (threshold - 0.1 margin) are skipped before MinHashing. This is what makes a full scan
     affordable; --no-prefilter disables it (several times slower).
   Held-out candidates that are near-duplicates of an earlier candidate are dropped too.
3. Every training shard (all downloaded train shards, or the first --max-train-shards in
   the order the loader reads them) is scanned in parallel (multiprocessing, one shard per
   task). Candidates with a near-duplicate in training are dropped.
4. Kept candidates, in sampled order, are taken until >= num_tokens tokens.

Outputs (in <base_dir>/heldout/<dataset>/ unless --output-dir):
- d0.parquet: columns text, source_shard (val shard filename), row_index (row in that
  shard). The document ID is (source_shard, row_index).
- d0_report.json: parameters, docs/tokens kept, dropped docs (with the training doc that
  matched), and the list of training shards scanned.

Cost: hashing dominates and is linear in the training data scanned. On a laptop core
(Apple M-series) one FineWeb-Edu shard (~94MB parquet, ~0.06B tokens) takes roughly
10-15 s with the prefilter, so the full ~1822-shard FineWeb-Edu-100B sample is ~6-8
CPU-hours: ~15-30 minutes on a 16-32 core node (more on slower cluster cores / network
filesystems). Memory is ~1-2 GB per worker. Run it as a CPU job on the cluster, e.g.
    sbatch -c 32 --mem=64G --wrap "python -m scripts.make_heldout --dataset fineweb-edu-100b --workers 32"
Make sure every training shard that will be used is downloaded first; with
--max-train-shards K only the first K train shards are checked (recorded in the report).

Example:
    python -m scripts.make_heldout --dataset fineweb-edu-100b --num-tokens 2000000 --seed 0
"""

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir
from nanochat.dataset import DATASETS, DEFAULT_DATASET, get_data_dir, list_parquet_files
from nanochat import dedup

CHARS_PER_TOKEN = 4.0


def make_token_counter(method="auto", num_threads=8, chars_per_token=CHARS_PER_TOKEN, tokenizer_name="gpt2"):
    """
    Return (count_fn(list[str]) -> list[int], method_used).
    method: "tokenizer" (the named tokenizer, no BOS), "chars" (ceil(len / chars_per_token)),
    or "auto" (the tokenizer if available, else chars). tokenizer_name: "gpt2" (tiktoken)
    or "nanochat" (<base_dir>/tokenizer).
    """
    if method == "auto":
        if tokenizer_name == "gpt2":
            method = "tokenizer"
        else:
            tok_path = os.path.join(get_base_dir(), "tokenizer", "tokenizer.pkl")
            method = "tokenizer" if os.path.exists(tok_path) else "chars"
            if method == "chars":
                print(f"WARNING: no tokenizer at {tok_path}; counting tokens as ceil(chars / {chars_per_token:g})")
    if method == "tokenizer":
        from nanochat.tokenizer import get_tokenizer
        tokenizer = get_tokenizer(tokenizer_name)
        return (lambda texts: [len(ids) for ids in tokenizer.encode(list(texts), num_threads=num_threads)]), f"tokenizer:{tokenizer_name}"
    if method == "chars":
        return (lambda texts: [math.ceil(len(t) / chars_per_token) for t in texts]), f"chars/{chars_per_token:g}"
    raise ValueError(f"unknown token count method {method!r}")


def sample_candidates(val_texts, target_tokens, count_tokens, seed, chunk=256):
    """Seeded random walk over val rows; returns (row_indices, token_counts) until target reached."""
    order = np.random.default_rng(seed).permutation(len(val_texts))
    rows, tokens, total = [], [], 0
    for start in range(0, len(order), chunk):
        idx = [int(i) for i in order[start:start + chunk]]
        texts = [val_texts[i] or "" for i in idx]
        keep = [(i, t) for i, t in zip(idx, texts) if t.strip()]
        if not keep:
            continue
        counts = count_tokens([t for _, t in keep])
        for (i, _), n in zip(keep, counts):
            rows.append(i)
            tokens.append(int(n))
            total += n
            if total >= target_tokens:
                return rows, tokens
    return rows, tokens


# -----------------------------------------------------------------------------
# Training-shard scan (worker side)

_INDEX = None

def _init_worker(index):
    global _INDEX
    _INDEX = index


def scan_shard(path):
    """Find near-duplicates of indexed (held-out) docs inside one training shard."""
    t0 = time.time()
    pf = pq.ParquetFile(path)
    row_offset, docs, candidates, matches = 0, 0, 0, []
    for rg_idx in range(pf.num_row_groups):
        texts = pf.read_row_group(rg_idx, columns=["text"]).column("text")
        found, stats = _INDEX.query_texts(texts)
        docs += stats["docs"]
        candidates += stats["candidates"]
        matches.extend((heldout_idx, row_offset + q, est) for q, heldout_idx, est in found)
        row_offset += len(texts)
    return {"shard": os.path.basename(path), "docs": docs, "prefilter_candidates": candidates,
            "matches": matches, "seconds": time.time() - t0}


def scan_training_shards(index, paths, workers):
    """Scan shards in parallel; returns per-shard results in input order."""
    results = {}
    t0 = time.time()
    if workers <= 1:
        _init_worker(index)
        iterator = map(scan_shard, paths)
        pool = None
    else:
        pool = Pool(processes=workers, initializer=_init_worker, initargs=(index,))
        iterator = pool.imap_unordered(scan_shard, paths)
    try:
        for k, res in enumerate(iterator, 1):
            results[res["shard"]] = res
            elapsed = time.time() - t0
            eta = elapsed / k * (len(paths) - k)
            print(f"[{k}/{len(paths)}] {res['shard']}: {res['docs']} docs, {res['prefilter_candidates']} prefilter "
                  f"candidates, {len(res['matches'])} matches, {res['seconds']:.1f}s | elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m",
                  flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return [results[os.path.basename(p)] for p in paths]


# -----------------------------------------------------------------------------

def build_heldout(data_dir, output_dir, dataset_label, num_tokens=2_000_000, seed=0, oversample=0.15,
                  max_train_shards=None, workers=None, token_count="auto", prefilter=True, tokenizer_name="gpt2",
                  minhash_seed=1, num_perm=dedup.NUM_PERM, bands=dedup.LSH_BANDS, rows=dedup.LSH_ROWS,
                  threshold=dedup.JACCARD_THRESHOLD, shingle_n=dedup.SHINGLE_N):
    t_start = time.time()
    paths = list_parquet_files(data_dir=data_dir)
    if len(paths) < 2:
        raise ValueError(f"need >= 2 shards in {data_dir} (train shards + val shard last), found {len(paths)}")
    val_path, train_paths = paths[-1], paths[:-1]
    if max_train_shards is not None:
        train_paths = train_paths[:max_train_shards]
    val_name = os.path.basename(val_path)
    print(f"Val shard (candidate pool): {val_path}")
    print(f"Training shards to scan: {len(train_paths)}")

    # 1) seeded candidate sample from the val shard
    count_tokens, count_method = make_token_counter(token_count, tokenizer_name=tokenizer_name)
    val_texts = pq.read_table(val_path, columns=["text"]).column("text").to_pylist()
    target = int(num_tokens * (1 + oversample))
    cand_rows, cand_tokens = sample_candidates(val_texts, target, count_tokens, seed)
    cand_texts = [val_texts[i] for i in cand_rows]
    del val_texts
    print(f"Sampled {len(cand_rows)} candidate docs, {sum(cand_tokens):,} tokens ({count_method})")

    # 2) MinHash/LSH index of the candidates, internal near-duplicates
    index = dedup.NearDupIndex(cand_texts, shingle_n=shingle_n, num_perm=num_perm, bands=bands, rows=rows,
                               threshold=threshold, seed=minhash_seed,
                               prefilter_margin=dedup.PREFILTER_MARGIN if prefilter else None)
    too_short = set(int(i) for i in np.flatnonzero(index.shingle_counts == 0))
    internal = index.internal_duplicates()
    internal_drop = {}
    for i, j, est in internal:
        internal_drop.setdefault(i, (j, est))

    # 3) scan training shards
    workers = workers or os.cpu_count() or 1
    shard_results = scan_training_shards(index, train_paths, workers) if train_paths else []
    train_dup = {}
    for res in shard_results:
        for heldout_idx, row, est in res["matches"]:
            prev = train_dup.get(heldout_idx)
            if prev is None or est > prev["est_jaccard"]:
                train_dup[heldout_idx] = {"train_shard": res["shard"], "train_row_index": int(row), "est_jaccard": est}

    # 4) select kept docs in sampled order until num_tokens
    kept, kept_tokens = [], 0
    for k in range(len(cand_rows)):
        if kept_tokens >= num_tokens:
            break
        if k in too_short or k in internal_drop or k in train_dup:
            continue
        kept.append(k)
        kept_tokens += cand_tokens[k]
    if kept_tokens < num_tokens:
        print(f"WARNING: only {kept_tokens:,} tokens survived dedup (< {num_tokens:,}); raise --oversample")

    os.makedirs(output_dir, exist_ok=True)
    table = pa.table({
        "text": pa.array([cand_texts[k] for k in kept], type=pa.string()),
        "source_shard": pa.array([val_name] * len(kept), type=pa.string()),
        "row_index": pa.array([cand_rows[k] for k in kept], type=pa.int64()),
    })
    out_parquet = os.path.join(output_dir, "d0.parquet")
    pq.write_table(table, out_parquet, row_group_size=128)

    doc_id = lambda k: {"source_shard": val_name, "row_index": int(cand_rows[k])}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_label,
        "data_dir": os.path.abspath(data_dir),
        "output": os.path.abspath(out_parquet),
        "params": {
            "num_tokens_target": num_tokens, "seed": seed, "oversample": oversample,
            "token_count_method": count_method,
            "shingles": f"word {shingle_n}-grams (lowercased, non-alphanumeric runs -> space)",
            "num_perm": num_perm, "lsh_bands": bands, "lsh_rows": rows,
            "lsh_s_curve_midpoint": dedup.lsh_threshold(bands, rows),
            "lsh_candidate_prob_at_threshold": dedup.lsh_collision_probability(threshold, bands, rows),
            "jaccard_threshold": threshold, "minhash_seed": minhash_seed,
            "prefilter": prefilter, "prefilter_margin": dedup.PREFILTER_MARGIN if prefilter else None,
            "max_train_shards": max_train_shards, "workers": workers,
        },
        "candidate_pool_shard": val_name,
        "candidates": {"docs": len(cand_rows), "tokens": int(sum(cand_tokens))},
        "kept": {"docs": len(kept), "tokens": int(kept_tokens)},
        "dropped": {
            "near_dup_in_training": len(train_dup),
            "near_dup_within_heldout": len(internal_drop),
            "too_short_for_shingles": len(too_short),
        },
        "dropped_near_dup_in_training": [
            {**doc_id(k), **v} for k, v in sorted(train_dup.items())
        ],
        "dropped_near_dup_within_heldout": [
            {**doc_id(i), "duplicate_of": doc_id(j), "est_jaccard": est} for i, (j, est) in sorted(internal_drop.items())
        ],
        "train_shards_scanned": {
            "count": len(shard_results),
            "docs": int(sum(r["docs"] for r in shard_results)),
            "prefilter_candidates": int(sum(r["prefilter_candidates"] for r in shard_results)),
            "cpu_seconds": round(sum(r["seconds"] for r in shard_results), 1),
            "files": [r["shard"] for r in shard_results],
        },
        "wall_seconds": round(time.time() - t_start, 1),
    }
    out_report = os.path.join(output_dir, "d0_report.json")
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Kept {len(kept)} docs / {kept_tokens:,} tokens; dropped {len(train_dup)} (training near-dups), "
          f"{len(internal_drop)} (internal near-dups), {len(too_short)} (too short)")
    print(f"Wrote {out_parquet} and {out_report}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the deduplicated held-out pretraining set D0")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=sorted(DATASETS), help=f"pretraining dataset (default: {DEFAULT_DATASET})")
    parser.add_argument("--data-dir", type=str, default=None, help="override the dataset's shard directory")
    parser.add_argument("--output-dir", type=str, default=None, help="default: <base_dir>/heldout/<dataset>")
    parser.add_argument("--num-tokens", type=int, default=2_000_000, help="target D0 size in tokens (default: 2M)")
    parser.add_argument("--seed", type=int, default=0, help="seed for sampling candidate docs from the val shard")
    parser.add_argument("--oversample", type=float, default=0.15, help="sample this fraction extra to absorb dedup drops")
    parser.add_argument("--token-count", type=str, default="auto", choices=["auto", "tokenizer", "chars"],
                        help=f"auto = nanochat tokenizer if present else chars/{CHARS_PER_TOKEN:g}")
    parser.add_argument("--tokenizer", type=str, default="gpt2", choices=["gpt2", "nanochat"], help="tokenizer for counting D0 tokens (default: gpt2)")
    parser.add_argument("--max-train-shards", type=int, default=None, help="scan only the first K train shards (default: all downloaded)")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes (default: all CPUs)")
    parser.add_argument("--no-prefilter", action="store_true", help="MinHash every training doc (slower; same results up to MinHash noise)")
    parser.add_argument("--minhash-seed", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=dedup.JACCARD_THRESHOLD)
    args = parser.parse_args(argv)

    data_dir = args.data_dir or get_data_dir(args.dataset)
    output_dir = args.output_dir or os.path.join(get_base_dir(), "heldout", args.dataset)
    return build_heldout(
        data_dir, output_dir, dataset_label=args.dataset, num_tokens=args.num_tokens, seed=args.seed,
        oversample=args.oversample, max_train_shards=args.max_train_shards, workers=args.workers,
        token_count=args.token_count, prefilter=not args.no_prefilter, tokenizer_name=args.tokenizer, minhash_seed=args.minhash_seed,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
