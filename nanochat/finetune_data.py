"""
Fine-tuning datasets (EXPERIMENT_SPEC.md §6) in nanochat's parquet layout.

- near: WikiText-103, raw version (`Salesforce/wikitext`, `wikitext-103-raw-v1`),
  grouped into articles and detokenized. Held-out D1 = the official validation split.
- far:  StarCoderData (`bigcode/starcoderdata`, gated: needs a Hugging Face token whose account
  accepted the terms), one language directory (default `python`). Training = the first
  parquet file(s) of that directory; held-out D1 = a seeded random sample of documents from
  its last file, which training never reads. StarCoderData has no validation split; it is
  near-deduplicated, so documents in different files are distinct. The sample is shuffled
  because D1 is scored from the start of the file (`build_eval_set`), and StarCoderData's
  files are not in random order.

Output directory: train_00000.parquet, train_00001.parquet, ... and val_00000.parquet
(the last file in sorted order is the validation split, as nanochat's loader expects),
each with a `text` column, plus manifest.json (sources, document and byte counts,
SHA-256 of every file, preprocessing version).
"""

import json
import os
import random
import re
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.train_config import file_sha256, write_json_atomic

DETOKENIZER_VERSION = "wikitext-detok-v1"
HF_PARQUET_API = "https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}"
HF_TREE_API = "https://huggingface.co/api/datasets/{repo}/tree/main/{path}"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
STARCODER_REPO = "bigcode/starcoderdata"
FINETUNE_SOURCES = {
    "wikitext103": {
        "train": ("Salesforce/wikitext", "wikitext-103-raw-v1", "train", "text"),
        "val": ("Salesforce/wikitext", "wikitext-103-raw-v1", "validation", "text"),
    },
}


# ---------------------------------------------------------------------------
# WikiText-103

ARTICLE_TITLE = re.compile(r"^ = [^=].* = $")  # " = Title = " (subsections use " = = ... = = ")


def wikitext_articles(lines):
    """Group raw WikiText lines into articles. A new article starts at each top-level
    title line; blank lines are dropped; text before the first title is discarded."""
    articles, current = [], None
    for line in lines:
        if ARTICLE_TITLE.match(line.rstrip("\n")):
            if current:
                articles.append(current)
            current = [line.strip()]
        elif current is not None and line.strip():
            current.append(line.strip())
    if current:
        articles.append(current)
    return ["\n".join(article) for article in articles]


def detokenize_wikitext(text):
    """Undo WikiText's tokenization artifacts (after the Megatron-LM wikitext detokenizer):
    @-@ / @,@ / @.@ number separators, spaces before punctuation and inside brackets and
    quotes, spaced heading markers and possessives."""
    s = text
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    for punct in (":", ";", ".", "!", "?", ","):
        s = s.replace(f" {punct} ", f"{punct} ")
    s = re.sub(r" ([.,;:!?])$", r"\1", s, flags=re.MULTILINE)
    s = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", s)
    s = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", s)
    s = re.sub(r"\{\s*([^}]*?)\s*\}", r"{\1}", s)
    s = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', s)
    s = s.replace("= = = =", "====").replace("= = =", "===").replace("= =", "==")
    s = s.replace(" \N{DEGREE SIGN} ", "\N{DEGREE SIGN}")
    s = s.replace(" 's", "'s").replace("s ' ", "s' ").replace(" n't", "n't")
    s = re.sub(r" +\n", "\n", s)
    s = re.sub(r"\n +", "\n", s)
    return s


# ---------------------------------------------------------------------------
# Download and write

def parquet_urls(repo, config, split):
    """URLs of the Hugging Face auto-converted parquet files of a dataset split."""
    with urllib.request.urlopen(HF_PARQUET_API.format(repo=repo, config=config, split=split), timeout=60) as resp:
        urls = json.load(resp)
    if not urls:
        raise RuntimeError(f"no parquet files listed for {repo}/{config}/{split}")
    return sorted(urls)


def hf_token():
    """Hugging Face token from $HF_TOKEN or ~/.cache/huggingface/token (None if neither exists)."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    return path.read_text().strip() if path.exists() else None


def hf_request(url, token=None):
    """A request that sends the token to huggingface.co only: it is added as an unredirected
    header, so the redirect to the storage CDN (a signed URL) does not carry it."""
    req = urllib.request.Request(url)
    if token:
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    return req


def download(url, path, token=None):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(hf_request(url, token), timeout=600) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 22):
            out.write(chunk)
    tmp.replace(path)
    return path


def write_shards(docs, out_dir, prefix, docs_per_shard=50_000, row_group_size=1024):
    """Write documents to <prefix>_00000.parquet, ... with a `text` column."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, max(len(docs), 1), docs_per_shard):
        path = out_dir / f"{prefix}_{len(paths):05d}.parquet"
        pq.write_table(pa.table({"text": docs[start:start + docs_per_shard]}), path, row_group_size=row_group_size)
        paths.append(path)
    return paths


def write_manifest(out_dir, info):
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("*.parquet"))
    info = {**info, "files": {f.name: {"sha256": file_sha256(f), "rows": pq.ParquetFile(f).metadata.num_rows}
                              for f in files}}
    write_json_atomic(out_dir / "manifest.json", info)
    return info


def prepare_wikitext103(out_dir, raw_dir):
    """Download WikiText-103 raw, group into articles, detokenize, write shards."""
    out = {}
    for split in ("train", "val"):
        repo, config, hf_split, column = FINETUNE_SOURCES["wikitext103"][split]
        lines = []
        for i, url in enumerate(parquet_urls(repo, config, hf_split)):
            lines += pq.read_table(download(url, Path(raw_dir) / f"wikitext103_{hf_split}_{i}.parquet"),
                                   columns=[column]).column(column).to_pylist()
        docs = [detokenize_wikitext(article) for article in wikitext_articles(lines)]
        write_shards(docs, out_dir, "train" if split == "train" else "val",
                     docs_per_shard=10**9 if split == "val" else 50_000)
        out[split] = {"source": f"{repo}/{config}/{hf_split}", "documents": len(docs),
                      "characters": sum(map(len, docs))}
    return write_manifest(out_dir, {"dataset": "wikitext103", "preprocessing": DETOKENIZER_VERSION, "splits": out})


def starcoder_files(language, token):
    """Sorted parquet paths of one StarCoderData language directory, e.g. python/train-00000-of-00059.parquet."""
    with urllib.request.urlopen(hf_request(HF_TREE_API.format(repo=STARCODER_REPO, path=language), token),
                                timeout=60) as resp:
        entries = json.load(resp)
    files = sorted(e["path"] for e in entries if e.get("type") == "file" and e["path"].endswith(".parquet"))
    if len(files) < 2:
        raise RuntimeError(f"{STARCODER_REPO}/{language}: need at least 2 parquet files (training + held-out), "
                           f"found {len(files)}")
    return files


def sample_heldout(docs, n_docs, seed):
    """A seeded random sample of n_docs documents, in random order."""
    if n_docs > len(docs):
        raise ValueError(f"held-out file has {len(docs)} documents; {n_docs} requested")
    return random.Random(seed).sample(docs, n_docs)


def prepare_starcoderdata(out_dir, raw_dir, language="python", train_files=1, val_docs=20_000, seed=0):
    """Training = the first `train_files` parquet files of one StarCoderData language;
    held-out D1 = `val_docs` documents sampled (seeded) from its last file."""
    token = hf_token()
    if not token:
        raise RuntimeError(f"{STARCODER_REPO} is gated: set HF_TOKEN or write ~/.cache/huggingface/token")
    files = starcoder_files(language, token)
    if train_files < 1 or train_files >= len(files):
        raise ValueError(f"train_files must be in [1, {len(files) - 1}]")
    plan = {"train": files[:train_files], "val": files[-1:]}
    out = {}
    for split, paths in plan.items():
        docs = []
        for path in paths:
            local = download(HF_RESOLVE.format(repo=STARCODER_REPO, path=path),
                             Path(raw_dir) / f"starcoderdata_{path.replace('/', '_')}", token)
            docs += pq.read_table(local, columns=["content"]).column("content").to_pylist()
        total = len(docs)
        if split == "val":
            docs = sample_heldout(docs, val_docs, seed)
        write_shards(docs, out_dir, split, docs_per_shard=10**9 if split == "val" else 50_000)
        out[split] = {"source": [f"{STARCODER_REPO}/{p}" for p in paths], "documents_in_source": total,
                      "documents": len(docs), "characters": sum(map(len, docs))}
    return write_manifest(out_dir, {"dataset": "starcoderdata", "language": language,
                                    "preprocessing": f"none; D1 = {val_docs} documents sampled with seed {seed}",
                                    "splits": out})
