"""
Fine-tuning datasets (EXPERIMENT_SPEC.md §6) in nanochat's parquet layout.

- near: WikiText-103, raw version (`Salesforce/wikitext`, `wikitext-103-raw-v1`),
  grouped into articles and detokenized. Held-out D1 = the official validation split.
- far:  codeparrot-clean (`codeparrot/codeparrot-clean`) for training and
  codeparrot-clean-valid (`codeparrot/codeparrot-clean-valid`) as held-out D1
  (spec's recommended option B: Python only, not gated, used by Hernanz).

Output directory: train_00000.parquet, train_00001.parquet, ... and val_00000.parquet
(the last file in sorted order is the validation split, as nanochat's loader expects),
each with a `text` column, plus manifest.json (sources, document and byte counts,
SHA-256 of every file, preprocessing version).
"""

import json
import re
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.train_config import file_sha256, write_json_atomic

DETOKENIZER_VERSION = "wikitext-detok-v1"
HF_PARQUET_API = "https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}"
FINETUNE_SOURCES = {
    "wikitext103": {
        "train": ("Salesforce/wikitext", "wikitext-103-raw-v1", "train", "text"),
        "val": ("Salesforce/wikitext", "wikitext-103-raw-v1", "validation", "text"),
    },
    "codeparrot": {
        "train": ("codeparrot/codeparrot-clean", "default", "train", "content"),
        "val": ("codeparrot/codeparrot-clean-valid", "default", "train", "content"),
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


def download(url, path):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as resp, open(tmp, "wb") as out:
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


def prepare_codeparrot(out_dir, raw_dir, max_train_files=2):
    """Download the first `max_train_files` parquet files of codeparrot-clean (training)
    and all of codeparrot-clean-valid (held-out); write shards."""
    out = {}
    for split in ("train", "val"):
        repo, config, hf_split, column = FINETUNE_SOURCES["codeparrot"][split]
        urls = parquet_urls(repo, config, hf_split)
        if split == "train":
            urls = urls[:max_train_files]
        docs = []
        for i, url in enumerate(urls):
            docs += pq.read_table(download(url, Path(raw_dir) / f"{repo.replace('/', '_')}_{i}.parquet"),
                                  columns=[column]).column(column).to_pylist()
        write_shards(docs, out_dir, "train" if split == "train" else "val",
                     docs_per_shard=10**9 if split == "val" else 50_000)
        out[split] = {"source": f"{repo}/{config}/{hf_split}", "parquet_files": len(urls), "documents": len(docs),
                      "characters": sum(map(len, docs))}
    return write_manifest(out_dir, {"dataset": "codeparrot", "preprocessing": "none", "splits": out})
