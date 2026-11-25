"""
Fixed evaluation token sets with document IDs (EXPERIMENT_SPEC.md §7, §9).

Every model in the study is scored on exactly the same tokens: documents are taken
in file order, each is prefixed with BOS and appended to one token stream, and the
stream is cut into rows of seq_len inputs (targets shifted by one). Each scored target
position records the index of the document it belongs to, so per-document loss sums
can be bootstrapped. Targets that are a document's BOS are not scored (doc_id -1).
"""

from dataclasses import dataclass

import pyarrow.parquet as pq
import torch

from nanochat.train_config import file_sha256


@dataclass
class EvalSet:
    name: str
    inputs: torch.Tensor    # (rows, seq_len) int64
    targets: torch.Tensor   # (rows, seq_len) int64
    doc_ids: torch.Tensor   # (rows, seq_len) int64, -1 = not scored
    n_docs: int
    source_file: str
    source_sha256: str

    @property
    def n_scored(self):
        return int((self.doc_ids >= 0).sum())

    def batches(self, batch_size):
        for start in range(0, self.inputs.size(0), batch_size):
            yield start, self.inputs[start:start + batch_size], self.targets[start:start + batch_size]

    def save(self, path):
        torch.save({k: getattr(self, k) for k in ("name", "inputs", "targets", "doc_ids", "n_docs",
                                                  "source_file", "source_sha256")}, path)

    @classmethod
    def load(cls, path):
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


def build_eval_set(name, parquet_file, tokenizer, seq_len, max_tokens, text_column="text"):
    """Build a fixed set of floor(max_tokens / seq_len) rows from the start of `parquet_file`."""
    n_rows = max_tokens // seq_len
    if n_rows < 1:
        raise ValueError(f"max_tokens={max_tokens} is smaller than one row of {seq_len}")
    need = n_rows * seq_len + 1
    bos = tokenizer.get_bos_token_id()
    stream, owner = [], []
    pf = pq.ParquetFile(parquet_file)
    doc = 0
    for rg in range(pf.num_row_groups):
        texts = pf.read_row_group(rg, columns=[text_column]).column(text_column).to_pylist()
        for tokens in tokenizer.encode(texts, prepend=bos):
            stream += tokens
            owner += [-1] + [doc] * (len(tokens) - 1)  # BOS is never a scored target
            doc += 1
            if len(stream) >= need:
                break
        if len(stream) >= need:
            break
    if len(stream) < need:
        raise ValueError(f"{parquet_file} has only {len(stream)} tokens; {need} needed")
    stream = torch.tensor(stream[:need], dtype=torch.long)
    owner = torch.tensor(owner[:need], dtype=torch.long)
    inputs = stream[:-1].view(n_rows, seq_len)
    targets = stream[1:].view(n_rows, seq_len)
    doc_ids = owner[1:].view(n_rows, seq_len)
    n_docs = int(doc_ids.max()) + 1
    return EvalSet(name, inputs, targets, doc_ids, n_docs, str(parquet_file), file_sha256(parquet_file))


def per_doc_sums(token_losses, doc_ids, n_docs):
    """Sum of per-token losses and number of scored tokens per document (float64)."""
    mask = doc_ids >= 0
    ids = doc_ids[mask]
    sums = torch.zeros(n_docs, dtype=torch.float64).index_add_(0, ids, token_losses[mask].double())
    counts = torch.zeros(n_docs, dtype=torch.float64).index_add_(0, ids, torch.ones_like(ids, dtype=torch.float64))
    return sums, counts
