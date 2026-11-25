"""
Fixed evaluation token sets with document IDs (EXPERIMENT_SPEC.md §7, §9).

Every model in the study is scored on exactly the same tokens. Each scored target position
records the document it belongs to, so per-document loss sums can be bootstrapped. Targets
that are a document's BOS are not scored (doc_id -1).

Packing (decided 2026-11-25, see LOG_CHANGES "D0 loss gap explained"):
  bestfit     (default) the rows the pretraining loader builds from the same file
              (`tokenizing_distributed_data_loader_bos_bestfit`, fixed order): every row starts
              with a BOS, documents are never split across rows (the largest document that fits
              is placed; when none fits, the shortest is cropped to fill the row and its rest
              is dropped), so every scored token sees its document from the start. Documents
              are read once (no second pass), so no document is scored twice. A base model's
              D0 loss matches its pretraining held-out evaluation.
  contiguous  documents in file order on one stream cut every seq_len tokens (the original
              version): a document crossing a row boundary loses its earlier context.
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
    packing: str = "contiguous"
    doc_rows: torch.Tensor = None  # (n_docs,) file row of each document ID (bestfit)

    @property
    def n_scored(self):
        return int((self.doc_ids >= 0).sum())

    def batches(self, batch_size):
        for start in range(0, self.inputs.size(0), batch_size):
            yield start, self.inputs[start:start + batch_size], self.targets[start:start + batch_size]

    def save(self, path):
        torch.save({k: getattr(self, k) for k in ("name", "inputs", "targets", "doc_ids", "n_docs",
                                                  "source_file", "source_sha256", "packing", "doc_rows")}, path)

    @classmethod
    def load(cls, path):
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


def document_chunks(parquet_file, tokenizer, text_column="text", chunk_size=128):
    """Tokenized documents (BOS first) in file order, in the chunks the pretraining loader
    reads: each row group split into chunks of `chunk_size` documents. Yields (first file
    row, list of token lists)."""
    bos = tokenizer.get_bos_token_id()
    pf = pq.ParquetFile(parquet_file)
    row = 0
    for rg in range(pf.num_row_groups):
        texts = pf.read_row_group(rg, columns=[text_column]).column(text_column).to_pylist()
        for i in range(0, len(texts), chunk_size):
            yield row + i, tokenizer.encode(texts[i:i + chunk_size], prepend=bos)
        row += len(texts)


def bestfit_rows(chunks, n_rows, row_capacity, buffer_size=1000):
    """The pretraining loader's best-fit packing over `chunks` (from document_chunks), read once.
    Returns rows (lists of row_capacity tokens) and, per token, the file row of its document."""
    buffer, rows, owners = [], [], []
    for _ in range(n_rows):
        row, owner = [], []
        while len(row) < row_capacity:
            while len(buffer) < buffer_size:
                chunk = next(chunks, None)
                if chunk is None:
                    break
                first, token_lists = chunk
                buffer += [(first + j, tokens) for j, tokens in enumerate(token_lists)]
            if not buffer:
                raise ValueError(f"not enough documents for {n_rows} best-fit rows ({len(rows)} built)")
            remaining = row_capacity - len(row)
            best, best_len = -1, 0
            for i, (_, tokens) in enumerate(buffer):
                if best_len < len(tokens) <= remaining:
                    best, best_len = i, len(tokens)
            if best >= 0:
                doc, tokens = buffer.pop(best)
            else:  # crop the shortest document to fill the row
                doc, tokens = buffer.pop(min(range(len(buffer)), key=lambda i: len(buffer[i][1])))
                tokens = tokens[:remaining]
            row += tokens
            owner += [doc] * len(tokens)
        rows.append(row)
        owners.append(owner)
    return rows, owners


def bestfit_capacity(doc_lengths, row_capacity, buffer_size=1000, chunk_size=128, row_group_size=1024):
    """Number of complete best-fit rows (as bestfit_rows builds them) that documents of these
    lengths (BOS included), in this order and written with this parquet row-group size, yield
    when read once. The packing depends on the lengths and the read chunks only."""
    lengths = list(doc_lengths)
    chunks = [lengths[i:min(i + chunk_size, g + row_group_size)]
              for g in range(0, len(lengths), row_group_size)
              for i in range(g, min(g + row_group_size, len(lengths)), chunk_size)]
    buffer, rows, next_chunk = [], 0, 0
    while True:
        filled = 0
        while filled < row_capacity:
            while len(buffer) < buffer_size and next_chunk < len(chunks):
                buffer += chunks[next_chunk]
                next_chunk += 1
            if not buffer:
                return rows
            remaining = row_capacity - filled
            best = max((n for n in buffer if n <= remaining), default=None)
            if best is not None:
                buffer.remove(best)
                filled += best
            else:
                buffer.remove(min(buffer))
                filled = row_capacity
        rows += 1


def build_eval_set(name, parquet_file, tokenizer, seq_len, max_tokens, text_column="text", packing="bestfit",
                   buffer_size=1000):
    """Build a fixed set of floor(max_tokens / seq_len) rows from `parquet_file` (see the module
    docstring for the two packings)."""
    n_rows = max_tokens // seq_len
    if n_rows < 1:
        raise ValueError(f"max_tokens={max_tokens} is smaller than one row of {seq_len}")
    if packing == "contiguous":
        return _build_contiguous(name, parquet_file, tokenizer, seq_len, n_rows, text_column)
    if packing != "bestfit":
        raise ValueError(f"unknown packing {packing!r}")
    rows, owners = bestfit_rows(document_chunks(parquet_file, tokenizer, text_column), n_rows, seq_len + 1,
                                buffer_size)
    rows = torch.tensor(rows, dtype=torch.long)
    owners = torch.tensor(owners, dtype=torch.long)
    # a target is scored unless it is a BOS (the first token of a document in the row)
    is_bos = torch.zeros_like(owners, dtype=torch.bool)
    is_bos[:, 0] = True
    is_bos[:, 1:] = owners[:, 1:] != owners[:, :-1]
    file_rows = owners[:, 1:].contiguous()
    used = torch.unique(file_rows)  # sorted file rows -> compact document IDs in file order
    compact = torch.searchsorted(used, file_rows)
    doc_ids = torch.where(is_bos[:, 1:], torch.full_like(compact, -1), compact)
    return EvalSet(name, rows[:, :-1].contiguous(), rows[:, 1:].contiguous(), doc_ids, int(used.numel()),
                   str(parquet_file), file_sha256(parquet_file), packing="bestfit", doc_rows=used)


def _build_contiguous(name, parquet_file, tokenizer, seq_len, n_rows, text_column):
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
