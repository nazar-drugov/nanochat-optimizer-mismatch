"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import numpy as np
import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size, data_dir=None, shuffle_seed=None, exclude_rows=None):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and exact resume for states with a document-batch cursor.
    Older states without that cursor retain approximate row-group resume.
    Yields (text_batch, cursor): text_batch is a list of document strings and
    cursor = (pq_idx, rg_idx, epoch, batch_idx) tracks position for resumption;
    epoch counts how many times we've cycled through the dataset (starts at 1).
    With shuffle_seed set, cursor has a fifth entry: the position in this rank's
    permuted list of (file, row group) units for the current epoch.

    data_dir: directory of nanochat-format parquet shards ('text' column, last file = val);
              None = the default pretraining dataset's directory.
    shuffle_seed: None = fixed file/row-group order (pretraining default);
              int = seeded shuffled order, see _shuffled_document_batches.
    exclude_rows: optional set of row indices (0-based, over the whole file) of the
              validation file to skip, e.g. the documents that D0 took from it. Only
              for split="val" in fixed order.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    parquet_paths = list_parquet_files(data_dir=data_dir)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    if split == "train" and len(parquet_paths) < 2:
        raise ValueError(f"Need >= 2 parquet files (train shard(s), then the val shard last), found {parquet_paths}")
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    if exclude_rows is not None and (split != "val" or shuffle_seed is not None):
        raise ValueError("exclude_rows is only supported for the validation split in fixed order")
    exclude_rows = frozenset(exclude_rows or ())

    if shuffle_seed is not None:
        yield from _shuffled_document_batches(
            parquet_paths, resume_state_dict, tokenizer_batch_size, shuffle_seed, ddp_rank, ddp_world_size,
        )
        return
    if resume_state_dict is not None and resume_state_dict.get("shuffle_seed") is not None:
        raise ValueError("resume state was produced by a shuffled loader; pass the same shuffle_seed")

    resume_pq_idx = resume_state_dict.get("next_pq_idx", resume_state_dict["pq_idx"]) if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict.get("next_rg_idx", resume_state_dict["rg_idx"]) if resume_state_dict is not None else None
    resume_batch_idx = resume_state_dict.get("next_batch_idx") if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        yielded_this_pass = False
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Index of each row group's first row in the file, to match exclude_rows
            rg_offsets = [0]
            for i in range(pf.num_row_groups - 1):
                rg_offsets.append(rg_offsets[-1] + pf.metadata.row_group(i).num_rows)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                if resume_batch_idx is not None:
                    rg_idx = resume_rg_idx
                else:
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1  # legacy state skips the whole row group
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                first_batch = resume_batch_idx if resume_rg_idx == rg_idx else 0
                for batch_idx in range(first_batch, (len(batch) + tokenizer_batch_size - 1) // tokenizer_batch_size):
                    i = batch_idx * tokenizer_batch_size
                    docs = batch[i:i+tokenizer_batch_size]
                    if exclude_rows:
                        first_row = rg_offsets[rg_idx] + i
                        docs = [doc for j, doc in enumerate(docs) if first_row + j not in exclude_rows]
                        if not docs:
                            continue
                    yielded_this_pass = True
                    yield docs, (pq_idx, rg_idx, epoch, batch_idx)
                resume_rg_idx = None  # apply the saved cursor only once
                resume_batch_idx = None
                rg_idx += ddp_world_size
            pq_idx += 1
        if not first_pass and not yielded_this_pass:
            # a complete pass gave this rank nothing: fail loudly instead of spinning forever
            raise RuntimeError(f"rank {ddp_rank} got no documents from the {split} split; every rank needs "
                               f">= 1 row group in some file (world_size={ddp_world_size})")
        first_pass = False
        epoch += 1


def _shuffled_document_batches(parquet_paths, resume_state_dict, tokenizer_batch_size, shuffle_seed, ddp_rank, ddp_world_size):
    """
    Seeded, shuffled variant of the _document_batches loop (e.g. fine-tuning data orders).

    Each epoch, the list of all (pq_idx, rg_idx) units is permuted with
    np.random.default_rng([shuffle_seed, epoch]); rank r takes every world_size-th unit
    of that permutation (disjoint across ranks). Documents inside a row group are permuted
    with np.random.default_rng([shuffle_seed, epoch, pq_idx, rg_idx]). No global RNG state
    is used. Exact resume uses (epoch, next_unit_pos, next_batch_idx) from the saved state.
    """
    if isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, (int, np.integer)) or shuffle_seed < 0:
        raise ValueError(f"shuffle_seed must be a non-negative int or None, got {shuffle_seed!r}")
    shuffle_seed = int(shuffle_seed)
    rg_counts = [pq.ParquetFile(path).metadata.num_row_groups for path in parquet_paths]
    units = [(pq_idx, rg_idx) for pq_idx, count in enumerate(rg_counts) for rg_idx in range(count)]
    if len(units) < ddp_world_size:
        raise RuntimeError(f"only {len(units)} row groups for world_size={ddp_world_size}; "
                           "every rank needs >= 1 row group (write smaller row groups)")

    if resume_state_dict is not None:
        if resume_state_dict.get("shuffle_seed") != shuffle_seed or "next_unit_pos" not in resume_state_dict:
            raise ValueError(f"resume state (shuffle_seed={resume_state_dict.get('shuffle_seed')!r}) does not match "
                             f"this loader (shuffle_seed={shuffle_seed}); exact resume needs the same seed")
        epoch = resume_state_dict["epoch"]
        unit_pos = resume_state_dict["next_unit_pos"]
        first_batch = resume_state_dict["next_batch_idx"]
    else:
        epoch, unit_pos, first_batch = 1, 0, 0

    while True:  # iterate infinitely (multi-epoch)
        permutation = np.random.default_rng([shuffle_seed, epoch]).permutation(len(units))
        rank_units = permutation[ddp_rank::ddp_world_size]
        while unit_pos < len(rank_units):
            pq_idx, rg_idx = units[rank_units[unit_pos]]
            rg = pq.ParquetFile(parquet_paths[pq_idx]).read_row_group(rg_idx, columns=['text'])
            texts = rg.column('text').to_pylist()
            order = np.random.default_rng([shuffle_seed, epoch, pq_idx, rg_idx]).permutation(len(texts))
            texts = [texts[i] for i in order]
            for batch_idx in range(first_batch, (len(texts) + tokenizer_batch_size - 1) // tokenizer_batch_size):
                i = batch_idx * tokenizer_batch_size
                yield texts[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch, batch_idx, unit_pos)
            first_batch = 0
            unit_pos += 1
        unit_pos = 0
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000, data_dir=None, shuffle_seed=None, exclude_rows=None,
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping

    data_dir: directory of nanochat-format parquet shards ('text' column, last file = val);
              None = the default pretraining dataset's dir (nanochat.dataset.get_data_dir()).
    shuffle_seed: None = fixed file/row-group order (pretraining). int = seeded shuffle of the
              row-group order and of documents within each row group, re-drawn every epoch; the
              same seed (and world size) gives the same batch sequence. Its states carry
              "shuffle_seed" and "next_unit_pos" and must be resumed with the same seed.
    exclude_rows: validation-file rows to skip (see _document_batches).
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size, data_dir=data_dir,
                                shuffle_seed=shuffle_seed, exclude_rows=exclude_rows)
    bos_token = tokenizer.get_bos_token_id()
    exact_resume = resume_state_dict is not None and "doc_buffer" in resume_state_dict
    doc_buffer = [list(tokens) for tokens in resume_state_dict["doc_buffer"]] if exact_resume else []
    pq_idx, rg_idx, epoch = (resume_state_dict[k] for k in ("pq_idx", "rg_idx", "epoch")) if exact_resume else (0, 0, 1)
    next_pq_idx = resume_state_dict["next_pq_idx"] if exact_resume else 0
    next_rg_idx = resume_state_dict["next_rg_idx"] if exact_resume else 0
    next_batch_idx = resume_state_dict["next_batch_idx"] if exact_resume else 0
    next_unit_pos = resume_state_dict.get("next_unit_pos", 0) if exact_resume else 0

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch, next_pq_idx, next_rg_idx, next_batch_idx, next_unit_pos
        doc_batch, cursor = next(batches)
        pq_idx, rg_idx, epoch, batch_idx = cursor[:4]
        next_pq_idx, next_rg_idx, next_batch_idx = pq_idx, rg_idx, batch_idx + 1
        if shuffle_seed is not None:
            next_unit_pos = cursor[4]
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    if exact_resume and "pending_inputs" in resume_state_dict:
        pending_inputs = torch.tensor(resume_state_dict["pending_inputs"], dtype=torch.long)
        pending_targets = torch.tensor(resume_state_dict["pending_targets"], dtype=torch.long)
        if pending_inputs.shape != (B, T) or pending_targets.shape != (B, T):
            raise ValueError("saved pending batch has different batch or sequence dimensions")
        cpu_inputs.copy_(pending_inputs)
        cpu_targets.copy_(pending_targets)
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        pending_state = {key: value for key, value in resume_state_dict.items()
                         if key not in {"pending_inputs", "pending_targets"}}
        yield inputs, targets, pending_state

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {
            "pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch,
            "next_pq_idx": next_pq_idx, "next_rg_idx": next_rg_idx,
            "next_batch_idx": next_batch_idx,
            "doc_buffer": doc_buffer.copy(),
        }
        if shuffle_seed is not None:
            state_dict["shuffle_seed"] = int(shuffle_seed)
            state_dict["next_unit_pos"] = next_unit_pos

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
