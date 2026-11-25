"""
Why does the fine-tuning evaluation report a higher base-model D0 loss than pretraining?

Evaluates one base model on D0 two ways, on the same GPU:
  A  pretraining: best-fit packer (`tokenizing_distributed_data_loader_bos_bestfit`, as
     base_train's held-out evaluation), 16 batches of 32 x 2048; every row starts with a BOS,
     documents are never split across rows, long documents are cropped.
  B  fine-tuning: `nanochat.eval_set.build_eval_set` (contiguous packing in file order with a
     BOS before each document); a document that crosses a row boundary loses its earlier
     context in the next row.
and breaks the difference down: B's tokens by whether their document started in the same row
(context intact) or in an earlier row (context cut), B by position in the row, and A and B on
the (document, position) pairs both score, which removes the difference in which tokens are scored.

  python -m dev.d0_eval_gap --base-run e8-main-adamw-s42 --d0-file <d0.parquet> --out gap.json
"""

import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
import torch

from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.common import autodetect_device_type, compute_init, get_base_dir
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.eval_set import build_eval_set
from nanochat.tokenizer import get_token_bytes

KEY = 24  # tokens after BOS that identify a document


@torch.no_grad()
def row_losses(model, x, y):
    return model(x, y, loss_reduction="none").float().view(y.shape)


def doc_keys(tokenizer, d0_file):
    texts = pq.read_table(d0_file, columns=["text"]).column("text").to_pylist()
    bos = tokenizer.get_bos_token_id()
    keys = {}
    for i, tokens in enumerate(tokenizer.encode(texts, prepend=bos)):
        keys.setdefault(tuple(tokens[1:1 + KEY]), i)
    return keys


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--base-checkpoints-dir", default=None)
    parser.add_argument("--d0-file", required=True)
    parser.add_argument("--tokens", type=int, default=1_048_576)
    parser.add_argument("--smoke-tokens", type=int, default=235_520, help="B on only this prefix (the smoke test)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type, seed=0)
    ckpt_dir = Path(args.base_checkpoints_dir or Path(get_base_dir()) / "base_checkpoints") / args.base_run
    model, tokenizer, meta = build_model(str(ckpt_dir), find_last_step(str(ckpt_dir)), device, phase="eval")
    model.eval()
    T = meta["max_seq_len"]
    bos = tokenizer.get_bos_token_id()
    token_bytes = get_token_bytes(device=device, name="gpt2")
    keys = doc_keys(tokenizer, args.d0_file)

    # A: best-fit packer, exactly as base_train's held-out evaluation
    loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.batch_size, T, split="val",
                                                            device=device, data_dir=str(Path(args.d0_file).parent))
    steps = args.tokens // (args.batch_size * T)
    a_nats = a_bytes = 0.0
    a_scored = []           # per scored token: loss
    a_by_doc_pos = {}       # (doc, position in doc) -> loss
    a_unmatched = 0
    for _ in range(steps):
        x, y = next(loader)
        loss = row_losses(model, x, y)
        nbytes = token_bytes[y]
        scored = nbytes > 0
        a_nats += float((loss * scored).sum())
        a_bytes += float(nbytes.sum())
        a_scored += loss[scored].tolist()
        xs, ls, sc = x.cpu().tolist(), loss.cpu().tolist(), scored.cpu().tolist()
        for r in range(len(xs)):
            doc, start = None, None
            for t in range(T):
                if xs[r][t] == bos:
                    key = tuple(xs[r][t + 1:t + 1 + KEY])
                    doc, start = keys.get(key), t
                    if doc is None:
                        a_unmatched += 1
                if sc[r][t] and doc is not None:
                    a_by_doc_pos[(doc, t - start)] = ls[r][t]  # position of the target within the document

    # B: the fine-tuning evaluation set (contiguous packing, file order)
    es = build_eval_set("D0", args.d0_file, tokenizer, T, args.tokens)
    b_loss = torch.cat([row_losses(model, es.inputs[i:i + args.batch_size].to(device),
                                   es.targets[i:i + args.batch_size].to(device)).cpu()
                        for i in range(0, es.inputs.size(0), args.batch_size)])
    ids = es.doc_ids
    scored = ids >= 0
    # position of each target within its document, counted across rows; whether its document
    # started (BOS) earlier in the same row
    flat_ids = ids.reshape(-1).tolist()
    flat_x = es.inputs.reshape(-1).tolist()
    pos_in_doc, intact = [], []
    cur, p, row_has_start = None, 0, False
    for k, (doc, xin) in enumerate(zip(flat_ids, flat_x)):
        if k % T == 0:
            row_has_start = False
        if xin == bos:
            cur, p, row_has_start = doc, 0, True
        elif doc != cur:  # cannot happen: a document only starts after its BOS
            cur, p = doc, 0
        pos_in_doc.append(p)
        intact.append(row_has_start)
        p += 1
    pos_in_doc = torch.tensor(pos_in_doc).view(ids.shape)
    intact = torch.tensor(intact).view(ids.shape)
    col = torch.arange(T).expand_as(ids)

    def b_mean(mask):
        m = mask & scored
        return float(b_loss[m].mean()), int(m.sum())

    b_all, n_all = b_mean(torch.ones_like(scored))
    b_bytes = float(token_bytes.cpu()[es.targets][scored].sum())
    result = {
        "base_run": args.base_run,
        "A_bestfit": {"nats_per_token": mean(a_scored), "scored_tokens": len(a_scored),
                      "bpb": a_nats / math.log(2) / a_bytes, "bytes_per_token": a_bytes / len(a_scored),
                      "unmatched_bos": a_unmatched},
        "B_contiguous": {"nats_per_token": b_all, "scored_tokens": n_all, "documents": es.n_docs,
                         "bpb": float(b_loss[scored].sum()) / math.log(2) / b_bytes,
                         "bytes_per_token": b_bytes / n_all},
        "B_smoke_prefix": dict(zip(("nats_per_token", "scored_tokens"),
                                   b_mean(torch.arange(ids.size(0)).unsqueeze(1).expand_as(ids)
                                          < args.smoke_tokens // T))),
        "B_context_intact": dict(zip(("nats_per_token", "scored_tokens"), b_mean(intact))),
        "B_context_cut": dict(zip(("nats_per_token", "scored_tokens"), b_mean(~intact))),
        "B_by_position_in_row": {f"{lo}-{hi - 1}": dict(zip(("nats_per_token", "scored_tokens"),
                                                             b_mean((col >= lo) & (col < hi))))
                                 for lo, hi in ((0, 16), (16, 128), (128, 512), (512, 1024), (1024, T))},
        "B_by_position_in_doc": {f"{lo}-{hi - 1}": dict(zip(("nats_per_token", "scored_tokens"),
                                                             b_mean((pos_in_doc >= lo) & (pos_in_doc < hi))))
                                 for lo, hi in ((0, 16), (16, 128), (128, 512), (512, 2048), (2048, 10**9))},
    }
    # Same (document, position) pairs under both packings
    common_a, common_b, common_b_intact, common_b_cut = [], [], [], []
    for doc, p_, l_, s_, it in zip(flat_ids, pos_in_doc.reshape(-1).tolist(), b_loss.reshape(-1).tolist(),
                                   scored.reshape(-1).tolist(), intact.reshape(-1).tolist()):
        if s_ and (doc, p_) in a_by_doc_pos:
            common_a.append(a_by_doc_pos[(doc, p_)])
            common_b.append(l_)
            (common_b_intact if it else common_b_cut).append((a_by_doc_pos[(doc, p_)], l_))
    result["common_pairs"] = {
        "pairs": len(common_a), "A_nats": mean(common_a), "B_nats": mean(common_b),
        "intact": {"pairs": len(common_b_intact), "A_nats": mean([a for a, _ in common_b_intact]),
                   "B_nats": mean([b for _, b in common_b_intact])},
        "cut": {"pairs": len(common_b_cut), "A_nats": mean([a for a, _ in common_b_cut]),
                "B_nats": mean([b for _, b in common_b_cut])},
        "A_documents": len({d for d, _ in a_by_doc_pos}), "B_documents": es.n_docs,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
