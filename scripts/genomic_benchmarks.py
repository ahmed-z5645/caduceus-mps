"""Caduceus Genomic Benchmarks — frozen-backbone linear probe (HUMAN VS WORM, HUMAN ENHANCERS COHN).

Loads a pretrained Caduceus checkpoint, freezes the backbone, pre-computes
mean-pooled features once over the entire dataset, then trains a small linear
classifier head with k-fold cross-validation.

This is a linear probe, not full fine-tuning. The paper's Table 1 numbers come
from full fine-tuning (freeze_backbone=false in their config). On MPS, full
fine-tune isn't tractable: autograd's backward through the sequential SSM
Python loop is ~50× slower than the forward (~10s/batch at B=64, L=200).

So we measure how separable the *frozen pretrained representations* are. This
is a meaningful research signal but **expect numbers a few percentage points
below Table 1**, especially on the harder enhancer-prediction tasks.

Recipe (head training):
    AdamW lr=1e-3 wd=0.0, cosine LR with 1% warmup, grad_clip=1.0,
    batch_size=512 (head is tiny — fit large batches for stable LR),
    k-fold CV over the combined train+test set.

Pooling:
    Caduceus-Ph: mean over L → (B, D)
    Caduceus-PS: sum h + flip(h, [-2,-1]) (RC-invariant), then mean over L → (B, 2D)

Usage:
    python scripts/genomic_benchmarks.py --task human_or_worm        --model ph
    python scripts/genomic_benchmarks.py --task human_enhancers_cohn --model ps
    python scripts/genomic_benchmarks.py --task human_or_worm        --model ph --smoke

Results are appended to runs/results.csv (one row per (task, model, fold)).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caduceus.modeling_caduceus import CaduceusForMaskedLM
from caduceus.tokenization_caduceus import CaduceusTokenizer

TASK_CFG = {
    "human_or_worm": {
        "hf_id": "katarinagresova/Genomic_Benchmarks_demo_human_or_worm",
        "seq_len": 200,
        "paper_ph": (0.973, 0.001),
        "paper_ps": (0.968, 0.002),
    },
    "human_enhancers_cohn": {
        "hf_id": "katarinagresova/Genomic_Benchmarks_human_enhancers_cohn",
        "seq_len": 500,
        "paper_ph": (0.747, 0.004),
        "paper_ps": (0.745, 0.007),
    },
}

MODEL_CFG = {
    "ph": "kuleshov-group/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3",
    "ps": "kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3",
}


def pretokenize(seqs, labels, tok, max_token_len, pad_id):
    n = len(seqs)
    out = np.full((n, max_token_len), pad_id, dtype=np.int64)
    for i, s in enumerate(seqs):
        ids = tok(s)["input_ids"]
        if len(ids) > max_token_len:
            ids = ids[:max_token_len]
        out[i, : len(ids)] = ids
    return torch.from_numpy(out), torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def compute_features(lm: CaduceusForMaskedLM, ids: torch.Tensor, *, rcps: bool, batch_size: int, device):
    """Run the frozen backbone over all sequences and return (N, feat_dim) features."""
    lm.eval()
    backbone = lm.caduceus.backbone
    feats = []
    n = ids.shape[0]
    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        chunk = ids[start : start + batch_size].to(device)
        h, _ = backbone(chunk, output_hidden_states=False)  # (B, L, D) or (B, L, 2D)
        if rcps:
            h = h + h.flip(dims=[-2, -1])  # RC-invariant
        feats.append(h.mean(dim=1).cpu())
        if (start // batch_size) % 10 == 0:
            done = min(start + batch_size, n)
            rate = done / (time.perf_counter() - t0 + 1e-9)
            eta = (n - done) / max(rate, 1e-9)
            print(f"  features: {done:>7,}/{n:,}  ({rate:>5.0f} ex/s, eta {eta:>5.1f}s)", flush=True)
    return torch.cat(feats, dim=0)


def cosine_lr(step, total, warmup, lr_max, lr_min):
    if step < warmup:
        return lr_max * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def evaluate_head(head, X, y, batch_size, device):
    head.eval()
    n_correct = n_total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = X[start : start + batch_size].to(device)
            yb = y[start : start + batch_size].to(device)
            logits = head(xb)
            loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
            n_correct += (logits.argmax(-1) == yb).sum().item()
            n_total += yb.numel()
    return n_correct / n_total, loss_sum / n_total


def train_head(
    feat_dim, num_classes, X_train, y_train, X_val, y_val, *,
    device, epochs, lr, weight_decay, batch_size, log_prefix=""
):
    head = nn.Linear(feat_dim, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = X_train.shape[0]
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    n_steps = steps_per_epoch * epochs
    n_warmup = max(int(0.01 * n_steps), 1)
    lr_min = 0.1 * lr

    best_val_acc = 0.0
    step = 0
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        perm = rng.permutation(n_train)
        head.train()
        loss_sum = 0.0
        n_seen = 0
        t0 = time.perf_counter()
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)
            for pg in opt.param_groups:
                pg["lr"] = cosine_lr(step, n_steps, n_warmup, lr, lr_min)
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * yb.numel()
            n_seen += yb.numel()
            step += 1
        train_loss = loss_sum / max(n_seen, 1)
        val_acc, val_loss = evaluate_head(head, X_val, y_val, batch_size, device)
        best_val_acc = max(best_val_acc, val_acc)
        dt = time.perf_counter() - t0
        print(
            f"{log_prefix}epoch {epoch+1}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"({dt:.1f}s)",
            flush=True,
        )
    return best_val_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASK_CFG.keys()))
    p.add_argument("--model", required=True, choices=["ph", "ps"])
    p.add_argument("--epochs", type=int, default=30, help="head training epochs (head is tiny, can afford more)")
    p.add_argument("--head-batch", type=int, default=512)
    p.add_argument("--feat-batch", type=int, default=128, help="batch size for feature extraction")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=2222)
    p.add_argument("--smoke", action="store_true",
                   help="1 fold, 3 epochs, 1024 examples — for sanity-checking the pipeline")
    p.add_argument("--results-csv", default="runs/results.csv")
    p.add_argument("--feat-cache-dir", default="runs/feature_cache",
                   help="features are cached per (task, model, seed) so re-running fold sweeps is cheap")
    args = p.parse_args()

    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.feat_cache_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = TASK_CFG[args.task]
    model_id = MODEL_CFG[args.model]
    paper_mean, paper_std = cfg[f"paper_{args.model}"]

    print(f"task={args.task} model={args.model} folds={args.folds} epochs={args.epochs} "
          f"head_batch={args.head_batch} feat_batch={args.feat_batch} lr={args.lr} device={device} smoke={args.smoke}")
    print(f"paper Table 1 (full fine-tune): {paper_mean:.3f} ± {paper_std:.3f}")
    print(f"  (we are running a frozen-backbone linear probe — expect lower numbers)")

    # Load + combine train+test for k-fold CV (matches the paper's CV protocol)
    ds = load_dataset(cfg["hf_id"])
    all_seqs = list(ds["train"]["seq"]) + list(ds["test"]["seq"])
    all_labels = list(ds["train"]["label"]) + list(ds["test"]["label"])
    print(f"loaded {len(all_seqs):,} sequences, seq_len={cfg['seq_len']}")

    if args.smoke:
        all_seqs, all_labels = all_seqs[:1024], all_labels[:1024]

    cache_key = f"{args.task}_{args.model}_smoke{int(args.smoke)}"
    cache_X = Path(args.feat_cache_dir) / f"{cache_key}_X.pt"
    cache_y = Path(args.feat_cache_dir) / f"{cache_key}_y.pt"

    if cache_X.exists() and cache_y.exists():
        print(f"cache hit: loading features from {cache_X}")
        X = torch.load(cache_X)
        y = torch.load(cache_y)
    else:
        # Pretokenize. seq_len + 1 to accommodate the SEP token the tokenizer appends.
        tok = CaduceusTokenizer(model_max_length=cfg["seq_len"] + 1)
        pad_id = tok._vocab_str_to_int["[PAD]"]
        t0 = time.perf_counter()
        all_ids, y = pretokenize(all_seqs, all_labels, tok, cfg["seq_len"] + 1, pad_id)
        print(f"pretokenized in {time.perf_counter()-t0:.1f}s, ids shape={tuple(all_ids.shape)}")

        # Compute features once (frozen backbone)
        print(f"loading {model_id}...")
        lm = CaduceusForMaskedLM.from_pretrained(model_id, fused_add_norm=False).to(device)
        for p_ in lm.parameters():
            p_.requires_grad_(False)
        X = compute_features(lm, all_ids, rcps=(args.model == "ps"), batch_size=args.feat_batch, device=device)
        print(f"features shape: {tuple(X.shape)}")
        torch.save(X, cache_X)
        torch.save(y, cache_y)
        del lm
        if device.type == "mps":
            torch.mps.empty_cache()

    feat_dim = X.shape[1]

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_accs = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        if args.smoke and fold_idx > 0:
            break
        log_prefix = f"[{args.task} {args.model.upper()} fold {fold_idx+1}/{args.folds}] "
        torch.manual_seed(args.seed + fold_idx)
        epochs = 3 if args.smoke else args.epochs
        best_acc = train_head(
            feat_dim, num_classes=2,
            X_train=X[train_idx], y_train=y[train_idx],
            X_val=X[val_idx], y_val=y[val_idx],
            device=device, epochs=epochs, lr=args.lr,
            weight_decay=args.weight_decay, batch_size=args.head_batch,
            log_prefix=log_prefix,
        )
        fold_accs.append(best_acc)

        write_header = not Path(args.results_csv).exists() or Path(args.results_csv).stat().st_size == 0
        with open(args.results_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["task", "model", "fold", "best_val_acc", "epochs", "lr", "head_batch", "seed", "frozen_backbone"])
            w.writerow([args.task, args.model, fold_idx, f"{best_acc:.6f}",
                        epochs, args.lr, args.head_batch, args.seed, True])

    if not args.smoke:
        accs = np.array(fold_accs)
        print(f"\n=== {args.task} / Caduceus-{args.model.upper()} (frozen backbone, linear probe) ===")
        print(f"  ours:  {accs.mean():.4f} ± {accs.std():.4f}  per-fold={[f'{a:.4f}' for a in accs]}")
        print(f"  paper: {paper_mean:.4f} ± {paper_std:.4f}  (full fine-tune)")
        print(f"  delta: {(accs.mean() - paper_mean)*100:+.2f} pp")


if __name__ == "__main__":
    main()
