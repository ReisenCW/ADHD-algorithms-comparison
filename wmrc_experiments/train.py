from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .common import (
    WMRCGraphDataset,
    accuracy_from_logits,
    balanced_accuracy_from_logits,
    collate_graphs,
    f1_from_logits,
    macro_f1_from_logits,
    make_loader,
    save_predictions_csv,
    split_by_subject,
)
from .models import build_model


def run_epoch(model, loader, criterion, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_count = 0
    all_logits = []
    all_targets = []

    for batch in loader:
        x = batch["x"].to(device)
        adj = batch["adj"].to(device)
        pe = batch["pe"].to(device)
        rws = batch["rws"].to(device)
        comm = batch["comm"].to(device)
        mask = batch["mask"].to(device)
        y = batch["y"].to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x, adj, pe=pe, rws=rws, comm=comm, mask=mask)
        loss = criterion(logits, y)
        if train_mode:
            loss.backward()
            optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_count += bs
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = {
        "loss": total_loss / max(1, total_count),
        "acc": accuracy_from_logits(logits, targets),
        "f1": f1_from_logits(logits, targets),
        "macro_f1": macro_f1_from_logits(logits, targets),
        "balanced_acc": balanced_accuracy_from_logits(logits, targets),
    }
    return metrics


def train_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="WMRC_general")
    parser.add_argument("--task", default="all", help="one of SLD/SLI/SSD/SSI/VLD/VLI/VSD/VSI or all")
    parser.add_argument("--model", required=True, choices=["gcn", "gat", "hgnn", "hypergcn", "gt"])
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pe-dim", type=int, default=8)
    parser.add_argument("--rw-dim", type=int, default=16, help="Random Walk SE dimension (0=disable)")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--monitor", default="val_loss", choices=["val_loss", "val_macro_f1", "val_balanced_acc"])
    parser.add_argument("--use-class-weight", action="store_true")
    parser.add_argument("--feature-mode", default="full", choices=["full", "adj_only", "adj_degree", "adj_coords"])
    parser.add_argument("--balance-batches", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--cosine-schedule", action="store_true", help="Use cosine LR with warmup")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

    dataset = WMRCGraphDataset(
        args.data_root,
        task_filter=args.task,
        max_samples=args.max_samples,
        pe_dim=args.pe_dim,
        rw_dim=args.rw_dim,
        feature_mode=args.feature_mode,
    )
    split = split_by_subject(dataset, seed=args.seed)
    if args.balance_batches:
        train_subset = Subset(dataset, split["indices"]["train"])
        train_labels = [dataset.records[i].label for i in split["indices"]["train"]]
        class_counts = torch.bincount(torch.tensor(train_labels), minlength=2).float()
        weights = torch.tensor([1.0 / class_counts[label] for label in train_labels], dtype=torch.float)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_graphs)
    else:
        train_loader = make_loader(dataset, split["indices"]["train"], args.batch_size, shuffle=True)
    val_loader = make_loader(dataset, split["indices"]["val"], args.batch_size, shuffle=False)
    test_loader = make_loader(dataset, split["indices"]["test"], args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = build_model(
        args.model,
        in_dim=dataset[0].x.size(1),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        pe_dim=args.pe_dim,
        topk=args.topk,
        rw_dim=args.rw_dim,
    ).to(device)

    class_weight = None
    if args.use_class_weight:
        train_labels = [dataset.records[i].label for i in split["indices"]["train"]]
        counts = torch.bincount(torch.tensor(train_labels), minlength=2).float()
        class_weight = counts.sum() / (2.0 * counts.clamp_min(1.0))
        class_weight = class_weight.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine LR scheduler with warmup (GPS paper recommendation)
    scheduler = None
    if args.cosine_schedule:
        def _cosine_schedule(epoch: int) -> float:
            if epoch < args.warmup_epochs:
                return (epoch + 1) / max(1, args.warmup_epochs)
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_cosine_schedule)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{args.model}_{args.task}.pt"
    best_score = float("inf") if args.monitor == "val_loss" else float("-inf")
    best_epoch = 0
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        if scheduler is not None:
            scheduler.step()
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        score = {
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_acc": val_metrics["balanced_acc"],
        }[args.monitor]
        improved = score < best_score - 1e-6 if args.monitor == "val_loss" else score > best_score + 1e-6

        if improved:
            best_score = score
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "model_name": args.model,
                    "task": args.task,
                    "state_dict": model.state_dict(),
                    "config": {
                        "in_dim": dataset[0].x.size(1),
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "heads": args.heads,
                        "dropout": args.dropout,
                        "pe_dim": args.pe_dim,
                        "rw_dim": args.rw_dim,
                        "topk": args.topk,
                    },
                    "split": split,
                    "seed": args.seed,
                },
                ckpt_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

        print(
            f"epoch {epoch:03d} "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} f1={train_metrics['f1']:.4f} "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} f1={val_metrics['f1']:.4f}"
        )

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, device, optimizer=None)

    summary = {
        "model": args.model,
        "task": args.task,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "monitor": args.monitor,
        "test": test_metrics,
        "num_samples": len(dataset),
        "num_train": len(split["indices"]["train"]),
        "num_val": len(split["indices"]["val"]),
        "num_test": len(split["indices"]["test"]),
        "checkpoint": str(ckpt_path),
    }
    with (out_dir / f"{args.model}_{args.task}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    train_main()
