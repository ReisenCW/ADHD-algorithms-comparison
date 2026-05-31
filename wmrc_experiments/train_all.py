from __future__ import annotations

import argparse
import itertools
import sys

from .common import TASK_NAMES
from .train import train_main


MODEL_NAMES = ["gcn", "gat", "hgnn", "hypergcn", "gt"]


COMMON_DEFAULTS = {
    "batch_size": 8,
    "monitor": "val_macro_f1",
    "feature_mode": "full",
    "label_smoothing": 0.05,
    "use_class_weight": True,
    "balance_batches": True,
}


MODEL_DEFAULTS = {
    "gcn": {
        "base": {"num_layers": 2, "heads": 2, "pe_dim": 8, "topk": 8},
        "tasks": {
            "SLD": {"lr": 0.002, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.1},
            "SLI": {"lr": 0.001, "weight_decay": 0.001, "hidden_dim": 16, "dropout": 0.2},
            "SSD": {"lr": 0.001, "weight_decay": 0.001, "hidden_dim": 16, "dropout": 0.2},
            "SSI": {"lr": 0.001, "weight_decay": 0.001, "hidden_dim": 16, "dropout": 0.2},
            "VLD": {"lr": 0.002, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.1},
            "VLI": {"lr": 0.002, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.1},
            "VSD": {"lr": 0.002, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.1},
            "VSI": {"lr": 0.003, "weight_decay": 1e-5, "hidden_dim": 32, "dropout": 0.1},
        },
    },
    "gat": {
        "base": {"num_layers": 2, "pe_dim": 8, "topk": 8},
        "tasks": {
            "SLD": {"lr": 0.001, "weight_decay": 0.001, "hidden_dim": 32, "heads": 2, "dropout": 0.3},
            "SLI": {"lr": 0.001, "weight_decay": 0.0001, "hidden_dim": 16, "heads": 1, "dropout": 0.2},
            "SSD": {"lr": 0.001, "weight_decay": 0.001, "hidden_dim": 32, "heads": 2, "dropout": 0.3},
            "SSI": {"lr": 0.001, "weight_decay": 0.001, "hidden_dim": 32, "heads": 2, "dropout": 0.3},
            "VLD": {"lr": 0.001, "weight_decay": 0.001, "hidden_dim": 32, "heads": 2, "dropout": 0.3},
            "VLI": {"lr": 0.001, "weight_decay": 0.001, "hidden_dim": 32, "heads": 2, "dropout": 0.3},
            "VSD": {"lr": 0.001, "weight_decay": 0.0001, "hidden_dim": 16, "heads": 1, "dropout": 0.2},
            "VSI": {"lr": 0.001, "weight_decay": 0.001, "hidden_dim": 32, "heads": 2, "dropout": 0.3},
        },
    },
    "hgnn": {
        "base": {"num_layers": 2, "heads": 2, "pe_dim": 8},
        "tasks": {
            "SLD": {"lr": 0.001, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.2, "topk": 4},
            "SLI": {"lr": 0.001, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.2, "topk": 4},
            "SSD": {"lr": 0.001, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.2, "topk": 4},
            "SSI": {"lr": 0.001, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.2, "topk": 4},
            "VLD": {"lr": 0.001, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.2, "topk": 4},
            "VLI": {"lr": 0.002, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.1, "topk": 8},
            "VSD": {"lr": 0.002, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.1, "topk": 8},
            "VSI": {"lr": 0.002, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.1, "topk": 8},
        },
    },
    "hypergcn": {
        "base": {"num_layers": 2, "heads": 2, "pe_dim": 8},
        "tasks": {
            "SLD": {"lr": 0.002, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.1, "topk": 8},
            "SLI": {"lr": 0.002, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.1, "topk": 8},
            "SSD": {"lr": 0.001, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.2, "topk": 4},
            "SSI": {"lr": 0.001, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.2, "topk": 4},
            "VLD": {"lr": 0.001, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.2, "topk": 4},
            "VLI": {"lr": 0.001, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.2, "topk": 4},
            "VSD": {"lr": 0.002, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.1, "topk": 8},
            "VSI": {"lr": 0.002, "weight_decay": 0.0001, "hidden_dim": 16, "dropout": 0.1, "topk": 8},
        },
    },
    "gt": {
        "base": {"heads": 2, "topk": 8},
        "tasks": {
            "SLD": {"lr": 0.0005, "weight_decay": 0.0001, "hidden_dim": 16, "num_layers": 2, "dropout": 0.3, "pe_dim": 4},
            "SLI": {"lr": 0.001, "weight_decay": 0.0001, "hidden_dim": 32, "num_layers": 1, "dropout": 0.3, "pe_dim": 8},
            "SSD": {"lr": 0.0005, "weight_decay": 0.0001, "hidden_dim": 16, "num_layers": 2, "dropout": 0.3, "pe_dim": 4},
            "SSI": {"lr": 0.0005, "weight_decay": 0.0001, "hidden_dim": 16, "num_layers": 2, "dropout": 0.3, "pe_dim": 4},
            "VLD": {"lr": 0.0005, "weight_decay": 0.0001, "hidden_dim": 16, "num_layers": 2, "dropout": 0.3, "pe_dim": 4},
            "VLI": {"lr": 0.001, "weight_decay": 0.0001, "hidden_dim": 16, "num_layers": 1, "dropout": 0.2, "pe_dim": 8},
            "VSD": {"lr": 0.0005, "weight_decay": 0.0001, "hidden_dim": 16, "num_layers": 2, "dropout": 0.3, "pe_dim": 4},
            "VSI": {"lr": 0.001, "weight_decay": 0.0001, "hidden_dim": 16, "num_layers": 1, "dropout": 0.2, "pe_dim": 8},
        },
    },
}


def build_default_config(task: str, model: str) -> dict:
    cfg = dict(COMMON_DEFAULTS)
    cfg.update(MODEL_DEFAULTS[model]["base"])
    cfg.update(MODEL_DEFAULTS[model]["tasks"][task])
    return cfg


def resolve_run_config(task: str, model: str, args: argparse.Namespace) -> dict:
    cfg = build_default_config(task, model)
    scalar_overrides = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "pe_dim": args.pe_dim,
        "topk": args.topk,
        "monitor": args.monitor,
        "feature_mode": args.feature_mode,
        "label_smoothing": args.label_smoothing,
    }
    for key, value in scalar_overrides.items():
        if value is not None:
            cfg[key] = value
    if args.use_class_weight is not None:
        cfg["use_class_weight"] = args.use_class_weight
    if args.balance_batches is not None:
        cfg["balance_batches"] = args.balance_batches
    return cfg


def train_all_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--tasks", nargs="+", default=TASK_NAMES, choices=TASK_NAMES)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--data-root", default="WMRC_general")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=None, help="override the tuned default")
    parser.add_argument("--lr", type=float, default=None, help="override the tuned default")
    parser.add_argument("--weight-decay", type=float, default=None, help="override the tuned default")
    parser.add_argument("--hidden-dim", type=int, default=None, help="override the tuned default")
    parser.add_argument("--num-layers", type=int, default=None, help="override the tuned default")
    parser.add_argument("--heads", type=int, default=None, help="override the tuned default")
    parser.add_argument("--dropout", type=float, default=None, help="override the tuned default")
    parser.add_argument("--pe-dim", type=int, default=None, help="override the tuned default")
    parser.add_argument("--topk", type=int, default=None, help="override the tuned default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--monitor", default=None, choices=["val_loss", "val_macro_f1", "val_balanced_acc"])
    use_weight = parser.add_mutually_exclusive_group()
    use_weight.add_argument("--use-class-weight", dest="use_class_weight", action="store_true")
    use_weight.add_argument("--no-class-weight", dest="use_class_weight", action="store_false")
    parser.set_defaults(use_class_weight=None)
    parser.add_argument("--feature-mode", default=None, choices=["full", "adj_only", "adj_degree", "adj_coords"])
    balance = parser.add_mutually_exclusive_group()
    balance.add_argument("--balance-batches", dest="balance_batches", action="store_true")
    balance.add_argument("--no-balance-batches", dest="balance_batches", action="store_false")
    parser.set_defaults(balance_batches=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    args = parser.parse_args(argv)

    failures = []
    total = len(args.models) * len(args.tasks)
    for idx, (task, model) in enumerate(itertools.product(args.tasks, args.models), start=1):
        print(f"\n[{idx}/{total}] training model={model} task={task}")
        cfg = resolve_run_config(task, model, args)
        train_args = [
            "--model",
            model,
            "--task",
            task,
            "--data-root",
            args.data_root,
            "--out-dir",
            args.out_dir,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(cfg["batch_size"]),
            "--lr",
            str(cfg["lr"]),
            "--weight-decay",
            str(cfg["weight_decay"]),
            "--hidden-dim",
            str(cfg["hidden_dim"]),
            "--num-layers",
            str(cfg["num_layers"]),
            "--heads",
            str(cfg["heads"]),
            "--dropout",
            str(cfg["dropout"]),
            "--pe-dim",
            str(cfg["pe_dim"]),
            "--topk",
            str(cfg["topk"]),
            "--seed",
            str(args.seed),
            "--patience",
            str(args.patience),
            "--monitor",
            cfg["monitor"],
            "--feature-mode",
            cfg["feature_mode"],
            "--label-smoothing",
            str(cfg["label_smoothing"]),
        ]
        if cfg["use_class_weight"]:
            train_args.append("--use-class-weight")
        if cfg["balance_batches"]:
            train_args.append("--balance-batches")
        if args.device:
            train_args.extend(["--device", args.device])
        if args.max_samples is not None:
            train_args.extend(["--max-samples", str(args.max_samples)])

        try:
            train_main(train_args)
        except Exception as exc:
            failures.append({"task": task, "model": model, "error": repr(exc)})
            print(f"failed model={model} task={task}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                raise

    if failures:
        print("\nfailed runs:")
        for item in failures:
            print(f"{item['task']}\t{item['model']}\t{item['error']}")
        raise SystemExit(1)


if __name__ == "__main__":
    train_all_main()
