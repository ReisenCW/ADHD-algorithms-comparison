from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .common import TASK_NAMES
from .train import train_main
from .train_all import MODEL_NAMES


BASE_CONFIG = {
    "batch_size": 8,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "hidden_dim": 16,
    "num_layers": 2,
    "heads": 2,
    "dropout": 0.1,
    "pe_dim": 8,
    "rw_dim": 16,
    "topk": 8,
    "monitor": "val_macro_f1",
    "feature_mode": "full",
    "label_smoothing": 0.05,
    "use_class_weight": True,
    "balance_batches": True,
}


MODEL_CANDIDATES = {
    "gcn": [
        {},
        {"hidden_dim": 32, "dropout": 0.1, "lr": 3e-3, "weight_decay": 1e-5},
        {"hidden_dim": 16, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-3},
    ],
    "gat": [
        {},
        {"hidden_dim": 16, "heads": 1, "dropout": 0.2, "lr": 1e-3},
        {"hidden_dim": 32, "heads": 2, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-3},
    ],
    "hgnn": [
        {},
        {"hidden_dim": 16, "topk": 4, "dropout": 0.2, "lr": 1e-3},
        {"hidden_dim": 32, "topk": 12, "dropout": 0.1, "lr": 2e-3},
    ],
    "hypergcn": [
        {},
        {"hidden_dim": 16, "topk": 4, "dropout": 0.2, "lr": 1e-3},
        {"hidden_dim": 32, "topk": 8, "dropout": 0.1, "lr": 2e-3},
    ],
    "gt": [
        {"hidden_dim": 32, "num_layers": 2, "heads": 4, "dropout": 0.1, "lr": 3e-3, "pe_dim": 8},
        {"hidden_dim": 32, "num_layers": 3, "heads": 4, "dropout": 0.15, "lr": 2e-3, "pe_dim": 8},
        {"hidden_dim": 64, "num_layers": 2, "heads": 4, "dropout": 0.15, "lr": 2e-3, "pe_dim": 8},
        {"hidden_dim": 32, "num_layers": 3, "heads": 2, "dropout": 0.1, "lr": 3e-3, "pe_dim": 8},
        {"hidden_dim": 16, "num_layers": 2, "heads": 2, "dropout": 0.2, "lr": 1e-3, "pe_dim": 8},
    ],
}


def merged_config(model: str, candidate_idx: int) -> dict:
    cfg = dict(BASE_CONFIG)
    cfg.update(MODEL_CANDIDATES[model][candidate_idx])
    return cfg


def config_to_args(config: dict) -> list[str]:
    args = [
        "--batch-size",
        str(config["batch_size"]),
        "--lr",
        str(config["lr"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--num-layers",
        str(config["num_layers"]),
        "--heads",
        str(config["heads"]),
        "--dropout",
        str(config["dropout"]),
        "--pe-dim",
        str(config["pe_dim"]),
        "--rw-dim",
        str(config.get("rw_dim", 16)),
        "--topk",
        str(config["topk"]),
        "--monitor",
        config["monitor"],
        "--feature-mode",
        config["feature_mode"],
        "--label-smoothing",
        str(config["label_smoothing"]),
        "--cosine-schedule",
    ]
    if config["use_class_weight"]:
        args.append("--use-class-weight")
    if config["balance_batches"]:
        args.append("--balance-batches")
    return args


def score_summary(summary: dict, metric: str) -> float:
    if metric == "best_score":
        return float(summary["best_score"])
    return float(summary["test"][metric])


def tune_all_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--tasks", nargs="+", default=TASK_NAMES, choices=TASK_NAMES)
    parser.add_argument("--data-root", default="WMRC_general")
    parser.add_argument("--out-dir", default="tuning_runs")
    parser.add_argument("--results-csv", default="tuning_runs/tuning_results.csv")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--select-metric", default="best_score", choices=["best_score", "macro_f1", "balanced_acc", "acc", "f1"])
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)

    rows = []
    failures = []
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        for model in args.models:
            candidates = MODEL_CANDIDATES[model]
            if args.max_candidates is not None:
                candidates = candidates[: args.max_candidates]
            for cand_idx in range(len(candidates)):
                cfg = merged_config(model, cand_idx)
                run_name = f"{task}_{model}_c{cand_idx}"
                run_dir = out_root / run_name
                train_args = [
                    "--model",
                    model,
                    "--task",
                    task,
                    "--data-root",
                    args.data_root,
                    "--out-dir",
                    str(run_dir),
                    "--epochs",
                    str(args.epochs),
                    "--patience",
                    str(args.patience),
                    "--seed",
                    str(args.seed),
                    *config_to_args(cfg),
                ]
                if args.device:
                    train_args.extend(["--device", args.device])
                if args.max_samples is not None:
                    train_args.extend(["--max-samples", str(args.max_samples)])

                print(f"\n[tune] task={task} model={model} candidate={cand_idx} cfg={cfg}")
                try:
                    train_main(train_args)
                    summary_path = run_dir / f"{model}_{task}_summary.json"
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    row = {
                        "task": task,
                        "model": model,
                        "candidate": cand_idx,
                        "run_dir": str(run_dir),
                        "best_epoch": summary["best_epoch"],
                        "best_score": summary["best_score"],
                        "monitor": summary["monitor"],
                        "test_loss": summary["test"]["loss"],
                        "test_acc": summary["test"]["acc"],
                        "test_f1": summary["test"]["f1"],
                        "test_macro_f1": summary["test"]["macro_f1"],
                        "test_balanced_acc": summary["test"]["balanced_acc"],
                        "config": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
                    }
                    rows.append(row)
                except Exception as exc:
                    failures.append({"task": task, "model": model, "candidate": cand_idx, "error": repr(exc)})
                    print(f"[failed] task={task} model={model} candidate={cand_idx}: {exc}")
                    if not args.continue_on_error:
                        raise

    results_path = Path(args.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with results_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if failures:
        fail_path = results_path.with_name(results_path.stem + "_failures.json")
        fail_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(1)

    print(f"wrote tuning results to {results_path}")


if __name__ == "__main__":
    tune_all_main()
