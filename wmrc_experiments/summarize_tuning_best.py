from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TASK_NAMES = ["SLD", "SLI", "SSD", "SSI", "VLD", "VLI", "VSD", "VSI"]
MODEL_NAMES = ["gcn", "gat", "hgnn", "hypergcn", "gt"]


METRICS = ["best_score", "test_macro_f1", "test_balanced_acc", "test_acc", "test_loss"]
CONFIG_KEYS = [
    "batch_size",
    "lr",
    "weight_decay",
    "hidden_dim",
    "num_layers",
    "heads",
    "dropout",
    "pe_dim",
    "topk",
    "monitor",
    "feature_mode",
    "label_smoothing",
    "use_class_weight",
    "balance_batches",
]


def load_rows(results_csv: str | Path) -> list[dict]:
    with Path(results_csv).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_config(row: dict) -> dict:
    raw = row.get("config", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def fmt_value(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        text = f"{value:.6g}"
        return text
    return str(value)


def config_cell(row: dict) -> str:
    cfg = parse_config(row)
    parts = [f"c{row['candidate']}"]
    for key in CONFIG_KEYS:
        if key in cfg:
            parts.append(f"{key}={fmt_value(cfg[key])}")
    return " | ".join(parts)


def summarize_best_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", default="tuning_runs/tuning_results.csv")
    parser.add_argument("--out-dir", default="tuning_runs/best_tables")
    args = parser.parse_args(argv)

    rows = load_rows(args.results_csv)
    best = {}
    for row in rows:
        key = (row["task"], row["model"])
        if key not in best or float(row["best_score"]) > float(best[key]["best_score"]):
            best[key] = row

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        out_path = out_dir / f"{metric}.csv"
        with out_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["task", *MODEL_NAMES])
            for task in TASK_NAMES:
                row = [task]
                for model in MODEL_NAMES:
                    item = best.get((task, model))
                    row.append("" if item is None else item[metric])
                writer.writerow(row)

    rows_out = out_dir / "best_rows.csv"
    selected = []
    for task in TASK_NAMES:
        for model in MODEL_NAMES:
            item = best.get((task, model))
            if item is None:
                continue
            cfg = parse_config(item)
            flat = {
                "task": task,
                "model": model,
                "candidate": item["candidate"],
                "best_score": item["best_score"],
                "monitor": item["monitor"],
                "test_loss": item["test_loss"],
                "test_acc": item["test_acc"],
                "test_f1": item["test_f1"],
                "test_macro_f1": item["test_macro_f1"],
                "test_balanced_acc": item["test_balanced_acc"],
                "config": item.get("config", ""),
            }
            for key in CONFIG_KEYS:
                if key in cfg:
                    flat[key] = fmt_value(cfg[key])
            selected.append(flat)

    if selected:
        fieldnames = list(selected[0].keys())
        with rows_out.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)

    config_table = out_dir / "best_configs.csv"
    with config_table.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["task", *MODEL_NAMES])
        for task in TASK_NAMES:
            row = [task]
            for model in MODEL_NAMES:
                item = best.get((task, model))
                row.append("" if item is None else config_cell(item))
            writer.writerow(row)

    print(f"wrote best tables to {out_dir}")


if __name__ == "__main__":
    summarize_best_main()
