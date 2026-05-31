from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .common import TASK_NAMES
from .train_all import MODEL_NAMES


METRICS = ["loss", "acc", "f1"]


def load_summaries(summary_dir: str | Path) -> dict[tuple[str, str], dict]:
    root = Path(summary_dir)
    summaries = {}
    for path in sorted(root.glob("*_*_summary.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        model = data.get("model")
        task = data.get("task")
        if model and task:
            summaries[(task, model)] = data
    return summaries


def build_metric_table(summaries: dict[tuple[str, str], dict], metric: str) -> list[dict[str, str]]:
    rows = []
    for task in TASK_NAMES:
        row = {"task": task}
        for model in MODEL_NAMES:
            item = summaries.get((task, model))
            value = None
            if item:
                value = item.get("test", {}).get(metric)
            row[model] = "" if value is None else f"{float(value):.6f}"
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["task", *MODEL_NAMES])
        writer.writeheader()
        writer.writerows(rows)


def print_markdown_table(title: str, rows: list[dict[str, str]]) -> None:
    headers = ["task", *MODEL_NAMES]
    print(f"\n## {title}")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row[h] for h in headers) + " |")


def summarize_results_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", default="checkpoints")
    parser.add_argument("--out-dir", default="output/results_tables")
    parser.add_argument("--print", action="store_true", help="also print markdown tables to stdout")
    args = parser.parse_args(argv)

    summaries = load_summaries(args.summary_dir)
    out_dir = Path(args.out_dir)
    for metric in METRICS:
        rows = build_metric_table(summaries, metric)
        write_csv(rows, out_dir / f"{metric}_table.csv")
        if args.print:
            print_markdown_table(metric, rows)

    print(f"loaded {len(summaries)} summaries from {args.summary_dir}")
    print(f"wrote tables to {out_dir}")


if __name__ == "__main__":
    summarize_results_main()

