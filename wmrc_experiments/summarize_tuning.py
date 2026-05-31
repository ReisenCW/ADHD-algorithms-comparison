from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("-inf")


def summarize_tuning_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", default="tuning_runs/tuning_results.csv")
    parser.add_argument("--out-csv", default="tuning_runs/best_by_model_task.csv")
    parser.add_argument("--select-column", default="best_score")
    args = parser.parse_args(argv)

    with Path(args.results_csv).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    best = {}
    for row in rows:
        key = (row["task"], row["model"])
        score = parse_float(row[args.select_column])
        if key not in best or score > parse_float(best[key][args.select_column]):
            best[key] = row

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected = list(best.values())
    if selected:
        with out_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
            writer.writeheader()
            writer.writerows(sorted(selected, key=lambda r: (r["task"], r["model"])))

    print(f"wrote {len(selected)} best rows to {out_path}")


if __name__ == "__main__":
    summarize_tuning_main()

