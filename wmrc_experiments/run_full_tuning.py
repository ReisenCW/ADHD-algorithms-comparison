from __future__ import annotations

import argparse

from .summarize_tuning_best import summarize_best_main
from .tune_all import tune_all_main


def run_full_tuning_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="WMRC_general")
    parser.add_argument("--out-dir", default="tuning_runs")
    parser.add_argument("--results-csv", default="tuning_runs/tuning_results.csv")
    parser.add_argument("--best-out-dir", default="tuning_runs/best_tables")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)

    tune_args = [
        "--data-root",
        args.data_root,
        "--out-dir",
        args.out_dir,
        "--results-csv",
        args.results_csv,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--continue-on-error",
    ]
    if args.device:
        tune_args.extend(["--device", args.device])
    if args.max_samples is not None:
        tune_args.extend(["--max-samples", str(args.max_samples)])
    if args.continue_on_error:
        tune_args.append("--continue-on-error")

    tune_all_main(tune_args)
    summarize_best_main(["--results-csv", args.results_csv, "--out-dir", args.best_out_dir])


if __name__ == "__main__":
    run_full_tuning_main()

