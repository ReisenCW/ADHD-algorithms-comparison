"""评估管线CLI入口。

用法:
    python -m wmrc_experiments.evaluation --all            # 运行全部
    python -m wmrc_experiments.evaluation --predictions    # 仅生成预测
    python -m wmrc_experiments.evaluation --metrics        # 仅计算指标
    python -m wmrc_experiments.evaluation --stats          # 仅统计检验
    python -m wmrc_experiments.evaluation --viz            # 仅可视化
    python -m wmrc_experiments.evaluation --errors         # 仅错误分析
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import torch

from ..common import TASK_NAMES, WMRCGraphDataset, split_by_subject, make_loader
from ..models import build_model
from ..train_all import MODEL_NAMES

from .metrics import aggregate_metrics_by_task_and_model
from .statistical_tests import (
    friedman_test_with_nemenyi,
    generate_latex_tables,
    mcnemar_test_per_task,
    wilcoxon_signed_rank_test,
)
from .visualization import plot_all
from .error_analysis import generate_error_report


def _remap_old_state_dict(state_dict: dict, model_name: str) -> dict:
    """将旧版checkpoint的state_dict键名映射到当前模型。

    旧版GAT模型用 layer1/layer2，新版用 layers.0/layers.1。
    旧版GCN模型用 convs.0/convs.1，新版用 layers.0/layers.1。
    """
    new_sd = {}
    for key, value in state_dict.items():
        # GAT: layer1 -> layers.0, layer2 -> layers.1
        if model_name == "gat" and key.startswith("layer"):
            parts = key.split(".", 1)
            layer_num = int(parts[0].replace("layer", "")) - 1
            new_key = f"layers.{layer_num}.{parts[1]}" if len(parts) > 1 else f"layers.{layer_num}"
            new_sd[new_key] = value
        # GCN/HyperGCN: convs.0 -> layers.0
        elif key.startswith("convs."):
            parts = key.split(".", 1)
            new_key = f"layers.{parts[1]}"
            new_sd[new_key] = value
        else:
            new_sd[key] = value
    return new_sd


def generate_all_predictions(
    data_root: str,
    checkpoints_dir: str,
    output_dir: str,
    device: str = "cpu",
) -> None:
    """为所有checkpoint生成逐样本预测CSV。

    复用 predict.py 的推理逻辑。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(checkpoints_dir)
    device_obj = torch.device(device)

    for task in TASK_NAMES:
        for model in MODEL_NAMES:
            output_path = out / f"{model}_{task}.csv"
            if output_path.exists():
                print(f"  skip existing: {output_path.name}")
                continue

            ckpt_path = ckpt_dir / f"{model}_{task}.pt"
            if not ckpt_path.exists():
                print(f"  missing checkpoint: {ckpt_path}")
                continue

            print(f"  generating predictions: {model}_{task}")

            # 加载模型（处理旧版checkpoint键名不兼容）
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt["state_dict"] = _remap_old_state_dict(ckpt["state_dict"], model)
            smodel = build_model(ckpt["model_name"], **ckpt["config"]).to(device_obj)
            smodel.load_state_dict(ckpt["state_dict"])
            smodel.eval()

            # 加载数据集（使用checkpoint里的配置）
            pe_dim = ckpt.get("config", {}).get("pe_dim", 8)
            dataset = WMRCGraphDataset(data_root, task_filter=task, pe_dim=pe_dim)
            split = split_by_subject(dataset, seed=ckpt.get("seed", 42))
            all_indices = split["indices"]["train"] + split["indices"]["val"] + split["indices"]["test"]
            loader = make_loader(dataset, all_indices, batch_size=32, shuffle=False)

            rows = []
            offset = 0
            with torch.no_grad():
                for batch in loader:
                    x = batch["x"].to(device_obj)
                    adj = batch["adj"].to(device_obj)
                    pe = batch["pe"].to(device_obj)
                    comm = batch["comm"].to(device_obj)
                    mask = batch["mask"].to(device_obj)
                    logits = smodel(x, adj, pe=pe, comm=comm, mask=mask)
                    probs = torch.softmax(logits, dim=1).cpu()
                    pred = probs.argmax(dim=1)
                    for i in range(probs.size(0)):
                        idx = offset + i
                        if idx < len(all_indices):
                            record = dataset.records[all_indices[idx]]
                            rows.append({
                                "path": record.path,
                                "task": record.task,
                                "subject": record.subject,
                                "label": record.label,
                                "pred": int(pred[i].item()),
                                "prob_0": float(probs[i, 0].item()),
                                "prob_1": float(probs[i, 1].item()),
                            })
                    offset += probs.size(0)

            with output_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
                if rows:
                    writer.writeheader()
                    writer.writerows(rows)

            print(f"    wrote {len(rows)} predictions to {output_path.name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="评估管线")
    parser.add_argument("--data-root", default="WMRC_general")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="output/evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Steps
    parser.add_argument("--all", action="store_true", help="运行全部步骤")
    parser.add_argument("--predictions", action="store_true", help="生成逐样本预测CSV")
    parser.add_argument("--metrics", action="store_true", help="计算扩展指标")
    parser.add_argument("--stats", action="store_true", help="统计显著性检验")
    parser.add_argument("--viz", action="store_true", help="可视化")
    parser.add_argument("--errors", action="store_true", help="错误分析")

    args = parser.parse_args(argv)

    # 如果没有任何step指定，默认--all
    run_all = args.all or not (args.predictions or args.metrics or args.stats or args.viz or args.errors)
    run_predictions = run_all or args.predictions
    run_metrics = run_all or args.metrics
    run_stats = run_all or args.stats
    run_viz = run_all or args.viz
    run_errors = run_all or args.errors

    out = Path(args.output_dir)
    pred_dir = out / "predictions"

    # Step 1: Predictions 预测, 得到结果CSV
    if run_predictions:
        print("=" * 60)
        print("Step 1: Generating per-sample predictions")
        print("=" * 60)
        generate_all_predictions(args.data_root, args.checkpoints_dir, str(pred_dir), args.device)

    # Step 2: Metrics 计算全套指标
    metrics_df = None
    if run_metrics:
        print("\n" + "=" * 60)
        print("Step 2: Computing extended metrics")
        print("=" * 60)
        metrics_rows = aggregate_metrics_by_task_and_model(str(pred_dir))
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_out = out / "metrics"
        metrics_out.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_out / "all_metrics.csv", index=False, encoding="utf-8-sig")
        print(f"  metrics saved to {metrics_out / 'all_metrics.csv'}")
        print(metrics_df[["task", "model", "accuracy", "f1", "macro_f1", "auc_roc", "mcc"]].to_string(index=False))
    else:
        # Load existing metrics if available
        existing = out / "metrics" / "all_metrics.csv"
        if existing.exists():
            metrics_df = pd.read_csv(existing)

    # Step 3: Statistical tests 统计性检验
    cd_data = None
    if run_stats:
        print("\n" + "=" * 60)
        print("Step 3: Statistical significance tests")
        print("=" * 60)
        stats_out = out / "stats"
        stats_out.mkdir(parents=True, exist_ok=True)

        # McNemar per task
        print("  running McNemar tests...")
        for task in TASK_NAMES:
            mcnemar = mcnemar_test_per_task(str(pred_dir), task, output_dir=stats_out / "mcnemar")
            if mcnemar:
                sig_count = sum(
                    1 for ma in mcnemar for mb in mcnemar[ma]
                    if ma < mb and mcnemar[ma][mb]["significant"]
                )
                print(f"    {task}: {sig_count} significant pairs out of {len(MODEL_NAMES) * (len(MODEL_NAMES)-1) // 2}")

        # Wilcoxon per task
        print("  running Wilcoxon tests...")
        for task in TASK_NAMES:
            wilcox = wilcoxon_signed_rank_test(str(pred_dir), task, output_dir=stats_out / "wilcoxon")
            if wilcox:
                sig_count = sum(
                    1 for ma in wilcox for mb in wilcox[ma]
                    if ma < mb and wilcox[ma][mb]["significant"]
                )
                print(f"    {task}: {sig_count} significant pairs out of {len(MODEL_NAMES) * (len(MODEL_NAMES)-1) // 2}")

        # Friedman + Nemenyi
        print("  running Friedman + Nemenyi tests...")
        for metric in ["accuracy", "macro_f1", "auc_roc"]:
            friedman = friedman_test_with_nemenyi(
                str(pred_dir), metric=metric, output_dir=stats_out
            )
            if "error" not in friedman:
                print(f"    {metric}: Friedman chi2={friedman['friedman_statistic']:.3f}, "
                      f"p={friedman['friedman_p_value']:.4f}, "
                      f"{'SIGNIFICANT' if friedman['friedman_significant'] else 'not significant'}")
                print(f"      Rankings: {friedman['average_rankings']}")
                print(f"      CD = {friedman['cd_threshold']:.3f}")
                if friedman.get("nemenyi_significant_pairs"):
                    print(f"      Significant pairs: {friedman['nemenyi_significant_pairs']}")

                # Use accuracy for CD diagram
                if metric == "accuracy":
                    from .statistical_tests import critical_difference_values
                    cd_data = critical_difference_values(
                        friedman["average_rankings"],
                        n_models=len(friedman["average_rankings"]),
                        n_tasks=len(TASK_NAMES),
                    )

        # 生成Latex表格用于插入论文
        print("  generating LaTeX tables...")
        generate_latex_tables(str(pred_dir), stats_out / "latex_tables")

    # Step 4: Visualization 可视化
    if run_viz and metrics_df is not None:
        print("\n" + "=" * 60)
        print("Step 4: Visualization")
        print("=" * 60)
        fig_out = out / "figures"
        plot_all(str(pred_dir), metrics_df, str(fig_out), cd_data=cd_data)
        print(f"  figures saved to {fig_out}")

    # Step 5: Error analysis 错误分析
    if run_errors:
        print("\n" + "=" * 60)
        print("Step 5: Error analysis")
        print("=" * 60)
        errors_out = out / "error_analysis"
        report_path = generate_error_report(str(pred_dir), str(errors_out))
        print(f"  error report saved to {report_path}")

    print("\n" + "=" * 60)
    print("Evaluation pipeline complete!")
    print(f"All outputs in: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
