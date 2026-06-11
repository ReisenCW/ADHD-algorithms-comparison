"""统计显著性检验。

包含：
- McNemar检验（配对二分类比较）
- Friedman检验 + Nemenyi事后检验（多模型全局比较）
- Wilcoxon符号秩检验（配对比较）
- Critical Difference 图数据
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import cohen_kappa_score

from ..common import TASK_NAMES

MODEL_NAMES = ["gcn", "gat", "hgnn", "hypergcn", "gt"]


def _load_prediction_rows(predictions_dir: str | Path, task: str, model: str) -> list[dict]:
    pred_dir = Path(predictions_dir)
    csv_path = pred_dir / f"{model}_{task}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"prediction file not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def mcnemar_test_per_task(
    predictions_dir: str | Path,
    task: str,
    output_dir: str | Path | None = None,
) -> dict:
    """对指定任务下所有模型组合做McNemar检验。

    McNemar检验统计量：
        χ² = (|b - c| - 1)² / (b + c)   (连续性校正)
    其中 b = 模型A对且模型B错的样本数, c = 模型A错且模型B对的样本数。

    Args:
        predictions_dir: 预测CSV所在目录。
        task: 任务名（如 "VSI"）。
        output_dir: 若提供，将结果写入CSV。

    Returns:
        {model_a: {model_b: {statistic, p_value, significant, b, c}}} 的嵌套字典。
    """
    # 加载每个模型的预测
    model_preds = {}
    for model in MODEL_NAMES:
        try:
            rows = _load_prediction_rows(predictions_dir, task, model)
            model_preds[model] = np.array([int(r["pred"]) for r in rows])
        except FileNotFoundError:
            continue

    available = list(model_preds.keys())
    if len(available) < 2:
        return {}

    # 验证所有模型预测相同数量的样本
    n_samples = len(next(iter(model_preds.values())))
    for m in available:
        if len(model_preds[m]) != n_samples:
            raise ValueError(f"sample count mismatch for {task}/{m}")

    # 初始化所有条目
    results: dict = {ma: {} for ma in available}
    for ma in available:
        for mb in available:
            if ma == mb:
                results[ma][mb] = {"statistic": 0, "p_value": 1.0, "significant": False, "b": 0, "c": 0}

    for i, ma in enumerate(available):
        for j, mb in enumerate(available):
            if i >= j:
                continue
            pred_a = model_preds[ma]
            pred_b = model_preds[mb]
            b = int(((pred_a == 1) & (pred_b == 0)).sum())
            c = int(((pred_a == 0) & (pred_b == 1)).sum())
            if b + c == 0:
                statistic = 0.0
                p_value = 1.0
            else:
                statistic = (abs(b - c) - 1) ** 2 / (b + c)
                from scipy.stats import chi2
                p_value = float(1 - chi2.cdf(statistic, 1))
            entry = {
                "statistic": statistic,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "b": b,
                "c": c,
            }
            results[ma][mb] = entry
            results[mb][ma] = entry

    if output_dir and results:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / f"mcnemar_{task}.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["model_a", "model_b", "statistic", "p_value", "significant", "b", "c"])
            for ma in available:
                for mb in available:
                    if ma >= mb:
                        continue
                    r = results[ma][mb]
                    writer.writerow([ma, mb, f"{r['statistic']:.4f}", f"{r['p_value']:.4f}", str(r["significant"]), r["b"], r["c"]])

    return results


def _load_task_metrics(predictions_dir: str | Path, metric: str = "accuracy") -> dict:
    """从预测CSV加载每个 (task, model) 的准确率（或指定指标）。"""
    from .metrics import load_predictions, compute_all_metrics

    all_preds = load_predictions(predictions_dir)
    results: dict[str, dict[str, float]] = {}
    for (task, model), rows in sorted(all_preds.items()):
        y_true = np.array([int(r["label"]) for r in rows])
        y_pred = np.array([int(r["pred"]) for r in rows])
        y_prob = np.array([[float(r["prob_0"]), float(r["prob_1"])] for r in rows])
        m = compute_all_metrics(y_true, y_pred, y_prob)
        results.setdefault(task, {})[model] = m.get(metric, m["accuracy"])
    return results


def friedman_test_with_nemenyi(
    predictions_dir: str | Path,
    metric: str = "accuracy",
    alpha: float = 0.05,
    output_dir: str | Path | None = None,
) -> dict:
    """Friedman检验 + Nemenyi事后检验。

    Args:
        predictions_dir: 预测CSV所在目录。
        metric: 用于排名的指标名。
        alpha: 显著性水平。
        output_dir: 若提供，结果写入CSV。

    Returns:
        {
            "friedman_statistic": float,
            "friedman_p_value": float,
            "friedman_significant": bool,
            "cd_threshold": float,
            "average_rankings": {model: rank},
            "nemenyi_significant_pairs": [(model_a, model_b), ...],
            "per_task_metrics": {task: {model: value}},
        }
    """
    task_metrics = _load_task_metrics(predictions_dir, metric)
    available_tasks = [t for t in TASK_NAMES if t in task_metrics and len(task_metrics[t]) >= 2]
    available_models = list(set().union(*[task_metrics[t].keys() for t in available_tasks]))

    if len(available_tasks) < 2 or len(available_models) < 2:
        return {"error": "insufficient data for Friedman test"}

    # 构建 matrix: tasks × models
    matrix = []
    for task in available_tasks:
        row = []
        for model in available_models:
            row.append(task_metrics[task].get(model, float("nan")))
        matrix.append(row)
    matrix = np.array(matrix)

    # Friedman test
    # scipy's friedmanchisquare expects columns to be samples (repeated measures)
    samples = [matrix[:, j] for j in range(len(available_models))]
    friedman_stat, friedman_p = friedmanchisquare(*samples)
    friedman_stat = float(friedman_stat)
    friedman_p = float(friedman_p)

    # Average rankings (rank 1 = best, lower rank = better)
    # For each task, rank models (1=best)
    rankings = np.zeros((len(available_tasks), len(available_models)))
    for i in range(len(available_tasks)):
        # Higher is better for most metrics
        # argsort ascending → 0 = worst; negate for descending
        ranks = np.argsort(np.argsort(-matrix[i])) + 1.0
        rankings[i] = ranks
    avg_ranks = rankings.mean(axis=0)
    avg_rankings = {m: float(avg_ranks[j]) for j, m in enumerate(available_models)}

    # Critical Difference (Nemenyi)
    # CD = q_alpha * sqrt(k*(k+1) / (6*N))
    # q_alpha from studentized range distribution
    k = len(available_models)
    N = len(available_tasks)
    # Approximate q_alpha values for alpha=0.05
    q_alpha_approx = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_alpha = q_alpha_approx.get(k, 3.0)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))

    # Nemenyi post-hoc: significant if |avg_rank_i - avg_rank_j| > CD
    sig_pairs = []
    for i, ma in enumerate(available_models):
        for j, mb in enumerate(available_models):
            if i >= j:
                continue
            if abs(avg_ranks[i] - avg_ranks[j]) > cd:
                sig_pairs.append((ma, mb))

    result = {
        "metric": metric,
        "friedman_statistic": friedman_stat,
        "friedman_p_value": friedman_p,
        "friedman_significant": friedman_p < alpha,
        "cd_threshold": float(cd),
        "average_rankings": avg_rankings,
        "nemenyi_significant_pairs": sig_pairs,
        "per_task_metrics": {t: task_metrics[t] for t in available_tasks},
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Save summary
        summary = {
            "metric": metric,
            "friedman_statistic": friedman_stat,
            "friedman_p_value": friedman_p,
            "friedman_significant": friedman_p < alpha,
            "cd_threshold": float(cd),
            "average_rankings": avg_rankings,
            "nemenyi_significant_pairs": [[a, b] for a, b in sig_pairs],
        }
        (out / "friedman_nemenyi.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Save per-task metrics
        with (out / "friedman_nemenyi_ranking.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["task", *available_models])
            for task in available_tasks:
                row = [task]
                for model in available_models:
                    row.append(f"{task_metrics[task].get(model, float('nan')):.6f}")
                writer.writerow(row)

    return result


def wilcoxon_signed_rank_test(
    predictions_dir: str | Path,
    task: str,
    output_dir: str | Path | None = None,
) -> dict:
    """对指定任务下所有模型组合做Wilcoxon符号秩检验。

    基于每个样本的预测正确性（0=错误, 1=正确）做配对检验。

    Returns:
        {model_a: {model_b: {statistic, p_value, significant}}} 的嵌套字典。
    """
    model_preds = {}
    model_correct = {}
    for model in MODEL_NAMES:
        try:
            rows = _load_prediction_rows(predictions_dir, task, model)
            labels = np.array([int(r["label"]) for r in rows])
            preds = np.array([int(r["pred"]) for r in rows])
            model_preds[model] = preds
            model_correct[model] = (preds == labels).astype(int)
        except FileNotFoundError:
            continue

    available = list(model_correct.keys())
    if len(available) < 2:
        return {}

    # 初始化所有条目
    results: dict = {ma: {} for ma in available}
    for ma in available:
        for mb in available:
            if ma == mb:
                results[ma][mb] = {"statistic": 0.0, "p_value": 1.0, "significant": False}

    for i, ma in enumerate(available):
        for j, mb in enumerate(available):
            if i >= j:
                continue
            try:
                stat, p = wilcoxon(model_correct[ma], model_correct[mb], zero_method="wilcox")
            except ValueError:
                stat, p = 0.0, 1.0
            entry = {
                "statistic": float(stat),
                "p_value": float(p),
                "significant": float(p) < 0.05,
            }
            results[ma][mb] = entry
            results[mb][ma] = entry

    if output_dir and results:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / f"wilcoxon_{task}.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["model_a", "model_b", "statistic", "p_value", "significant"])
            for ma in available:
                for mb in available:
                    if ma >= mb:
                        continue
                    r = results[ma][mb]
                    writer.writerow([ma, mb, f"{r['statistic']:.4f}", f"{r['p_value']:.4f}", str(r["significant"])])

    return results


def critical_difference_values(
    rankings: dict[str, float],
    n_models: int,
    n_tasks: int,
    alpha: float = 0.05,
) -> dict:
    """计算Critical Difference图所需数据。

    Args:
        rankings: {model_name: average_rank} 字典。
        n_models: 模型数量。
        n_tasks: 任务（数据集）数量。
        alpha: 显著性水平。

    Returns:
        {"models": [...], "ranks": [...], "cd": float, "alpha": alpha}
    """
    q_alpha_approx = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_alpha = q_alpha_approx.get(n_models, 3.0)
    cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6.0 * n_tasks))

    sorted_models = sorted(rankings.items(), key=lambda x: x[1])
    return {
        "models": [m for m, _ in sorted_models],
        "ranks": [r for _, r in sorted_models],
        "cd": float(cd),
        "alpha": alpha,
    }


def generate_latex_tables(
    predictions_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """生成LaTeX格式的统计检验结果表。

    包括:
    - McNemar检验的p值矩阵
    - Friedman + Nemenyi检验结果
    - Wilcoxon检验的p值矩阵
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Friedman for accuracy and macro_f1
    for metric in ["accuracy", "macro_f1"]:
        friedman = friedman_test_with_nemenyi(predictions_dir, metric=metric)

        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Friedman检验与Nemenyi事后检验结果（指标: " + metric.replace("_", r"\_") + "）}")
        lines.append(r"\label{tab:friedman_" + metric + "}")
        lines.append(r"\begin{tabular}{l" + "c" * len(MODEL_NAMES) + "}")
        lines.append(r"\toprule")
        lines.append(r" & " + " & ".join(m.replace("_", r"\_") for m in MODEL_NAMES) + r" \\")
        lines.append(r"\midrule")

        # 平均排名
        if "average_rankings" in friedman:
            ranks = friedman["average_rankings"]
            rank_str = " & ".join(f"{ranks.get(m, 0):.2f}" for m in MODEL_NAMES)
            lines.append(r"平均排名 & " + rank_str + r" \\")
            lines.append(r"\midrule")

        # Friedman statistic
        lines.append(r"\multicolumn{" + str(len(MODEL_NAMES) + 1) + r"}{l}{Friedman $\chi^2 = "
                     + f"{friedman.get('friedman_statistic', 0):.3f}, "
                     + f"p = {friedman.get('friedman_p_value', 1):.4f}$"
                     + (", 显著" if friedman.get("friedman_significant") else ", 不显著") + r"} \\")
        lines.append(r"\multicolumn{" + str(len(MODEL_NAMES) + 1) + r"}{l}{CD = "
                     + f"{friedman.get('cd_threshold', 0):.3f}" + r"} \\")

        sig_pairs = friedman.get("nemenyi_significant_pairs", [])
        if sig_pairs:
            pairs_str = ", ".join(f"{a.upper()}-{b.upper()}" for a, b in sig_pairs)
        else:
            pairs_str = "无显著差异"
        lines.append(r"\multicolumn{" + str(len(MODEL_NAMES) + 1) + r"}{l}{Nemenyi显著差异对: " + pairs_str + r"} \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        (out / f"friedman_{metric}.tex").write_text("\n".join(lines), encoding="utf-8")

    # McNemar per task
    for task in TASK_NAMES:
        mcnemar = mcnemar_test_per_task(predictions_dir, task)
        if not mcnemar:
            continue
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{McNemar检验p值矩阵 — 任务: " + task + "}")
        lines.append(r"\label{tab:mcnemar_" + task + "}")
        available = sorted(mcnemar.keys())
        lines.append(r"\begin{tabular}{l" + "c" * len(available) + "}")
        lines.append(r"\toprule")
        lines.append(r" & " + " & ".join(m.upper() for m in available) + r" \\")
        lines.append(r"\midrule")
        for ma in available:
            row = [ma.upper()]
            for mb in available:
                if ma == mb:
                    row.append("--")
                else:
                    p = mcnemar[ma][mb]["p_value"]
                    row.append(f"{p:.4f}" + (r"$^*$" if p < 0.05 else ""))
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        (out / f"mcnemar_{task}.tex").write_text("\n".join(lines), encoding="utf-8")
