"""结果可视化。

使用 matplotlib + seaborn 生成学术风格图表，输出 PNG (300dpi) + PDF。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from ..common import TASK_NAMES

# 模型名称和显示名称映射
MODEL_NAMES = ["gcn", "gat", "hgnn", "hypergcn", "gt"]
MODEL_DISPLAY = {
    "gcn": "GCN",
    "gat": "GAT",
    "hgnn": "HGNN",
    "hypergcn": "HyperGCN",
    "gt": "Graph Transformer",
}

# plt全局配置
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
})

# 色板
PALETTE = sns.color_palette("Set2", n_colors=5)
MODEL_COLORS = dict(zip(MODEL_NAMES, PALETTE))


def _load_prediction_data(predictions_dir: str | Path, task: str, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """加载预测数据，返回 (labels, preds, probs_1)。"""
    import csv
    csv_path = Path(predictions_dir) / f"{model}_{task}.csv"
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    labels = np.array([int(r["label"]) for r in rows])
    preds = np.array([int(r["pred"]) for r in rows])
    probs = np.array([float(r["prob_1"]) for r in rows])
    return labels, preds, probs


def _save_figure(fig, out_dir: Path, name: str) -> None:
    """保存图表为PNG和PDF。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["png", "pdf"]:
        fig.savefig(out_dir / f"{name}.{fmt}", format=fmt)
    plt.close(fig)


def plot_confusion_matrices(
    predictions_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """绘制混淆矩阵：每任务一行、每模型一列的子图布局。

    输出: confusion_matrices.png/pdf
    """
    out_dir = Path(output_dir) / "confusion_matrices"
    n_tasks = len(TASK_NAMES)
    n_models = len(MODEL_NAMES)

    fig, axes = plt.subplots(n_tasks, n_models, figsize=(n_models * 3, n_tasks * 2.8))
    fig.suptitle("混淆矩阵 (行=任务, 列=模型)", fontsize=14, y=1.01)

    for ti, task in enumerate(TASK_NAMES):
        for mi, model in enumerate(MODEL_NAMES):
            ax = axes[ti, mi] if n_tasks > 1 else axes[mi]
            data = _load_prediction_data(predictions_dir, task, model)
            # 如果无数据(比如某模型/任务没训练成功)，显示N/A
            if data is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{MODEL_DISPLAY[model]}" if ti == 0 else "")
                continue
            # 计算混淆矩阵并绘制热力图
            labels, preds, _ = data
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["HC (0)", "ADHD (1)"],
                yticklabels=["HC (0)", "ADHD (1)"],
                ax=ax, cbar=False, square=True,
            )
            
            # 只在第一行显示模型名称，第一列显示任务名称
            if ti == 0:
                ax.set_title(MODEL_DISPLAY[model], fontsize=11)
            if mi == 0:
                ax.set_ylabel(task, fontsize=11)

    # 调整布局，避免标题和标签重叠
    plt.tight_layout()
    _save_figure(fig, out_dir, "confusion_matrices")


def plot_roc_curves(
    predictions_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """每任务绘制一张ROC曲线图，5条模型曲线叠加。

    输出: roc_curves_{task}.png/pdf
    """
    out_dir = Path(output_dir) / "roc_curves"

    for task in TASK_NAMES:
        # 绘制 y=x 参考线
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
        # 绘制每个模型的ROC曲线
        for model in MODEL_NAMES:
            data = _load_prediction_data(predictions_dir, task, model)
            if data is None:
                continue
            labels, _, probs = data
            if len(np.unique(labels)) < 2:
                continue
            # 计算ROC曲线和AUC
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            # 绘制曲线
            ax.plot(fpr, tpr, color=MODEL_COLORS[model], lw=1.5,
                    label=f"{MODEL_DISPLAY[model]} (AUC={roc_auc:.3f})")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves — {task}")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_aspect("equal")
        plt.tight_layout()
        _save_figure(fig, out_dir, f"roc_{task}")


def plot_metric_heatmap(
    metrics_df,
    output_dir: str | Path,
    metric: str = "accuracy",
) -> None:
    """绘制 tasks x models 热力图, 数值为指标值, 越红表现越差, 越绿表现越好

    输出: heatmap_{metric}.png/pdf
    """
    out_dir = Path(output_dir) / "heatmaps"

    # 将 metrics指标数据 转换成 task x model 的矩阵形式
    pivot = metrics_df.pivot(index="task", columns="model", values=metric)
    pivot = pivot.reindex(index=TASK_NAMES, columns=MODEL_NAMES)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.0, vmax=1.0, ax=ax,
        xticklabels=[MODEL_DISPLAY[m] for m in MODEL_NAMES],
        cbar_kws={"label": metric},
    )
    ax.set_title(f"{metric} Heatmap (Tasks × Models)")
    ax.set_ylabel("Task")
    ax.set_xlabel("Model")
    plt.tight_layout()
    _save_figure(fig, out_dir, f"heatmap_{metric}")


def plot_model_comparison_bar(
    metrics_df,
    output_dir: str | Path,
    metric: str = "accuracy",
) -> None:
    """分组柱状图: x轴-任务, y轴-指标值, 每组5个模型对比

    输出: bar_{metric}.png/pdf
    """
    out_dir = Path(output_dir) / "bar_charts"

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(TASK_NAMES))
    width = 0.16
    n_models = len(MODEL_NAMES)

    for i, model in enumerate(MODEL_NAMES):
        values = []
        for task in TASK_NAMES:
            row = metrics_df[(metrics_df["task"] == task) & (metrics_df["model"] == model)]
            if row.empty:
                values.append(0)
            else:
                values.append(row[metric].values[0])
        bars = ax.bar(x + i * width, values, width, label=MODEL_DISPLAY[model],
                      color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)

        # 在柱上标数值
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xlabel("Task")
    ax.set_ylabel(metric)
    ax.set_title(f"Model Comparison — {metric}")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(TASK_NAMES)
    ax.legend(loc="upper right", fontsize=8, ncol=n_models)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    _save_figure(fig, out_dir, f"bar_{metric}")


def plot_critical_difference(
    cd_data: dict,
    output_dir: str | Path,
    metric: str = "accuracy",
) -> None:
    """绘制Critical Difference图。

    横轴为平均排名（越小越好），模型从左到右按排名排列。
    用粗线连接排名差异在CD阈值内的模型（表示无显著差异）。

    输出: cd_diagram_{metric}.png/pdf
    """
    out_dir = Path(output_dir)
    models = cd_data["models"]
    ranks = cd_data["ranks"]
    cd = cd_data["cd"]

    n = len(models)
    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.6)))

    # 排名从左到右递增（最佳在左）
    y_positions = list(range(n))
    colors = [MODEL_COLORS.get(m, "gray") for m in models]
    display_names = [MODEL_DISPLAY.get(m, m) for m in models]

    ax.barh(y_positions, ranks, color=colors, edgecolor="white", height=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(display_names)
    ax.invert_yaxis()
    ax.set_xlabel("Average Rank (lower is better)")
    ax.set_title(f"Critical Difference Diagram — {metric}\n(CD = {cd:.3f}, Friedman test)")

    # 标注排名
    for i, r in enumerate(ranks):
        ax.text(r + 0.05, i, f"{r:.2f}", va="center", fontsize=9)

    # CD bar (画在顶部)
    bar_y = -0.8
    ax.plot([1, 1 + cd], [bar_y, bar_y], "k-", lw=2)
    ax.plot([1, 1], [bar_y - 0.15, bar_y + 0.15], "k-", lw=1.5)
    ax.plot([1 + cd, 1 + cd], [bar_y - 0.15, bar_y + 0.15], "k-", lw=1.5)
    ax.text(1 + cd / 2, bar_y - 0.3, f"CD = {cd:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    _save_figure(fig, out_dir, f"cd_diagram_{metric}")


def plot_performance_distribution(
    metrics_df,
    output_dir: str | Path,
    metric: str = "accuracy",
) -> None:
    """箱线图：5个模型在8任务上的性能分布。

    输出: boxplot_{metric}.png/pdf
    """
    out_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(8, 5))
    data_for_box = []
    for model in MODEL_NAMES:
        values = metrics_df[metrics_df["model"] == model][metric].dropna().values
        if len(values) > 0:
            data_for_box.append(values)
        else:
            data_for_box.append([0])

    bp = ax.boxplot(data_for_box, tick_labels=[MODEL_DISPLAY[m] for m in MODEL_NAMES],
                    patch_artist=True, widths=0.5)

    for patch, model in zip(bp["boxes"], MODEL_NAMES):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.7)

    # 叠加散点
    for i, model in enumerate(MODEL_NAMES):
        values = metrics_df[metrics_df["model"] == model][metric].dropna().values
        if len(values) > 0:
            jitter = np.random.normal(0, 0.05, size=len(values))
            ax.scatter(np.ones(len(values)) * (i + 1) + jitter, values,
                       color=MODEL_COLORS[model], alpha=0.6, s=30, edgecolors="white", linewidth=0.5)

    ax.set_ylabel(metric)
    ax.set_title(f"Performance Distribution — {metric}")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    _save_figure(fig, out_dir, f"boxplot_{metric}")

def plot_all(
    predictions_dir: str | Path,
    metrics_df,
    output_dir: str | Path,
    cd_data: dict | None = None,
) -> None:
    """一键生成所有可视化图表。"""
    out = Path(output_dir)
    # 绘制9个任务 * 5个模型的混淆矩阵
    print("  generating confusion matrices...")
    plot_confusion_matrices(predictions_dir, out)
    # 绘制每个任务的ROC曲线
    print("  generating ROC curves...")
    plot_roc_curves(predictions_dir, out)
    # 绘制指标热力图、柱状图和箱线图
    for metric in ["accuracy", "f1", "macro_f1", "auc_roc", "mcc"]:
        if metric in metrics_df.columns:
            print(f"  generating heatmap for {metric}...")
            plot_metric_heatmap(metrics_df, out, metric=metric)

            print(f"  generating bar chart for {metric}...")
            plot_model_comparison_bar(metrics_df, out, metric=metric)

            print(f"  generating boxplot for {metric}...")
            plot_performance_distribution(metrics_df, out, metric=metric)
    # 绘制CD图
    if cd_data:
        print("  generating critical difference diagram...")
        plot_critical_difference(cd_data, out, metric=cd_data.get("metric", "accuracy"))
