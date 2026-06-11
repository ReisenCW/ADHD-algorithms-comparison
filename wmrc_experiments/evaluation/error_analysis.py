"""错误分析。

分析模型预测错误的模式：难易样本识别、模型间一致性、错误类型偏向等。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ..common import TASK_NAMES

MODEL_NAMES = ["gcn", "gat", "hgnn", "hypergcn", "gt"]


def _load_all_predictions(predictions_dir: str | Path) -> dict[tuple[str, str], pd.DataFrame]:
    """加载所有预测CSV为DataFrame字典。"""
    pred_dir = Path(predictions_dir)
    results = {}
    for csv_path in sorted(pred_dir.glob("*.csv")):
        filename = csv_path.stem
        parts = filename.rsplit("_", 1)
        if len(parts) != 2:
            continue
        model, task = parts
        if model not in MODEL_NAMES or task not in TASK_NAMES:
            continue
        df = pd.read_csv(csv_path)
        results[(task, model)] = df
    return results


def identify_hard_samples(
    predictions_dir: str | Path,
    output_dir: str | Path,
    agreement_threshold: float = 0.6,
) -> dict:
    """识别难/易样本。

    为每个task内的样本计算"模型共识度"：有多少模型将该样本分类正确。
    - 难样本 (hard): 正确模型数 / 总模型数 <= (1 - agreement_threshold)
    - 简单样本 (easy): 所有模型都正确
    - 中等样本 (medium): 介于两者之间

    输出: hard_samples.csv, easy_samples.csv
    """
    all_preds = _load_all_predictions(predictions_dir)
    out = Path(output_dir)

    hard_samples = []
    easy_samples = []
    summary = {}

    for task in TASK_NAMES:
        task_preds = {m: df for (t, m), df in all_preds.items() if t == task}
        if not task_preds:
            continue

        # 获取公共样本集（按path匹配）
        first_model = next(iter(task_preds.keys()))
        ref_df = task_preds[first_model]
        n_models = len(task_preds)

        correctness = np.zeros((len(ref_df), n_models), dtype=int)
        for mi, model in enumerate(sorted(task_preds.keys())):
            df = task_preds[model]
            correctness[:, mi] = (df["label"] == df["pred"]).values.astype(int)

        agreement = correctness.sum(axis=1) / n_models

        hard_mask = agreement <= (1 - agreement_threshold)
        easy_mask = agreement == 1.0

        n_hard = int(hard_mask.sum())
        n_easy = int(easy_mask.sum())
        n_medium = len(ref_df) - n_hard - n_easy

        summary[task] = {
            "total": len(ref_df),
            "hard": n_hard,
            "medium": n_medium,
            "easy": n_easy,
            "hard_ratio": n_hard / len(ref_df) if len(ref_df) > 0 else 0,
            "easy_ratio": n_easy / len(ref_df) if len(ref_df) > 0 else 0,
        }

        for idx in np.where(hard_mask)[0]:
            row = ref_df.iloc[idx]
            hard_samples.append({
                "task": task,
                "subject": row["subject"],
                "path": row["path"],
                "label": int(row["label"]),
                "agreement": round(agreement[idx], 3),
                "difficulty": "hard",
            })

        for idx in np.where(easy_mask)[0]:
            row = ref_df.iloc[idx]
            easy_samples.append({
                "task": task,
                "subject": row["subject"],
                "path": row["path"],
                "label": int(row["label"]),
                "agreement": round(agreement[idx], 3),
                "difficulty": "easy",
            })

    out.mkdir(parents=True, exist_ok=True)

    if hard_samples:
        pd.DataFrame(hard_samples).to_csv(out / "hard_samples.csv", index=False, encoding="utf-8-sig")
    if easy_samples:
        pd.DataFrame(easy_samples).to_csv(out / "easy_samples.csv", index=False, encoding="utf-8-sig")

    # Summary
    summary_df = pd.DataFrame([
        {"task": t, **s} for t, s in summary.items()
    ])
    summary_df.to_csv(out / "difficulty_summary.csv", index=False, encoding="utf-8-sig")
    (out / "difficulty_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return summary


def model_agreement_matrix(
    predictions_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """计算模型间预测一致性矩阵（Cohen's Kappa）。

    对每对模型，在所有任务和样本上计算预测一致性。

    输出: model_agreement.csv, model_agreement.json
    """
    all_preds = _load_all_predictions(predictions_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 收集所有(模型, 任务)对的预测
    model_all_preds = {m: [] for m in MODEL_NAMES}
    model_all_labels = {m: [] for m in MODEL_NAMES}

    for (task, model), df in all_preds.items():
        model_all_preds[model].extend(df["pred"].tolist())
        model_all_labels[model].extend(df["label"].tolist())

    # Kappa矩阵
    kappa_matrix = {}
    results = []
    for i, ma in enumerate(MODEL_NAMES):
        kappa_matrix[ma] = {}
        for j, mb in enumerate(MODEL_NAMES):
            if i == j:
                kappa_matrix[ma][mb] = 1.0
            elif j < i:
                kappa_matrix[ma][mb] = kappa_matrix[mb][ma]
            else:
                preds_a = model_all_preds[ma]
                preds_b = model_all_preds[mb]
                # 确保长度一致
                min_len = min(len(preds_a), len(preds_b))
                k = cohen_kappa_score(preds_a[:min_len], preds_b[:min_len])
                kappa_matrix[ma][mb] = round(float(k), 4)
            if i < j:
                results.append({
                    "model_a": ma,
                    "model_b": mb,
                    "cohens_kappa": kappa_matrix[ma][mb],
                })

    # Save
    df_results = pd.DataFrame(results)
    df_results.to_csv(out / "model_agreement_kappa.csv", index=False, encoding="utf-8-sig")

    (out / "model_agreement_kappa.json").write_text(
        json.dumps(kappa_matrix, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return kappa_matrix


def per_task_error_summary(
    predictions_dir: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    """每任务每模型的FP/FN分布分析。

    输出: error_summary.csv
    """
    all_preds = _load_all_predictions(predictions_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for (task, model), df in sorted(all_preds.items()):
        labels = df["label"].values
        preds = df["pred"].values
        assert len(labels) == len(preds)

        tp = int(((labels == 1) & (preds == 1)).sum())
        tn = int(((labels == 0) & (preds == 0)).sum())
        fp = int(((labels == 0) & (preds == 1)).sum())
        fn = int(((labels == 1) & (preds == 0)).sum())

        n_positive = int((labels == 1).sum())
        n_negative = int((labels == 0).sum())

        rows.append({
            "task": task,
            "model": model,
            "n_total": len(labels),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "fpr": fp / n_negative if n_negative > 0 else 0,
            "fnr": fn / n_positive if n_positive > 0 else 0,
            "error_rate": (fp + fn) / len(labels),
            "false_positive_rate": fp / n_negative if n_negative > 0 else 0,
            "false_negative_rate": fn / n_positive if n_positive > 0 else 0,
            "bias_toward": "FP" if fp > fn else ("FN" if fn > fp else "balanced"),
        })

    df_errors = pd.DataFrame(rows)
    df_errors.to_csv(out / "error_summary.csv", index=False, encoding="utf-8-sig")
    return df_errors


def confusion_pattern_analysis(
    predictions_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """分析误分类模式。

    找出所有5个模型都判错的样本（系统误差），以及标签类别和误分类之间的模式。

    输出: confusion_patterns.json, all_models_wrong.csv
    """
    all_preds = _load_all_predictions(predictions_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    patterns = {}
    all_wrong_records = []

    for task in TASK_NAMES:
        task_preds = {m: df for (t, m), df in all_preds.items() if t == task}
        if len(task_preds) < 2:
            continue

        ref_df = next(iter(task_preds.values()))
        n_models = len(task_preds)

        wrong_matrix = np.zeros((len(ref_df), n_models), dtype=int)
        model_order = sorted(task_preds.keys())

        for mi, model in enumerate(model_order):
            df = task_preds[model]
            wrong_matrix[:, mi] = (df["label"] != df["pred"]).values.astype(int)

        all_wrong_mask = wrong_matrix.sum(axis=1) == n_models
        n_all_wrong = int(all_wrong_mask.sum())

        # 分析：将ADHD判为HC的模型数 vs 将HC判为ADHD的模型数
        adhd_as_hc = np.zeros(len(ref_df), dtype=int)  # FN per sample
        hc_as_adhd = np.zeros(len(ref_df), dtype=int)   # FP per sample
        for mi, model in enumerate(model_order):
            df = task_preds[model]
            fn_mask = (df["label"] == 1) & (df["pred"] == 0)
            fp_mask = (df["label"] == 0) & (df["pred"] == 1)
            adhd_as_hc += fn_mask.values.astype(int)
            hc_as_adhd += fp_mask.values.astype(int)

        patterns[task] = {
            "n_total": len(ref_df),
            "n_all_models_wrong": n_all_wrong,
            "all_wrong_ratio": n_all_wrong / len(ref_df) if len(ref_df) > 0 else 0,
            "avg_models_wrong_per_sample": float(wrong_matrix.sum(axis=1).mean()),
            "samples_with_high_fn_agreement": int((adhd_as_hc >= n_models // 2).sum()),
            "samples_with_high_fp_agreement": int((hc_as_adhd >= n_models // 2).sum()),
        }

        # 记录所有模型都判错的样本
        for idx in np.where(all_wrong_mask)[0]:
            row = ref_df.iloc[idx]
            all_wrong_records.append({
                "task": task,
                "subject": row["subject"],
                "path": row["path"],
                "true_label": int(row["label"]),
                "fn_models": int(adhd_as_hc[idx]),
                "fp_models": int(hc_as_adhd[idx]),
            })

    out.mkdir(parents=True, exist_ok=True)

    (out / "confusion_patterns.json").write_text(
        json.dumps(patterns, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if all_wrong_records:
        pd.DataFrame(all_wrong_records).to_csv(
            out / "all_models_wrong.csv", index=False, encoding="utf-8-sig"
        )

    return patterns


def generate_error_report(
    predictions_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """生成综合错误分析报告（Markdown格式）。

    输出: error_report.md
    """
    out = Path(output_dir)

    # 运行各项分析
    print("  running difficulty analysis...")
    difficulty = identify_hard_samples(predictions_dir, out)

    print("  running model agreement analysis...")
    agreement = model_agreement_matrix(predictions_dir, out)

    print("  running per-task error summary...")
    error_df = per_task_error_summary(predictions_dir, out)

    print("  running confusion pattern analysis...")
    patterns = confusion_pattern_analysis(predictions_dir, out)

    # 生成报告
    lines = []
    lines.append("# 错误分析报告 (Error Analysis Report)")
    lines.append("")
    lines.append("## 1. 样本难度分析")
    lines.append("")
    lines.append("| 任务 | 总数 | 困难 | 中等 | 简单 | 困难比例 |")
    lines.append("|------|------|------|------|------|----------|")
    for task in TASK_NAMES:
        if task in difficulty:
            d = difficulty[task]
            lines.append(f"| {task} | {d['total']} | {d['hard']} | {d['medium']} | {d['easy']} | {d['hard_ratio']:.3f} |")
    lines.append("")

    lines.append("## 2. 模型预测一致性 (Cohen's Kappa)")
    lines.append("")
    lines.append("| 模型A | 模型B | Kappa |")
    lines.append("|-------|-------|-------|")
    for i, ma in enumerate(MODEL_NAMES):
        for j, mb in enumerate(MODEL_NAMES):
            if i < j and ma in agreement and mb in agreement[ma]:
                lines.append(f"| {ma.upper()} | {mb.upper()} | {agreement[ma][mb]:.4f} |")
    lines.append("")

    lines.append("## 3. 每任务每模型错误分布")
    lines.append("")
    lines.append("| 任务 | 模型 | 样本数 | TP | TN | FP | FN | 错误率 | 偏向 |")
    lines.append("|------|------|--------|----|----|----|----|--------|------|")
    for _, row in error_df.iterrows():
        lines.append(
            f"| {row['task']} | {row['model'].upper()} | {int(row['n_total'])} | "
            f"{int(row['tp'])} | {int(row['tn'])} | {int(row['fp'])} | {int(row['fn'])} | "
            f"{row['error_rate']:.3f} | {row['bias_toward']} |"
        )
    lines.append("")

    lines.append("## 4. 系统误差模式分析")
    lines.append("")
    lines.append("| 任务 | 样本总数 | 全模型错误 | 全错比例 | 平均错误模型数 |")
    lines.append("|------|----------|------------|----------|----------------|")
    for task in TASK_NAMES:
        if task in patterns:
            p = patterns[task]
            lines.append(
                f"| {task} | {p['n_total']} | {p['n_all_models_wrong']} | "
                f"{p['all_wrong_ratio']:.3f} | {p['avg_models_wrong_per_sample']:.2f} |"
            )
    lines.append("")

    report_path = out / "error_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
