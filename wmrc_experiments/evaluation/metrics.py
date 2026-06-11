"""扩展评估指标。

复用 common.py 中已有的 accuracy, f1, macro_f1, balanced_accuracy，
本模块新增 AUC-ROC, AUC-PR, Sensitivity, Specificity, MCC, Cohen's Kappa 等。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from ..common import TASK_NAMES

MODEL_NAMES = ["gcn", "gat", "hgnn", "hypergcn", "gt"]


def confusion_matrix_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """返回混淆矩阵的四个值：tn, fp, fn, tp。"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """召回率 / 真阳性率 = TP / (TP + FN)。"""
    return float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """真阴性率 = TN / (TN + FP)。"""
    cm = confusion_matrix_values(y_true, y_pred)
    tn, fp = cm["tn"], cm["fp"]
    denom = tn + fp
    return float(tn / denom) if denom > 0 else 0.0


def precision_score_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """精确率 = TP / (TP + FP)。"""
    return float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))


def auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC（二分类，取正类概率）。"""
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def auc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """PR-AUC（二分类，取正类概率）。"""
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews相关系数。"""
    return float(matthews_corrcoef(y_true, y_pred))


def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cohen's Kappa。"""
    return float(cohen_kappa_score(y_true, y_pred))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    """一次性计算所有指标。

    Args:
        y_true: 真实标签，shape (N,)
        y_pred: 预测标签，shape (N,)
        y_prob: 预测概率（可选），shape (N,) 或 (N, 2)

    Returns:
        包含所有指标的字典。
    """
    cm = confusion_matrix_values(y_true, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score_binary(y_true, y_pred),
        "sensitivity": sensitivity(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
        "mcc": mcc(y_true, y_pred),
        "cohens_kappa": cohens_kappa(y_true, y_pred),
        "tn": cm["tn"],
        "fp": cm["fp"],
        "fn": cm["fn"],
        "tp": cm["tp"],
    }
    if y_prob is not None:
        metrics["auc_roc"] = auc_roc(y_true, y_prob)
        metrics["auc_pr"] = auc_pr(y_true, y_prob)
    return metrics


def load_predictions(predictions_dir: str | Path) -> dict[tuple[str, str], list[dict]]:
    """加载所有预测CSV文件。

    Returns:
        {(task, model): [rows...]} 的字典。
    """
    pred_dir = Path(predictions_dir)
    results: dict[tuple[str, str], list[dict]] = {}
    for csv_path in sorted(pred_dir.glob("*.csv")):
        filename = csv_path.stem  # e.g. gcn_VSI
        parts = filename.rsplit("_", 1)
        if len(parts) != 2:
            continue
        model, task = parts
        if model not in MODEL_NAMES or task not in TASK_NAMES:
            continue
        with csv_path.open("r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        results[(task, model)] = rows
    return results


def aggregate_metrics_by_task_and_model(predictions_dir: str | Path) -> list[dict]:
    """汇总所有预测结果，计算每个 (task, model) 的全套指标。

    Returns:
        每行包含 task, model, accuracy, f1, macro_f1, balanced_accuracy,
        precision, sensitivity, specificity, auc_roc, auc_pr, mcc, cohens_kappa
        等指标的列表。
    """
    all_preds = load_predictions(predictions_dir)
    rows = []
    for (task, model), pred_rows in sorted(all_preds.items()):
        y_true = np.array([int(r["label"]) for r in pred_rows])
        y_pred = np.array([int(r["pred"]) for r in pred_rows])
        y_prob = np.array([[float(r["prob_0"]), float(r["prob_1"])] for r in pred_rows])
        m = compute_all_metrics(y_true, y_pred, y_prob)
        row = {"task": task, "model": model}
        row.update(m)
        rows.append(row)
    return rows
