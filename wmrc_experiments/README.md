# wmrc_experiments

WMRC General 图分类实验代码，包括 `GCN`、`GAT`、`HGNN`、`HyperGCN`、`Graph Transformer` 五个模型。

## 核心文件

- `common.py`: 数据读取、特征构造、padding、划分、指标计算。
- `models.py`: `GCN`、`GAT`、`HGNN`、`HyperGCN`、`Graph Transformer` 模型实现。
- `train.py`: 单个 `任务 + 模型` 的训练入口，训练后保存 checkpoint 和 summary。
- `predict.py`: 读取 checkpoint，对单个文件、目录或数据集划分做预测，输出 CSV。
- `train_all.py`: 批量训练 8 个任务 × 5 个模型；默认使用 `tuning_runs` 中筛出的每个任务 × 模型最佳超参数。
- `tune_all.py`: 按预设候选超参数做扫描式调参。
- `run_full_tuning.py`: 先调参，再生成最佳汇总表。
- `summarize_results.py`: 汇总普通训练结果。
- `summarize_tuning.py`: 从调参结果里筛出每个任务 × 模型的最佳行。
- `summarize_tuning_best.py`: 生成最佳指标表和最佳超参数表。

## 常用命令

训练单个模型：

```powershell
python -m wmrc_experiments.train --model gcn --task VSI
```

批量训练：

```powershell
python -m wmrc_experiments.train_all
```

如需覆盖调参默认值，可以继续传入参数，例如：

```powershell
python -m wmrc_experiments.train_all --lr 0.001 --dropout 0.2
```

对于批量训练的指标结果汇总为三张表格，默认输出到 output/results_tables/ 下。

```powershell
python -m wmrc_experiments.summarize_results
```

预测：

```powershell
python -m wmrc_experiments.predict --checkpoint checkpoints/gcn_VSI.pt
```

## 输出位置

- 普通训练结果默认写入 `checkpoints/`
- 批量调参结果默认写入 `tuning_runs/`
- 汇总表默认写入 `results_tables/` 或 `tuning_runs/best_tables/`
