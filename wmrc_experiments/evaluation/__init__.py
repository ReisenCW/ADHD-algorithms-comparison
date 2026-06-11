"""评估管线：指标计算、统计检验、可视化、错误分析。"""

from .metrics import (
    compute_all_metrics,
    confusion_matrix_values,
    sensitivity,
    specificity,
    precision_score_binary,
    auc_roc,
    auc_pr,
    mcc,
    cohens_kappa,
    aggregate_metrics_by_task_and_model,
)
from .statistical_tests import (
    mcnemar_test_per_task,
    friedman_test_with_nemenyi,
    wilcoxon_signed_rank_test,
    critical_difference_values,
)
from .visualization import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_metric_heatmap,
    plot_model_comparison_bar,
    plot_critical_difference,
    plot_performance_distribution,
)
from .error_analysis import (
    identify_hard_samples,
    model_agreement_matrix,
    per_task_error_summary,
    confusion_pattern_analysis,
    generate_error_report,
)
