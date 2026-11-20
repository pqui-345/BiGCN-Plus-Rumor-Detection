# -*- coding: utf-8 -*-
"""
Evaluation metrics (8 metric groups / 11 scalars) and multi-seed statistical testing
(Wilcoxon signed-rank test & paired t-test). Works with scikit-learn or pure NumPy/SciPy.
"""
import numpy as np
from scipy import stats

from .data_loader import LABELS

METRIC_KEYS = [
    "Accuracy",
    "Macro-Precision",
    "Macro-Recall",
    "Macro-F1",
    "F1-NR",
    "F1-FR",
    "F1-TR",
    "F1-UR",
    "Cohen-Kappa",
    "MCC",
    "AUC-ROC",
]


def confusion_matrix_np(y_true, y_pred, n_cls=4):
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _roc_auc_binary(y_bin, scores):
    pos = int(y_bin.sum())
    neg = len(y_bin) - pos
    if pos == 0 or neg == 0:
        return np.nan
    ranks = stats.rankdata(scores)
    return float((ranks[y_bin == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def roc_auc_ovr_np(y_true, prob, n_cls=4):
    aucs = []
    for c in range(n_cls):
        a = _roc_auc_binary((y_true == c).astype(np.int64), prob[:, c])
        if not np.isnan(a):
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan")


def compute_metrics(y_true, y_pred, prob):
    """Compute all 11 standard metrics across 8 evaluation groups."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)

    try:
        from sklearn.metrics import (
            accuracy_score,
            cohen_kappa_score,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        m = {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Macro-Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "Macro-Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        cf1 = f1_score(y_true, y_pred, average=None, labels=list(range(4)), zero_division=0)
        for i, label in enumerate(LABELS):
            m[f"F1-{label}"] = float(cf1[i])
        m["Cohen-Kappa"] = float(cohen_kappa_score(y_true, y_pred))
        m["MCC"] = float(matthews_corrcoef(y_true, y_pred))
        try:
            m["AUC-ROC"] = float(roc_auc_score(y_true, prob, multi_class="ovr", labels=list(range(4))))
        except ValueError:
            m["AUC-ROC"] = roc_auc_ovr_np(y_true, prob)
        return m
    except ImportError:
        cm = confusion_matrix_np(y_true, y_pred, n_cls=4)
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp
        total = float(cm.sum())

        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)

        acc = float(tp.sum() / max(total, 1.0))
        row_sum = cm.sum(axis=1).astype(np.float64)
        col_sum = cm.sum(axis=0).astype(np.float64)
        pe = float(np.dot(row_sum, col_sum) / max(total * total, 1.0))
        kappa = float((acc - pe) / (1.0 - pe)) if (1.0 - pe) > 1e-12 else 0.0

        t_sum = total
        c_val = float(tp.sum())
        num = c_val * t_sum - float(np.dot(row_sum, col_sum))
        den = np.sqrt(max((t_sum**2 - np.sum(col_sum**2)) * (t_sum**2 - np.sum(row_sum**2)), 0.0))
        mcc = float(num / den) if den > 1e-12 else 0.0

        m = {
            "Accuracy": acc,
            "Macro-Precision": float(np.mean(prec)),
            "Macro-Recall": float(np.mean(rec)),
            "Macro-F1": float(np.mean(f1)),
        }
        for i, label in enumerate(LABELS):
            m[f"F1-{label}"] = float(f1[i])
        m["Cohen-Kappa"] = kappa
        m["MCC"] = mcc
        m["AUC-ROC"] = roc_auc_ovr_np(y_true, prob)
        return m


def summarize_multiseed_records(records):
    """
    Summarize list of per-(seed, fold) metric dicts.
    Returns both all-run Mean/Std (N_seeds * N_folds) and seed-level Mean/Std (across N_seeds).
    """
    seeds = sorted({r["seed"] for r in records})
    by_seed = {}
    for s in seeds:
        s_rows = [r for r in records if r["seed"] == s]
        by_seed[s] = {k: float(np.nanmean([r[k] for r in s_rows])) for k in METRIC_KEYS}

    overall_mean = {k: float(np.nanmean([r[k] for r in records])) for k in METRIC_KEYS}
    overall_std = {k: float(np.nanstd([r[k] for r in records])) for k in METRIC_KEYS}
    seed_std = {k: float(np.nanstd([by_seed[s][k] for s in seeds])) for k in METRIC_KEYS}
    return {
        "n_seeds": len(seeds),
        "n_runs": len(records),
        "mean": overall_mean,
        "std_all_runs": overall_std,
        "std_across_seeds": seed_std,
        "per_seed": by_seed,
    }


def paired_significance_test(scores_a, scores_b):
    """Run paired t-test and Wilcoxon signed-rank test on aligned (seed, fold) accuracies."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    diff = a - b
    res = {"mean_diff": float(diff.mean()), "n": int(len(diff))}
    if len(diff) < 2 or np.allclose(diff, 0.0):
        res["t_pvalue"] = 1.0
        res["wilcoxon_pvalue"] = 1.0
        return res
    _, t_p = stats.ttest_rel(a, b)
    try:
        _, w_p = stats.wilcoxon(diff, zero_method="wilcox")
    except ValueError:
        w_p = 1.0
    res["t_pvalue"] = float(t_p)
    res["wilcoxon_pvalue"] = float(w_p)
    return res
