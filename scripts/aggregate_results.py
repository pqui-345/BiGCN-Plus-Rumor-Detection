# -*- coding: utf-8 -*-
"""
Aggregate multi-seed benchmark outputs into publication-ready tables,
statistical significance tests (Paired t-test & Wilcoxon), tree-sparsity
stratification, early-detection curves, external SOTA positioning, and
400-DPI Vietnamese figures suitable for A4 academic printing.
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.metrics import compute_metrics, paired_significance_test

MODEL_ORDER = [
    "TF-IDF-Ridge",
    "TF-IDF-LR",
    "TF-IDF-SVC",
    "Text-PLM",
    "MLP-BoW",
    "GCN-Single",
    "GraphSAGE-Single",
    "GAT-Single",
    "BiGCN-Base",
    "BiGCN-NoRoot",
    "BiGCN-NoDropEdge",
    "BiGCN-TDOnly",
    "BiGCN-BUOnly",
    "BiGCN-Base+",
    "EBGCN",
    "RAGCL",
    "BiGCN-Plus-B1Only",
    "BiGCN-Plus-B2Only",
    "BiGCN-Plus-B12",
    "BiGCN-Plus-Clean",
    "BiGCN-Plus-PLM",
    "BiGCN-Plus-UserID",
]

DEADLINE_LABELS_VI = {
    0: "0p (Bài gốc)",
    10: "10 phút",
    30: "30 phút",
    60: "1 giờ",
    120: "2 giờ",
    240: "4 giờ",
    720: "12 giờ",
    -1: "Toàn cảnh (Full)",
}

DEADLINE_SHORT_VI = {
    0: "0p",
    10: "10p",
    30: "30p",
    60: "1g",
    120: "2g",
    240: "4g",
    720: "12g",
    -1: "Full",
}

KEY_COMPARE_ORDER = [
    "TF-IDF-Ridge",
    "MLP-BoW",
    "GraphSAGE-Single",
    "BiGCN-Base",
    "EBGCN",
    "RAGCL",
    "BiGCN-Plus-Clean",
    "BiGCN-Plus-UserID",
]

# Verified external literature numbers extracted directly from surveyed PDFs
EXTERNAL_BENCHMARK = [
    ("Group B (Full replies + Heterogeneous Graph)", "MLI-GRA (Sci. Rep. 2025)", 0.9200, 0.9290, "70/15/15 split"),
    ("Group B (Full replies + BERT + GAE)", "GARD (WWW 2024)", 0.9110, 0.9320, "5-Fold CV"),
    ("Group B (GloVe + User Follow/Reply Net)", "PESTO (WASSA 2023)", 0.9150, 0.9080, "5-Fold CV"),
    ("Group B (Full replies + BERT + BiGCN)", "ICP-BGCN (Appl. Sci. 2025)", 0.8970, 0.9170, "5-Fold CV"),
    ("Group B (Full replies + Bayesian Edge)", "EBGCN (ACL 2021, orig.)", 0.8920, 0.9150, "5-Fold (re-impl 2025: 85.1/85.8)"),
    ("Group B (Full replies TF-IDF + Trans-GCN)", "TRGCN (Nature HSSC 2026)", 0.8940, 0.9010, "Early stop on Val"),
    ("Group B (Wikidata/Probase + Attn-GCN)", "KAGN (J. Big Data 2023)", 0.8923, 0.9013, "5-Fold CV"),
    ("Group B (RL Key Graph + BERT)", "KPG (WWW 2025)", 0.8930, 0.8890, "5-Fold CV"),
    ("Group B (Tree Transformer + Pretrain)", "P2T3 (2026)", 0.8740, 0.9110, "10 random splits"),
    ("Group B (Full replies + RoBERTa + InfoNCE)", "RAGCL (AAAI 2024)", 0.8670, 0.9050, "10 random splits"),
    ("Group B (Full replies TF-IDF)", "Bi-GCN Orig. (AAAI 2020)", 0.8860, 0.8800, "5-Fold CV"),
    ("Group B (Global Struct + Publisher Cred.)", "GS-PC (2025)", 0.8470, 0.8860, "30 runs + Wilcoxon"),
    ("Group A (Open rumdetect2017, source-only)", "BiGCN Re-impl (RAGCL AAAI 2024)", 0.8440, 0.8800, "10 random splits"),
    ("Group A (Open rumdetect2017, source-only)", "BiGCN Re-impl (BiTGN EMNLP 2023)", 0.8402, 0.8733, "5-Fold CV"),
    ("Group A (Open rumdetect2017, source TF-IDF)", "BiGCN (TF-IDF, COLING 2025)", 0.8180, 0.8470, "5-Fold / Multi-split"),
]


def _vi_comma_fmt(decimals=2):
    return FuncFormatter(lambda val, _: f"{val:.{decimals}f}".replace(".", ","))


def load_all_results(results_dir):
    paths = sorted(glob.glob(os.path.join(results_dir, "*", "results_multiseed.json")))
    items = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            items.append(json.load(f))
    return items


def sort_key(row):
    m = row["Model"]
    idx = MODEL_ORDER.index(m) if m in MODEL_ORDER else 999
    return (row["Dataset"], idx, m)


def build_summary_tables(items, tables_dir):
    rows = []
    same_eval_rows = []
    for d in items:
        ds = d["dataset"]
        model = d["model"]
        s = d["summary"]
        rows.append(
            {
                "Dataset": ds,
                "Model": model,
                "EvalMode": d["primary_eval_mode"],
                "Device": d["device"],
                "Seeds": s["n_seeds"],
                "Runs": s["n_runs"],
                "Acc_Mean": s["mean"]["Accuracy"],
                "Acc_Std": s["std_all_runs"]["Accuracy"],
                "Acc_SeedStd": s["std_across_seeds"]["Accuracy"],
                "F1_Mean": s["mean"]["Macro-F1"],
                "F1_Std": s["std_all_runs"]["Macro-F1"],
                "F1_SeedStd": s["std_across_seeds"]["Macro-F1"],
                "F1_NR": s["mean"]["F1-NR"],
                "F1_FR": s["mean"]["F1-FR"],
                "F1_TR": s["mean"]["F1-TR"],
                "F1_UR": s["mean"]["F1-UR"],
                "Kappa": s["mean"]["Cohen-Kappa"],
                "MCC": s["mean"]["MCC"],
                "AUC_ROC": s["mean"]["AUC-ROC"],
                "Sec_Per_Ep": d.get("sec_per_epoch_mean", 0.0),
            }
        )

        eval_dict = {d["primary_eval_mode"]: s}
        if d.get("alt_eval_mode") and d.get("alt_summary"):
            eval_dict[d["alt_eval_mode"]] = d["alt_summary"]
        for em, es in eval_dict.items():
            same_eval_rows.append(
                {
                    "Dataset": ds,
                    "Model": model,
                    "EvalMode": em,
                    "Runs": es["n_runs"],
                    "Acc_Mean": es["mean"]["Accuracy"],
                    "Acc_Std": es["std_all_runs"]["Accuracy"],
                    "F1_Mean": es["mean"]["Macro-F1"],
                    "F1_Std": es["std_all_runs"]["Macro-F1"],
                }
            )

    rows.sort(key=sort_key)
    same_eval_rows.sort(
        key=lambda r: (
            r["Dataset"],
            r["EvalMode"],
            MODEL_ORDER.index(r["Model"]) if r["Model"] in MODEL_ORDER else 999,
        )
    )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(tables_dir, "master_multiseed_summary.csv"), index=False)

    df_same = pd.DataFrame(same_eval_rows)
    df_same.to_csv(os.path.join(tables_dir, "same_eval_comparison.csv"), index=False)

    md_lines = [
        "# Master Multi-Seed Summary Table",
        "",
        "| Dataset | Model | Eval | Runs | Device | Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) | Seed-Std (Acc) | Kappa | MCC | AUC-ROC |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['Dataset']} | `{r['Model']}` | {r['EvalMode']} | {int(r['Runs'])} | {r['Device']} | "
            f"{r['Acc_Mean']:.5f} ± {r['Acc_Std']:.5f} | {r['F1_Mean']:.5f} ± {r['F1_Std']:.5f} | "
            f"{r['Acc_SeedStd']:.5f} | {r['Kappa']:.5f} | {r['MCC']:.5f} | {r['AUC_ROC']:.5f} |"
        )
    with open(os.path.join(tables_dir, "master_multiseed_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    md_se = [
        "# Same-Checkpoint Evaluation Comparison (`last` vs `snapshot5`)",
        "",
        "| Dataset | EvalMode | Model | Runs | Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) |",
        "| :--- | :---: | :--- | :---: | :---: | :---: |",
    ]
    for _, r in df_same.iterrows():
        md_se.append(
            f"| {r['Dataset']} | {r['EvalMode']} | `{r['Model']}` | {int(r['Runs'])} | "
            f"{r['Acc_Mean']:.5f} ± {r['Acc_Std']:.5f} | {r['F1_Mean']:.5f} ± {r['F1_Std']:.5f} |"
        )
    with open(os.path.join(tables_dir, "same_eval_comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_se) + "\n")

    return df


def build_statistical_tests(items, tables_dir):
    by_ds = {}
    for d in items:
        ds = d["dataset"]
        m = d["model"]
        pm = d["primary_eval_mode"]
        recs = {(r["seed"], r["fold"]): r for r in d["per_run_records"] if r["eval_mode"] == pm}
        by_ds.setdefault(ds, {})[m] = recs

    refs = ["BiGCN-Base", "BiGCN-Base+", "BiGCN-Plus-Clean"]
    test_rows = []
    for ds, mmap in sorted(by_ds.items()):
        for m, recs in sorted(mmap.items()):
            for ref in refs:
                if ref not in mmap or ref == m:
                    continue
                common_keys = sorted(set(recs.keys()) & set(mmap[ref].keys()))
                if len(common_keys) < 2:
                    continue
                acc_a = [recs[k]["Accuracy"] for k in common_keys]
                acc_b = [mmap[ref][k]["Accuracy"] for k in common_keys]
                f1_a = [recs[k]["Macro-F1"] for k in common_keys]
                f1_b = [mmap[ref][k]["Macro-F1"] for k in common_keys]
                sig_acc = paired_significance_test(acc_a, acc_b)
                sig_f1 = paired_significance_test(f1_a, f1_b)
                test_rows.append(
                    {
                        "Dataset": ds,
                        "Model_A": m,
                        "Ref_B": ref,
                        "Paired_Runs": sig_acc["n"],
                        "Acc_Diff_A_minus_B": sig_acc["mean_diff"],
                        "Acc_t_pval": sig_acc["t_pvalue"],
                        "Acc_Wilcoxon_pval": sig_acc["wilcoxon_pvalue"],
                        "F1_Diff_A_minus_B": sig_f1["mean_diff"],
                        "F1_t_pval": sig_f1["t_pvalue"],
                        "F1_Wilcoxon_pval": sig_f1["wilcoxon_pvalue"],
                    }
                )

    df_t = pd.DataFrame(test_rows)
    df_t.to_csv(os.path.join(tables_dir, "statistical_tests.csv"), index=False)

    md_lines = [
        "# Paired Statistical Significance Tests (Paired t-test & Wilcoxon Signed-Rank)",
        "",
        "| Dataset | Model A | Reference B | N Pairs | ΔAcc (A-B) | Paired-t p (Acc) | Wilcoxon p (Acc) | ΔF1 (A-B) | Wilcoxon p (F1) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in df_t.iterrows():
        sig_tag = "*" if r["Acc_Wilcoxon_pval"] < 0.05 else ""
        md_lines.append(
            f"| {r['Dataset']} | `{r['Model_A']}` | `{r['Ref_B']}` | {int(r['Paired_Runs'])} | "
            f"{r['Acc_Diff_A_minus_B']:+.5f} | {r['Acc_t_pval']:.4g} | {r['Acc_Wilcoxon_pval']:.4g}{sig_tag} | "
            f"{r['F1_Diff_A_minus_B']:+.5f} | {r['F1_Wilcoxon_pval']:.4g} |"
        )
    with open(os.path.join(tables_dir, "statistical_tests.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def build_tree_sparsity_analysis(items, results_dir, tables_dir, fig_dir):
    """Stratify OOF event predictions by tree node tertiles (Sparse / Medium / Dense)."""
    datasets = sorted({d["dataset"] for d in items})
    rows = []

    for ds in datasets:
        model_dfs = {}
        for d in items:
            if d["dataset"] != ds:
                continue
            oof_path = os.path.join(results_dir, f"{ds}_{d['model']}", "oof_predictions.csv")
            if os.path.isfile(oof_path):
                df_oof = pd.read_csv(oof_path)
                if not df_oof.empty:
                    model_dfs[d["model"]] = df_oof

        if not model_dfs:
            continue

        ref_model = "BiGCN-Plus-Clean" if "BiGCN-Plus-Clean" in model_dfs else sorted(model_dfs.keys())[0]
        ref_df = model_dfs[ref_model]
        nodes_arr = ref_df["n_nodes"].to_numpy(dtype=np.float64)
        q33 = int(np.floor(np.quantile(nodes_arr, 1.0 / 3.0)))
        q67 = int(np.ceil(np.quantile(nodes_arr, 2.0 / 3.0)))
        if q67 <= q33:
            q67 = q33 + 1

        strata_defs = [
            ("Sparse", f"Tầng 1 — Cây thưa/nhỏ (≤ {q33} nút)", lambda n: n <= q33),
            ("Medium", f"Tầng 2 — Cây trung bình ({q33 + 1}–{q67} nút)", lambda n: (n > q33) & (n <= q67)),
            ("Dense", f"Tầng 3 — Cây dày/lớn (> {q67} nút)", lambda n: n > q67),
        ]

        ordered_models = sorted(
            model_dfs.keys(),
            key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999,
        )
        for m in ordered_models:
            df_m = model_dfs[m]
            for s_key, s_label, mask_fn in strata_defs:
                sub = df_m[mask_fn(df_m["n_nodes"])]
                if sub.empty:
                    continue
                prob = sub[["p_NR", "p_FR", "p_TR", "p_UR"]].to_numpy(dtype=np.float64)
                met = compute_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(), prob)
                rows.append(
                    {
                        "Dataset": ds,
                        "Model": m,
                        "Stratum_Key": s_key,
                        "Stratum": s_label,
                        "Q33": q33,
                        "Q67": q67,
                        "N_Events": int(len(sub)),
                        "Accuracy": met["Accuracy"],
                        "Macro_F1": met["Macro-F1"],
                    }
                )

    df_sp = pd.DataFrame(rows)
    df_sp.to_csv(os.path.join(tables_dir, "tree_sparsity_analysis.csv"), index=False)

    md_lines = [
        "# Phân tích Độ bền theo Quy mô / Độ thưa Cây Lan truyền (OOF Stratified)",
        "",
        "| Dataset | Mô hình | Phân tầng cây | Số mẫu | Accuracy | Macro-F1 |",
        "| :--- | :--- | :--- | :---: | :---: | :---: |",
    ]
    for _, r in df_sp.iterrows():
        md_lines.append(
            f"| {r['Dataset']} | `{r['Model']}` | {r['Stratum']} | {int(r['N_Events'])} | "
            f"{r['Accuracy']:.5f} | {r['Macro_F1']:.5f} |"
        )
    with open(os.path.join(tables_dir, "tree_sparsity_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    if df_sp.empty:
        return

    for ds, suffix in [("Twitter15", "t15"), ("Twitter16", "t16")]:
        sub_ds = df_sp[df_sp["Dataset"] == ds]
        if sub_ds.empty:
            continue
        models_plot = [m for m in KEY_COMPARE_ORDER if m in set(sub_ds["Model"])]
        if not models_plot:
            models_plot = sorted(sub_ds["Model"].unique())

        strata_keys = ["Sparse", "Medium", "Dense"]
        strata_labels = {
            k: sub_ds[sub_ds["Stratum_Key"] == k]["Stratum"].iloc[0]
            for k in strata_keys
            if not sub_ds[sub_ds["Stratum_Key"] == k].empty
        }
        active_keys = [k for k in strata_keys if k in strata_labels]
        if not active_keys:
            continue

        fig, ax = plt.subplots(figsize=(10.5, 4.9))
        x = np.arange(len(models_plot))
        width = 0.25
        colors = {"Sparse": "#2563eb", "Medium": "#0d9488", "Dense": "#f59e0b"}

        for idx_k, sk in enumerate(active_keys):
            offset = (idx_k - (len(active_keys) - 1) / 2.0) * width
            vals = []
            for m in models_plot:
                row_m = sub_ds[(sub_ds["Model"] == m) & (sub_ds["Stratum_Key"] == sk)]
                vals.append(float(row_m["Accuracy"].iloc[0]) if not row_m.empty else 0.0)
            bars = ax.bar(x + offset, vals, width, label=strata_labels[sk], color=colors[sk], alpha=0.9)
            if len(models_plot) <= 8:
                for bar, v in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        v + 0.006,
                        f"{v:.3f}".replace(".", ","),
                        ha="center",
                        va="bottom",
                        fontsize=7.2,
                        rotation=90,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(models_plot, rotation=20, ha="right", fontsize=10)
        ymin = max(0.35, float(sub_ds["Accuracy"].min()) - 0.10)
        ax.set_ylim(ymin, min(1.03, float(sub_ds["Accuracy"].max()) + 0.09))
        ax.yaxis.set_major_formatter(_vi_comma_fmt(2))
        ax.set_ylabel("Độ chính xác Out-Of-Fold (Accuracy)", fontsize=10.5)
        ax.set_title(f"{ds} — Độ bền hiệu năng theo phân tầng quy mô cây lan truyền", fontsize=12)
        ax.legend(fontsize=9.2, loc="lower right", framealpha=0.92)
        ax.grid(True, axis="y", ls="--", alpha=0.35)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"tree_sparsity_{suffix}.png"), dpi=400)
        plt.close(fig)


def build_early_detection_summary(items, results_dir, tables_dir, fig_dir):
    """Aggregate Early Rumor Detection trajectories across deadlines with Wilcoxon p-values."""
    datasets = sorted({d["dataset"] for d in items})
    summary_rows = []
    raw_by_ds_model_dt = {}

    for ds in datasets:
        for d in items:
            if d["dataset"] != ds:
                continue
            m = d["model"]
            e_path = os.path.join(results_dir, f"{ds}_{m}", "early_detection.csv")
            if not os.path.isfile(e_path):
                continue
            df_e = pd.read_csv(e_path)
            if df_e.empty:
                continue
            for dt, grp in df_e.groupby("deadline_min"):
                dt_int = int(dt)
                key_map = {(int(r["seed"]), int(r["fold"])): r for _, r in grp.iterrows()}
                raw_by_ds_model_dt.setdefault(ds, {}).setdefault(m, {})[dt_int] = key_map

    for ds, mmap in sorted(raw_by_ds_model_dt.items()):
        ordered_models = sorted(mmap.keys(), key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999)
        all_dts = sorted({dt for m_dts in mmap.values() for dt in m_dts.keys()}, key=lambda v: (v < 0, v))
        for dt in all_dts:
            for m in ordered_models:
                recs = mmap[m].get(dt)
                if not recs:
                    continue
                accs = [r["Accuracy"] for r in recs.values()]
                f1s = [r["Macro-F1"] for r in recs.values()]

                p_vs_bigcn = float("nan")
                p_vs_ragcl = float("nan")
                if "BiGCN-Base" in mmap and dt in mmap["BiGCN-Base"] and m != "BiGCN-Base":
                    ck = sorted(set(recs.keys()) & set(mmap["BiGCN-Base"][dt].keys()))
                    if len(ck) >= 2:
                        p_vs_bigcn = paired_significance_test(
                            [recs[k]["Accuracy"] for k in ck],
                            [mmap["BiGCN-Base"][dt][k]["Accuracy"] for k in ck],
                        )["wilcoxon_pvalue"]
                if "RAGCL" in mmap and dt in mmap["RAGCL"] and m != "RAGCL":
                    ck = sorted(set(recs.keys()) & set(mmap["RAGCL"][dt].keys()))
                    if len(ck) >= 2:
                        p_vs_ragcl = paired_significance_test(
                            [recs[k]["Accuracy"] for k in ck],
                            [mmap["RAGCL"][dt][k]["Accuracy"] for k in ck],
                        )["wilcoxon_pvalue"]

                summary_rows.append(
                    {
                        "Dataset": ds,
                        "Model": m,
                        "Deadline_Min": dt,
                        "Deadline_Label": DEADLINE_LABELS_VI.get(dt, f"{dt}p"),
                        "Runs": len(accs),
                        "Acc_Mean": float(np.mean(accs)),
                        "Acc_Std": float(np.std(accs)),
                        "F1_Mean": float(np.mean(f1s)),
                        "F1_Std": float(np.std(f1s)),
                        "Wilcoxon_vs_BiGCNBase": p_vs_bigcn,
                        "Wilcoxon_vs_RAGCL": p_vs_ragcl,
                    }
                )

    df_ed = pd.DataFrame(summary_rows)
    df_ed.to_csv(os.path.join(tables_dir, "early_detection_summary.csv"), index=False)

    md_lines = [
        "# Tổng hợp Đường cong Phát hiện Sớm (Early Rumor Detection)",
        "",
        "| Dataset | Mốc thời gian | Mô hình | Lượt chạy | Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) | p vs BiGCN-Base | p vs RAGCL |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in df_ed.iterrows():
        pb = f"{r['Wilcoxon_vs_BiGCNBase']:.4g}" if pd.notna(r["Wilcoxon_vs_BiGCNBase"]) else "—"
        pr = f"{r['Wilcoxon_vs_RAGCL']:.4g}" if pd.notna(r["Wilcoxon_vs_RAGCL"]) else "—"
        md_lines.append(
            f"| {r['Dataset']} | {r['Deadline_Label']} | `{r['Model']}` | {int(r['Runs'])} | "
            f"{r['Acc_Mean']:.5f} ± {r['Acc_Std']:.5f} | {r['F1_Mean']:.5f} ± {r['F1_Std']:.5f} | {pb} | {pr} |"
        )
    with open(os.path.join(tables_dir, "early_detection_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    if df_ed.empty:
        return

    style_map = {
        "BiGCN-Plus-Clean": ("#dc2626", "-", "o", 2.6, 5),
        "BiGCN-Base": ("#2563eb", "-", "s", 2.0, 4),
        "RAGCL": ("#7c3aed", "--", "^", 1.9, 3),
        "EBGCN": ("#059669", "-.", "D", 1.8, 3),
        "GraphSAGE-Single": ("#0891b2", ":", "v", 1.8, 3),
        "MLP-BoW": ("#64748b", ":", "x", 1.5, 2),
        "TF-IDF-Ridge": ("#d97706", "--", "", 1.8, 2),
    }

    for ds, suffix in [("Twitter15", "t15"), ("Twitter16", "t16")]:
        sub_ds = df_ed[df_ed["Dataset"] == ds]
        if sub_ds.empty:
            continue
        dts_sorted = sorted(sub_ds["Deadline_Min"].unique(), key=lambda v: (v < 0, v))
        x = np.arange(len(dts_sorted))
        x_labels = [DEADLINE_SHORT_VI.get(int(d), str(d)) for d in dts_sorted]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8))
        models_present = sorted(
            sub_ds["Model"].unique(),
            key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999,
        )

        for m in models_present:
            sub_m = sub_ds[sub_ds["Model"] == m].set_index("Deadline_Min").reindex(dts_sorted)
            color, ls, marker, lw, zorder = style_map.get(m, ("#475569", "-", ".", 1.5, 2))
            ax1.plot(x, sub_m["Acc_Mean"], color=color, ls=ls, marker=marker or None, lw=lw, zorder=zorder, label=m, markersize=5.5)
            ax2.plot(x, sub_m["F1_Mean"], color=color, ls=ls, marker=marker or None, lw=lw, zorder=zorder, label=m, markersize=5.5)

        for ax, ylab, title_s in [
            (ax1, "Độ chính xác trung bình (Accuracy)", f"{ds} — Accuracy phát hiện sớm theo thời gian"),
            (ax2, "Macro-F1 trung bình", f"{ds} — Macro-F1 phát hiện sớm theo thời gian"),
        ]:
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=9.8)
            ax.set_xlabel("Ngưỡng thời gian quan sát sau bài đăng gốc", fontsize=10.2)
            ax.set_ylabel(ylab, fontsize=10.2)
            ax.set_title(title_s, fontsize=11.5)
            ax.yaxis.set_major_formatter(_vi_comma_fmt(2))
            ax.grid(True, ls="--", alpha=0.35)
            ax.legend(fontsize=8.3, loc="lower right", framealpha=0.92)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"early_detection_{suffix}.png"), dpi=400)
        plt.close(fig)


def build_external_benchmark_md(df_local, tables_dir):
    md_lines = [
        "# External Benchmark Positioning on Twitter15 & Twitter16",
        "",
        "Separation into **Group A** (open `rumdetect2017` source-tweet text + structure only, identical constraint to ours) and **Group B** (full reply-tweet texts crawled via legacy Twitter API, external KGs, or user follow graphs).",
        "",
        "## Verified Literature Results (from Surveyed PDFs)",
        "",
        "| Data Regime | Method & Source | Twitter15 Acc | Twitter16 Acc | Protocol |",
        "| :--- | :--- | :---: | :---: | :--- |",
    ]
    for regime, name, t15, t16, proto in EXTERNAL_BENCHMARK:
        md_lines.append(f"| {regime} | {name} | {t15:.5f} | {t16:.5f} | {proto} |")

    if not df_local.empty:
        md_lines.extend(
            [
                "",
                "## Local / Current Multi-Seed Results Observed",
                "",
                "| Model | Runs | Twitter15 Acc (Mean ± Std) | Twitter16 Acc (Mean ± Std) |",
                "| :--- | :---: | :---: | :---: |",
            ]
        )
        models_present = sorted(df_local["Model"].unique(), key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999)
        for m in models_present:
            r15 = df_local[(df_local["Model"] == m) & (df_local["Dataset"] == "Twitter15")]
            r16 = df_local[(df_local["Model"] == m) & (df_local["Dataset"] == "Twitter16")]
            s15 = f"{r15.iloc[0]['Acc_Mean']:.5f} ± {r15.iloc[0]['Acc_Std']:.5f}" if not r15.empty else "—"
            s16 = f"{r16.iloc[0]['Acc_Mean']:.5f} ± {r16.iloc[0]['Acc_Std']:.5f}" if not r16.empty else "—"
            runs = int(r15.iloc[0]["Runs"]) if not r15.empty else (int(r16.iloc[0]["Runs"]) if not r16.empty else 0)
            md_lines.append(f"| `{m}` | {runs} | {s15} | {s16} |")

    with open(os.path.join(tables_dir, "external_benchmark_sota.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def build_latex_tables(df, tables_dir):
    if df.empty:
        return
    lines = [
        "% Auto-generated by scripts/aggregate_results.py",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Kết quả thực nghiệm đa lượt chạy (Mean $\\pm$ Std trên toàn bộ các lượt chạy)}",
        "\\label{tab:multiseed-results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Mô hình} & \\textbf{Lượt chạy} & \\textbf{Accuracy} & \\textbf{Macro-F1} & \\textbf{Kappa} & \\textbf{AUC-ROC} \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        m_tex = str(r["Model"]).replace("_", "\\_")
        acc_m = f"{r['Acc_Mean']:.5f}".replace(".", "{,}")
        acc_s = f"{r['Acc_Std']:.5f}".replace(".", "{,}")
        f1_m = f"{r['F1_Mean']:.5f}".replace(".", "{,}")
        f1_s = f"{r['F1_Std']:.5f}".replace(".", "{,}")
        kap = f"{r['Kappa']:.5f}".replace(".", "{,}")
        auc = f"{r['AUC_ROC']:.5f}".replace(".", "{,}")
        lines.append(
            f"{r['Dataset']} & \\texttt{{{m_tex}}} & {int(r['Runs'])} & "
            f"${acc_m} \\pm {acc_s}$ & ${f1_m} \\pm {f1_s}$ & "
            f"${kap}$ & ${auc}$ \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table}"])
    with open(os.path.join(tables_dir, "latex_tables.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _plot_single_dataset_bars(sub, ds, out_path):
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    x = np.arange(len(sub))
    colors = [
        "#dc2626" if "BiGCN-Plus-UserID" in m else ("#b91c1c" if "BiGCN-Plus-Clean" in m else ("#0284c7" if "BiGCN-Plus" in m else "#3b82f6"))
        for m in sub["Model"]
    ]
    ax.bar(x, sub["Acc_Mean"], yerr=sub["Acc_Std"], capsize=3.5, color=colors, alpha=0.88, label="Độ chính xác (Accuracy)")
    ax.plot(x, sub["F1_Mean"], color="#f97316", marker="o", lw=2.0, markersize=5.0, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["Model"], rotation=32, ha="right", fontsize=10.5)
    ymin = max(0.35, float(sub["Acc_Mean"].min()) - 0.08)
    ax.set_ylim(ymin, 1.0)
    ax.yaxis.set_major_formatter(_vi_comma_fmt(2))
    ax.set_title(f"{ds} — Đối sánh {len(sub)} mô hình (25 lượt chạy: 5 seeds × 5 folds)", fontsize=12.5)
    ax.set_ylabel("Điểm số trung bình", fontsize=11)
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close(fig)


def _plot_ablation_bars(sub, ds, out_path):
    ablation_order = [
        "BiGCN-NoRoot",
        "BiGCN-NoDropEdge",
        "BiGCN-TDOnly",
        "BiGCN-BUOnly",
        "BiGCN-Base",
        "BiGCN-Base+",
        "BiGCN-Plus-B1Only",
        "BiGCN-Plus-B2Only",
        "BiGCN-Plus-B12",
        "BiGCN-Plus-Clean",
        "BiGCN-Plus-UserID",
    ]
    sub_ab = sub[sub["Model"].isin(ablation_order)].copy()
    if sub_ab.empty:
        return
    sub_ab["__ord"] = sub_ab["Model"].apply(lambda m: ablation_order.index(m))
    sub_ab = sub_ab.sort_values("__ord")

    fig, ax = plt.subplots(figsize=(10.8, 4.7))
    x = np.arange(len(sub_ab))
    colors = [
        "#dc2626" if "BiGCN-Plus-UserID" in m else ("#b91c1c" if m == "BiGCN-Plus-Clean" else ("#0284c7" if "BiGCN-Plus" in m else "#64748b"))
        for m in sub_ab["Model"]
    ]
    bars = ax.bar(x, sub_ab["Acc_Mean"], yerr=sub_ab["Acc_Std"], capsize=3.5, color=colors, alpha=0.9, label="Accuracy")
    ax.plot(x, sub_ab["F1_Mean"], color="#f59e0b", marker="s", lw=1.9, markersize=5, label="Macro-F1")
    for bar, v in zip(bars, sub_ab["Acc_Mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, v + 0.008, f"{v:.4f}".replace(".", ","), ha="center", va="bottom", fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels(sub_ab["Model"], rotation=26, ha="right", fontsize=10)
    ymin = max(0.35, float(sub_ab["Acc_Mean"].min()) - 0.10)
    ax.set_ylim(ymin, min(1.03, float(sub_ab["Acc_Mean"].max()) + 0.08))
    ax.yaxis.set_major_formatter(_vi_comma_fmt(2))
    ax.set_title(f"{ds} — Giải phẫu đóng góp từng thành phần kiến trúc", fontsize=12)
    ax.set_ylabel("Độ chính xác / Macro-F1", fontsize=10.8)
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    ax.legend(fontsize=9.8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_figures(df, fig_dir):
    if df.empty:
        return

    # 1a. Individual wide per-dataset figures for stacked A4 layout (400 DPI)
    for ds, suffix in [("Twitter15", "t15"), ("Twitter16", "t16")]:
        sub = df[df["Dataset"] == ds].copy()
        _plot_single_dataset_bars(sub, ds, os.path.join(fig_dir, f"multiseed_{suffix}.png"))
        _plot_ablation_bars(sub, ds, os.path.join(fig_dir, f"ablation_{suffix}.png"))

    # 1b. Combined overview figure (kept for README compatibility)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=False)
    for ax, ds in zip(axes, ["Twitter15", "Twitter16"]):
        sub = df[df["Dataset"] == ds].copy()
        if sub.empty:
            ax.set_title(f"{ds} (Chưa có dữ liệu)")
            continue
        x = np.arange(len(sub))
        ax.bar(x, sub["Acc_Mean"], yerr=sub["Acc_Std"], capsize=3.5, color="#3b82f6", alpha=0.85, label="Accuracy")
        ax.plot(x, sub["F1_Mean"], color="#ef4444", marker="o", lw=1.8, label="Macro-F1")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["Model"], rotation=35, ha="right", fontsize=8.5)
        ax.set_ylim(max(0.4, float(sub["Acc_Mean"].min()) - 0.08), 1.0)
        ax.yaxis.set_major_formatter(_vi_comma_fmt(2))
        ax.set_title(f"{ds} — Thực nghiệm đa hạt giống (25 lượt)", fontsize=11)
        ax.set_ylabel("Điểm số", fontsize=10)
        ax.grid(True, axis="y", ls="--", alpha=0.35)
        ax.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "multiseed_comparison.png"), dpi=400)
    plt.close(fig)

    # 2. External positioning scatter chart (Twitter15 vs Twitter16 Acc in [0, 1])
    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    seen_grp = set()
    for regime, name, t15, t16, _ in EXTERNAL_BENCHMARK:
        is_grp_a = regime.startswith("Group A")
        lbl = None
        if is_grp_a and "A" not in seen_grp:
            lbl = "Nhóm A quốc tế (Source-only / rumdetect2017)"
            seen_grp.add("A")
        elif (not is_grp_a) and "B" not in seen_grp:
            lbl = "Nhóm B quốc tế (Full replies + BERT/KG)"
            seen_grp.add("B")
        ax.scatter(
            t15,
            t16,
            c="#f59e0b" if is_grp_a else "#94a3b8",
            marker="s" if is_grp_a else "o",
            s=70,
            alpha=0.88,
            edgecolors="black",
            linewidths=0.6,
            label=lbl,
        )
        short = name.split(" (")[0]
        ax.annotate(short, (t15 + 0.0014, t16 + 0.0012), fontsize=8.0, color="#334155")

    models_both = set(df[df["Dataset"] == "Twitter15"]["Model"]) & set(df[df["Dataset"] == "Twitter16"]["Model"])
    key_ours = {"TF-IDF-Ridge", "GraphSAGE-Single", "BiGCN-Base", "EBGCN", "RAGCL", "BiGCN-Plus-B12", "BiGCN-Plus-Clean", "BiGCN-Plus-UserID"}
    for m in sorted(models_both & key_ours, key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999):
        a15 = float(df[(df["Dataset"] == "Twitter15") & (df["Model"] == m)]["Acc_Mean"].iloc[0])
        a16 = float(df[(df["Dataset"] == "Twitter16") & (df["Model"] == m)]["Acc_Mean"].iloc[0])
        is_plus = "BiGCN-Plus" in m
        ax.scatter(
            a15,
            a16,
            c="#dc2626" if is_plus else "#2563eb",
            marker="*",
            s=185 if is_plus else 130,
            edgecolors="black",
            zorder=5,
        )
        dy = -0.0028 if m in {"BiGCN-Plus-Clean", "EBGCN"} else 0.0015
        ax.annotate(
            f"{m} (nhóm)",
            (a15 + 0.0016, a16 + dy),
            fontsize=8.4,
            fontweight="bold" if is_plus else "normal",
            color="#b91c1c" if is_plus else "#1d4ed8",
        )

    ax.xaxis.set_major_formatter(_vi_comma_fmt(2))
    ax.yaxis.set_major_formatter(_vi_comma_fmt(2))
    ax.set_xlabel("Độ chính xác trên Twitter15 (Accuracy)", fontsize=11)
    ax.set_ylabel("Độ chính xác trên Twitter16 (Accuracy)", fontsize=11)
    ax.set_title("Định vị hiệu năng trên Twitter15 và Twitter16 so với công trình quốc tế", fontsize=12)
    ax.legend(fontsize=8.8, loc="lower right", framealpha=0.92)
    ax.grid(True, ls="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "external_positioning.png"), dpi=400)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Aggregate CT255 Multi-Seed Rumor Detection Results")
    parser.add_argument("--results_dir", type=str, default=os.path.join(ROOT_DIR, "results", "multi_seed"))
    parser.add_argument("--out_dir", type=str, default=os.path.join(ROOT_DIR, "results"))
    args = parser.parse_args()

    tables_dir = os.path.join(args.out_dir, "tables")
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    items = load_all_results(args.results_dir)
    if not items:
        print(f"No results_multiseed.json found under {args.results_dir}")
        df = pd.DataFrame()
        build_external_benchmark_md(df, tables_dir)
        return

    df = build_summary_tables(items, tables_dir)
    build_statistical_tests(items, tables_dir)
    build_tree_sparsity_analysis(items, args.results_dir, tables_dir, fig_dir)
    build_early_detection_summary(items, args.results_dir, tables_dir, fig_dir)
    build_external_benchmark_md(df, tables_dir)
    build_latex_tables(df, tables_dir)
    plot_figures(df, fig_dir)
    print(f"Aggregated {len(items)} result files into {tables_dir} and {fig_dir}")


if __name__ == "__main__":
    main()
