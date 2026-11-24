# -*- coding: utf-8 -*-
"""
Aggregate multi-seed benchmark outputs into publication-ready tables,
statistical significance tests (Paired t-test & Wilcoxon), LaTeX tables,
external SOTA positioning (Group A vs Group B), and 300 DPI figures.
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.metrics import METRIC_KEYS, paired_significance_test

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

        # Dual evaluation comparison row (last vs snapshot5)
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
    same_eval_rows.sort(key=lambda r: (r["Dataset"], r["EvalMode"], MODEL_ORDER.index(r["Model"]) if r["Model"] in MODEL_ORDER else 999))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(tables_dir, "master_multiseed_summary.csv"), index=False)

    df_same = pd.DataFrame(same_eval_rows)
    df_same.to_csv(os.path.join(tables_dir, "same_eval_comparison.csv"), index=False)

    # Markdown master table
    md_lines = [
        "# Master Multi-Seed Summary Table",
        "",
        "| Dataset | Model | Eval | Runs | Device | Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) | Seed-Std (Acc) | Kappa | MCC | AUC-ROC |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['Dataset']} | `{r['Model']}` | {r['EvalMode']} | {int(r['Runs'])} | {r['Device']} | "
            f"{r['Acc_Mean']:.4f} ± {r['Acc_Std']:.4f} | {r['F1_Mean']:.4f} ± {r['F1_Std']:.4f} | "
            f"{r['Acc_SeedStd']:.4f} | {r['Kappa']:.4f} | {r['MCC']:.4f} | {r['AUC_ROC']:.4f} |"
        )
    with open(os.path.join(tables_dir, "master_multiseed_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # Markdown same-eval comparison
    md_se = [
        "# Same-Checkpoint Evaluation Comparison (`last` vs `snapshot5`)",
        "",
        "| Dataset | EvalMode | Model | Runs | Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) |",
        "| :--- | :---: | :--- | :---: | :---: | :---: |",
    ]
    for _, r in df_same.iterrows():
        md_se.append(
            f"| {r['Dataset']} | {r['EvalMode']} | `{r['Model']}` | {int(r['Runs'])} | "
            f"{r['Acc_Mean']:.4f} ± {r['Acc_Std']:.4f} | {r['F1_Mean']:.4f} ± {r['F1_Std']:.4f} |"
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
            f"{r['Acc_Diff_A_minus_B']:+.4f} | {r['Acc_t_pval']:.4g} | {r['Acc_Wilcoxon_pval']:.4g}{sig_tag} | "
            f"{r['F1_Diff_A_minus_B']:+.4f} | {r['F1_Wilcoxon_pval']:.4g} |"
        )
    with open(os.path.join(tables_dir, "statistical_tests.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


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
        md_lines.append(f"| {regime} | {name} | {t15*100:.2f}% | {t16*100:.2f}% | {proto} |")

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
            s15 = f"{r15.iloc[0]['Acc_Mean']*100:.2f}% ± {r15.iloc[0]['Acc_Std']*100:.2f}%" if not r15.empty else "—"
            s16 = f"{r16.iloc[0]['Acc_Mean']*100:.2f}% ± {r16.iloc[0]['Acc_Std']*100:.2f}%" if not r16.empty else "—"
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
        lines.append(
            f"{r['Dataset']} & \\texttt{{{m_tex}}} & {int(r['Runs'])} & "
            f"${r['Acc_Mean']:.4f} \\pm {r['Acc_Std']:.4f}$ & ${r['F1_Mean']:.4f} \\pm {r['F1_Std']:.4f}$ & "
            f"{r['Kappa']:.4f} & {r['AUC_ROC']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table}"])
    with open(os.path.join(tables_dir, "latex_tables.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_figures(df, fig_dir):
    if df.empty:
        return

    # 1. Multi-seed comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=False)
    for ax, ds in zip(axes, ["Twitter15", "Twitter16"]):
        sub = df[df["Dataset"] == ds].copy()
        if sub.empty:
            ax.set_title(f"{ds} (No results yet)")
            continue
        x = np.arange(len(sub))
        ax.bar(x, sub["Acc_Mean"], yerr=sub["Acc_Std"], capsize=3.5, color="#3b82f6", alpha=0.85, label="Accuracy")
        ax.plot(x, sub["F1_Mean"], color="#ef4444", marker="o", lw=1.8, label="Macro-F1")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["Model"], rotation=35, ha="right", fontsize=8.5)
        ax.set_ylim(max(0.4, float(sub["Acc_Mean"].min()) - 0.08), 1.0)
        ax.set_title(f"{ds} Multi-Seed Benchmark")
        ax.set_ylabel("Score")
        ax.grid(True, axis="y", ls="--", alpha=0.35)
        ax.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "multiseed_comparison.png"), dpi=400)
    plt.close(fig)

    # 2. External positioning scatter chart (Twitter15 vs Twitter16 Acc)
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    for regime, name, t15, t16, _ in EXTERNAL_BENCHMARK:
        is_grp_a = regime.startswith("Group A")
        ax.scatter(
            t15 * 100,
            t16 * 100,
            c="#f59e0b" if is_grp_a else "#94a3b8",
            marker="s" if is_grp_a else "o",
            s=65,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.6,
        )
        short = name.split(" (")[0]
        ax.annotate(short, (t15 * 100 + 0.15, t16 * 100 + 0.12), fontsize=7.8, color="#334155")

    models_both = set(df[df["Dataset"] == "Twitter15"]["Model"]) & set(df[df["Dataset"] == "Twitter16"]["Model"])
    for m in sorted(models_both, key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999):
        a15 = float(df[(df["Dataset"] == "Twitter15") & (df["Model"] == m)]["Acc_Mean"].iloc[0]) * 100
        a16 = float(df[(df["Dataset"] == "Twitter16") & (df["Model"] == m)]["Acc_Mean"].iloc[0]) * 100
        ax.scatter(a15, a16, c="#dc2626" if "BiGCN-Plus" in m else "#2563eb", marker="*", s=160, edgecolors="black", zorder=5)
        ax.annotate(f"{m} (ours)", (a15 + 0.18, a16 - 0.25), fontsize=8.2, fontweight="bold", color="#b91c1c" if "BiGCN-Plus" in m else "#1d4ed8")

    ax.set_xlabel("Twitter15 Accuracy (%)")
    ax.set_ylabel("Twitter16 Accuracy (%)")
    ax.set_title("Positioning Against Published Twitter15 / Twitter16 Literature")
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
    build_external_benchmark_md(df, tables_dir)
    build_latex_tables(df, tables_dir)
    plot_figures(df, fig_dir)
    print(f"Aggregated {len(items)} result files into {tables_dir} and {fig_dir}")


if __name__ == "__main__":
    main()
