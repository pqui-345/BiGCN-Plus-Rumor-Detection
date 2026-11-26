# -*- coding: utf-8 -*-
"""
Unified Multi-Seed x 5-Fold Benchmark Runner for CT255 Rumor Detection.
Runs Level 1 (NLP & Single-View GNN), Level 2 (BiGCN, ablations, BiGCN-Base+, EBGCN, RAGCL),
and Level 3 (BiGCN-Plus-Clean, User-ID ablation, Branch ablations, BiGCN-Plus-PLM).

Outputs per (dataset, model) directory under --save_dir:
  - fold_seed_results.csv   (all seed x fold metrics for both eval_mode='last' and 'snapshot5')
  - epoch_logs.csv          (per-epoch loss/acc/f1 for neural models)
  - results_multiseed.json  (aggregated multi-seed mean, std_all_runs, std_across_seeds, alt_eval)
  - summary_report.txt
  - confusion_matrix.png
  - training_curves.png
"""
import argparse
import json
import os
import random
import sys
import time
import types
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.baselines_gnn import build_baseline_model
from src.baselines_nlp import (
    extract_plm_embeddings,
    run_plm_fold,
    run_tfidf_linear_fold,
    run_tfidf_ridge_fold,
)
from src.bigcn_plus import BiGCNPlus, compute_branch3_logits
from src.data_loader import (
    EARLY_DEADLINES_MIN,
    LABELS,
    N_STRUCT_BASE,
    N_STRUCT_EXT,
    build_sublinear_tfidf_features,
    find_data_root,
    fit_transform_bow_count,
    get_kfold_splits,
    load_dataset,
    make_batch,
    truncate_tree_by_delay,
)
from src.metrics import (
    METRIC_KEYS,
    compute_metrics,
    confusion_matrix_np,
    summarize_multiseed_records,
)

NLP_MODELS = ["TF-IDF-Ridge", "TF-IDF-LR", "TF-IDF-SVC", "Text-PLM"]
LEVEL1_MODELS = ["MLP-BoW", "GCN-Single", "GraphSAGE-Single", "GAT-Single"]
LEVEL2_MODELS = [
    "BiGCN-Base",
    "BiGCN-NoRoot",
    "BiGCN-NoDropEdge",
    "BiGCN-TDOnly",
    "BiGCN-BUOnly",
    "BiGCN-Base+",
    "EBGCN",
    "RAGCL",
]
LEVEL3_MODELS = [
    "BiGCN-Plus-B1Only",
    "BiGCN-Plus-B2Only",
    "BiGCN-Plus-B12",
    "BiGCN-Plus-Clean",
    "BiGCN-Plus-UserID",
    "BiGCN-Plus-PLM",
]
CORE_MODELS = [
    "TF-IDF-Ridge",
    "TF-IDF-LR",
    "MLP-BoW",
    "GCN-Single",
    "GraphSAGE-Single",
    "GAT-Single",
    "BiGCN-Base",
    "BiGCN-Base+",
    "EBGCN",
    "RAGCL",
    "BiGCN-Plus-B12",
    "BiGCN-Plus-Clean",
    "BiGCN-Plus-UserID",
]
ALL_MODELS = NLP_MODELS + LEVEL1_MODELS + LEVEL2_MODELS + LEVEL3_MODELS

EXT_STRUCT_MODELS = {
    "BiGCN-Base+",
    "EBGCN",
    "RAGCL",
    "BiGCN-Plus-B1Only",
    "BiGCN-Plus-B2Only",
    "BiGCN-Plus-B12",
    "BiGCN-Plus-Clean",
    "BiGCN-Plus-UserID",
    "BiGCN-Plus-PLM",
}
DEFAULT_SNAPSHOT_MODELS = {
    "BiGCN-Base+",
    "BiGCN-Plus-B1Only",
    "BiGCN-Plus-B2Only",
    "BiGCN-Plus-B12",
    "BiGCN-Plus-Clean",
    "BiGCN-Plus-UserID",
    "BiGCN-Plus-PLM",
}
EARLY_EVAL_MODELS = {
    "TF-IDF-Ridge",
    "MLP-BoW",
    "GraphSAGE-Single",
    "BiGCN-Base",
    "EBGCN",
    "RAGCL",
    "BiGCN-Plus-Clean",
}
ARCH_VERSION = "v3_auto_loo_oof"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(path):
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def make_optimizer(model, lr, weight_decay):
    """Apply differential lr/5 on bottom-up convolutional branch as in Bian et al. (2020)."""
    if hasattr(model, "bu") and model.bu is not None:
        bu_trainable = [p for p in model.bu.parameters() if p.requires_grad]
        bu_ids = {id(p) for p in bu_trainable}
        base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in bu_ids]
        if bu_trainable:
            return torch.optim.Adam(
                [{"params": base_params, "lr": lr}, {"params": bu_trainable, "lr": lr / 5.0}],
                weight_decay=weight_decay,
            )
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def save_confusion_matrix(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(LABELS)),
        yticks=np.arange(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Nhãn dự đoán",
        ylabel="Nhãn thực tế",
        title=title,
    )
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    ax.title.set_fontsize(12)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    row_sums = cm.sum(axis=1, keepdims=True)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = 100.0 * cm[i, j] / max(row_sums[i, 0], 1)
            pct_str = f"{pct:.1f}%".replace(".", ",")
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}\n({pct_str})", ha="center", va="center", color=color, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def save_training_curves(epoch_df, title, out_path):
    if epoch_df.empty:
        return
    from matplotlib.ticker import FuncFormatter

    fmt_vi2 = FuncFormatter(lambda x, pos: f"{x:.2f}".replace(".", ","))
    fmt_vi3 = FuncFormatter(lambda x, pos: f"{x:.3f}".replace(".", ","))
    agg = epoch_df.groupby("epoch")[["train_loss", "test_acc", "test_f1"]].agg(["mean", "std"]).reset_index()
    epochs = agg["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6))

    loss_m = agg["train_loss"]["mean"].fillna(0.0)
    loss_s = agg["train_loss"]["std"].fillna(0.0)
    ax1.plot(epochs, loss_m, color="#1f77b4", lw=2.0, label="Mean Loss")
    if (loss_s > 0).any():
        ax1.fill_between(epochs, (loss_m - loss_s).clip(lower=0.0), loss_m + loss_s, color="#1f77b4", alpha=0.16)
    ax1.set_title(f"{title} — Mất mát huấn luyện", fontsize=11.0)
    ax1.set_xlabel("Vòng lặp huấn luyện (Epoch)", fontsize=10.5)
    ax1.set_ylabel("Hàm mất mát (Loss)", fontsize=10.5)
    ax1.yaxis.set_major_formatter(fmt_vi2)
    ax1.grid(True, ls="--", alpha=0.35)

    acc_m = agg["test_acc"]["mean"].fillna(0.0)
    acc_s = agg["test_acc"]["std"].fillna(0.0)
    f1_m = agg["test_f1"]["mean"].fillna(0.0)
    ax2.plot(epochs, acc_m, label="Accuracy (TB 5 seeds)", color="#2ca02c", lw=2.0)
    if (acc_s > 0).any():
        ax2.fill_between(epochs, acc_m - acc_s, (acc_m + acc_s).clip(upper=1.0), color="#2ca02c", alpha=0.14)
    ax2.plot(epochs, f1_m, label="Macro-F1 (TB 5 seeds)", color="#ff7f0e", lw=2.0)
    ax2.set_title(f"{title} — Accuracy & Macro-F1 kiểm thử", fontsize=11.0)
    ax2.set_xlabel("Vòng lặp huấn luyện (Epoch)", fontsize=10.5)
    ax2.set_ylabel("Điểm số", fontsize=10.5)
    ax2.yaxis.set_major_formatter(fmt_vi3)
    ax2.legend(fontsize=9.5, loc="lower right")
    ax2.grid(True, ls="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _build_device_tree_map(trees, indices, n_struct, device):
    return {
        int(i): {
            "n": int(trees[i]["n"]),
            "edge_index": torch.tensor(trees[i]["edge_index"], dtype=torch.long, device=device),
            "struct": torch.tensor(trees[i]["struct"][:, :n_struct], dtype=torch.float32, device=device),
            "centrality": torch.tensor(trees[i]["centrality"], dtype=torch.float32, device=device),
        }
        for i in indices
    }


def _eval_batches_probs(model, batches):
    probs = []
    with torch.no_grad():
        for b in batches:
            probs.append(torch.exp(model(b)).cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_gnn_fold(
    model_name,
    docs,
    trees,
    y,
    tr_idx,
    te_idx,
    args,
    device,
    seed,
    fold,
    plm_embs=None,
    trees_by_deadline=None,
    primary_mode="last",
):
    set_seed(seed * 100 + fold)
    n_struct = N_STRUCT_EXT if model_name in EXT_STRUCT_MODELS else N_STRUCT_BASE

    if model_name == "BiGCN-Base+":
        text_mat = (
            build_sublinear_tfidf_features(docs, tr_idx, max_features=args.vocab_size).toarray().astype(np.float32)
            * np.float32(np.sqrt(12.0))
        )
    else:
        text_mat = fit_transform_bow_count(docs, tr_idx, max_features=args.vocab_size)
    vocab = text_mat.shape[1]

    trees_t = [
        {
            "n": int(t["n"]),
            "edge_index": torch.tensor(t["edge_index"], dtype=torch.long, device=device),
            "struct": torch.tensor(t["struct"][:, :n_struct], dtype=torch.float32, device=device),
            "centrality": torch.tensor(t["centrality"], dtype=torch.float32, device=device),
        }
        for t in trees
    ]
    text_t = torch.tensor(text_mat, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    b3_t = None
    if model_name in {"BiGCN-Plus-Clean", "BiGCN-Plus-UserID", "BiGCN-Plus-PLM"}:
        use_uid = model_name == "BiGCN-Plus-UserID"
        b3_np = compute_branch3_logits(
            docs,
            trees,
            y,
            tr_idx,
            te_idx,
            use_user_id=use_uid,
            u_weight=args.u_weight_full if use_uid else 0.0,
            g_weight=0.0 if use_uid else 0.25,
            ridge_lam=args.ridge_lam,
            max_features=args.b3_max_features,
            auto_tune=not use_uid,
        )
        b3_t = torch.tensor(b3_np, dtype=torch.float32, device=device)

    plm_t = torch.tensor(plm_embs, dtype=torch.float32, device=device) if (model_name == "BiGCN-Plus-PLM" and plm_embs is not None) else None

    if model_name.startswith("BiGCN-Plus-"):
        mode_map = {
            "BiGCN-Plus-B1Only": "b1_only",
            "BiGCN-Plus-B2Only": "b2_only",
            "BiGCN-Plus-B12": "b12",
            "BiGCN-Plus-Clean": "clean",
            "BiGCN-Plus-UserID": "userid",
            "BiGCN-Plus-PLM": "clean",
        }
        plm_dim = int(plm_t.size(1)) if plm_t is not None else 0
        model = BiGCNPlus(
            vocab=vocab,
            hid=args.hidden_dim,
            sage_hid=args.sage_dim,
            n_cls=4,
            td_drop=args.td_droprate,
            bu_drop=args.bu_droprate,
            mode=mode_map[model_name],
            plm_dim=plm_dim,
        ).to(device)
    else:
        m_args = types.SimpleNamespace(
            hidden_dim=args.hidden_dim,
            out_dim=args.hidden_dim,
            td_droprate=args.td_droprate,
            bu_droprate=args.bu_droprate,
        )
        model = build_baseline_model(model_name, vocab, m_args).to(device)

    optimizer = make_optimizer(model, args.lr, args.weight_decay)

    te_batches = []
    for s in range(0, len(te_idx), args.batch_size):
        b_idx = te_idx[s : s + args.batch_size]
        te_batches.append(make_batch(b_idx, trees_t, text_t, y_t, device, n_struct=n_struct, branch3_t=b3_t, plm_t=plm_t))

    eval_early = bool(getattr(args, "run_early", True) and trees_by_deadline is not None and model_name in EARLY_EVAL_MODELS)
    dt_te_batches = {}
    early_snap_probs = {}
    if eval_early and model_name != "MLP-BoW":
        for dt in EARLY_DEADLINES_MIN:
            if dt == -1:
                continue
            trees_dt = trees_by_deadline[dt]
            dt_tree_map = _build_device_tree_map(trees_dt, te_idx, n_struct, device)
            b3_dt_t = None
            if model_name in {"BiGCN-Plus-Clean", "BiGCN-Plus-UserID", "BiGCN-Plus-PLM"}:
                b3_dt_np = compute_branch3_logits(
                    docs,
                    trees_dt,
                    y,
                    tr_idx,
                    te_idx,
                    use_user_id=False,
                    u_weight=0.0,
                    g_weight=0.25,
                    ridge_lam=args.ridge_lam,
                    max_features=args.b3_max_features,
                    auto_tune=True,
                )
                b3_dt_t = torch.tensor(b3_dt_np, dtype=torch.float32, device=device)
            batches_dt = []
            for s in range(0, len(te_idx), args.batch_size):
                b_idx = te_idx[s : s + args.batch_size]
                batches_dt.append(
                    make_batch(b_idx, dt_tree_map, text_t, y_t, device, n_struct=n_struct, branch3_t=b3_dt_t, plm_t=plm_t)
                )
            dt_te_batches[dt] = batches_dt
            early_snap_probs[dt] = []

    epoch_rows = []
    snap_probs = []
    last_prob = None
    y_te_np = y[te_idx]
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(tr_idx)
        tot_loss = 0.0
        n_steps = 0
        for s in range(0, len(perm), args.batch_size):
            b_idx = perm[s : s + args.batch_size]
            b = make_batch(b_idx, trees_t, text_t, y_t, device, n_struct=n_struct, branch3_t=b3_t, plm_t=plm_t)
            optimizer.zero_grad()
            if model_name.startswith("BiGCN-Plus-"):
                out, l_b1, l_b2 = model(b, return_aux=True)
                loss = F.nll_loss(out, b.y)
                if model.mode not in {"b1_only", "b2_only"}:
                    loss = loss + 0.35 * F.cross_entropy(l_b1, b.y) + 0.25 * F.cross_entropy(l_b2, b.y)
            elif model_name == "EBGCN":
                out, reg_loss = model(b, return_reg=True)
                loss = F.nll_loss(out, b.y) + reg_loss
            elif model_name == "RAGCL":
                out, cl_loss = model(b, return_cl=True)
                loss = F.nll_loss(out, b.y) + cl_loss
            else:
                out = model(b)
                loss = F.nll_loss(out, b.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            tot_loss += float(loss.item())
            n_steps += 1

        model.eval()
        last_prob = _eval_batches_probs(model, te_batches)
        ep_pred = last_prob.argmax(axis=1)
        ep_m = compute_metrics(y_te_np, ep_pred, last_prob)

        if ep > max(0, args.epochs - args.snapshot_k):
            snap_probs.append(last_prob)
            if eval_early and primary_mode == "snapshot5" and model_name != "MLP-BoW":
                for dt, batches_dt in dt_te_batches.items():
                    early_snap_probs[dt].append(_eval_batches_probs(model, batches_dt))

        if seed == args.seeds[0] or ep == args.epochs:
            epoch_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "epoch": ep,
                    "train_loss": tot_loss / max(n_steps, 1),
                    "test_acc": ep_m["Accuracy"],
                    "test_f1": ep_m["Macro-F1"],
                }
            )

    sec_per_ep = (time.time() - t0) / max(args.epochs, 1)
    snap_prob = np.mean(snap_probs, axis=0) if snap_probs else last_prob

    early_probs = None
    if eval_early:
        early_probs = {}
        full_prob = snap_prob if primary_mode == "snapshot5" else last_prob
        if model_name == "MLP-BoW":
            for dt in EARLY_DEADLINES_MIN:
                early_probs[dt] = full_prob
        else:
            model.eval()
            for dt in EARLY_DEADLINES_MIN:
                if dt == -1:
                    early_probs[dt] = full_prob
                elif primary_mode == "snapshot5" and early_snap_probs.get(dt):
                    early_probs[dt] = np.mean(early_snap_probs[dt], axis=0)
                else:
                    early_probs[dt] = _eval_batches_probs(model, dt_te_batches[dt])

    return y_te_np, last_prob, snap_prob, sec_per_ep, epoch_rows, early_probs


def run_one_model(dataset, model_name, ids, docs, trees, y, plm_embs, args, device, trees_by_deadline=None):
    out_dir = os.path.join(args.save_dir, f"{dataset}_{model_name}")
    os.makedirs(out_dir, exist_ok=True)
    is_nlp = model_name in NLP_MODELS
    primary_mode = "snapshot5" if (not is_nlp and model_name in DEFAULT_SNAPSHOT_MODELS) else "last"

    all_records = []
    epoch_logs = []
    early_rows = []
    cm_primary = np.zeros((4, 4), dtype=np.int64)
    sec_list = []

    n_total = len(y)
    oof_prob_sum = np.zeros((n_total, 4), dtype=np.float64)
    oof_counts = np.zeros(n_total, dtype=np.int64)
    record_early = bool(getattr(args, "run_early", True) and model_name in EARLY_EVAL_MODELS and (is_nlp or trees_by_deadline is not None))

    for seed in args.seeds:
        splits = get_kfold_splits(y, n_folds=args.n_folds, seed=seed)
        for fold in args.folds:
            tr_idx, te_idx = splits[fold]
            t_start = time.time()

            if is_nlp:
                if model_name == "TF-IDF-Ridge":
                    y_te, pred, prob = run_tfidf_ridge_fold(
                        docs, y, tr_idx, te_idx, max_features=args.b3_max_features, ridge_lam=args.ridge_lam
                    )
                elif model_name == "TF-IDF-LR":
                    y_te, pred, prob = run_tfidf_linear_fold(
                        docs, y, tr_idx, te_idx, variant="lr", max_features=args.b3_max_features, seed=seed
                    )
                elif model_name == "TF-IDF-SVC":
                    y_te, pred, prob = run_tfidf_linear_fold(
                        docs, y, tr_idx, te_idx, variant="svc", max_features=args.b3_max_features, seed=seed
                    )
                elif model_name == "Text-PLM":
                    y_te, pred, prob = run_plm_fold(
                        docs,
                        y,
                        tr_idx,
                        te_idx,
                        model_name=args.plm_name,
                        epochs=args.plm_epochs,
                        batch_size=args.plm_batch_size,
                        lr=args.plm_lr,
                        max_length=args.plm_max_length,
                        seed=seed * 10 + fold,
                        device=device,
                    )
                else:
                    raise ValueError(model_name)

                sec_ep = time.time() - t_start
                sec_list.append(sec_ep)
                m = compute_metrics(y_te, pred, prob)
                all_records.append({"seed": seed, "fold": fold, "eval_mode": "last", "sec_per_ep": sec_ep, **m})
                cm_primary += confusion_matrix_np(y_te, pred)
                oof_prob_sum[te_idx] += prob
                oof_counts[te_idx] += 1
                if record_early:
                    for dt in EARLY_DEADLINES_MIN:
                        early_rows.append({"seed": seed, "fold": fold, "deadline_min": dt, **m})
                print(f"[{dataset}][{model_name}] seed={seed} fold={fold} Acc={m['Accuracy']:.4f} F1={m['Macro-F1']:.4f} ({sec_ep:.2f}s)")
            else:
                y_te, last_prob, snap_prob, sec_ep, ep_rows, early_probs = train_gnn_fold(
                    model_name,
                    docs,
                    trees,
                    y,
                    tr_idx,
                    te_idx,
                    args,
                    device,
                    seed,
                    fold,
                    plm_embs=plm_embs,
                    trees_by_deadline=trees_by_deadline if record_early else None,
                    primary_mode=primary_mode,
                )
                sec_list.append(sec_ep)
                epoch_logs.extend(ep_rows)

                pred_last = last_prob.argmax(axis=1)
                m_last = compute_metrics(y_te, pred_last, last_prob)
                all_records.append({"seed": seed, "fold": fold, "eval_mode": "last", "sec_per_ep": sec_ep, **m_last})

                pred_snap = snap_prob.argmax(axis=1)
                m_snap = compute_metrics(y_te, pred_snap, snap_prob)
                all_records.append({"seed": seed, "fold": fold, "eval_mode": "snapshot5", "sec_per_ep": sec_ep, **m_snap})

                chosen_prob = snap_prob if primary_mode == "snapshot5" else last_prob
                chosen_pred = pred_snap if primary_mode == "snapshot5" else pred_last
                cm_primary += confusion_matrix_np(y_te, chosen_pred)
                oof_prob_sum[te_idx] += chosen_prob
                oof_counts[te_idx] += 1

                if record_early and early_probs is not None:
                    for dt in EARLY_DEADLINES_MIN:
                        dt_prob = early_probs[dt]
                        dt_pred = dt_prob.argmax(axis=1)
                        dt_m = compute_metrics(y_te, dt_pred, dt_prob)
                        early_rows.append({"seed": seed, "fold": fold, "deadline_min": dt, **dt_m})

                print(
                    f"[{dataset}][{model_name}] seed={seed} fold={fold} "
                    f"Last=({m_last['Accuracy']:.4f}/{m_last['Macro-F1']:.4f}) "
                    f"Snap5=({m_snap['Accuracy']:.4f}/{m_snap['Macro-F1']:.4f}) "
                    f"({sec_ep:.3f}s/ep)"
                )

    df_res = pd.DataFrame(all_records)
    df_res.to_csv(os.path.join(out_dir, "fold_seed_results.csv"), index=False)

    valid_idx = np.where(oof_counts > 0)[0]
    oof_rows = []
    for idx in valid_idx:
        mean_p = oof_prob_sum[idx] / float(oof_counts[idx])
        oof_rows.append(
            {
                "tid": ids[idx],
                "y_true": int(y[idx]),
                "y_pred": int(mean_p.argmax()),
                "p_NR": float(mean_p[0]),
                "p_FR": float(mean_p[1]),
                "p_TR": float(mean_p[2]),
                "p_UR": float(mean_p[3]),
                "n_nodes": int(trees[idx]["n_full"]),
                "max_depth": int(trees[idx]["max_depth"]),
            }
        )
    pd.DataFrame(oof_rows).to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)

    if early_rows:
        pd.DataFrame(early_rows).to_csv(os.path.join(out_dir, "early_detection.csv"), index=False)

    if epoch_logs:
        df_ep = pd.DataFrame(epoch_logs)
        df_ep.to_csv(os.path.join(out_dir, "epoch_logs.csv"), index=False)
        save_training_curves(df_ep, f"{dataset} — {model_name}", os.path.join(out_dir, "training_curves.png"))

    save_confusion_matrix(cm_primary, f"{dataset} — {model_name} ({primary_mode})", os.path.join(out_dir, "confusion_matrix.png"))

    prim_rows = [r for r in all_records if r["eval_mode"] == primary_mode]
    prim_summary = summarize_multiseed_records(prim_rows)
    alt_mode = "last" if primary_mode == "snapshot5" else "snapshot5"
    alt_rows = [r for r in all_records if r["eval_mode"] == alt_mode]
    alt_summary = summarize_multiseed_records(alt_rows) if alt_rows else None

    payload = {
        "dataset": dataset,
        "model": model_name,
        "arch_version": ARCH_VERSION,
        "primary_eval_mode": primary_mode,
        "has_oof": True,
        "has_early": bool(early_rows),
        "device": str(device),
        "seeds": list(args.seeds),
        "folds": list(args.folds),
        "epochs": 1 if is_nlp else args.epochs,
        "snapshot_k": 1 if is_nlp else args.snapshot_k,
        "sec_per_epoch_mean": float(np.mean(sec_list)) if sec_list else 0.0,
        "confusion_matrix_sum": cm_primary.tolist(),
        "summary": prim_summary,
        "alt_eval_mode": alt_mode if alt_summary else None,
        "alt_summary": alt_summary,
        "per_run_records": all_records,
    }
    with open(os.path.join(out_dir, "results_multiseed.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        f"Dataset: {dataset} | Model: {model_name}",
        f"Device: {device} | Primary Eval: {primary_mode}",
        f"Seeds ({len(args.seeds)}): {args.seeds} | Folds ({len(args.folds)}): {args.folds} | Total Runs: {prim_summary['n_runs']}",
        f"Mean Time: {payload['sec_per_epoch_mean']:.4f} s/ep (or fold)",
        "-" * 64,
    ]
    for k in METRIC_KEYS:
        m_v = prim_summary["mean"][k]
        s_v = prim_summary["std_all_runs"][k]
        ss_v = prim_summary["std_across_seeds"][k]
        lines.append(f"{k:<18}: {m_v:.4f} +/- {s_v:.4f} (seed-std={ss_v:.4f})")
    with open(os.path.join(out_dir, "summary_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(
        f"==> DONE [{dataset}][{model_name}] ({primary_mode}, {prim_summary['n_runs']} runs): "
        f"Acc={prim_summary['mean']['Accuracy']:.4f}+/-{prim_summary['std_all_runs']['Accuracy']:.4f} | "
        f"F1={prim_summary['mean']['Macro-F1']:.4f}+/-{prim_summary['std_all_runs']['Macro-F1']:.4f}"
    )


def resolve_models(suite, explicit_models, skip_plm):
    if explicit_models:
        models = list(explicit_models)
    else:
        suite_map = {
            "nlp": NLP_MODELS,
            "level1": LEVEL1_MODELS,
            "level2": LEVEL2_MODELS,
            "level3": LEVEL3_MODELS,
            "core": CORE_MODELS,
            "all": ALL_MODELS,
        }
        models = list(suite_map[suite])
    if skip_plm:
        models = [m for m in models if m not in {"Text-PLM", "BiGCN-Plus-PLM"}]
    return models


def main():
    parser = argparse.ArgumentParser(description="CT255 Rumor Detection Multi-Seed Benchmark")
    parser.add_argument("--config", type=str, default=os.path.join(ROOT_DIR, "configs", "default.yaml"))
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="ALL", choices=["Twitter15", "Twitter16", "ALL"])
    parser.add_argument("--suite", type=str, default="core", choices=["nlp", "level1", "level2", "level3", "core", "all"])
    parser.add_argument("--models", nargs="+", default=None, choices=ALL_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--sage_dim", type=int, default=None)
    parser.add_argument("--snapshot_k", type=int, default=None)
    parser.add_argument("--plm_name", type=str, default=None)
    parser.add_argument("--skip_plm", action="store_true")
    parser.add_argument("--run_early", action="store_true", help="Evaluate early-detection trajectory on key models (default enabled)")
    parser.add_argument("--disable_early", action="store_true", help="Disable early-detection evaluation")
    parser.add_argument("--save_dir", type=str, default=os.path.join(ROOT_DIR, "results", "multi_seed"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    args.run_early = not args.disable_early

    cfg = load_yaml_config(args.config)
    exp = cfg.get("experiment", {})
    b3 = cfg.get("branch3", {})
    plm = cfg.get("plm", {})

    args.seeds = args.seeds or exp.get("seeds", [42, 43, 44, 45, 46])
    args.n_folds = exp.get("n_folds", 5)
    args.epochs = args.epochs or exp.get("epochs", 100)
    args.batch_size = args.batch_size or exp.get("batch_size", 128)
    args.lr = args.lr or exp.get("lr", 0.0005)
    args.weight_decay = args.weight_decay or exp.get("weight_decay", 0.0001)
    args.hidden_dim = args.hidden_dim or exp.get("hidden_dim", 64)
    args.sage_dim = args.sage_dim or exp.get("sage_dim", 32)
    args.td_droprate = exp.get("td_droprate", 0.2)
    args.bu_droprate = exp.get("bu_droprate", 0.2)
    args.vocab_size = exp.get("vocab_size", 5000)
    args.snapshot_k = args.snapshot_k or exp.get("snapshot_k", 5)

    args.b3_max_features = b3.get("max_features", 40000)
    args.ridge_lam = b3.get("ridge_lam", 0.45)
    args.u_weight_full = b3.get("u_weight_full", 0.95)

    args.plm_name = args.plm_name or plm.get("model_name", "roberta-base")
    args.plm_epochs = plm.get("epochs", 6)
    args.plm_batch_size = plm.get("batch_size", 16)
    args.plm_lr = plm.get("lr", 2e-5)
    args.plm_max_length = plm.get("max_length", 128)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = find_data_root(args.data_root)
    datasets = ["Twitter15", "Twitter16"] if args.dataset == "ALL" else [args.dataset]
    models = resolve_models(args.suite, args.models, args.skip_plm)

    print(f"Data root  : {data_root}")
    print(f"Device     : {device}")
    print(f"Datasets   : {datasets}")
    print(f"Models     : {models}")
    print(f"Seeds      : {args.seeds} | Folds: {args.folds} | Epochs: {args.epochs}")
    print(f"Early Eval : {args.run_early}")

    for ds in datasets:
        ids, y, docs, trees = load_dataset(data_root, ds)
        print(f"\nLoaded {ds}: N={len(y)}, class_counts={np.bincount(y).tolist()}")

        trees_by_deadline = None
        if args.run_early and any(m in EARLY_EVAL_MODELS for m in models):
            print(f"Preparing temporally truncated trees for {ds} across {EARLY_DEADLINES_MIN}...")
            trees_by_deadline = {
                dt: ([truncate_tree_by_delay(t, dt) for t in trees] if dt >= 0 else trees)
                for dt in EARLY_DEADLINES_MIN
            }

        plm_embs = None
        if "BiGCN-Plus-PLM" in models:
            print(f"Extracting frozen PLM embeddings ({args.plm_name}) for {ds}...")
            plm_embs = extract_plm_embeddings(
                docs, model_name=args.plm_name, batch_size=64, max_length=args.plm_max_length, device=device
            )

        for model_name in models:
            run_one_model(ds, model_name, ids, docs, trees, y, plm_embs, args, device, trees_by_deadline=trees_by_deadline)


if __name__ == "__main__":
    main()
