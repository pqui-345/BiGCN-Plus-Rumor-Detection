# -*- coding: utf-8 -*-
"""
Export compact float16 inference bundle (<Dataset>_demo_bundle.pt) from a single
leakage-free fold (default seed=42, fold=0) for the interactive Web Demo.
Trains MLP-BoW, BiGCN-Base, RAGCL, and BiGCN-Plus-Clean + Branch 3 Auto-LOO primal weights.
"""
import argparse
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.baselines_gnn import build_baseline_model
from src.bigcn_plus import BiGCNPlus, compute_branch3_logits
from src.data_loader import (
    N_STRUCT_BASE,
    N_STRUCT_EXT,
    build_sublinear_tfidf_features,
    find_data_root,
    fit_transform_bow_count,
    get_kfold_splits,
    load_dataset,
    make_batch,
    transform_sublinear_tfidf_texts,
)
from src.metrics import compute_metrics

DEMO_MODELS = ["MLP-BoW", "BiGCN-Base", "RAGCL", "BiGCN-Plus-Clean"]
EXT_STRUCT = {"RAGCL", "BiGCN-Plus-Clean"}


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_yaml(path):
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _make_optimizer(model, lr, weight_decay):
    if hasattr(model, "bu") and model.bu is not None:
        bu_p = [p for p in model.bu.parameters() if p.requires_grad]
        bu_ids = {id(p) for p in bu_p}
        base_p = [p for p in model.parameters() if p.requires_grad and id(p) not in bu_ids]
        if bu_p:
            return torch.optim.Adam(
                [{"params": base_p, "lr": lr}, {"params": bu_p, "lr": lr / 5.0}],
                weight_decay=weight_decay,
            )
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def _state_to_fp16(state_dict):
    return {
        k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu())
        for k, v in state_dict.items()
    }


def _reconstruct_branch3_logits(docs, trees, artifacts):
    """Primal inference matching compute_branch3_logits without fitting on test samples."""
    Xt = transform_sublinear_tfidf_texts(docs, artifacts["vocab"], artifacts["idf"])
    raw = (Xt @ artifacts["W_text"]).astype(np.float32)
    if artifacts["has_graph_var"] and artifacts["best_g_weight"] > 0.0:
        gs = np.stack([t["graph_stats"] for t in trees], axis=0).astype(np.float32)
        gs_z = (gs - artifacts["gs_mu"]) / artifacts["gs_sd"]
        norms = np.maximum(np.linalg.norm(gs_z, axis=1, keepdims=True), 1e-6)
        gs_n = gs_z / norms
        raw = raw + (gs_n @ artifacts["W_graph"]).astype(np.float32)
    return ((raw - raw.mean(axis=1, keepdims=True)) / artifacts["std_tr"]).astype(np.float32)


def export_dataset_bundle(dataset, data_root, out_dir, cfg, args, device):
    exp = cfg.get("experiment", {})
    b3 = cfg.get("branch3", {})

    hidden_dim = args.hidden_dim or exp.get("hidden_dim", 64)
    sage_dim = args.sage_dim or exp.get("sage_dim", 32)
    td_drop = exp.get("td_droprate", 0.2)
    bu_drop = exp.get("bu_droprate", 0.2)
    vocab_size = exp.get("vocab_size", 5000)
    lr = exp.get("lr", 0.0005)
    weight_decay = exp.get("weight_decay", 0.0001)
    batch_size = exp.get("batch_size", 128)
    b3_max_features = b3.get("max_features", 40000)
    ridge_lam = b3.get("ridge_lam", 0.45)

    ids, y, docs, trees = load_dataset(data_root, dataset)
    splits = get_kfold_splits(y, n_folds=args.n_folds, seed=args.seed)
    tr_idx, te_idx = splits[args.fold]

    bow_mat, bow_vocab = fit_transform_bow_count(docs, tr_idx, max_features=vocab_size, return_vectorizer=True)
    vocab = bow_mat.shape[1]

    b3_logits, b3_artifacts = compute_branch3_logits(
        docs,
        trees,
        y,
        tr_idx,
        te_idx,
        use_user_id=False,
        u_weight=0.0,
        g_weight=0.25,
        ridge_lam=ridge_lam,
        max_features=b3_max_features,
        auto_tune=True,
        return_artifacts=True,
    )

    # Pure-text TF-IDF-Ridge primal weights (for fast live Demo comparison)
    Xt_pure, pure_vocab, pure_idf = build_sublinear_tfidf_features(
        docs, tr_idx, max_features=b3_max_features, return_vectorizer=True
    )
    Xt_pure_tr = Xt_pure[tr_idx]
    K_pure_tr = (Xt_pure_tr @ Xt_pure_tr.T).toarray().astype(np.float64)
    Y_tr = np.eye(4, dtype=np.float64)[y[tr_idx]]
    alpha_pure = np.linalg.solve(K_pure_tr + ridge_lam * np.eye(len(tr_idx)), Y_tr)
    W_ridge_pure = (Xt_pure_tr.T @ alpha_pure).astype(np.float32)

    # Verify Branch 3 primal reconstruction fidelity on unseen test split
    b3_recon_te = _reconstruct_branch3_logits([docs[i] for i in te_idx], [trees[i] for i in te_idx], b3_artifacts)
    b3_recon_err = float(np.max(np.abs(b3_recon_te - b3_logits[te_idx])))
    print(
        f"[{dataset}] Branch 3 Auto-LOO selected (g_weight={b3_artifacts['best_g_weight']}, "
        f"lam={b3_artifacts['best_lam']}); test primal recon error = {b3_recon_err:.2e}"
    )

    text_t = torch.tensor(bow_mat, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    b3_t = torch.tensor(b3_logits, dtype=torch.float32, device=device)

    trees_t_base = [
        {
            "n": int(t["n"]),
            "edge_index": torch.tensor(t["edge_index"], dtype=torch.long, device=device),
            "struct": torch.tensor(t["struct"][:, :N_STRUCT_BASE], dtype=torch.float32, device=device),
            "centrality": torch.tensor(t["centrality"], dtype=torch.float32, device=device),
        }
        for t in trees
    ]
    trees_t_ext = [
        {
            "n": int(t["n"]),
            "edge_index": trees_t_base[i]["edge_index"],
            "struct": torch.tensor(t["struct"][:, :N_STRUCT_EXT], dtype=torch.float32, device=device),
            "centrality": trees_t_base[i]["centrality"],
        }
        for i, t in enumerate(trees)
    ]

    exported_states = {}
    te_metrics = {}

    for m_name in DEMO_MODELS:
        _set_seed(args.seed * 100 + args.fold)
        n_struct = N_STRUCT_EXT if m_name in EXT_STRUCT else N_STRUCT_BASE
        t_store = trees_t_ext if m_name in EXT_STRUCT else trees_t_base
        use_b3 = b3_t if m_name == "BiGCN-Plus-Clean" else None

        if m_name == "BiGCN-Plus-Clean":
            model = BiGCNPlus(
                vocab=vocab,
                hid=hidden_dim,
                sage_hid=sage_dim,
                n_cls=4,
                td_drop=td_drop,
                bu_drop=bu_drop,
                mode="clean",
                plm_dim=0,
            ).to(device)
        else:
            m_args = types.SimpleNamespace(
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                td_droprate=td_drop,
                bu_droprate=bu_drop,
            )
            model = build_baseline_model(m_name, vocab, m_args).to(device)

        optimizer = _make_optimizer(model, lr, weight_decay)
        te_batches = [
            make_batch(te_idx[s : s + batch_size], t_store, text_t, y_t, device, n_struct=n_struct, branch3_t=use_b3)
            for s in range(0, len(te_idx), batch_size)
        ]

        for _ in range(args.epochs):
            model.train()
            perm = np.random.permutation(tr_idx)
            for s in range(0, len(perm), batch_size):
                b = make_batch(perm[s : s + batch_size], t_store, text_t, y_t, device, n_struct=n_struct, branch3_t=use_b3)
                optimizer.zero_grad()
                if m_name == "BiGCN-Plus-Clean":
                    out, l_b1, l_b2 = model(b, return_aux=True)
                    loss = F.nll_loss(out, b.y) + 0.35 * F.cross_entropy(l_b1, b.y) + 0.25 * F.cross_entropy(l_b2, b.y)
                elif m_name == "RAGCL":
                    out, cl_loss = model(b, return_cl=True)
                    loss = F.nll_loss(out, b.y) + cl_loss
                else:
                    loss = F.nll_loss(model(b), b.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

        model.eval()
        probs = []
        with torch.no_grad():
            for b in te_batches:
                probs.append(torch.exp(model(b)).cpu().numpy())
        prob_np = np.concatenate(probs, axis=0)
        met = compute_metrics(y[te_idx], prob_np.argmax(axis=1), prob_np)
        te_metrics[m_name] = met
        exported_states[m_name] = _state_to_fp16(model.state_dict())
        print(f"[{dataset}][{m_name}] Export Fold {args.fold} Test Acc={met['Accuracy']:.5f} F1={met['Macro-F1']:.5f}")

    # Pure TF-IDF-Ridge test metric check
    ridge_logits_te = (Xt_pure[te_idx] @ W_ridge_pure).astype(np.float32)
    z = ridge_logits_te - ridge_logits_te.max(axis=1, keepdims=True)
    e = np.exp(z)
    ridge_prob_te = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)
    te_metrics["TF-IDF-Ridge"] = compute_metrics(y[te_idx], ridge_prob_te.argmax(axis=1), ridge_prob_te)

    bundle = {
        "bundle_version": "v1_demo_fp16",
        "dataset": dataset,
        "seed": int(args.seed),
        "fold": int(args.fold),
        "epochs": int(args.epochs),
        "device_trained": str(device),
        "tr_idx": tr_idx.tolist(),
        "te_idx": te_idx.tolist(),
        "te_tids": [ids[i] for i in te_idx],
        "bow_vocab": bow_vocab,
        "hparams": {
            "hidden_dim": int(hidden_dim),
            "sage_dim": int(sage_dim),
            "vocab_size": int(vocab),
            "td_droprate": float(td_drop),
            "bu_droprate": float(bu_drop),
        },
        "branch3_artifacts": b3_artifacts,
        "ridge_pure_artifacts": {
            "vocab": pure_vocab,
            "idf": pure_idf,
            "W_text": W_ridge_pure,
        },
        "models": exported_states,
        "te_metrics": te_metrics,
        "b3_recon_max_err": b3_recon_err,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset}_demo_bundle.pt")
    torch.save(bundle, out_path)
    mb = os.path.getsize(out_path) / (1024.0 * 1024.0)
    print(f"==> Saved demo bundle: {out_path} ({mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export trained fp16 model bundle for interactive Demo")
    parser.add_argument("--config", type=str, default=os.path.join(ROOT_DIR, "configs", "default.yaml"))
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="ALL", choices=["Twitter15", "Twitter16", "ALL"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--sage_dim", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=os.path.join(ROOT_DIR, "results", "demo_models"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = find_data_root(args.data_root)
    datasets = ["Twitter15", "Twitter16"] if args.dataset == "ALL" else [args.dataset]

    for ds in datasets:
        export_dataset_bundle(ds, data_root, args.out_dir, cfg, args, device)


if __name__ == "__main__":
    main()
