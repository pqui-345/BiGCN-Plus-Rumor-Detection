# -*- coding: utf-8 -*-
"""
Proposed Level 3 Architecture: BiGCN-Plus (Multi-Expert Structural-Semantic Fusion)
Supports clean (zero User-ID) proposed model, V2 User-ID ablation, Branch ablations,
and optional PLM semantic upgrade (BiGCN-Plus-PLM).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines_gnn import InputLinear, RumorGCN, drop_edges, scatter_mean
from .data_loader import (
    N_STRUCT_EXT,
    build_sublinear_tfidf_features,
    build_user_tfidf_features,
)


def _solve_ridge_loo_eigh(eigvals, eigvecs, Y_tr, lam):
    """Fast eigendecomposition-based dual Ridge solve + exact Leave-One-Out logits on tr_idx."""
    inv_d = 1.0 / np.maximum(eigvals + lam, 1e-8)
    alpha = eigvecs @ (inv_d[:, None] * (eigvecs.T @ Y_tr))
    diag_G = np.maximum((eigvecs ** 2) @ inv_d, 1e-6)
    loo_tr = Y_tr - (alpha / diag_G[:, None])
    return alpha, loo_tr


def compute_branch3_logits(
    docs,
    trees,
    y,
    tr_idx,
    te_idx,
    use_user_id=False,
    u_weight=0.95,
    g_weight=0.25,
    ridge_lam=0.45,
    max_features=40000,
    auto_tune=True,
    return_artifacts=False,
):
    """
    Branch 3: LOO-calibrated Kernel Ridge Residual Logits fit strictly on tr_idx.
      - Text kernel: Sublinear TF-IDF word (1,2)-gram + char (3,4)-gram.
      - Graph kernel (clean): 10-dim normalized topological/temporal graph summary.
      - With auto_tune=True and not use_user_id, selects (g_weight, ridge_lam) strictly
        via Leave-One-Out objective on tr_idx (zero test leakage).
      - User kernel (ablation-only if use_user_id=True): tree user ID frequencies.
    """
    n_tr = len(tr_idx)
    if return_artifacts:
        Xt, tfidf_vocab, tfidf_idf = build_sublinear_tfidf_features(
            docs, tr_idx, max_features=max_features, return_vectorizer=True
        )
    else:
        Xt = build_sublinear_tfidf_features(docs, tr_idx, max_features=max_features)

    Xt_tr = Xt[tr_idx]
    K_text_all_tr = (Xt @ Xt_tr.T).toarray().astype(np.float64)
    K_text_tr = K_text_all_tr[tr_idx]
    Y_tr = np.eye(4, dtype=np.float64)[y[tr_idx]]
    y_tr = y[tr_idx]

    gs_n = None
    gs_mu = np.zeros((1, 10), dtype=np.float32)
    gs_sd = np.ones((1, 10), dtype=np.float32)
    Kg_all = None
    Kg_tr = None
    has_graph_var = False

    if not use_user_id and g_weight > 0.0:
        gs = np.stack([t["graph_stats"] for t in trees], axis=0).astype(np.float32)
        gs_mu = gs[tr_idx].mean(axis=0, keepdims=True)
        sd_raw = gs[tr_idx].std(axis=0, keepdims=True)
        if not np.all(sd_raw < 1e-6):
            has_graph_var = True
            gs_sd = np.maximum(sd_raw, 1e-4)
            gs_z = (gs - gs_mu) / gs_sd
            norms = np.maximum(np.linalg.norm(gs_z, axis=1, keepdims=True), 1e-6)
            gs_n = gs_z / norms
            Kg_all = (gs_n @ gs_n[tr_idx].T).astype(np.float64)
            Kg_tr = Kg_all[tr_idx]

    best_g_weight = 0.0
    best_lam = float(ridge_lam)
    best_alpha = None
    best_loo_tr = None

    if not use_user_id:
        g_grid = (
            (0.05, 0.15, 0.25, 0.35)
            if (auto_tune and g_weight > 0.0 and has_graph_var)
            else ((g_weight,) if (g_weight > 0.0 and has_graph_var) else (0.0,))
        )
        lam_grid = (0.25, 0.35, 0.45, 0.65, 1.00) if auto_tune else (float(ridge_lam),)

        best_score = -1e18
        for g_cand in g_grid:
            K_cand = K_text_tr if g_cand == 0.0 else (K_text_tr + g_cand * Kg_tr)
            K_cand = 0.5 * (K_cand + K_cand.T)
            eigvals, eigvecs = np.linalg.eigh(K_cand)
            eigvals = np.maximum(eigvals, 0.0)
            for lam_cand in lam_grid:
                alpha_c, loo_c = _solve_ridge_loo_eigh(eigvals, eigvecs, Y_tr, lam_cand)
                loo_acc = float((loo_c.argmax(axis=1) == y_tr).mean())
                loo_mse = float(np.mean((loo_c - Y_tr) ** 2))
                score = loo_acc - 0.05 * loo_mse
                if score > best_score:
                    best_score = score
                    best_g_weight = float(g_cand)
                    best_lam = float(lam_cand)
                    best_alpha = alpha_c
                    best_loo_tr = loo_c

        K_all_tr = K_text_all_tr if (best_g_weight == 0.0 or Kg_all is None) else (K_text_all_tr + best_g_weight * Kg_all)
    else:
        Xu = build_user_tfidf_features(trees, tr_idx, max_features=max_features)
        Ku_all = (Xu @ Xu[tr_idx].T).toarray().astype(np.float64)
        K_all_tr = K_text_all_tr + u_weight * Ku_all
        K_tr = 0.5 * (K_all_tr[tr_idx] + K_all_tr[tr_idx].T)
        eigvals, eigvecs = np.linalg.eigh(K_tr)
        eigvals = np.maximum(eigvals, 0.0)
        best_alpha, best_loo_tr = _solve_ridge_loo_eigh(eigvals, eigvecs, Y_tr, ridge_lam)

    raw_scores = (K_all_tr @ best_alpha).astype(np.float32)
    raw_scores[tr_idx] = best_loo_tr.astype(np.float32)

    std_tr = float(np.maximum(raw_scores[tr_idx].std(), 1e-4))
    logits = (raw_scores - raw_scores.mean(axis=1, keepdims=True)) / std_tr
    logits = logits.astype(np.float32)

    if not return_artifacts:
        return logits

    W_text = (Xt_tr.T @ best_alpha).astype(np.float32)
    W_graph = (
        (best_g_weight * (gs_n[tr_idx].T @ best_alpha)).astype(np.float32)
        if (gs_n is not None and best_g_weight > 0.0)
        else np.zeros((10, 4), dtype=np.float32)
    )
    artifacts = {
        "best_g_weight": float(best_g_weight),
        "best_lam": float(best_lam),
        "std_tr": float(std_tr),
        "vocab": tfidf_vocab,
        "idf": tfidf_idf,
        "W_text": W_text,
        "W_graph": W_graph,
        "gs_mu": gs_mu.astype(np.float32),
        "gs_sd": gs_sd.astype(np.float32),
        "has_graph_var": bool(has_graph_var),
    }
    return logits, artifacts


class DualViewSAGEEncoder(nn.Module):
    """Bidirectional Edge-Preserving GraphSAGE with Mean + Multi-Scale Root Skip Pooling."""

    def __init__(self, vocab, hid=32, n_struct=N_STRUCT_EXT, dropout=0.4):
        super().__init__()
        self.s1 = InputLinear(vocab, hid, n_struct=n_struct, bias=True)
        self.n1_td = InputLinear(vocab, hid, n_struct=n_struct, bias=False)
        self.n1_bu = InputLinear(vocab, hid, n_struct=n_struct, bias=False)
        self.s2 = nn.Linear(hid, hid)
        self.n2_td = nn.Linear(hid, hid, bias=False)
        self.n2_bu = nn.Linear(hid, hid, bias=False)
        self.dropout = dropout
        self.out_dim = hid * 2

    @staticmethod
    def mean_agg(h, ei, n):
        if ei.size(1) == 0:
            return torch.zeros_like(h)
        s = torch.zeros(n, h.size(1), device=h.device, dtype=h.dtype).index_add_(0, ei[1], h[ei[0]])
        c = torch.zeros(n, device=h.device, dtype=h.dtype).index_add_(
            0, ei[1], torch.ones(ei.size(1), device=h.device, dtype=h.dtype)
        )
        return s / c.clamp(min=1).unsqueeze(1)

    def forward(self, b):
        n = b.num_nodes
        ei_td = b.edge_index
        ei_bu = b.edge_index.flip(0)
        h1 = F.relu(self.s1(b) + self.mean_agg(self.n1_td(b), ei_td, n) + self.mean_agg(self.n1_bu(b), ei_bu, n))
        h1_drop = F.dropout(h1, self.dropout, self.training)
        h2 = F.relu(
            self.s2(h1_drop)
            + self.mean_agg(self.n2_td(h1_drop), ei_td, n)
            + self.mean_agg(self.n2_bu(h1_drop), ei_bu, n)
        )
        g_mean = scatter_mean(h2, b.batch, b.num_graphs)
        g_root = h2[b.root] + h1[b.root]
        return torch.cat([g_mean, g_root], dim=1)


class BiGCNPlus(nn.Module):
    """Multi-Expert Structural-Semantic Fusion with configurable branch ablation."""

    def __init__(
        self,
        vocab,
        hid=64,
        sage_hid=32,
        n_cls=4,
        td_drop=0.2,
        bu_drop=0.2,
        mode="clean",
        plm_dim=0,
    ):
        super().__init__()
        self.mode = mode
        self.td = RumorGCN(vocab, hid, hid, n_struct=N_STRUCT_EXT, use_root=True, dropout=0.5)
        self.bu = RumorGCN(vocab, hid, hid, n_struct=N_STRUCT_EXT, use_root=True, dropout=0.5)
        self.fc_bigcn = nn.Linear(self.td.out_dim + self.bu.out_dim, n_cls)
        self.td_drop = td_drop
        self.bu_drop = bu_drop

        self.sage_dual = DualViewSAGEEncoder(vocab, sage_hid, n_struct=N_STRUCT_EXT, dropout=0.4)
        self.fc_sage = nn.Linear(self.sage_dual.out_dim, n_cls)

        self.residual_scale = nn.Parameter(torch.ones(n_cls, dtype=torch.float32))
        self.residual_bias = nn.Parameter(torch.zeros(n_cls, dtype=torch.float32))

        if mode == "userid":
            init_logits = [-0.6, -0.9, 1.2]
        else:
            init_logits = [0.0, 0.0, 0.0]
        self.fusion_logits = nn.Parameter(torch.tensor(init_logits, dtype=torch.float32))

        self.plm_head = None
        if plm_dim > 0:
            self.plm_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(plm_dim, hid),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hid, n_cls),
            )
            self.plm_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, b, return_aux=False):
        h_td = self.td(b, drop_edges(b.edge_index, self.td_drop, self.training))
        h_bu = self.bu(b, drop_edges(b.edge_index.flip(0), self.bu_drop, self.training))
        logits_b1 = self.fc_bigcn(torch.cat([h_td, h_bu], dim=1))

        h_sage = self.sage_dual(b)
        logits_b2 = self.fc_sage(h_sage)

        if self.mode == "b1_only":
            log_prob = F.log_softmax(logits_b1, dim=1)
            return (log_prob, logits_b1, logits_b2) if return_aux else log_prob
        if self.mode == "b2_only":
            log_prob = F.log_softmax(logits_b2, dim=1)
            return (log_prob, logits_b1, logits_b2) if return_aux else log_prob
        if self.mode == "b12":
            w2 = F.softmax(self.fusion_logits[:2], dim=0)
            fused = w2[0] * logits_b1 + w2[1] * logits_b2
            log_prob = F.log_softmax(fused, dim=1)
            return (log_prob, logits_b1, logits_b2) if return_aux else log_prob

        logits_b3 = b.tfidf_logits * self.residual_scale + self.residual_bias
        w = F.softmax(self.fusion_logits, dim=0)
        fused = w[0] * logits_b1 + w[1] * logits_b2 + w[2] * logits_b3
        if self.plm_head is not None and b.plm_emb is not None:
            fused = fused + torch.sigmoid(self.plm_weight) * self.plm_head(b.plm_emb)

        log_prob = F.log_softmax(fused, dim=1)
        if return_aux:
            return log_prob, logits_b1, logits_b2
        return log_prob

    def forward_explain(self, b):
        """Return end probabilities, per-branch expert probabilities, and softmax fusion weights."""
        with torch.no_grad():
            h_td = self.td(b, b.edge_index)
            h_bu = self.bu(b, b.edge_index.flip(0))
            logits_b1 = self.fc_bigcn(torch.cat([h_td, h_bu], dim=1))

            h_sage = self.sage_dual(b)
            logits_b2 = self.fc_sage(h_sage)

            if self.mode == "b1_only":
                p = F.softmax(logits_b1, dim=1)
                return {
                    "probs": p,
                    "probs_b1": p,
                    "probs_b2": F.softmax(logits_b2, dim=1),
                    "probs_b3": None,
                    "fusion_weights": torch.tensor([1.0, 0.0, 0.0], device=p.device),
                }
            if self.mode == "b2_only":
                p = F.softmax(logits_b2, dim=1)
                return {
                    "probs": p,
                    "probs_b1": F.softmax(logits_b1, dim=1),
                    "probs_b2": p,
                    "probs_b3": None,
                    "fusion_weights": torch.tensor([0.0, 1.0, 0.0], device=p.device),
                }
            if self.mode == "b12":
                w2 = F.softmax(self.fusion_logits[:2], dim=0)
                fused = w2[0] * logits_b1 + w2[1] * logits_b2
                return {
                    "probs": F.softmax(fused, dim=1),
                    "probs_b1": F.softmax(logits_b1, dim=1),
                    "probs_b2": F.softmax(logits_b2, dim=1),
                    "probs_b3": None,
                    "fusion_weights": torch.tensor([w2[0], w2[1], 0.0], device=fused.device),
                }

            logits_b3 = b.tfidf_logits * self.residual_scale + self.residual_bias
            w = F.softmax(self.fusion_logits, dim=0)
            fused = w[0] * logits_b1 + w[1] * logits_b2 + w[2] * logits_b3
            if self.plm_head is not None and b.plm_emb is not None:
                fused = fused + torch.sigmoid(self.plm_weight) * self.plm_head(b.plm_emb)
            return {
                "probs": F.softmax(fused, dim=1),
                "probs_b1": F.softmax(logits_b1, dim=1),
                "probs_b2": F.softmax(logits_b2, dim=1),
                "probs_b3": F.softmax(logits_b3, dim=1),
                "fusion_weights": w,
            }
