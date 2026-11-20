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
):
    """
    Branch 3: LOO-calibrated Kernel Ridge Residual Logits fit strictly on tr_idx.
      - Text kernel: Sublinear TF-IDF word (1,2)-gram + char (3,4)-gram.
      - Graph kernel (clean): 10-dim normalized topological/temporal graph summary.
      - User kernel (ablation-only if use_user_id=True): tree user ID frequencies.
    """
    n_tr = len(tr_idx)
    Xt = build_sublinear_tfidf_features(docs, tr_idx, max_features=max_features)
    K_all_tr = (Xt @ Xt[tr_idx].T).toarray().astype(np.float32)

    if not use_user_id and g_weight > 0.0:
        gs = np.stack([t["graph_stats"] for t in trees], axis=0).astype(np.float32)
        mu = gs[tr_idx].mean(axis=0, keepdims=True)
        sd = np.maximum(gs[tr_idx].std(axis=0, keepdims=True), 1e-4)
        gs_z = (gs - mu) / sd
        norms = np.maximum(np.linalg.norm(gs_z, axis=1, keepdims=True), 1e-6)
        gs_n = gs_z / norms
        Kg_all = gs_n @ gs_n[tr_idx].T
        K_all_tr = K_all_tr + g_weight * Kg_all

    if use_user_id:
        Xu = build_user_tfidf_features(trees, tr_idx, max_features=max_features)
        Ku_all = (Xu @ Xu[tr_idx].T).toarray().astype(np.float32)
        K_all_tr = K_all_tr + u_weight * Ku_all

    K_tr = K_all_tr[tr_idx]
    Y_tr = np.eye(4, dtype=np.float32)[y[tr_idx]]
    G_inv = np.linalg.inv(K_tr + ridge_lam * np.eye(n_tr, dtype=np.float32))
    alpha = G_inv @ Y_tr

    raw_scores = K_all_tr @ alpha
    diag_G = np.maximum(np.diag(G_inv)[:, None], 1e-6)
    loo_tr = Y_tr - (alpha / diag_G)
    raw_scores[tr_idx] = loo_tr

    std_tr = np.maximum(raw_scores[tr_idx].std(), 1e-4)
    logits = (raw_scores - raw_scores.mean(axis=1, keepdims=True)) / std_tr
    return logits.astype(np.float32)


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
