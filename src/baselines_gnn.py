# -*- coding: utf-8 -*-
"""
Level 1 & Level 2 Graph Baselines:
  Level 1 : MLP-BoW, GCN-Single, GraphSAGE-Single, GAT-Single
  Level 2 : BiGCN-Base (+ 4 ablations: NoRoot, NoDropEdge, TDOnly, BUOnly),
            BiGCN-Base+ (controlled 8-dim + sublinear TF-IDF projection),
            EBGCN (Edge-enhanced Bayesian GCN, ACL 2021 style),
            RAGCL (Rumor Adaptive Graph Contrastive Learning, AAAI 2024 style)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_loader import N_STRUCT_BASE, N_STRUCT_EXT


def drop_edges(ei, p, training):
    if not training or p <= 0 or ei.size(1) == 0:
        return ei
    keep = torch.rand(ei.size(1), device=ei.device) >= p
    return ei[:, keep]


def adaptive_drop_edges(ei, centrality, base_p, training):
    """RAGCL-inspired adaptive edge dropping: higher drop probability on low-centrality leaves."""
    if not training or base_p <= 0 or ei.size(1) == 0:
        return ei
    dst_cent = centrality[ei[1]].clamp(min=0.0, max=1.0)
    keep_prob = (1.0 - base_p) + 0.5 * base_p * dst_cent
    keep_prob = keep_prob.clamp(min=0.25, max=0.98)
    keep = torch.rand(ei.size(1), device=ei.device) < keep_prob
    return ei[:, keep]


def gcn_norm(ei, n):
    loop = torch.arange(n, device=ei.device)
    row = torch.cat([ei[0], loop])
    col = torch.cat([ei[1], loop])
    deg = torch.zeros(n, device=ei.device).index_add_(0, col, torch.ones_like(col, dtype=torch.float32))
    dinv = deg.clamp(min=1.0).pow(-0.5)
    w = dinv[row] * dinv[col]
    return row, col, w


def propagate(h, row, col, w, n):
    out = torch.zeros(n, h.size(1), device=h.device, dtype=h.dtype)
    return out.index_add_(0, col, h[row] * w.unsqueeze(1))


def weighted_propagate(h, ei, edge_weights, n):
    loop = torch.arange(n, device=ei.device)
    src = torch.cat([ei[0], loop])
    dst = torch.cat([ei[1], loop])
    w = torch.cat([edge_weights, torch.ones(n, device=ei.device, dtype=edge_weights.dtype)])
    deg = torch.zeros(n, device=ei.device, dtype=h.dtype).index_add_(0, dst, w).clamp(min=1e-6)
    msg = h[src] * (w / deg[dst]).unsqueeze(1)
    return torch.zeros(n, h.size(1), device=h.device, dtype=h.dtype).index_add_(0, dst, msg)


def scatter_mean(h, batch, num_graphs):
    s = torch.zeros(num_graphs, h.size(1), device=h.device, dtype=h.dtype).index_add_(0, batch, h)
    c = torch.zeros(num_graphs, device=h.device, dtype=h.dtype).index_add_(
        0, batch, torch.ones(h.size(0), device=h.device, dtype=h.dtype)
    )
    return s / c.clamp(min=1).unsqueeze(1)


class InputLinear(nn.Module):
    """Sparse root text linear + dense structural feature linear."""

    def __init__(self, vocab, out, n_struct=N_STRUCT_BASE, bias=True):
        super().__init__()
        self.wt = nn.Linear(vocab, out, bias=False)
        self.ws = nn.Linear(n_struct, out, bias=bias)

    def forward(self, b):
        h = self.ws(b.struct)
        h.index_add_(0, b.root, self.wt(b.text))
        return h

    def root_only(self, b, p=0.0):
        t, s = b.text, b.struct[b.root]
        if p > 0.0:
            t = F.dropout(t, p, True)
            s = F.dropout(s, p, True)
        return self.wt(t) + self.ws(s)


class RumorGCN(nn.Module):
    """Directional 2-layer GCN with optional Root Feature Enhancement."""

    def __init__(self, vocab, hid, out, n_struct=N_STRUCT_BASE, use_root=True, dropout=0.5):
        super().__init__()
        self.use_root = use_root
        self.dropout = dropout
        self.conv1 = InputLinear(vocab, hid, n_struct=n_struct, bias=False)
        self.conv1_bias = nn.Parameter(torch.zeros(hid))
        self.conv2_h = nn.Linear(hid, out, bias=False)
        self.conv2_r = InputLinear(vocab, out, n_struct=n_struct, bias=False) if use_root else None
        self.conv2_bias = nn.Parameter(torch.zeros(out))
        self.out_dim = hid + out if use_root else out

    def forward(self, b, ei):
        n = b.num_nodes
        row, col, w = gcn_norm(ei, n)
        h1 = propagate(self.conv1(b), row, col, w, n) + self.conv1_bias
        x2 = h1
        if self.use_root:
            r = self.conv2_r.root_only(b, self.dropout if self.training else 0.0)[b.batch]
            h2_in = self.conv2_h(F.dropout(F.relu(h1), self.dropout, self.training)) + r
        else:
            h2_in = self.conv2_h(F.dropout(F.relu(h1), self.dropout, self.training))
        h2 = F.relu(propagate(h2_in, row, col, w, n) + self.conv2_bias)
        pooled = scatter_mean(h2, b.batch, b.num_graphs)
        if self.use_root:
            return torch.cat([pooled, x2[b.root]], dim=1)
        return pooled


class BiGCN(nn.Module):
    def __init__(self, vocab, hid=64, out=64, n_struct=N_STRUCT_BASE, use_root=True, td=True, bu=True, n_cls=4):
        super().__init__()
        self.td = RumorGCN(vocab, hid, out, n_struct=n_struct, use_root=use_root) if td else None
        self.bu = RumorGCN(vocab, hid, out, n_struct=n_struct, use_root=use_root) if bu else None
        dim = (self.td.out_dim if td else 0) + (self.bu.out_dim if bu else 0)
        self.fc = nn.Linear(dim, n_cls)
        self.td_drop = 0.2
        self.bu_drop = 0.2

    def forward(self, b):
        outs = []
        if self.td is not None:
            outs.append(self.td(b, drop_edges(b.edge_index, self.td_drop, self.training)))
        if self.bu is not None:
            outs.append(self.bu(b, drop_edges(b.edge_index.flip(0), self.bu_drop, self.training)))
        return F.log_softmax(self.fc(torch.cat(outs, 1)), dim=1)


class MLPBoW(nn.Module):
    def __init__(self, vocab, hid=64, n_cls=4, dropout=0.5):
        super().__init__()
        self.l1 = nn.Linear(vocab, hid)
        self.l2 = nn.Linear(hid, hid)
        self.fc = nn.Linear(hid, n_cls)
        self.dropout = dropout

    def forward(self, b):
        h = F.dropout(F.relu(self.l1(b.text)), self.dropout, self.training)
        h = F.dropout(F.relu(self.l2(h)), self.dropout, self.training)
        return F.log_softmax(self.fc(h), dim=1)


def undirected(ei):
    return torch.cat([ei, ei.flip(0)], 1)


class GCNSingle(nn.Module):
    def __init__(self, vocab, hid=64, n_struct=N_STRUCT_BASE, n_cls=4, dropout=0.5):
        super().__init__()
        self.l1 = InputLinear(vocab, hid, n_struct=n_struct)
        self.l2 = nn.Linear(hid, hid)
        self.fc = nn.Linear(hid, n_cls)
        self.dropout = dropout

    def forward(self, b):
        ei = undirected(b.edge_index)
        row, col, w = gcn_norm(ei, b.num_nodes)
        h = F.relu(propagate(self.l1(b), row, col, w, b.num_nodes))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(propagate(self.l2(h), row, col, w, b.num_nodes))
        return F.log_softmax(self.fc(scatter_mean(h, b.batch, b.num_graphs)), dim=1)


class GraphSAGESingle(nn.Module):
    def __init__(self, vocab, hid=64, n_struct=N_STRUCT_BASE, n_cls=4, dropout=0.5):
        super().__init__()
        self.s1 = InputLinear(vocab, hid, n_struct=n_struct)
        self.n1 = InputLinear(vocab, hid, n_struct=n_struct, bias=False)
        self.s2, self.n2 = nn.Linear(hid, hid), nn.Linear(hid, hid, bias=False)
        self.fc = nn.Linear(hid, n_cls)
        self.dropout = dropout

    @staticmethod
    def mean_agg(h, ei, n):
        s = torch.zeros(n, h.size(1), device=h.device, dtype=h.dtype).index_add_(0, ei[1], h[ei[0]])
        c = torch.zeros(n, device=h.device, dtype=h.dtype).index_add_(
            0, ei[1], torch.ones(ei.size(1), device=h.device, dtype=h.dtype)
        )
        return s / c.clamp(min=1).unsqueeze(1)

    def forward(self, b):
        n = b.num_nodes
        ei = undirected(b.edge_index)
        h = F.relu(self.s1(b) + self.mean_agg(self.n1(b), ei, n))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.s2(h) + self.mean_agg(self.n2(h), ei, n))
        return F.log_softmax(self.fc(scatter_mean(h, b.batch, b.num_graphs)), dim=1)


class GATLayer(nn.Module):
    def __init__(self, lin, out, heads, concat=True):
        super().__init__()
        self.lin, self.heads, self.out, self.concat = lin, heads, out, concat
        self.att_src = nn.Parameter(torch.empty(heads, out))
        self.att_dst = nn.Parameter(torch.empty(heads, out))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x_or_b, ei, n):
        h = self.lin(x_or_b).view(n, self.heads, self.out)
        loop = torch.arange(n, device=h.device)
        src, dst = torch.cat([ei[0], loop]), torch.cat([ei[1], loop])
        e = F.leaky_relu((h * self.att_src).sum(-1)[src] + (h * self.att_dst).sum(-1)[dst], 0.2)
        emax = torch.full((n, self.heads), -1e9, device=h.device).scatter_reduce(
            0, dst.unsqueeze(1).expand_as(e), e, reduce="amax", include_self=True
        )
        a = torch.exp(e - emax[dst])
        den = torch.zeros(n, self.heads, device=h.device).index_add_(0, dst, a)
        a = a / den[dst].clamp(min=1e-16)
        msg = h[src] * a.unsqueeze(-1)
        o = torch.zeros(n, self.heads, self.out, device=h.device).index_add_(0, dst, msg)
        return o.reshape(n, -1) if self.concat else o.mean(1)


class GATSingle(nn.Module):
    def __init__(self, vocab, hid=64, n_struct=N_STRUCT_BASE, n_cls=4, heads=4, dropout=0.5):
        super().__init__()
        self.g1 = GATLayer(InputLinear(vocab, hid * heads, n_struct=n_struct, bias=False), hid, heads, True)
        self.g2 = GATLayer(nn.Linear(hid * heads, hid, bias=False), hid, 1, False)
        self.b1, self.b2 = nn.Parameter(torch.zeros(hid * heads)), nn.Parameter(torch.zeros(hid))
        self.fc = nn.Linear(hid, n_cls)
        self.dropout = dropout

    def forward(self, b):
        n = b.num_nodes
        ei = undirected(b.edge_index)
        h = F.elu(self.g1(b, ei, n) + self.b1)
        h = F.dropout(h, self.dropout, self.training)
        h = F.elu(self.g2(h, ei, n) + self.b2)
        return F.log_softmax(self.fc(scatter_mean(h, b.batch, b.num_graphs)), dim=1)


class BayesianDirRumorGCN(nn.Module):
    """Edge-uncertainty aware directional GCN for EBGCN (Wei et al., ACL 2021)."""

    def __init__(self, vocab, hid=64, out=64, n_struct=N_STRUCT_EXT, dropout=0.5):
        super().__init__()
        self.conv1 = InputLinear(vocab, hid, n_struct=n_struct, bias=False)
        self.conv1_bias = nn.Parameter(torch.zeros(hid))
        self.edge_gate = nn.Sequential(
            nn.Linear(hid * 2, hid // 2),
            nn.ReLU(),
            nn.Linear(hid // 2, 1),
        )
        self.conv2_h = nn.Linear(hid, out, bias=False)
        self.conv2_r = InputLinear(vocab, out, n_struct=n_struct, bias=False)
        self.conv2_bias = nn.Parameter(torch.zeros(out))
        self.dropout = dropout
        self.out_dim = hid + out

    def forward(self, b, ei, return_edge_probs=False):
        n = b.num_nodes
        row, col, w_norm = gcn_norm(ei, n)
        h1_raw = propagate(self.conv1(b), row, col, w_norm, n) + self.conv1_bias
        h1 = F.relu(h1_raw)

        if ei.size(1) > 0:
            edge_feat = torch.cat([h1[ei[0]], h1[ei[1]]], dim=1)
            logits_e = self.edge_gate(edge_feat).squeeze(-1)
            if self.training:
                noise = -torch.log(-torch.log(torch.rand_like(logits_e).clamp(min=1e-6, max=1 - 1e-6)))
                p_edge = torch.sigmoid(logits_e + 0.15 * noise)
            else:
                p_edge = torch.sigmoid(logits_e)
        else:
            p_edge = torch.zeros(0, device=h1.device)

        h1_drop = F.dropout(h1, self.dropout, self.training)
        r = self.conv2_r.root_only(b, self.dropout if self.training else 0.0)[b.batch]
        h2_in = self.conv2_h(h1_drop) + r
        h2 = F.relu(weighted_propagate(h2_in, ei, p_edge, n) + self.conv2_bias)
        pooled = torch.cat([scatter_mean(h2, b.batch, b.num_graphs), h1_raw[b.root]], dim=1)
        if return_edge_probs:
            return pooled, p_edge
        return pooled


class EBGCN(nn.Module):
    """Edge-Enhanced Bayesian Bi-Directional GCN (Wei et al., ACL 2021)."""

    def __init__(self, vocab, hid=64, out=64, n_struct=N_STRUCT_EXT, n_cls=4):
        super().__init__()
        self.td = BayesianDirRumorGCN(vocab, hid, out, n_struct=n_struct)
        self.bu = BayesianDirRumorGCN(vocab, hid, out, n_struct=n_struct)
        self.fc = nn.Linear(self.td.out_dim + self.bu.out_dim, n_cls)
        self.td_drop = 0.2
        self.bu_drop = 0.2

    def forward(self, b, return_reg=False):
        h_td, p_td = self.td(b, drop_edges(b.edge_index, self.td_drop, self.training), return_edge_probs=True)
        h_bu, p_bu = self.bu(b, drop_edges(b.edge_index.flip(0), self.bu_drop, self.training), return_edge_probs=True)
        log_prob = F.log_softmax(self.fc(torch.cat([h_td, h_bu], dim=1)), dim=1)
        if return_reg:
            # Weakly encourage edge posterior near informative propagation prior (0.7) + entropy balance
            prior = 0.7
            reg = 0.0
            for p in (p_td, p_bu):
                if p.numel() > 0:
                    reg = reg + ((p.mean() - prior) ** 2)
            return log_prob, 0.1 * reg
        return log_prob


class RAGCL(nn.Module):
    """Rumor Adaptive Graph Contrastive Learning (Cui & Jia, AAAI 2024)."""

    def __init__(self, vocab, hid=64, out=64, n_struct=N_STRUCT_EXT, n_cls=4, base_drop=0.25):
        super().__init__()
        self.td = RumorGCN(vocab, hid, out, n_struct=n_struct, use_root=True)
        self.bu = RumorGCN(vocab, hid, out, n_struct=n_struct, use_root=True)
        dim = self.td.out_dim + self.bu.out_dim
        self.proj = nn.Sequential(nn.Linear(dim, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.fc = nn.Linear(dim, n_cls)
        self.base_drop = base_drop

    def _encode(self, b, aug=False):
        if aug:
            ei_td = adaptive_drop_edges(b.edge_index, b.centrality, self.base_drop, self.training)
            ei_bu = adaptive_drop_edges(b.edge_index.flip(0), b.centrality, self.base_drop, self.training)
        else:
            ei_td = drop_edges(b.edge_index, 0.15, self.training)
            ei_bu = drop_edges(b.edge_index.flip(0), 0.15, self.training)
        return torch.cat([self.td(b, ei_td), self.bu(b, ei_bu)], dim=1)

    @staticmethod
    def _infonce(z1, z2, temp=0.3):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim = torch.mm(z1, z2.T) / temp
        labels = torch.arange(z1.size(0), device=z1.device)
        return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))

    def forward(self, b, return_cl=False):
        h = self._encode(b, aug=False)
        log_prob = F.log_softmax(self.fc(h), dim=1)
        if return_cl and self.training and b.num_graphs > 1:
            z1 = self.proj(self._encode(b, aug=True))
            z2 = self.proj(self._encode(b, aug=True))
            return log_prob, 0.2 * self._infonce(z1, z2)
        if return_cl:
            return log_prob, log_prob.new_tensor(0.0)
        return log_prob


def build_baseline_model(name, vocab, args):
    hid = args.hidden_dim
    if name == "MLP-BoW":
        return MLPBoW(vocab, hid)
    if name == "GCN-Single":
        return GCNSingle(vocab, hid, n_struct=N_STRUCT_BASE)
    if name == "GraphSAGE-Single":
        return GraphSAGESingle(vocab, hid, n_struct=N_STRUCT_BASE)
    if name == "GAT-Single":
        return GATSingle(vocab, hid, n_struct=N_STRUCT_BASE)
    if name == "BiGCN-Base+":
        m = BiGCN(vocab, hid, args.out_dim or hid, n_struct=N_STRUCT_EXT, use_root=True)
        m.td_drop, m.bu_drop = args.td_droprate, args.bu_droprate
        return m
    if name == "EBGCN":
        m = EBGCN(vocab, hid, args.out_dim or hid, n_struct=N_STRUCT_EXT)
        m.td_drop, m.bu_drop = args.td_droprate, args.bu_droprate
        return m
    if name == "RAGCL":
        return RAGCL(vocab, hid, args.out_dim or hid, n_struct=N_STRUCT_EXT)

    cfg = {
        "BiGCN-Base": dict(),
        "BiGCN-NoDropEdge": dict(),
        "BiGCN-NoRoot": dict(use_root=False),
        "BiGCN-TDOnly": dict(bu=False),
        "BiGCN-BUOnly": dict(td=False),
    }[name]
    m = BiGCN(vocab, hid, args.out_dim or hid, n_struct=N_STRUCT_BASE, **cfg)
    if name == "BiGCN-NoDropEdge":
        m.td_drop = 0.0
        m.bu_drop = 0.0
    else:
        m.td_drop, m.bu_drop = args.td_droprate, args.bu_droprate
    return m
