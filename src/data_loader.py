# -*- coding: utf-8 -*-
"""
Data loading, propagation tree parsing, structural feature extraction,
and zero-leakage text feature builders for rumdetect2017 (Twitter15/Twitter16).
"""
import ast
import os
import re
from collections import Counter, defaultdict

import numpy as np
import torch
from scipy import sparse

LABELS = ["NR", "FR", "TR", "UR"]
LABEL_MAP = {"non-rumor": 0, "false": 1, "true": 2, "unverified": 3}
TOKEN_PAT = re.compile(r"(?u)[#@]?\b\w+\b")

# 8 node-level structural/temporal features computed strictly from tree topology & delays:
# [0] is_root, [1] log1p(delay)/10, [2] log1p(depth), [3] log1p(out_degree),
# [4] temporal_order/(n-1), [5] is_leaf, [6] log1p(subtree_size)/log1p(n),
# [7] log1p(max(delay - parent_delay, 0))/10
N_STRUCT_BASE = 5
N_STRUCT_EXT = 8
N_GRAPH_STATS = 10
EARLY_DEADLINES_MIN = [0, 10, 30, 60, 120, 240, 720, -1]


def find_data_root(explicit_root=None):
    """Locate rumdetect2017 dataset across Colab, Google Drive, and local paths."""
    if explicit_root and os.path.exists(explicit_root):
        return explicit_root

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    workspace_root = os.path.abspath(os.path.join(repo_root, ".."))
    candidates = [
        os.path.join(repo_root, "data", "rumdetect2017", "rumor_detection_acl2017"),
        os.path.join(repo_root, "data", "rumor_detection_acl2017"),
        os.path.join(workspace_root, "rumdetect2017", "rumor_detection_acl2017"),
        "/content/data_local/rumor_detection_acl2017",
        "/content/drive/MyDrive/NghiepVuTM/dataset/rumdetect2017/rumor_detection_acl2017",
        "/content/drive/MyDrive/NghiepVuTM/dataset/rumor_detection_acl2017",
        r"G:\My Drive\NghiepVuTM\dataset\rumdetect2017\rumor_detection_acl2017",
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "twitter15")):
            return path
    raise FileNotFoundError(
        "Khong tim thay thu muc du lieu rumdetect2017. "
        "Vui long truyen --data_root hoac dat du lieu theo data/README.md."
    )


def _compute_pagerank_centrality(edges, n_nodes, damping=0.85, iters=12):
    """Fast power-iteration node centrality on undirected + root-anchored tree."""
    if n_nodes <= 1 or not edges:
        return np.ones(n_nodes, dtype=np.float32) / max(n_nodes, 1)
    adj = [[] for _ in range(n_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    pr = np.full(n_nodes, 1.0 / n_nodes, dtype=np.float64)
    base = (1.0 - damping) / n_nodes
    for _ in range(iters):
        new_pr = np.full(n_nodes, base, dtype=np.float64)
        for u in range(n_nodes):
            deg = len(adj[u])
            if deg > 0:
                share = damping * pr[u] / deg
                for v in adj[u]:
                    new_pr[v] += share
            else:
                new_pr += damping * pr[u] / n_nodes
        pr = new_pr
    return (pr / max(pr.max(), 1e-8)).astype(np.float32)


def build_tree_from_raw(n_full, edges, delays, u_cnt=None, max_tree_nodes=0):
    """Compute topology edge_index, 8-dim node features, centrality, and 10-dim graph_stats."""
    edges = sorted({(int(a), int(b)) for a, b in edges if int(a) != int(b)})
    delays = np.asarray(delays, dtype=np.float32)
    e_full = len(edges)

    ei_full = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    children = np.bincount(ei_full[0], minlength=n_full) if ei_full.shape[1] else np.zeros(n_full, dtype=np.int64)

    depth = np.full(n_full, -1, dtype=np.int64)
    depth[0] = 0
    parent_delay = np.zeros(n_full, dtype=np.float32)
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
    bfs_order = [0]
    for u in bfs_order:
        for v in adj[u]:
            if depth[v] < 0:
                depth[v] = depth[u] + 1
                parent_delay[v] = delays[u]
                bfs_order.append(v)
    depth[depth < 0] = 1

    subtree = np.ones(n_full, dtype=np.float32)
    for u in reversed(bfs_order):
        for v in adj[u]:
            subtree[u] += subtree[v]

    order = np.argsort(np.argsort(delays, kind="stable"), kind="stable")
    struct = np.zeros((n_full, N_STRUCT_EXT), dtype=np.float32)
    struct[0, 0] = 1.0
    struct[:, 1] = np.log1p(delays) / 10.0
    struct[:, 2] = np.log1p(depth)
    struct[:, 3] = np.log1p(children)
    struct[:, 4] = order / max(n_full - 1, 1)
    struct[:, 5] = (children == 0).astype(np.float32)
    struct[:, 6] = np.log1p(subtree) / max(np.log1p(n_full), 1.0)
    struct[:, 7] = np.log1p(np.maximum(delays - parent_delay, 0.0)) / 10.0

    # Purely topological & temporal graph-level summary (no User IDs)
    non_root_delays = delays[1:] if n_full > 1 else np.array([0.0], dtype=np.float32)
    graph_stats = np.array(
        [
            np.log1p(n_full) / 8.0,
            np.log1p(e_full) / 8.0,
            float(depth.max()) / 10.0,
            float(depth.mean()) / 5.0,
            float((children == 0).sum()) / max(n_full, 1),
            float((depth == 1).sum()) / max(n_full - 1, 1),
            float(children[0]) / max(n_full - 1, 1),
            float(np.log1p(non_root_delays).mean()) / 10.0,
            float(np.log1p(np.percentile(non_root_delays, 90))) / 10.0,
            float((non_root_delays <= 60.0).sum()) / max(len(non_root_delays), 1),
        ],
        dtype=np.float32,
    )
    centrality = _compute_pagerank_centrality(edges, n_full)

    if max_tree_nodes > 0 and n_full > max_tree_nodes:
        keep = bfs_order[:max_tree_nodes]
        if 0 not in keep:
            keep[0] = 0
        keep = sorted(keep)
        k_map = {old: idx for idx, old in enumerate(keep)}
        sub_edges = [(k_map[a], k_map[b]) for a, b in edges if a in k_map and b in k_map]
        ei = np.array(sub_edges, dtype=np.int64).T if sub_edges else np.zeros((2, 0), dtype=np.int64)
        struct = struct[keep]
        centrality = centrality[keep]
        n_use = len(keep)
    else:
        ei = ei_full
        n_use = n_full

    return {
        "edge_index": ei,
        "struct": struct,
        "centrality": centrality,
        "graph_stats": graph_stats,
        "n": n_use,
        "n_full": n_full,
        "e_full": e_full,
        "max_depth": int(depth.max()) if len(depth) > 0 else 0,
        "u_cnt": u_cnt if u_cnt is not None else Counter(),
        "raw_edges": edges,
        "raw_delays": delays,
        "max_tree_nodes": max_tree_nodes,
    }


def truncate_tree_by_delay(tree, max_delay_minutes):
    """Return structurally recomputed subtree reachable from root within max_delay_minutes."""
    if max_delay_minutes is None or max_delay_minutes < 0:
        return tree
    raw_edges = tree["raw_edges"]
    raw_delays = tree["raw_delays"]
    cutoff = float(max_delay_minutes)

    adj = defaultdict(list)
    for a, b in raw_edges:
        adj[a].append(b)

    keep_old = {0}
    queue = [0]
    for u in queue:
        for v in adj[u]:
            if v not in keep_old and float(raw_delays[v]) <= cutoff:
                keep_old.add(v)
                queue.append(v)

    keep_list = [0] + sorted(idx for idx in keep_old if idx != 0)
    old_to_new = {old: new for new, old in enumerate(keep_list)}
    sub_edges = sorted(
        {(old_to_new[a], old_to_new[b]) for a, b in raw_edges if a in old_to_new and b in old_to_new and a != b}
    )
    sub_delays = np.array([raw_delays[old] for old in keep_list], dtype=np.float32)
    if len(sub_delays) > 0:
        sub_delays[0] = 0.0

    return build_tree_from_raw(
        n_full=len(keep_list),
        edges=sub_edges,
        delays=sub_delays,
        u_cnt=Counter(),
        max_tree_nodes=tree.get("max_tree_nodes", 0),
    )


def parse_tree(path, tid, max_tree_nodes=0):
    """Parse one propagation tree file into topology edges, 8-dim node features, and graph stats."""
    node_id, delays = {}, []
    edges = set()
    root = None
    u_cnt = Counter()

    def nid(u, t, d_str):
        key = (u, t, d_str)
        if key not in node_id:
            node_id[key] = len(node_id)
            try:
                delays.append(max(float(d_str), 0.0))
            except ValueError:
                delays.append(0.0)
        return node_id[key]

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "->" not in line:
                continue
            q = line.split("'")
            if len(q) >= 12:
                p0, p1, p2 = q[1], q[3], q[5]
                c0, c1, c2 = q[7], q[9], q[11]
            else:
                p_str, c_str = line.split("->")
                p, c = ast.literal_eval(p_str), ast.literal_eval(c_str)
                p0, p1, p2 = str(p[0]), str(p[1]), str(p[2])
                c0, c1, c2 = str(c[0]), str(c[1]), str(c[2])
            if p0 == "ROOT":
                root = nid(c0, c1, c2)
                u_cnt["ROOT_U_" + c0] += 3
                continue
            u_cnt["U_" + c0] += 1
            u_cnt["P_" + p0] += 1
            a, b = nid(p0, p1, p2), nid(c0, c1, c2)
            if a != b:
                edges.add((a, b))

    if root is None:
        cands = [v for k, v in node_id.items() if k[1] == tid]
        root = cands[0] if cands else 0
    if not node_id:
        node_id[("x", tid, "0.0")] = 0
        delays.append(0.0)
        root = 0

    n_full = len(node_id)
    perm = list(range(n_full))
    perm[0], perm[root] = perm[root], perm[0]
    inv = {old: new for new, old in enumerate(perm)}
    edges = sorted({(inv[a], inv[b]) for a, b in edges})
    delays = np.array([delays[old] for old in perm], dtype=np.float32)

    return build_tree_from_raw(
        n_full=n_full,
        edges=edges,
        delays=delays,
        u_cnt=u_cnt,
        max_tree_nodes=max_tree_nodes,
    )


def load_dataset(data_root, dataset, max_tree_nodes=0):
    """Load sorted event IDs, integer labels, source tweet texts, and parsed trees."""
    ddir = os.path.join(data_root, dataset.lower())
    labels = {}
    with open(os.path.join(ddir, "label.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lab, tid = line.split(":")
            labels[tid] = LABEL_MAP[lab]

    texts = {}
    with open(os.path.join(ddir, "source_tweets.txt"), encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                texts[parts[0]] = parts[1]

    ids, ys, docs, trees = [], [], [], []
    for tid in sorted(labels.keys()):
        path = os.path.join(ddir, "tree", tid + ".txt")
        if not os.path.exists(path) or tid not in texts:
            continue
        trees.append(parse_tree(path, tid, max_tree_nodes=max_tree_nodes))
        ids.append(tid)
        ys.append(labels[tid])
        docs.append(texts[tid])
    return ids, np.array(ys, dtype=np.int64), docs, trees


def get_kfold_splits(y, n_folds=5, seed=42):
    """Stratified K-Fold splits using sklearn when available, deterministic fallback otherwise."""
    try:
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return [(np.asarray(tr), np.asarray(te)) for tr, te in skf.split(np.zeros(len(y)), y)]
    except ImportError:
        rng = np.random.RandomState(seed)
        fold_bins = [[] for _ in range(n_folds)]
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0].copy()
            rng.shuffle(cls_idx)
            for i, idx in enumerate(cls_idx):
                fold_bins[i % n_folds].append(int(idx))
        all_idx = np.arange(len(y))
        splits = []
        for k in range(n_folds):
            te = np.array(sorted(fold_bins[k]), dtype=np.int64)
            tr_mask = np.ones(len(y), dtype=bool)
            tr_mask[te] = False
            tr = all_idx[tr_mask]
            splits.append((tr, te))
        return splits


def fit_transform_bow_count(docs, tr_idx, max_features=5000, return_vectorizer=False):
    """Fit raw BoW CountVectorizer strictly on training indices."""
    try:
        from sklearn.feature_extraction.text import CountVectorizer

        vec = CountVectorizer(max_features=max_features, token_pattern=r"(?u)[#@]?\b\w+\b", lowercase=True)
        vec.fit([docs[i] for i in tr_idx])
        X = vec.transform(docs).toarray().astype(np.float32)
        if return_vectorizer:
            vocab = [str(w) for w in vec.get_feature_names_out()]
            return X, vocab
        return X
    except ImportError:
        doc_tokens = [TOKEN_PAT.findall(d.lower()) for d in docs]
        tf_counts = Counter()
        for i in tr_idx:
            for tok in doc_tokens[i]:
                tf_counts[tok] += 1
        vocab = sorted(tf_counts.keys(), key=lambda t: (-tf_counts[t], t))[:max_features]
        X = transform_bow_texts(docs, vocab)
        if return_vectorizer:
            return X, vocab
        return X


def transform_bow_texts(docs, vocab):
    """Transform list of strings into dense BoW count matrix matching an exported vocab."""
    v2i = {t: idx for idx, t in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for i, d in enumerate(docs):
        for tok in TOKEN_PAT.findall(d.lower()):
            j = v2i.get(tok)
            if j is not None:
                X[i, j] += 1.0
    return X


def extract_sublinear_tfidf_counter(text):
    """Extract word (1,2)-gram + char (3,4)-gram counts for a single source text."""
    s = text.lower()
    w = TOKEN_PAT.findall(s)
    w2 = [w[j] + " " + w[j + 1] for j in range(len(w) - 1)]
    c3 = ["c3_" + s[j : j + 3] for j in range(len(s) - 2)]
    c4 = ["c4_" + s[j : j + 4] for j in range(len(s) - 3)]
    return Counter(w + w2 + c3 + c4)


def transform_sublinear_tfidf_counters(counters, vocab, idf):
    """Transform list of Counters into L2-normalized sparse CSR sublinear TF-IDF matrix."""
    v2i = {t: idx for idx, t in enumerate(vocab)}
    rows, cols, vals = [], [], []
    for i, cnt in enumerate(counters):
        for t, v in cnt.items():
            idx = v2i.get(t)
            if idx is not None and v > 0:
                rows.append(i)
                cols.append(idx)
                vals.append((1.0 + np.log(v)) * idf[idx])
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(counters), len(vocab)), dtype=np.float32)
    norms = np.sqrt(np.array(X.power(2).sum(axis=1)).ravel())
    return sparse.diags(1.0 / np.maximum(norms, 1e-8)) @ X


def transform_sublinear_tfidf_texts(docs, vocab, idf):
    """Transform raw strings into L2-normalized sparse CSR sublinear TF-IDF matrix."""
    counters = [extract_sublinear_tfidf_counter(d) for d in docs]
    return transform_sublinear_tfidf_counters(counters, vocab, idf)


def build_sublinear_tfidf_features(docs, tr_idx, max_features=40000, min_df=2, return_vectorizer=False):
    """Fit sublinear word (1,2)-gram + char (3,4)-gram TF-IDF strictly on tr_idx."""
    counters = [extract_sublinear_tfidf_counter(d) for d in docs]
    return _fit_sublinear_tfidf_from_counters(
        counters, tr_idx, max_features=max_features, min_df=min_df, return_vectorizer=return_vectorizer
    )


def build_user_tfidf_features(trees, tr_idx, max_features=40000, min_df=2, return_vectorizer=False):
    """Optional ablation-only: user participation tokens from propagation tree."""
    counters = [t["u_cnt"] for t in trees]
    return _fit_sublinear_tfidf_from_counters(
        counters, tr_idx, max_features=max_features, min_df=min_df, return_vectorizer=return_vectorizer
    )


def _fit_sublinear_tfidf_from_counters(counters, tr_idx, max_features=40000, min_df=2, return_vectorizer=False):
    n_tr = len(tr_idx)
    df_c = Counter()
    for i in tr_idx:
        for tok in counters[i]:
            df_c[tok] += 1
    vocab = [t for t, c in df_c.most_common(max_features) if c >= min_df]
    if not vocab:
        vocab = [t for t, _ in df_c.most_common(max_features)]
    idf = np.array([np.log((1.0 + n_tr) / (1.0 + df_c[t])) + 1.0 for t in vocab], dtype=np.float32)
    X = transform_sublinear_tfidf_counters(counters, vocab, idf)
    if return_vectorizer:
        return X, vocab, idf
    return X


class GraphBatch:
    """Compact batched propagation graph container without PyG dependency."""
    pass


def make_batch(idx, trees_t, text_t, y_t, device, n_struct=8, branch3_t=None, gstats_t=None, plm_t=None):
    xs, eis, batch, roots, cents = [], [], [], [], []
    off = 0
    for g, i in enumerate(idx):
        t = trees_t[i]
        xs.append(t["struct"][:, :n_struct])
        eis.append(t["edge_index"] + off)
        cents.append(t["centrality"])
        batch.append(torch.full((t["n"],), g, dtype=torch.long, device=device))
        roots.append(off)
        off += t["n"]

    b = GraphBatch()
    b.struct = torch.cat(xs, 0)
    b.edge_index = torch.cat(eis, 1)
    b.centrality = torch.cat(cents, 0)
    b.batch = torch.cat(batch, 0)
    b.root = torch.tensor(roots, dtype=torch.long, device=device)
    b.text = text_t[idx]
    b.y = y_t[idx]
    b.tfidf_logits = branch3_t[idx] if branch3_t is not None else None
    b.graph_stats = gstats_t[idx] if gstats_t is not None else None
    b.plm_emb = plm_t[idx] if plm_t is not None else None
    b.num_nodes = off
    b.num_graphs = len(idx)
    return b
