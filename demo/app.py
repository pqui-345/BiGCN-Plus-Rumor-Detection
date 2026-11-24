# -*- coding: utf-8 -*-
"""
Interactive 4-Tab Web Demo Backend (Python stdlib HTTP server + PyTorch CPU inference)
for BiGCN-Plus Rumor Detection XAI Lab.

Tabs supported:
  1. Live Event XAI & Model Showdown (/api/event)
  2. Real-Time Early Detection Trajectory (/api/early_trajectory)
  3. Counterfactual What-If Sandbox (/api/what_if)
  4. Verified Colab Benchmark Explorer (/api/benchmarks)
"""
import argparse
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
import sys
import threading
import types
from urllib.parse import parse_qs, urlparse
import webbrowser

import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.baselines_gnn import build_baseline_model
from src.bigcn_plus import BiGCNPlus
from src.data_loader import (
    EARLY_DEADLINES_MIN,
    LABELS,
    N_STRUCT_BASE,
    N_STRUCT_EXT,
    build_tree_from_raw,
    find_data_root,
    load_dataset,
    make_batch,
    transform_bow_texts,
    transform_sublinear_tfidf_texts,
    truncate_tree_by_delay,
)

LABEL_NAMES_VI = {
    0: "NR — Tin thật / Bình luận thường",
    1: "FR — Tin đồn sai sự thật (False Rumor)",
    2: "TR — Tin đồn được xác thực (True Rumor)",
    3: "UR — Tin chưa kiểm chứng (Unverified)",
}

GRAPH_STAT_NAMES_VI = [
    ("log1p(|V|)/8", "Quy mô nút cây (log)"),
    ("log1p(|E|)/8", "Mật độ cạnh lan truyền (log)"),
    ("max_depth/10", "Độ sâu phân tầng tối đa"),
    ("mean_depth/5", "Độ sâu trung bình các nút"),
    ("leaf_ratio", "Tỷ lệ nút lá (cụt nhánh)"),
    ("depth1_ratio", "Tỷ lệ phản hồi trực tiếp tầng 1"),
    ("root_out_ratio", "Tỷ lệ lan tỏa thẳng từ gốc"),
    ("mean_log_delay", "Độ trễ phản hồi trung bình (log)"),
    ("p90_log_delay", "Phân vị 90% thời gian lan truyền"),
    ("ratio_le_60m", "Tỷ lệ phản hồi trong 60 phút đầu"),
]

DEADLINE_OPTIONS = [
    {"minutes": 0, "label": "0 phút (Chỉ bài gốc)"},
    {"minutes": 10, "label": "10 phút"},
    {"minutes": 30, "label": "30 phút"},
    {"minutes": 60, "label": "1 giờ"},
    {"minutes": 120, "label": "2 giờ"},
    {"minutes": 240, "label": "4 giờ"},
    {"minutes": 720, "label": "12 giờ"},
    {"minutes": -1, "label": "Toàn cảnh (Full)"},
]


def _torch_load_bundle(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _restore_fp32_state(sd):
    return {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}


def _softmax_1d(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.maximum(np.sum(e), 1e-12)


def _df_to_records(df):
    if df is None or df.empty:
        return []
    clean = df.replace({np.nan: None})
    return clean.to_dict(orient="records")


def _prune_tree_keep_ratio(tree, keep_ratio):
    """Subsample tree edges strictly connected from root 0 according to keep_ratio in (0, 1]."""
    ratio = float(np.clip(keep_ratio, 0.05, 1.0))
    if ratio >= 0.999 or tree["n_full"] <= 1:
        return tree

    raw_edges = tree["raw_edges"]
    raw_delays = tree["raw_delays"]
    target_non_root = max(0, int(round((tree["n_full"] - 1) * ratio)))
    if target_non_root == 0:
        return build_tree_from_raw(1, [], np.array([0.0], dtype=np.float32))

    adj = defaultdict(list)
    for a, b in raw_edges:
        adj[a].append(b)
    for k in adj:
        adj[k].sort(key=lambda nid: float(raw_delays[nid]))

    keep_old = {0}
    queue = [0]
    while queue and len(keep_old) - 1 < target_non_root:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in keep_old:
                keep_old.add(v)
                queue.append(v)
                if len(keep_old) - 1 >= target_non_root:
                    break

    keep_list = [0] + sorted(idx for idx in keep_old if idx != 0)
    old_to_new = {old: new for new, old in enumerate(keep_list)}
    sub_edges = sorted({(old_to_new[a], old_to_new[b]) for a, b in raw_edges if a in old_to_new and b in old_to_new})
    sub_delays = np.array([raw_delays[old] for old in keep_list], dtype=np.float32)
    sub_delays[0] = 0.0
    return build_tree_from_raw(len(keep_list), sub_edges, sub_delays)


def _build_template_tree(mode):
    """Construct controlled synthetic propagation topologies for counterfactual What-If."""
    if mode == "root_only":
        return build_tree_from_raw(1, [], np.array([0.0], dtype=np.float32))

    if mode == "flat_star":
        # Official news broadcast pattern: broad 1-hop fast replies, shallow depth=1
        n = 65
        edges = [(0, i) for i in range(1, n)]
        delays = np.zeros(n, dtype=np.float32)
        for i in range(1, n):
            delays[i] = round(1.5 + i * 0.7, 2)
        return build_tree_from_raw(n, edges, delays)

    if mode == "deep_viral":
        # Viral multi-hop cascading discussion: 115 nodes reaching depth 5
        edges = []
        delays = [0.0]
        cur_level = [0]
        next_id = 1
        branch_plan = [6, 3, 2, 2, 1]
        for lvl_idx, b_fac in enumerate(branch_plan):
            nxt_level = []
            for p in cur_level:
                p_delay = delays[p]
                for c_idx in range(b_fac):
                    if next_id >= 120:
                        break
                    edges.append((p, next_id))
                    delays.append(round(p_delay + 2.0 + (lvl_idx + 1) * 4.5 + c_idx * 1.8, 2))
                    nxt_level.append(next_id)
                    next_id += 1
            cur_level = nxt_level
        return build_tree_from_raw(len(delays), edges, np.array(delays, dtype=np.float32))

    # Default "slow_sparse": sparse small tree (14 nodes, depth 3, delayed reactions)
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (2, 6), (2, 7), (3, 8),
        (4, 9), (4, 10), (6, 11), (8, 12), (9, 13),
    ]
    delays = np.array([0.0, 28.0, 45.0, 92.0, 115.0, 140.0, 180.0, 210.0, 265.0, 340.0, 410.0, 520.0, 640.0, 780.0], dtype=np.float32)
    return build_tree_from_raw(14, edges, delays)


def _serialize_tree_for_viz(tree, max_viz_nodes=220):
    n_full = int(tree["n_full"])
    edges = tree["raw_edges"]
    delays = tree["raw_delays"]
    centrality = tree["centrality"]

    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)

    depth = np.zeros(n_full, dtype=np.int64)
    bfs = [0]
    seen = {0}
    for u in bfs:
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                depth[v] = depth[u] + 1
                bfs.append(v)

    keep = set(bfs[:max_viz_nodes])
    nodes = []
    for u in bfs[:max_viz_nodes]:
        nodes.append(
            {
                "id": int(u),
                "is_root": bool(u == 0),
                "depth": int(depth[u]),
                "delay": round(float(delays[u]), 2),
                "centrality": round(float(centrality[u]), 4),
                "children_count": int(len(adj[u])),
            }
        )
    vis_edges = [{"from": int(a), "to": int(b)} for a, b in edges if a in keep and b in keep]
    return {
        "n_full": n_full,
        "e_full": int(tree["e_full"]),
        "max_depth": int(tree["max_depth"]),
        "n_shown": len(nodes),
        "truncated_for_viz": bool(n_full > max_viz_nodes),
        "nodes": nodes,
        "edges": vis_edges,
    }


class DemoEngine:
    def __init__(self, data_root, bundle_dir, results_dir, tables_dir):
        self.data_root = data_root
        self.bundle_dir = bundle_dir
        self.results_dir = results_dir
        self.tables_dir = tables_dir
        self.bundles = {}
        self.models = {}
        self.datasets = {}
        self.event_catalog = {}
        self._load_all()

    def _load_all(self):
        for ds in ["Twitter15", "Twitter16"]:
            b_path = os.path.join(self.bundle_dir, f"{ds}_demo_bundle.pt")
            if not os.path.isfile(b_path):
                continue

            bundle = _torch_load_bundle(b_path)
            hp = bundle["hparams"]
            vocab = int(hp["vocab_size"])
            hid = int(hp["hidden_dim"])
            sage_hid = int(hp["sage_dim"])

            m_args = types.SimpleNamespace(
                hidden_dim=hid,
                out_dim=hid,
                td_droprate=float(hp["td_droprate"]),
                bu_droprate=float(hp["bu_droprate"]),
            )
            loaded_models = {}
            for m_name, sd in bundle["models"].items():
                if m_name == "BiGCN-Plus-Clean":
                    mdl = BiGCNPlus(
                        vocab=vocab,
                        hid=hid,
                        sage_hid=sage_hid,
                        n_cls=4,
                        td_drop=float(hp["td_droprate"]),
                        bu_drop=float(hp["bu_droprate"]),
                        mode="clean",
                        plm_dim=0,
                    )
                else:
                    mdl = build_baseline_model(m_name, vocab, m_args)
                mdl.load_state_dict(_restore_fp32_state(sd))
                mdl.eval()
                loaded_models[m_name] = mdl

            ids, y, docs, trees = load_dataset(self.data_root, ds)
            tid_to_idx = {tid: i for i, tid in enumerate(ids)}

            self.bundles[ds] = bundle
            self.models[ds] = loaded_models
            self.datasets[ds] = {
                "ids": ids,
                "y": y,
                "docs": docs,
                "trees": trees,
                "tid_to_idx": tid_to_idx,
            }
            self.event_catalog[ds] = self._build_event_catalog(ds)

    def _build_event_catalog(self, ds):
        bundle = self.bundles[ds]
        ddata = self.datasets[ds]
        te_idx = bundle["te_idx"]
        catalog = []
        for i in te_idx:
            tid = ddata["ids"][i]
            t = ddata["trees"][i]
            res = self.infer_single(ds, ddata["docs"][i], t, include_tokens=False)
            preds = res["models"]
            y_true = int(ddata["y"][i])
            plus_ok = preds["BiGCN-Plus-Clean"]["pred"] == y_true
            base_ok = preds["BiGCN-Base"]["pred"] == y_true
            rag_ok = preds["RAGCL"]["pred"] == y_true
            ridge_ok = preds["TF-IDF-Ridge"]["pred"] == y_true
            if plus_ok and ((not base_ok) or (not rag_ok) or (not ridge_ok)):
                win_tag = "highlight"
            elif plus_ok:
                win_tag = "correct"
            else:
                win_tag = "wrong"
            catalog.append(
                {
                    "tid": tid,
                    "y_true": y_true,
                    "true_label": LABELS[y_true],
                    "pred_plus": LABELS[preds["BiGCN-Plus-Clean"]["pred"]],
                    "pred_base": LABELS[preds["BiGCN-Base"]["pred"]],
                    "pred_ragcl": LABELS[preds["RAGCL"]["pred"]],
                    "n_nodes": int(t["n_full"]),
                    "max_depth": int(t["max_depth"]),
                    "win_tag": win_tag,
                    "snippet": ddata["docs"][i][:110],
                }
            )
        rank_map = {"highlight": 0, "correct": 1, "wrong": 2}
        catalog.sort(key=lambda r: (rank_map[r["win_tag"]], r["n_nodes"]))
        return catalog

    def _compute_b3_single(self, ds, text, tree, top_k_tokens=10):
        b3a = self.bundles[ds]["branch3_artifacts"]
        Xt = transform_sublinear_tfidf_texts([text], b3a["vocab"], b3a["idf"])
        raw = (Xt @ b3a["W_text"]).astype(np.float32)

        gs_raw = tree["graph_stats"].astype(np.float32)
        if b3a["has_graph_var"] and b3a["best_g_weight"] > 0.0:
            gs_z = (gs_raw[None, :] - b3a["gs_mu"]) / b3a["gs_sd"]
            norm = float(np.maximum(np.linalg.norm(gs_z, axis=1, keepdims=True), 1e-6)[0, 0])
            gs_n = gs_z / norm
            raw = raw + (gs_n @ b3a["W_graph"]).astype(np.float32)

        logits = ((raw - raw.mean(axis=1, keepdims=True)) / b3a["std_tr"]).astype(np.float32)

        # Pure TF-IDF-Ridge for side-by-side baseline comparison
        rp = self.bundles[ds]["ridge_pure_artifacts"]
        Xt_p = transform_sublinear_tfidf_texts([text], rp["vocab"], rp["idf"])
        ridge_logits = (Xt_p @ rp["W_text"]).astype(np.float32)[0]
        ridge_probs = _softmax_1d(ridge_logits)

        return logits, ridge_probs, Xt, gs_raw

    def _extract_top_tokens(self, ds, Xt, pred_cls, top_k=10):
        b3a = self.bundles[ds]["branch3_artifacts"]
        vocab = b3a["vocab"]
        W_text = b3a["W_text"]
        row = Xt.getrow(0)
        items = []
        for idx, val in zip(row.indices, row.data):
            contrib = float(val * W_text[idx, pred_cls])
            tok = vocab[idx]
            if tok.startswith("c3_") or tok.startswith("c4_"):
                kind = "Ký tự n-gram"
                display = tok[3:]
            elif " " in tok:
                kind = "Cụm 2 từ"
                display = tok
            else:
                kind = "Từ đơn"
                display = tok
            items.append({"token": display, "kind": kind, "score": round(contrib, 5)})

        pos_items = sorted([x for x in items if x["score"] > 0], key=lambda x: -x["score"])
        if len(pos_items) >= top_k:
            return pos_items[:top_k]
        items.sort(key=lambda x: -abs(x["score"]))
        return items[:top_k]

    def infer_single(self, ds, text, tree, include_tokens=True):
        bundle = self.bundles[ds]
        mdls = self.models[ds]
        device = torch.device("cpu")

        bow_np = transform_bow_texts([text], bundle["bow_vocab"])
        text_t = torch.tensor(bow_np, dtype=torch.float32, device=device)
        y_dummy = torch.zeros(1, dtype=torch.long, device=device)

        b3_np, ridge_probs, Xt_b3, gs_raw = self._compute_b3_single(ds, text, tree)
        b3_t = torch.tensor(b3_np, dtype=torch.float32, device=device)

        t_base = [
            {
                "n": int(tree["n"]),
                "edge_index": torch.tensor(tree["edge_index"], dtype=torch.long, device=device),
                "struct": torch.tensor(tree["struct"][:, :N_STRUCT_BASE], dtype=torch.float32, device=device),
                "centrality": torch.tensor(tree["centrality"], dtype=torch.float32, device=device),
            }
        ]
        t_ext = [
            {
                "n": int(tree["n"]),
                "edge_index": t_base[0]["edge_index"],
                "struct": torch.tensor(tree["struct"][:, :N_STRUCT_EXT], dtype=torch.float32, device=device),
                "centrality": t_base[0]["centrality"],
            }
        ]

        b_base = make_batch([0], t_base, text_t, y_dummy, device, n_struct=N_STRUCT_BASE)
        b_ext = make_batch([0], t_ext, text_t, y_dummy, device, n_struct=N_STRUCT_EXT, branch3_t=b3_t)

        model_results = {}
        model_results["TF-IDF-Ridge"] = {
            "probs": [round(float(v), 5) for v in ridge_probs],
            "pred": int(np.argmax(ridge_probs)),
            "label": LABELS[int(np.argmax(ridge_probs))],
        }

        with torch.no_grad():
            for m_name in ["MLP-BoW", "BiGCN-Base"]:
                p = torch.exp(mdls[m_name](b_base))[0].numpy()
                pred_c = int(np.argmax(p))
                model_results[m_name] = {
                    "probs": [round(float(v), 5) for v in p],
                    "pred": pred_c,
                    "label": LABELS[pred_c],
                }

            p_rag = torch.exp(mdls["RAGCL"](b_ext))[0].numpy()
            pred_r = int(np.argmax(p_rag))
            model_results["RAGCL"] = {
                "probs": [round(float(v), 5) for v in p_rag],
                "pred": pred_r,
                "label": LABELS[pred_r],
            }

            exp_out = mdls["BiGCN-Plus-Clean"].forward_explain(b_ext)
            p_plus = exp_out["probs"][0].numpy()
            pred_plus = int(np.argmax(p_plus))
            model_results["BiGCN-Plus-Clean"] = {
                "probs": [round(float(v), 5) for v in p_plus],
                "pred": pred_plus,
                "label": LABELS[pred_plus],
            }

        xai = {
            "fusion_weights": [round(float(w), 5) for w in exp_out["fusion_weights"].numpy()],
            "branch1_probs": [round(float(v), 5) for v in exp_out["probs_b1"][0].numpy()],
            "branch2_probs": [round(float(v), 5) for v in exp_out["probs_b2"][0].numpy()],
            "branch3_probs": [round(float(v), 5) for v in exp_out["probs_b3"][0].numpy()],
            "auto_loo_params": {
                "g_weight": bundle["branch3_artifacts"]["best_g_weight"],
                "ridge_lam": bundle["branch3_artifacts"]["best_lam"],
            },
            "graph_stats": [
                {"key": k, "name": nm, "value": round(float(val), 4)}
                for (k, nm), val in zip(GRAPH_STAT_NAMES_VI, gs_raw)
            ],
            "top_tokens": self._extract_top_tokens(ds, Xt_b3, pred_plus, top_k=10) if include_tokens else [],
        }

        return {"models": model_results, "xai": xai}

    def get_meta(self):
        return {
            "available_datasets": sorted(self.bundles.keys()),
            "labels": LABELS,
            "label_names_vi": LABEL_NAMES_VI,
            "deadlines": DEADLINE_OPTIONS,
            "bundles_info": {
                ds: {
                    "seed": b["seed"],
                    "fold": b["fold"],
                    "epochs": b["epochs"],
                    "device_trained": b["device_trained"],
                    "n_test_events": len(b["te_idx"]),
                    "b3_recon_max_err": b.get("b3_recon_max_err", 0.0),
                    "te_metrics": b.get("te_metrics", {}),
                }
                for ds, b in self.bundles.items()
            },
            "events": self.event_catalog,
        }

    def get_event_detail(self, ds, tid, max_delay=-1):
        ddata = self.datasets[ds]
        idx = ddata["tid_to_idx"][tid]
        base_tree = ddata["trees"][idx]
        sub_tree = truncate_tree_by_delay(base_tree, max_delay) if (max_delay is not None and max_delay >= 0) else base_tree
        res = self.infer_single(ds, ddata["docs"][idx], sub_tree, include_tokens=True)
        return {
            "dataset": ds,
            "tid": tid,
            "text": ddata["docs"][idx],
            "y_true": int(ddata["y"][idx]),
            "true_label": LABELS[int(ddata["y"][idx])],
            "true_label_vi": LABEL_NAMES_VI[int(ddata["y"][idx])],
            "max_delay": int(max_delay),
            "tree_viz": _serialize_tree_for_viz(sub_tree),
            **res,
        }

    def get_early_trajectory(self, ds, tid):
        ddata = self.datasets[ds]
        idx = ddata["tid_to_idx"][tid]
        text = ddata["docs"][idx]
        base_tree = ddata["trees"][idx]
        y_true = int(ddata["y"][idx])

        points = []
        for opt in DEADLINE_OPTIONS:
            dt = opt["minutes"]
            sub_tree = truncate_tree_by_delay(base_tree, dt) if dt >= 0 else base_tree
            res = self.infer_single(ds, text, sub_tree, include_tokens=False)
            points.append(
                {
                    "minutes": dt,
                    "label": opt["label"],
                    "n_nodes": int(sub_tree["n_full"]),
                    "max_depth": int(sub_tree["max_depth"]),
                    "plus_probs": res["models"]["BiGCN-Plus-Clean"]["probs"],
                    "plus_pred": res["models"]["BiGCN-Plus-Clean"]["label"],
                    "base_probs": res["models"]["BiGCN-Base"]["probs"],
                    "base_pred": res["models"]["BiGCN-Base"]["label"],
                    "ragcl_probs": res["models"]["RAGCL"]["probs"],
                    "ragcl_pred": res["models"]["RAGCL"]["label"],
                    "ridge_probs": res["models"]["TF-IDF-Ridge"]["probs"],
                    "fusion_weights": res["xai"]["fusion_weights"],
                }
            )
        return {
            "dataset": ds,
            "tid": tid,
            "text": text,
            "y_true": y_true,
            "true_label": LABELS[y_true],
            "points": points,
        }

    def run_what_if(self, payload):
        ds = payload.get("dataset") or sorted(self.bundles.keys())[0]
        text = str(payload.get("text", "")).strip() or "BREAKING: Unverified claim spreading rapidly on social media"
        template_mode = payload.get("template_mode", "use_base")
        base_tid = payload.get("base_tid")
        keep_ratio = float(payload.get("keep_edge_ratio", 1.0))
        max_delay = int(payload.get("max_delay", -1))

        if template_mode == "use_base" and base_tid and base_tid in self.datasets[ds]["tid_to_idx"]:
            idx = self.datasets[ds]["tid_to_idx"][base_tid]
            tree = self.datasets[ds]["trees"][idx]
            if max_delay >= 0:
                tree = truncate_tree_by_delay(tree, max_delay)
            tree = _prune_tree_keep_ratio(tree, keep_ratio)
        else:
            tree = _build_template_tree(template_mode)

        res = self.infer_single(ds, text, tree, include_tokens=True)
        return {
            "dataset": ds,
            "text": text,
            "template_mode": template_mode,
            "tree_viz": _serialize_tree_for_viz(tree),
            **res,
        }

    def get_benchmarks(self):
        def read_csv_safe(name):
            p = os.path.join(self.tables_dir, name)
            if os.path.isfile(p):
                return _df_to_records(pd.read_csv(p))
            return []

        return {
            "master": read_csv_safe("master_multiseed_summary.csv"),
            "same_eval": read_csv_safe("same_eval_comparison.csv"),
            "statistical_tests": read_csv_safe("statistical_tests.csv"),
            "tree_sparsity": read_csv_safe("tree_sparsity_analysis.csv"),
            "early_detection": read_csv_safe("early_detection_summary.csv"),
        }


def make_handler(engine):
    class DemoHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _send_json(self, obj, status=200):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, path):
            if not os.path.isfile(path):
                self._send_json({"error": "Not found"}, status=404)
                return
            mime, _ = mimetypes.guess_type(path)
            mime = mime or "application/octet-stream"
            if mime.startswith("text/") or mime == "application/javascript":
                mime += "; charset=utf-8"
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            if path == "/" or path == "/index.html":
                return self._send_file(os.path.join(STATIC_DIR, "index.html"))
            if path.startswith("/static/"):
                rel = os.path.normpath(path[len("/static/"):]).lstrip("\\/")
                if ".." in rel:
                    return self._send_json({"error": "Forbidden"}, status=403)
                return self._send_file(os.path.join(STATIC_DIR, rel))

            try:
                if path == "/api/meta":
                    return self._send_json(engine.get_meta())
                if path == "/api/event":
                    ds = qs.get("dataset", [None])[0] or sorted(engine.bundles.keys())[0]
                    tid = qs.get("tid", [None])[0] or engine.event_catalog[ds][0]["tid"]
                    max_delay = int(qs.get("max_delay", ["-1"])[0])
                    return self._send_json(engine.get_event_detail(ds, tid, max_delay=max_delay))
                if path == "/api/early_trajectory":
                    ds = qs.get("dataset", [None])[0] or sorted(engine.bundles.keys())[0]
                    tid = qs.get("tid", [None])[0] or engine.event_catalog[ds][0]["tid"]
                    return self._send_json(engine.get_early_trajectory(ds, tid))
                if path == "/api/benchmarks":
                    return self._send_json(engine.get_benchmarks())
                return self._send_json({"error": f"Unknown route: {path}"}, status=404)
            except Exception as exc:
                return self._send_json({"error": str(exc)}, status=500)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/what_if":
                return self._send_json({"error": f"Unknown route: {parsed.path}"}, status=404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw_body.decode("utf-8"))
                return self._send_json(engine.run_what_if(payload))
            except Exception as exc:
                return self._send_json({"error": str(exc)}, status=500)

    return DemoHandler


def main():
    parser = argparse.ArgumentParser(description="BiGCN-Plus XAI Interactive Demo Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--bundle_dir", type=str, default=os.path.join(ROOT_DIR, "results", "demo_models"))
    parser.add_argument("--results_dir", type=str, default=os.path.join(ROOT_DIR, "results", "multi_seed"))
    parser.add_argument("--tables_dir", type=str, default=os.path.join(ROOT_DIR, "results", "tables"))
    parser.add_argument("--no_browser", action="store_true")
    args = parser.parse_args()

    bundle_dir = args.bundle_dir
    has_official = any(os.path.isfile(os.path.join(bundle_dir, f"{d}_demo_bundle.pt")) for d in ("Twitter15", "Twitter16"))
    smoke_bundle_dir = os.path.join(ROOT_DIR, "results", "smoke_test", "demo_models")
    if not has_official and any(os.path.isfile(os.path.join(smoke_bundle_dir, f"{d}_demo_bundle.pt")) for d in ("Twitter15", "Twitter16")):
        bundle_dir = smoke_bundle_dir
        print(f"[INFO] Chua co goi chinh thuc tai '{args.bundle_dir}'; dang nap tam goi thu nghiem tu '{bundle_dir}'.")

    data_root = find_data_root(args.data_root)
    engine = DemoEngine(data_root, bundle_dir, args.results_dir, args.tables_dir)
    if not engine.bundles:
        print(
            f"[CANH BAO] Chua tim thay goi demo .pt tai '{bundle_dir}'. "
            "Hay chay 'python scripts/export_demo_bundle.py' hoac truyen --bundle_dir."
        )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    url = f"http://{args.host}:{args.port}"
    print(f"BiGCN-Plus Demo Server running at {url} | Bundles loaded: {list(engine.bundles.keys())}")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Demo Server...")
        server.server_close()


if __name__ == "__main__":
    main()
