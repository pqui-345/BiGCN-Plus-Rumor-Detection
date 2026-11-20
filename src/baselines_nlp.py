# -*- coding: utf-8 -*-
"""
Level 1 NLP Baselines (pure source-tweet baselines without propagation graph or User IDs):
  - TF-IDF-Ridge : Sublinear Word (1,2)-gram + Char (3,4)-gram L2 Ridge Classifier
  - TF-IDF-LR    : Sublinear TF-IDF + Multinomial Logistic Regression
  - TF-IDF-SVC   : Sublinear TF-IDF + Calibrated LinearSVC
  - Text-PLM     : Fine-tuned Pre-trained Language Model (e.g. roberta-base / bertweet-base)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_loader import build_sublinear_tfidf_features


def _softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def run_tfidf_ridge_fold(docs, y, tr_idx, te_idx, max_features=40000, ridge_lam=0.45):
    """Exact dual-form Kernel Ridge Classification strictly fit on tr_idx (zero User IDs)."""
    X = build_sublinear_tfidf_features(docs, tr_idx, max_features=max_features)
    X_tr = X[tr_idx]
    X_te = X[te_idx]
    n_tr = len(tr_idx)
    K_tr = (X_tr @ X_tr.T).toarray().astype(np.float64)
    K_te = (X_te @ X_tr.T).toarray().astype(np.float64)
    Y_tr = np.eye(4, dtype=np.float64)[y[tr_idx]]
    alpha = np.linalg.solve(K_tr + ridge_lam * np.eye(n_tr), Y_tr)
    logits_te = K_te @ alpha
    prob = _softmax_np(logits_te)
    pred = logits_te.argmax(axis=1)
    return y[te_idx], pred, prob


def run_tfidf_linear_fold(docs, y, tr_idx, te_idx, variant="lr", max_features=40000, seed=42):
    """Fit LogisticRegression or Calibrated LinearSVC strictly on tr_idx."""
    X = build_sublinear_tfidf_features(docs, tr_idx, max_features=max_features)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    try:
        if variant == "lr":
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(C=2.0, max_iter=1000, random_state=seed)
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)
            pred = prob.argmax(axis=1)
            return y_te, pred, prob
        elif variant == "svc":
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.svm import LinearSVC

            base = LinearSVC(C=1.0, random_state=seed)
            clf = CalibratedClassifierCV(base, cv=3)
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)
            pred = prob.argmax(axis=1)
            return y_te, pred, prob
    except ImportError:
        pass

    # Pure-PyTorch fallback if sklearn is not installed
    torch.manual_seed(seed)
    device = torch.device("cpu")
    Xtr_t = torch.tensor(X_tr.toarray(), dtype=torch.float32, device=device)
    Xte_t = torch.tensor(X_te.toarray(), dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    layer = nn.Linear(Xtr_t.size(1), 4).to(device)
    opt = torch.optim.AdamW(layer.parameters(), lr=0.05, weight_decay=1e-3 if variant == "svc" else 1e-4)
    for _ in range(120):
        opt.zero_grad()
        out = layer(Xtr_t)
        loss = F.multi_margin_loss(out, ytr_t) if variant == "svc" else F.cross_entropy(out, ytr_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = layer(Xte_t).numpy()
    prob = _softmax_np(logits)
    return y_te, prob.argmax(axis=1), prob


def run_plm_fold(
    docs,
    y,
    tr_idx,
    te_idx,
    model_name="roberta-base",
    epochs=6,
    batch_size=16,
    lr=2e-5,
    max_length=128,
    seed=42,
    device=None,
):
    """Fine-tune HuggingFace PLM (RoBERTa / BERTweet) on source tweets for one fold."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    tr_docs = [docs[i] for i in tr_idx]
    te_docs = [docs[i] for i in te_idx]
    y_tr = y[tr_idx]
    y_te = y[te_idx]

    rng = np.random.RandomState(seed)
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(len(tr_docs))
        for s in range(0, len(perm), batch_size):
            batch_ids = perm[s : s + batch_size]
            enc = tokenizer(
                [tr_docs[j] for j in batch_ids],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            labels = torch.tensor(y_tr[batch_ids], dtype=torch.long, device=device)
            opt.zero_grad()
            out = model(**enc, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(te_docs), batch_size * 2):
            enc = tokenizer(
                te_docs[s : s + batch_size * 2],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.cpu().numpy()
            probs.append(_softmax_np(logits))
    prob = np.concatenate(probs, axis=0)
    return y_te, prob.argmax(axis=1), prob


@torch.no_grad()
def extract_plm_embeddings(docs, model_name="roberta-base", batch_size=64, max_length=128, device=None):
    """Extract frozen [CLS]/mean-pooled source tweet embeddings (no label usage -> zero fold leakage)."""
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()

    all_embs = []
    for s in range(0, len(docs), batch_size):
        enc = tokenizer(
            docs[s : s + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = encoder(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        pooled = F.normalize(pooled, p=2, dim=1)
        all_embs.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(all_embs, axis=0)
