# External Benchmark Positioning on Twitter15 & Twitter16

Separation into **Group A** (open `rumdetect2017` source-tweet text + structure only, identical constraint to ours) and **Group B** (full reply-tweet texts crawled via legacy Twitter API, external KGs, or user follow graphs).

## Verified Literature Results (from Surveyed PDFs)

| Data Regime | Method & Source | Twitter15 Acc | Twitter16 Acc | Protocol |
| :--- | :--- | :---: | :---: | :--- |
| Group B (Full replies + Heterogeneous Graph) | MLI-GRA (Sci. Rep. 2025) | 92.00% | 92.90% | 70/15/15 split |
| Group B (Full replies + BERT + GAE) | GARD (WWW 2024) | 91.10% | 93.20% | 5-Fold CV |
| Group B (GloVe + User Follow/Reply Net) | PESTO (WASSA 2023) | 91.50% | 90.80% | 5-Fold CV |
| Group B (Full replies + BERT + BiGCN) | ICP-BGCN (Appl. Sci. 2025) | 89.70% | 91.70% | 5-Fold CV |
| Group B (Full replies + Bayesian Edge) | EBGCN (ACL 2021, orig.) | 89.20% | 91.50% | 5-Fold (re-impl 2025: 85.1/85.8) |
| Group B (Full replies TF-IDF + Trans-GCN) | TRGCN (Nature HSSC 2026) | 89.40% | 90.10% | Early stop on Val |
| Group B (Wikidata/Probase + Attn-GCN) | KAGN (J. Big Data 2023) | 89.23% | 90.13% | 5-Fold CV |
| Group B (RL Key Graph + BERT) | KPG (WWW 2025) | 89.30% | 88.90% | 5-Fold CV |
| Group B (Tree Transformer + Pretrain) | P2T3 (2026) | 87.40% | 91.10% | 10 random splits |
| Group B (Full replies + RoBERTa + InfoNCE) | RAGCL (AAAI 2024) | 86.70% | 90.50% | 10 random splits |
| Group B (Full replies TF-IDF) | Bi-GCN Orig. (AAAI 2020) | 88.60% | 88.00% | 5-Fold CV |
| Group B (Global Struct + Publisher Cred.) | GS-PC (2025) | 84.70% | 88.60% | 30 runs + Wilcoxon |
| Group A (Open rumdetect2017, source-only) | BiGCN Re-impl (RAGCL AAAI 2024) | 84.40% | 88.00% | 10 random splits |
| Group A (Open rumdetect2017, source-only) | BiGCN Re-impl (BiTGN EMNLP 2023) | 84.02% | 87.33% | 5-Fold CV |
| Group A (Open rumdetect2017, source TF-IDF) | BiGCN (TF-IDF, COLING 2025) | 81.80% | 84.70% | 5-Fold / Multi-split |

## Local / Current Multi-Seed Results Observed

| Model | Runs | Twitter15 Acc (Mean ± Std) | Twitter16 Acc (Mean ± Std) |
| :--- | :---: | :---: | :---: |
| `MLP-BoW` | 1 | — | 42.42% ± 0.00% |
| `GCN-Single` | 1 | — | 26.67% ± 0.00% |
| `GraphSAGE-Single` | 1 | — | 25.45% ± 0.00% |
| `GAT-Single` | 1 | — | 28.48% ± 0.00% |
| `BiGCN-Base` | 1 | — | 26.06% ± 0.00% |
| `BiGCN-NoRoot` | 1 | — | 24.85% ± 0.00% |
| `BiGCN-NoDropEdge` | 1 | — | 26.06% ± 0.00% |
| `BiGCN-TDOnly` | 1 | — | 26.06% ± 0.00% |
| `BiGCN-BUOnly` | 1 | — | 25.45% ± 0.00% |
| `BiGCN-Base+` | 1 | — | 23.64% ± 0.00% |
| `EBGCN` | 1 | — | 39.39% ± 0.00% |
| `RAGCL` | 1 | — | 24.85% ± 0.00% |
| `BiGCN-Plus-B1Only` | 1 | — | 26.67% ± 0.00% |
| `BiGCN-Plus-B2Only` | 1 | — | 26.67% ± 0.00% |
| `BiGCN-Plus-B12` | 1 | — | 24.85% ± 0.00% |
| `BiGCN-Plus-Clean` | 1 | — | 83.03% ± 0.00% |
| `BiGCN-Plus-UserID` | 1 | — | 88.48% ± 0.00% |
