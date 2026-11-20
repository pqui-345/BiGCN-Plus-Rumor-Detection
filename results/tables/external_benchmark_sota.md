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
| `TF-IDF-Ridge` | 25 | 83.69% ± 1.52% | 85.23% ± 2.95% |
| `TF-IDF-LR` | 25 | 83.09% ± 1.63% | 84.28% ± 2.64% |
| `TF-IDF-SVC` | 25 | 79.49% ± 2.57% | 83.42% ± 2.38% |
| `MLP-BoW` | 25 | 81.73% ± 2.16% | 84.45% ± 2.26% |
| `GCN-Single` | 25 | 77.38% ± 2.11% | 65.36% ± 3.56% |
| `GraphSAGE-Single` | 25 | 83.02% ± 2.60% | 86.14% ± 2.49% |
| `GAT-Single` | 25 | 80.23% ± 2.51% | 84.21% ± 2.72% |
| `BiGCN-Base` | 25 | 83.49% ± 1.94% | 81.34% ± 2.77% |
| `BiGCN-NoRoot` | 25 | 76.63% ± 2.02% | 71.37% ± 3.88% |
| `BiGCN-NoDropEdge` | 25 | 83.53% ± 2.33% | 81.69% ± 2.39% |
| `BiGCN-TDOnly` | 25 | 81.36% ± 2.25% | 83.06% ± 2.19% |
| `BiGCN-BUOnly` | 25 | 47.53% ± 2.71% | 50.61% ± 3.89% |
| `BiGCN-Base+` | 25 | 76.56% ± 2.70% | 74.01% ± 3.44% |
| `EBGCN` | 25 | 83.15% ± 1.82% | 85.48% ± 2.34% |
| `RAGCL` | 25 | 83.01% ± 2.46% | 79.98% ± 2.85% |
| `BiGCN-Plus-B1Only` | 25 | 83.87% ± 2.33% | 81.52% ± 2.96% |
| `BiGCN-Plus-B2Only` | 25 | 84.26% ± 2.25% | 86.09% ± 2.71% |
| `BiGCN-Plus-B12` | 25 | 84.72% ± 2.31% | 86.80% ± 2.84% |
| `BiGCN-Plus-Clean` | 25 | 85.95% ± 2.19% | 87.12% ± 2.72% |
| `BiGCN-Plus-UserID` | 25 | 88.39% ± 1.86% | 89.71% ± 2.41% |
