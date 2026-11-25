# External Benchmark Positioning on Twitter15 & Twitter16

Separation into **Group A** (open `rumdetect2017` source-tweet text + structure only, identical constraint to ours) and **Group B** (full reply-tweet texts crawled via legacy Twitter API, external KGs, or user follow graphs).

## Verified Literature Results (from Surveyed PDFs)

| Data Regime | Method & Source | Twitter15 Acc | Twitter16 Acc | Protocol |
| :--- | :--- | :---: | :---: | :--- |
| Group B (Full replies + Heterogeneous Graph) | MLI-GRA (Sci. Rep. 2025) | 0.92000 | 0.92900 | 70/15/15 split |
| Group B (Full replies + BERT + GAE) | GARD (WWW 2024) | 0.91100 | 0.93200 | 5-Fold CV |
| Group B (GloVe + User Follow/Reply Net) | PESTO (WASSA 2023) | 0.91500 | 0.90800 | 5-Fold CV |
| Group B (Full replies + BERT + BiGCN) | ICP-BGCN (Appl. Sci. 2025) | 0.89700 | 0.91700 | 5-Fold CV |
| Group B (Full replies + Bayesian Edge) | EBGCN (ACL 2021, orig.) | 0.89200 | 0.91500 | 5-Fold (re-impl 2025: 85.1/85.8) |
| Group B (Full replies TF-IDF + Trans-GCN) | TRGCN (Nature HSSC 2026) | 0.89400 | 0.90100 | Early stop on Val |
| Group B (Wikidata/Probase + Attn-GCN) | KAGN (J. Big Data 2023) | 0.89230 | 0.90130 | 5-Fold CV |
| Group B (RL Key Graph + BERT) | KPG (WWW 2025) | 0.89300 | 0.88900 | 5-Fold CV |
| Group B (Tree Transformer + Pretrain) | P2T3 (2026) | 0.87400 | 0.91100 | 10 random splits |
| Group B (Full replies + RoBERTa + InfoNCE) | RAGCL (AAAI 2024) | 0.86700 | 0.90500 | 10 random splits |
| Group B (Full replies TF-IDF) | Bi-GCN Orig. (AAAI 2020) | 0.88600 | 0.88000 | 5-Fold CV |
| Group B (Global Struct + Publisher Cred.) | GS-PC (2025) | 0.84700 | 0.88600 | 30 runs + Wilcoxon |
| Group A (Open rumdetect2017, source-only) | BiGCN Re-impl (RAGCL AAAI 2024) | 0.84400 | 0.88000 | 10 random splits |
| Group A (Open rumdetect2017, source-only) | BiGCN Re-impl (BiTGN EMNLP 2023) | 0.84020 | 0.87330 | 5-Fold CV |
| Group A (Open rumdetect2017, source TF-IDF) | BiGCN (TF-IDF, COLING 2025) | 0.81800 | 0.84700 | 5-Fold / Multi-split |

## Local / Current Multi-Seed Results Observed

| Model | Runs | Twitter15 Acc (Mean ± Std) | Twitter16 Acc (Mean ± Std) |
| :--- | :---: | :---: | :---: |
| `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.85230 ± 0.02954 |
| `TF-IDF-LR` | 25 | 0.83087 ± 0.01630 | 0.84277 ± 0.02640 |
| `TF-IDF-SVC` | 25 | 0.79490 ± 0.02574 | 0.83422 ± 0.02382 |
| `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.84450 ± 0.02257 |
| `GCN-Single` | 25 | 0.77356 ± 0.02084 | 0.65356 ± 0.03560 |
| `GraphSAGE-Single` | 25 | 0.83020 ± 0.02617 | 0.86137 ± 0.02466 |
| `GAT-Single` | 25 | 0.80228 ± 0.02515 | 0.84206 ± 0.02723 |
| `BiGCN-Base` | 25 | 0.83745 ± 0.02294 | 0.86334 ± 0.02678 |
| `BiGCN-NoRoot` | 25 | 0.83248 ± 0.02458 | 0.86726 ± 0.02642 |
| `BiGCN-NoDropEdge` | 25 | 0.83624 ± 0.02133 | 0.86335 ± 0.02839 |
| `BiGCN-TDOnly` | 25 | 0.82725 ± 0.02515 | 0.85722 ± 0.02888 |
| `BiGCN-BUOnly` | 25 | 0.80698 ± 0.02462 | 0.76286 ± 0.03682 |
| `BiGCN-Base+` | 25 | 0.82054 ± 0.01952 | 0.85988 ± 0.02327 |
| `EBGCN` | 25 | 0.83597 ± 0.02206 | 0.86334 ± 0.02631 |
| `RAGCL` | 25 | 0.83638 ± 0.02373 | 0.86579 ± 0.02938 |
| `BiGCN-Plus-B1Only` | 25 | 0.83987 ± 0.02463 | 0.86383 ± 0.02974 |
| `BiGCN-Plus-B2Only` | 25 | 0.84282 ± 0.02363 | 0.86187 ± 0.02642 |
| `BiGCN-Plus-B12` | 25 | 0.84268 ± 0.02331 | 0.86798 ± 0.02848 |
| `BiGCN-Plus-Clean` | 25 | 0.85315 ± 0.02114 | 0.87312 ± 0.02826 |
| `BiGCN-Plus-UserID` | 25 | 0.87987 ± 0.01998 | 0.89267 ± 0.02165 |
