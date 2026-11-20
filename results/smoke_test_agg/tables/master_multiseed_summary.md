# Master Multi-Seed Summary Table

| Dataset | Model | Eval | Runs | Device | Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) | Seed-Std (Acc) | Kappa | MCC | AUC-ROC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Twitter16 | `MLP-BoW` | last | 1 | cpu | 0.4242 ± 0.0000 | 0.2952 ± 0.0000 | 0.0000 | 0.2318 | 0.2995 | 0.7924 |
| Twitter16 | `GCN-Single` | last | 1 | cpu | 0.2667 ± 0.0000 | 0.1339 ± 0.0000 | 0.0000 | 0.0240 | 0.0857 | 0.6260 |
| Twitter16 | `GraphSAGE-Single` | last | 1 | cpu | 0.2545 ± 0.0000 | 0.1121 ± 0.0000 | 0.0000 | 0.0081 | 0.0452 | 0.7455 |
| Twitter16 | `GAT-Single` | last | 1 | cpu | 0.2848 ± 0.0000 | 0.1822 ± 0.0000 | 0.0000 | 0.0484 | 0.0630 | 0.7583 |
| Twitter16 | `BiGCN-Base` | last | 1 | cpu | 0.2606 ± 0.0000 | 0.1138 ± 0.0000 | 0.0000 | 0.0082 | 0.0641 | 0.4932 |
| Twitter16 | `BiGCN-NoRoot` | last | 1 | cpu | 0.2485 ± 0.0000 | 0.0995 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6483 |
| Twitter16 | `BiGCN-NoDropEdge` | last | 1 | cpu | 0.2606 ± 0.0000 | 0.1138 ± 0.0000 | 0.0000 | 0.0082 | 0.0641 | 0.4912 |
| Twitter16 | `BiGCN-TDOnly` | last | 1 | cpu | 0.2606 ± 0.0000 | 0.1676 ± 0.0000 | 0.0000 | 0.0099 | 0.0145 | 0.6576 |
| Twitter16 | `BiGCN-BUOnly` | last | 1 | cpu | 0.2545 ± 0.0000 | 0.1014 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4567 |
| Twitter16 | `BiGCN-Base+` | snapshot5 | 1 | cpu | 0.2364 ± 0.0000 | 0.1163 ± 0.0000 | 0.0000 | -0.0161 | -0.0414 | 0.5889 |
| Twitter16 | `EBGCN` | last | 1 | cpu | 0.3939 ± 0.0000 | 0.3145 ± 0.0000 | 0.0000 | 0.1922 | 0.2551 | 0.7050 |
| Twitter16 | `RAGCL` | last | 1 | cpu | 0.2485 ± 0.0000 | 0.1000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7273 |
| Twitter16 | `BiGCN-Plus-B1Only` | snapshot5 | 1 | cpu | 0.2667 ± 0.0000 | 0.1262 ± 0.0000 | 0.0000 | 0.0167 | 0.0427 | 0.5362 |
| Twitter16 | `BiGCN-Plus-B2Only` | snapshot5 | 1 | cpu | 0.2667 ± 0.0000 | 0.1376 ± 0.0000 | 0.0000 | 0.0242 | 0.0566 | 0.6702 |
| Twitter16 | `BiGCN-Plus-B12` | snapshot5 | 1 | cpu | 0.2485 ± 0.0000 | 0.1000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5903 |
| Twitter16 | `BiGCN-Plus-Clean` | snapshot5 | 1 | cpu | 0.8303 ± 0.0000 | 0.8316 ± 0.0000 | 0.0000 | 0.7738 | 0.7882 | 0.9596 |
| Twitter16 | `BiGCN-Plus-UserID` | snapshot5 | 1 | cpu | 0.8848 ± 0.0000 | 0.8841 ± 0.0000 | 0.0000 | 0.8465 | 0.8472 | 0.9840 |
