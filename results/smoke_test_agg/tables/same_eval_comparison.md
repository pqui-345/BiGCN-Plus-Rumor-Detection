# Same-Checkpoint Evaluation Comparison (`last` vs `snapshot5`)

| Dataset | EvalMode | Model | Runs | Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| Twitter16 | last | `MLP-BoW` | 1 | 0.4242 ± 0.0000 | 0.2952 ± 0.0000 |
| Twitter16 | last | `GCN-Single` | 1 | 0.2667 ± 0.0000 | 0.1339 ± 0.0000 |
| Twitter16 | last | `GraphSAGE-Single` | 1 | 0.2545 ± 0.0000 | 0.1121 ± 0.0000 |
| Twitter16 | last | `GAT-Single` | 1 | 0.2848 ± 0.0000 | 0.1822 ± 0.0000 |
| Twitter16 | last | `BiGCN-Base` | 1 | 0.2606 ± 0.0000 | 0.1138 ± 0.0000 |
| Twitter16 | last | `BiGCN-NoRoot` | 1 | 0.2485 ± 0.0000 | 0.0995 ± 0.0000 |
| Twitter16 | last | `BiGCN-NoDropEdge` | 1 | 0.2606 ± 0.0000 | 0.1138 ± 0.0000 |
| Twitter16 | last | `BiGCN-TDOnly` | 1 | 0.2606 ± 0.0000 | 0.1676 ± 0.0000 |
| Twitter16 | last | `BiGCN-BUOnly` | 1 | 0.2545 ± 0.0000 | 0.1014 ± 0.0000 |
| Twitter16 | last | `BiGCN-Base+` | 1 | 0.2667 ± 0.0000 | 0.1361 ± 0.0000 |
| Twitter16 | last | `EBGCN` | 1 | 0.3939 ± 0.0000 | 0.3145 ± 0.0000 |
| Twitter16 | last | `RAGCL` | 1 | 0.2485 ± 0.0000 | 0.1000 ± 0.0000 |
| Twitter16 | last | `BiGCN-Plus-B1Only` | 1 | 0.3273 ± 0.0000 | 0.2192 ± 0.0000 |
| Twitter16 | last | `BiGCN-Plus-B2Only` | 1 | 0.2909 ± 0.0000 | 0.2099 ± 0.0000 |
| Twitter16 | last | `BiGCN-Plus-B12` | 1 | 0.2606 ± 0.0000 | 0.1245 ± 0.0000 |
| Twitter16 | last | `BiGCN-Plus-Clean` | 1 | 0.8182 ± 0.0000 | 0.8192 ± 0.0000 |
| Twitter16 | last | `BiGCN-Plus-UserID` | 1 | 0.8909 ± 0.0000 | 0.8905 ± 0.0000 |
| Twitter16 | snapshot5 | `MLP-BoW` | 1 | 0.3879 ± 0.0000 | 0.2637 ± 0.0000 |
| Twitter16 | snapshot5 | `GCN-Single` | 1 | 0.2667 ± 0.0000 | 0.1339 ± 0.0000 |
| Twitter16 | snapshot5 | `GraphSAGE-Single` | 1 | 0.2545 ± 0.0000 | 0.1119 ± 0.0000 |
| Twitter16 | snapshot5 | `GAT-Single` | 1 | 0.2909 ± 0.0000 | 0.1784 ± 0.0000 |
| Twitter16 | snapshot5 | `BiGCN-Base` | 1 | 0.2545 ± 0.0000 | 0.1014 ± 0.0000 |
| Twitter16 | snapshot5 | `BiGCN-NoRoot` | 1 | 0.2485 ± 0.0000 | 0.0995 ± 0.0000 |
| Twitter16 | snapshot5 | `BiGCN-NoDropEdge` | 1 | 0.2545 ± 0.0000 | 0.1014 ± 0.0000 |
| Twitter16 | snapshot5 | `BiGCN-TDOnly` | 1 | 0.2788 ± 0.0000 | 0.1778 ± 0.0000 |
| Twitter16 | snapshot5 | `BiGCN-BUOnly` | 1 | 0.2545 ± 0.0000 | 0.1014 ± 0.0000 |
| Twitter16 | snapshot5 | `BiGCN-Base+` | 1 | 0.2364 ± 0.0000 | 0.1163 ± 0.0000 |
| Twitter16 | snapshot5 | `EBGCN` | 1 | 0.3515 ± 0.0000 | 0.2562 ± 0.0000 |
| Twitter16 | snapshot5 | `RAGCL` | 1 | 0.2606 ± 0.0000 | 0.1238 ± 0.0000 |
| Twitter16 | snapshot5 | `BiGCN-Plus-B1Only` | 1 | 0.2667 ± 0.0000 | 0.1262 ± 0.0000 |
| Twitter16 | snapshot5 | `BiGCN-Plus-B2Only` | 1 | 0.2667 ± 0.0000 | 0.1376 ± 0.0000 |
| Twitter16 | snapshot5 | `BiGCN-Plus-B12` | 1 | 0.2485 ± 0.0000 | 0.1000 ± 0.0000 |
| Twitter16 | snapshot5 | `BiGCN-Plus-Clean` | 1 | 0.8303 ± 0.0000 | 0.8316 ± 0.0000 |
| Twitter16 | snapshot5 | `BiGCN-Plus-UserID` | 1 | 0.8848 ± 0.0000 | 0.8841 ± 0.0000 |
