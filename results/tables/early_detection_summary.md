# Tổng hợp Đường cong Phát hiện Sớm (Early Rumor Detection)

| Dataset | Mốc thời gian | Mô hình | Lượt chạy | Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) | p vs BiGCN-Base | p vs RAGCL |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Twitter15 | 0p (Bài gốc) | `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.83761 ± 0.01511 | — | — |
| Twitter15 | 0p (Bài gốc) | `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.81735 ± 0.02148 | — | — |
| Twitter15 | 0p (Bài gốc) | `GraphSAGE-Single` | 25 | 0.39195 ± 0.10665 | 0.29524 ± 0.13113 | — | — |
| Twitter15 | 0p (Bài gốc) | `BiGCN-Plus-Clean` | 25 | 0.69664 ± 0.03958 | 0.65608 ± 0.04893 | — | — |
| Twitter15 | 10 phút | `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.83761 ± 0.01511 | — | — |
| Twitter15 | 10 phút | `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.81735 ± 0.02148 | — | — |
| Twitter15 | 10 phút | `GraphSAGE-Single` | 25 | 0.76604 ± 0.02674 | 0.76121 ± 0.02825 | — | — |
| Twitter15 | 10 phút | `BiGCN-Plus-Clean` | 25 | 0.81745 ± 0.02533 | 0.81556 ± 0.02632 | — | — |
| Twitter15 | 30 phút | `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.83761 ± 0.01511 | — | — |
| Twitter15 | 30 phút | `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.81735 ± 0.02148 | — | — |
| Twitter15 | 30 phút | `GraphSAGE-Single` | 25 | 0.79396 ± 0.02508 | 0.79216 ± 0.02642 | — | — |
| Twitter15 | 30 phút | `BiGCN-Plus-Clean` | 25 | 0.83302 ± 0.02261 | 0.83255 ± 0.02294 | — | — |
| Twitter15 | 1 giờ | `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.83761 ± 0.01511 | — | — |
| Twitter15 | 1 giờ | `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.81735 ± 0.02148 | — | — |
| Twitter15 | 1 giờ | `GraphSAGE-Single` | 25 | 0.80685 ± 0.02464 | 0.80595 ± 0.02536 | — | — |
| Twitter15 | 1 giờ | `BiGCN-Plus-Clean` | 25 | 0.84309 ± 0.02247 | 0.84287 ± 0.02258 | — | — |
| Twitter15 | 2 giờ | `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.83761 ± 0.01511 | — | — |
| Twitter15 | 2 giờ | `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.81735 ± 0.02148 | — | — |
| Twitter15 | 2 giờ | `GraphSAGE-Single` | 25 | 0.81638 ± 0.02600 | 0.81589 ± 0.02645 | — | — |
| Twitter15 | 2 giờ | `BiGCN-Plus-Clean` | 25 | 0.84617 ± 0.01979 | 0.84601 ± 0.01979 | — | — |
| Twitter15 | 4 giờ | `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.83761 ± 0.01511 | — | — |
| Twitter15 | 4 giờ | `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.81735 ± 0.02148 | — | — |
| Twitter15 | 4 giờ | `GraphSAGE-Single` | 25 | 0.82121 ± 0.02378 | 0.82096 ± 0.02410 | — | — |
| Twitter15 | 4 giờ | `BiGCN-Plus-Clean` | 25 | 0.84980 ± 0.01904 | 0.84967 ± 0.01903 | — | — |
| Twitter15 | 12 giờ | `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.83761 ± 0.01511 | — | — |
| Twitter15 | 12 giờ | `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.81735 ± 0.02148 | — | — |
| Twitter15 | 12 giờ | `GraphSAGE-Single` | 25 | 0.82685 ± 0.02531 | 0.82682 ± 0.02547 | — | — |
| Twitter15 | 12 giờ | `BiGCN-Plus-Clean` | 25 | 0.85208 ± 0.01986 | 0.85208 ± 0.01984 | — | — |
| Twitter15 | Toàn cảnh (Full) | `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.83761 ± 0.01511 | — | — |
| Twitter15 | Toàn cảnh (Full) | `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.81735 ± 0.02148 | — | — |
| Twitter15 | Toàn cảnh (Full) | `GraphSAGE-Single` | 25 | 0.83020 ± 0.02617 | 0.83032 ± 0.02610 | — | — |
| Twitter15 | Toàn cảnh (Full) | `BiGCN-Plus-Clean` | 25 | 0.85315 ± 0.02114 | 0.85319 ± 0.02097 | — | — |
| Twitter16 | 0p (Bài gốc) | `TF-IDF-Ridge` | 25 | 0.85230 ± 0.02954 | 0.85304 ± 0.02880 | — | — |
| Twitter16 | 0p (Bài gốc) | `MLP-BoW` | 25 | 0.84450 ± 0.02257 | 0.84437 ± 0.02267 | — | — |
| Twitter16 | 0p (Bài gốc) | `GraphSAGE-Single` | 25 | 0.46977 ± 0.13547 | 0.36905 ± 0.15634 | — | — |
| Twitter16 | 0p (Bài gốc) | `BiGCN-Plus-Clean` | 25 | 0.75136 ± 0.03246 | 0.71638 ± 0.04520 | — | — |
| Twitter16 | 10 phút | `TF-IDF-Ridge` | 25 | 0.85230 ± 0.02954 | 0.85304 ± 0.02880 | — | — |
| Twitter16 | 10 phút | `MLP-BoW` | 25 | 0.84450 ± 0.02257 | 0.84437 ± 0.02267 | — | — |
| Twitter16 | 10 phút | `GraphSAGE-Single` | 25 | 0.78458 ± 0.03263 | 0.77947 ± 0.03361 | — | — |
| Twitter16 | 10 phút | `BiGCN-Plus-Clean` | 25 | 0.84769 ± 0.02038 | 0.84527 ± 0.02168 | — | — |
| Twitter16 | 30 phút | `TF-IDF-Ridge` | 25 | 0.85230 ± 0.02954 | 0.85304 ± 0.02880 | — | — |
| Twitter16 | 30 phút | `MLP-BoW` | 25 | 0.84450 ± 0.02257 | 0.84437 ± 0.02267 | — | — |
| Twitter16 | 30 phút | `GraphSAGE-Single` | 25 | 0.81589 ± 0.03019 | 0.81381 ± 0.03103 | — | — |
| Twitter16 | 30 phút | `BiGCN-Plus-Clean` | 25 | 0.86406 ± 0.02330 | 0.86293 ± 0.02386 | — | — |
| Twitter16 | 1 giờ | `TF-IDF-Ridge` | 25 | 0.85230 ± 0.02954 | 0.85304 ± 0.02880 | — | — |
| Twitter16 | 1 giờ | `MLP-BoW` | 25 | 0.84450 ± 0.02257 | 0.84437 ± 0.02267 | — | — |
| Twitter16 | 1 giờ | `GraphSAGE-Single` | 25 | 0.83227 ± 0.02633 | 0.83077 ± 0.02723 | — | — |
| Twitter16 | 1 giờ | `BiGCN-Plus-Clean` | 25 | 0.87092 ± 0.02635 | 0.87030 ± 0.02701 | — | — |
| Twitter16 | 2 giờ | `TF-IDF-Ridge` | 25 | 0.85230 ± 0.02954 | 0.85304 ± 0.02880 | — | — |
| Twitter16 | 2 giờ | `MLP-BoW` | 25 | 0.84450 ± 0.02257 | 0.84437 ± 0.02267 | — | — |
| Twitter16 | 2 giờ | `GraphSAGE-Single` | 25 | 0.84156 ± 0.02569 | 0.84070 ± 0.02652 | — | — |
| Twitter16 | 2 giờ | `BiGCN-Plus-Clean` | 25 | 0.87410 ± 0.02723 | 0.87369 ± 0.02771 | — | — |
| Twitter16 | 4 giờ | `TF-IDF-Ridge` | 25 | 0.85230 ± 0.02954 | 0.85304 ± 0.02880 | — | — |
| Twitter16 | 4 giờ | `MLP-BoW` | 25 | 0.84450 ± 0.02257 | 0.84437 ± 0.02267 | — | — |
| Twitter16 | 4 giờ | `GraphSAGE-Single` | 25 | 0.85012 ± 0.02361 | 0.84957 ± 0.02414 | — | — |
| Twitter16 | 4 giờ | `BiGCN-Plus-Clean` | 25 | 0.87508 ± 0.02708 | 0.87488 ± 0.02759 | — | — |
| Twitter16 | 12 giờ | `TF-IDF-Ridge` | 25 | 0.85230 ± 0.02954 | 0.85304 ± 0.02880 | — | — |
| Twitter16 | 12 giờ | `MLP-BoW` | 25 | 0.84450 ± 0.02257 | 0.84437 ± 0.02267 | — | — |
| Twitter16 | 12 giờ | `GraphSAGE-Single` | 25 | 0.85991 ± 0.02379 | 0.85962 ± 0.02428 | — | — |
| Twitter16 | 12 giờ | `BiGCN-Plus-Clean` | 25 | 0.87337 ± 0.02672 | 0.87344 ± 0.02704 | — | — |
| Twitter16 | Toàn cảnh (Full) | `TF-IDF-Ridge` | 25 | 0.85230 ± 0.02954 | 0.85304 ± 0.02880 | — | — |
| Twitter16 | Toàn cảnh (Full) | `MLP-BoW` | 25 | 0.84450 ± 0.02257 | 0.84437 ± 0.02267 | — | — |
| Twitter16 | Toàn cảnh (Full) | `GraphSAGE-Single` | 25 | 0.86137 ± 0.02466 | 0.86124 ± 0.02500 | — | — |
| Twitter16 | Toàn cảnh (Full) | `BiGCN-Plus-Clean` | 25 | 0.87312 ± 0.02826 | 0.87362 ± 0.02835 | — | — |
