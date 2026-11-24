# Tổng hợp Đường cong Phát hiện Sớm (Early Rumor Detection)

| Dataset | Mốc thời gian | Mô hình | Lượt chạy | Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) | p vs BiGCN-Base | p vs RAGCL |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Twitter15 | 0p (Bài gốc) | `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.83761 ± 0.01511 | — | — |
| Twitter15 | 0p (Bài gốc) | `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.81735 ± 0.02148 | — | — |
| Twitter15 | 0p (Bài gốc) | `GraphSAGE-Single` | 25 | 0.39195 ± 0.10665 | 0.29524 ± 0.13113 | — | — |
| Twitter15 | 10 phút | `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.83761 ± 0.01511 | — | — |
| Twitter15 | 10 phút | `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.81735 ± 0.02148 | — | — |
| Twitter15 | 10 phút | `GraphSAGE-Single` | 25 | 0.76604 ± 0.02674 | 0.76121 ± 0.02825 | — | — |
| Twitter15 | 30 phút | `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.83761 ± 0.01511 | — | — |
| Twitter15 | 30 phút | `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.81735 ± 0.02148 | — | — |
| Twitter15 | 30 phút | `GraphSAGE-Single` | 25 | 0.79396 ± 0.02508 | 0.79216 ± 0.02642 | — | — |
| Twitter15 | 1 giờ | `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.83761 ± 0.01511 | — | — |
| Twitter15 | 1 giờ | `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.81735 ± 0.02148 | — | — |
| Twitter15 | 1 giờ | `GraphSAGE-Single` | 25 | 0.80685 ± 0.02464 | 0.80595 ± 0.02536 | — | — |
| Twitter15 | 2 giờ | `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.83761 ± 0.01511 | — | — |
| Twitter15 | 2 giờ | `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.81735 ± 0.02148 | — | — |
| Twitter15 | 2 giờ | `GraphSAGE-Single` | 25 | 0.81638 ± 0.02600 | 0.81589 ± 0.02645 | — | — |
| Twitter15 | 4 giờ | `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.83761 ± 0.01511 | — | — |
| Twitter15 | 4 giờ | `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.81735 ± 0.02148 | — | — |
| Twitter15 | 4 giờ | `GraphSAGE-Single` | 25 | 0.82121 ± 0.02378 | 0.82096 ± 0.02410 | — | — |
| Twitter15 | 12 giờ | `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.83761 ± 0.01511 | — | — |
| Twitter15 | 12 giờ | `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.81735 ± 0.02148 | — | — |
| Twitter15 | 12 giờ | `GraphSAGE-Single` | 25 | 0.82685 ± 0.02531 | 0.82682 ± 0.02547 | — | — |
| Twitter15 | Toàn cảnh (Full) | `TF-IDF-Ridge` | 25 | 0.83691 ± 0.01519 | 0.83761 ± 0.01511 | — | — |
| Twitter15 | Toàn cảnh (Full) | `MLP-BoW` | 25 | 0.81732 ± 0.02162 | 0.81735 ± 0.02148 | — | — |
| Twitter15 | Toàn cảnh (Full) | `GraphSAGE-Single` | 25 | 0.83020 ± 0.02617 | 0.83032 ± 0.02610 | — | — |
| Twitter16 | 0p (Bài gốc) | `TF-IDF-Ridge` | 25 | 0.85230 ± 0.02954 | 0.85304 ± 0.02880 | — | — |
| Twitter16 | 0p (Bài gốc) | `MLP-BoW` | 25 | 0.84450 ± 0.02257 | 0.84437 ± 0.02267 | — | — |
| Twitter16 | 0p (Bài gốc) | `GraphSAGE-Single` | 25 | 0.46977 ± 0.13547 | 0.36905 ± 0.15634 | — | — |
| Twitter16 | 10 phút | `TF-IDF-Ridge` | 25 | 0.85230 ± 0.02954 | 0.85304 ± 0.02880 | — | — |
| Twitter16 | 10 phút | `MLP-BoW` | 25 | 0.84450 ± 0.02257 | 0.84437 ± 0.02267 | — | — |
| Twitter16 | 10 phút | `GraphSAGE-Single` | 25 | 0.78458 ± 0.03263 | 0.77947 ± 0.03361 | — | — |
| Twitter16 | 30 phút | `TF-IDF-Ridge` | 25 | 0.85230 ± 0.02954 | 0.85304 ± 0.02880 | — | — |
| Twitter16 | 30 phút | `MLP-BoW` | 25 | 0.84450 ± 0.02257 | 0.84437 ± 0.02267 | — | — |
| Twitter16 | 30 phút | `GraphSAGE-Single` | 25 | 0.81589 ± 0.03019 | 0.81381 ± 0.03103 | — | — |
| Twitter16 | 1 giờ | `TF-IDF-Ridge` | 25 | 0.85230 ± 0.02954 | 0.85304 ± 0.02880 | — | — |
| Twitter16 | 1 giờ | `MLP-BoW` | 25 | 0.84450 ± 0.02257 | 0.84437 ± 0.02267 | — | — |
| Twitter16 | 1 giờ | `GraphSAGE-Single` | 25 | 0.83227 ± 0.02633 | 0.83077 ± 0.02723 | — | — |
| Twitter16 | 2 giờ | `TF-IDF-Ridge` | 25 | 0.85230 ± 0.02954 | 0.85304 ± 0.02880 | — | — |
| Twitter16 | 2 giờ | `MLP-BoW` | 25 | 0.84450 ± 0.02257 | 0.84437 ± 0.02267 | — | — |
| Twitter16 | 2 giờ | `GraphSAGE-Single` | 25 | 0.84156 ± 0.02569 | 0.84070 ± 0.02652 | — | — |
| Twitter16 | 4 giờ | `TF-IDF-Ridge` | 25 | 0.85230 ± 0.02954 | 0.85304 ± 0.02880 | — | — |
| Twitter16 | 4 giờ | `MLP-BoW` | 25 | 0.84450 ± 0.02257 | 0.84437 ± 0.02267 | — | — |
| Twitter16 | 4 giờ | `GraphSAGE-Single` | 25 | 0.85012 ± 0.02361 | 0.84957 ± 0.02414 | — | — |
| Twitter16 | 12 giờ | `TF-IDF-Ridge` | 25 | 0.85230 ± 0.02954 | 0.85304 ± 0.02880 | — | — |
| Twitter16 | 12 giờ | `MLP-BoW` | 25 | 0.84450 ± 0.02257 | 0.84437 ± 0.02267 | — | — |
| Twitter16 | 12 giờ | `GraphSAGE-Single` | 25 | 0.85991 ± 0.02379 | 0.85962 ± 0.02428 | — | — |
| Twitter16 | Toàn cảnh (Full) | `TF-IDF-Ridge` | 25 | 0.85230 ± 0.02954 | 0.85304 ± 0.02880 | — | — |
| Twitter16 | Toàn cảnh (Full) | `MLP-BoW` | 25 | 0.84450 ± 0.02257 | 0.84437 ± 0.02267 | — | — |
| Twitter16 | Toàn cảnh (Full) | `GraphSAGE-Single` | 25 | 0.86137 ± 0.02466 | 0.86124 ± 0.02500 | — | — |
