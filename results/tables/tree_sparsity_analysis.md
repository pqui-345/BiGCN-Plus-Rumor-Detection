# Phân tích Độ bền theo Quy mô / Độ thưa Cây Lan truyền (OOF Stratified)

| Dataset | Mô hình | Phân tầng cây | Số mẫu | Accuracy | Macro-F1 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Twitter15 | `TF-IDF-Ridge` | Tầng 1 — Cây thưa/nhỏ (≤ 174 nút) | 499 | 0.83768 | 0.65656 |
| Twitter15 | `TF-IDF-Ridge` | Tầng 2 — Cây trung bình (175–380 nút) | 497 | 0.87123 | 0.87085 |
| Twitter15 | `TF-IDF-Ridge` | Tầng 3 — Cây dày/lớn (> 380 nút) | 494 | 0.85223 | 0.83335 |
| Twitter15 | `TF-IDF-LR` | Tầng 1 — Cây thưa/nhỏ (≤ 174 nút) | 499 | 0.81363 | 0.64583 |
| Twitter15 | `TF-IDF-LR` | Tầng 2 — Cây trung bình (175–380 nút) | 497 | 0.85111 | 0.85143 |
| Twitter15 | `TF-IDF-LR` | Tầng 3 — Cây dày/lớn (> 380 nút) | 494 | 0.85020 | 0.83047 |
| Twitter15 | `TF-IDF-SVC` | Tầng 1 — Cây thưa/nhỏ (≤ 174 nút) | 499 | 0.82966 | 0.64116 |
| Twitter15 | `TF-IDF-SVC` | Tầng 2 — Cây trung bình (175–380 nút) | 497 | 0.82093 | 0.81602 |
| Twitter15 | `TF-IDF-SVC` | Tầng 3 — Cây dày/lớn (> 380 nút) | 494 | 0.79757 | 0.78256 |
| Twitter15 | `MLP-BoW` | Tầng 1 — Cây thưa/nhỏ (≤ 174 nút) | 499 | 0.84369 | 0.65500 |
| Twitter15 | `MLP-BoW` | Tầng 2 — Cây trung bình (175–380 nút) | 497 | 0.85111 | 0.84954 |
| Twitter15 | `MLP-BoW` | Tầng 3 — Cây dày/lớn (> 380 nút) | 494 | 0.81579 | 0.80041 |
| Twitter15 | `GCN-Single` | Tầng 1 — Cây thưa/nhỏ (≤ 174 nút) | 499 | 0.84569 | 0.63729 |
| Twitter15 | `GCN-Single` | Tầng 2 — Cây trung bình (175–380 nút) | 497 | 0.79276 | 0.79139 |
| Twitter15 | `GCN-Single` | Tầng 3 — Cây dày/lớn (> 380 nút) | 494 | 0.72672 | 0.65670 |
| Twitter15 | `GraphSAGE-Single` | Tầng 1 — Cây thưa/nhỏ (≤ 174 nút) | 499 | 0.85772 | 0.65149 |
| Twitter15 | `GraphSAGE-Single` | Tầng 2 — Cây trung bình (175–380 nút) | 497 | 0.83501 | 0.83265 |
| Twitter15 | `GraphSAGE-Single` | Tầng 3 — Cây dày/lớn (> 380 nút) | 494 | 0.83401 | 0.81059 |
| Twitter15 | `GAT-Single` | Tầng 1 — Cây thưa/nhỏ (≤ 174 nút) | 499 | 0.81964 | 0.63804 |
| Twitter15 | `GAT-Single` | Tầng 2 — Cây trung bình (175–380 nút) | 497 | 0.80483 | 0.80236 |
| Twitter15 | `GAT-Single` | Tầng 3 — Cây dày/lớn (> 380 nút) | 494 | 0.82996 | 0.80623 |
| Twitter16 | `TF-IDF-Ridge` | Tầng 1 — Cây thưa/nhỏ (≤ 183 nút) | 274 | 0.88321 | 0.69031 |
| Twitter16 | `TF-IDF-Ridge` | Tầng 2 — Cây trung bình (184–378 nút) | 273 | 0.86081 | 0.86213 |
| Twitter16 | `TF-IDF-Ridge` | Tầng 3 — Cây dày/lớn (> 378 nút) | 271 | 0.85240 | 0.84977 |
| Twitter16 | `TF-IDF-LR` | Tầng 1 — Cây thưa/nhỏ (≤ 183 nút) | 274 | 0.84672 | 0.67246 |
| Twitter16 | `TF-IDF-LR` | Tầng 2 — Cây trung bình (184–378 nút) | 273 | 0.84249 | 0.84552 |
| Twitter16 | `TF-IDF-LR` | Tầng 3 — Cây dày/lớn (> 378 nút) | 271 | 0.85609 | 0.85323 |
| Twitter16 | `TF-IDF-SVC` | Tầng 1 — Cây thưa/nhỏ (≤ 183 nút) | 274 | 0.90876 | 0.69772 |
| Twitter16 | `TF-IDF-SVC` | Tầng 2 — Cây trung bình (184–378 nút) | 273 | 0.82418 | 0.82471 |
| Twitter16 | `TF-IDF-SVC` | Tầng 3 — Cây dày/lớn (> 378 nút) | 271 | 0.80443 | 0.80367 |
| Twitter16 | `MLP-BoW` | Tầng 1 — Cây thưa/nhỏ (≤ 183 nút) | 274 | 0.91241 | 0.70411 |
| Twitter16 | `MLP-BoW` | Tầng 2 — Cây trung bình (184–378 nút) | 273 | 0.84249 | 0.84835 |
| Twitter16 | `MLP-BoW` | Tầng 3 — Cây dày/lớn (> 378 nút) | 271 | 0.82288 | 0.82185 |
| Twitter16 | `GCN-Single` | Tầng 1 — Cây thưa/nhỏ (≤ 183 nút) | 274 | 0.71898 | 0.55162 |
| Twitter16 | `GCN-Single` | Tầng 2 — Cây trung bình (184–378 nút) | 273 | 0.66667 | 0.66170 |
| Twitter16 | `GCN-Single` | Tầng 3 — Cây dày/lớn (> 378 nút) | 271 | 0.59779 | 0.51672 |
| Twitter16 | `GraphSAGE-Single` | Tầng 1 — Cây thưa/nhỏ (≤ 183 nút) | 274 | 0.92336 | 0.70082 |
| Twitter16 | `GraphSAGE-Single` | Tầng 2 — Cây trung bình (184–378 nút) | 273 | 0.87179 | 0.87194 |
| Twitter16 | `GraphSAGE-Single` | Tầng 3 — Cây dày/lớn (> 378 nút) | 271 | 0.85978 | 0.86017 |
| Twitter16 | `GAT-Single` | Tầng 1 — Cây thưa/nhỏ (≤ 183 nút) | 274 | 0.89781 | 0.69863 |
| Twitter16 | `GAT-Single` | Tầng 2 — Cây trung bình (184–378 nút) | 273 | 0.86081 | 0.86666 |
| Twitter16 | `GAT-Single` | Tầng 3 — Cây dày/lớn (> 378 nút) | 271 | 0.86716 | 0.86103 |
