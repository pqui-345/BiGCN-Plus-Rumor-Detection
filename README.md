# CT255 — Multi-Seed Rumor Detection Benchmark & BiGCN-Plus (`Twitter15` & `Twitter16`)

Kho mã nguồn chuẩn hóa cho đồ án **Phát hiện tin đồn trên mạng xã hội dựa trên đồ thị lan truyền hai chiều (`BiGCN` & `BiGCN-Plus`)**, thiết kế sạch để công bố công khai (GitHub) hoặc nộp kèm báo cáo khoa học.

---

## 1. Cấu trúc thư mục

```text
CT255_RumorDetection_BiGCN_Plus/
├── README.md                        # Hướng dẫn sử dụng & kiến trúc dự án
├── requirements.txt                 # Danh sách thư viện chuẩn
├── .gitignore                       # Loại trừ dữ liệu nặng / cache
├── configs/
│   └── default.yaml                 # Siêu tham số chuẩn (5 seeds × 5 folds = 25 lượt)
├── data/
│   └── README.md                    # Hướng dẫn đặt bộ dữ liệu rumdetect2017 (Twitter15/16)
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Đọc Twitter15/16, 8 đặc trưng cấu trúc, tách CV zero-leakage
│   ├── metrics.py                   # 11 chỉ số đánh giá (8 nhóm) & Wilcoxon / Paired t-test
│   ├── baselines_nlp.py             # TF-IDF-Ridge, TF-IDF-LR, TF-IDF-SVC, Text-PLM (RoBERTa)
│   ├── baselines_gnn.py             # MLP-BoW, GCN, GraphSAGE, GAT, BiGCN(+ablation), EBGCN, RAGCL
│   └── bigcn_plus.py                # BiGCN-Plus-Clean (không User-ID), UserID ablation, B1/B2/PLM
├── scripts/
│   ├── run_benchmark.py             # Runner CLI hợp nhất mọi mô hình, mọi seed, lưu song song Last & Snap5
│   ├── aggregate_results.py         # Tổng hợp bảng Mean±Std, kiểm định p-value, bảng LaTeX & biểu đồ 300 DPI
│   └── run_colab_all.ipynb          # Notebook 1-click chạy toàn bộ trên Google Colab T4 GPU
├── report/
│   └── README.md                    # Liên kết tới báo cáo LaTeX (BaoCao_CT255_NghiepVuThongMinh)
└── results/
    ├── multi_seed/                  # Kết quả chi tiết từng (Dataset, Model)
    ├── tables/                      # master_multiseed_summary.md/.csv, statistical_tests.md, latex_tables.tex
    └── figures/                     # Biểu đồ multiseed_comparison.png, external_positioning.png
```

---

## 2. Những điểm khắc phục chuẩn hóa so với phiên bản 1-seed ban đầu

1. **Giao thức đa lượt chạy + kiểm định thống kê:** Chạy mặc định `5 seeds × 5 folds = 25 lượt/mô hình` (`--seeds 42 43 44 45 46`, hỗ trợ mở rộng `10 seeds = 50 lượt`), tính cả `Std trên toàn bộ lượt` lẫn `Seed-Std`, và kiểm định `Wilcoxon signed-rank` & `Paired t-test`.
2. **Loại bỏ rò rỉ / học thuộc User-ID (`BiGCN-Plus-Clean`):**
   - Trong `rumdetect2017` mở, **không dùng** mã định danh tài khoản (`ROOT_U_`, `U_`, `P_`) trong mô hình đề xuất chính `BiGCN-Plus-Clean` (chỉ dùng text bài gốc `Word (1,2) + Char (3,4) TF-IDF` kết hợp `10` đặc trưng thống kê cấu trúc/thời gian đồ thị + `BiGCN` + `DualViewSAGE`).
   - Phiên bản gốc có `User-ID` (`BiGCN-Plus-UserID`) được giữ riêng để làm đối chứng ablation trung thực.
3. **Đo đúng 8 chiều đặc trưng cấu trúc ($\mathbf{s}_v^+ \in \mathbb{R}^8$):**
   `[is_root, log1p(depth)/10, log1p(delay)/10, log1p(degree)/10, text_len_norm, is_leaf, log1p(subtree)/log1p(n), log1p(parent_dt)/10]`.
4. **Đối sánh công bằng & đầy đủ Baseline:**
   - **NLP thuần văn bản:** `TF-IDF-Ridge`, `TF-IDF-LR`, `TF-IDF-SVC`, `Text-PLM` (`roberta-base`).
   - **GNN Cấp 1–2:** `MLP-BoW`, `GCN-Single`, `GraphSAGE-Single`, `GAT-Single`, `BiGCN-Base` (+ 4 ablations), **`BiGCN-Base+`** (cấp đủ 8-dim + Sublinear TF-IDF + Snapshot-5), **`EBGCN`** (ACL 2021), **`RAGCL`** (AAAI 2024).
   - **Đánh giá song song cả `last` (epoch 100) lẫn `snapshot5` (trung bình 5 epoch cuối)** trong cùng một lượt chạy.

---

## 3. Hướng dẫn chạy nhanh

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Chạy cục bộ (CPU) cho các Baseline NLP (5 seeds × 5 folds = 25 lượt)
```bash
python scripts/run_benchmark.py --dataset ALL --models TF-IDF-Ridge TF-IDF-LR TF-IDF-SVC --seeds 42 43 44 45 46
python scripts/aggregate_results.py
```

### Chạy toàn bộ mô hình GNN & BiGCN-Plus trên GPU (hoặc Google Colab T4)
Mở `scripts/run_colab_all.ipynb` trên Google Colab (chọn GPU T4) hoặc chạy lệnh CLI:
```bash
python scripts/run_benchmark.py --dataset ALL --suite core --seeds 42 43 44 45 46 --epochs 100 --snapshot_k 5
python scripts/aggregate_results.py
```
