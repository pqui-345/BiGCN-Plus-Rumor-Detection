# Hướng dẫn Làm việc Nhóm Trực tiếp trên GitHub (3 Thành viên)

Kho lưu trữ chính thức: `https://github.com/pqui-345/BiGCN-Plus-Rumor-Detection.git`

---

## 1. Thiết lập lần đầu cho Thành viên (Khang - TV1, Phúc - TV2, Quí - TV3)

1. **Chấp nhận lời mời Collaborator:** Trưởng nhóm (Quí) vào `Settings -> Collaborators` trên repo GitHub để mời tài khoản GitHub của Khang và Phúc (`Write` access). Khang và Phúc mở email để bấm **Accept invitation**.
2. **Tạo GitHub Personal Access Token (PAT)** (dùng để đẩy kết quả từ Google Colab lên thẳng GitHub):
   - Vào GitHub $\rightarrow$ góc trên phải chọn ảnh đại diện $\rightarrow$ **Settings** $\rightarrow$ **Developer settings** $\rightarrow$ **Personal access tokens** $\rightarrow$ **Tokens (classic)**.
   - Bấm **Generate new token (classic)** $\rightarrow$ đặt tên (vd: `colab-ct255`), tích chọn quyền **`repo`**, bấm **Generate token** và copy cất mã token (`ghp_...`).

---

## 2. Chạy Colab T4 GPU và Push thẳng kết quả lên GitHub

1. Mở trang chủ GitHub của nhóm $\rightarrow$ bấm nút **`Open In Colab`** (hoặc mở tệp `scripts/CT255_RumorDetection_BiGCN_Plus_MultiSeed_5x5_Colab_T4GPU.ipynb`).
2. Trên Colab, chọn **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**.
3. Đảm bảo thư mục dữ liệu `rumor_detection_acl2017` (chứa `twitter15/`, `twitter16/`) nằm trên Google Drive cá nhân (tại `MyDrive/rumor_detection_acl2017` hoặc `MyDrive/NghiepVuTM/dataset/rumdetect2017/rumor_detection_acl2017`).
4. Chạy **Cell 1** (clone/pull code từ GitHub + nạp dữ liệu từ Drive) và **Cell 2** (nạp hàm chạy tự động bỏ qua model đã xong).
5. Chạy **đúng Cell được phân công** dưới đây:
   - **Khang (TV1):** Chạy **Cell 3** (`BiGCN-Base+`, `EBGCN`, `RAGCL` — 6 thư mục, ~12 phút).
   - **Phúc (TV2):** Chạy **Cell 4** (`BiGCN-Base`, `BiGCN-NoRoot`, `BiGCN-NoDropEdge`, `BiGCN-TDOnly`, `BiGCN-BUOnly` — 10 thư mục, ~15 phút).
   - **Quí (TV3):** Chạy **Cell 5** (`BiGCN-Plus-UserID`, `BiGCN-Plus-Clean`, `BiGCN-Plus-B1Only`, `BiGCN-Plus-B12`; riêng `B2Only` tự động skip — ~15 phút).
6. Chạy **Cell 6** trên Colab:
   - Nhập GitHub Username, Email, dán PAT (`ghp_...`) và ghi Commit Message (ví dụ `feat(tv1): update EBGCN RAGCL BiGCN-Base+ 25 runs`).
   - Cell 6 tự động chạy `scripts/aggregate_results.py`, `git pull --rebase` để ghép với phần của thành viên khác và `git push` thẳng lên nhánh `main` của GitHub.

---

## 3. Làm việc với Mã nguồn & Báo cáo LaTeX trên máy cá nhân

- Kéo thay đổi mới nhất của đồng đội trước khi sửa:
  ```bash
  git pull --rebase origin main
  ```
- Sửa xong code (`src/`, `scripts/`) hoặc báo cáo (`report/main.tex`, `report/main.pdf`) thì kiểm tra và đẩy lên:
  ```bash
  git status
  git add .
  git commit -m "mo ta ngan gon thay doi"
  git push origin main
  ```
