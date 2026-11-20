# Báo cáo Khoa học (`report/`)

Thư mục này lưu trọn bộ mã nguồn LaTeX của báo cáo đồ án (đồng bộ trực tiếp qua GitHub cho cả 3 thành viên và hỗ trợ kết nối Overleaf).

## Biên dịch trên máy cục bộ
```bash
cd report
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Quy trình cập nhật báo cáo cùng nhóm trên GitHub
1. Luôn kéo bản mới nhất trước khi sửa: `git pull --rebase origin main`
2. Chép bảng/biểu đồ mới từ `../results/tables/` hoặc `../results/figures/` vào đây và cập nhật `main.tex`.
3. Biên dịch kiểm tra `main.pdf` không lỗi (các file phụ như `.aux`, `.log`, `.bbl`, `.toc` tự động bị `.gitignore` bỏ qua).
4. Đẩy lên GitHub:
   ```bash
   git add report/main.tex report/ref.bib report/main.pdf report/figures/
   git commit -m "docs(report): cap nhat ket qua va phan tich Chuong ..."
   git push origin main
   ```
