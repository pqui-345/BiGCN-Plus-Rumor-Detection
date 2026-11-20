# Hướng dẫn Dữ liệu (`rumdetect2017`)

Repository không đính kèm dữ liệu thô để giữ dung lượng gọn nhẹ khi nộp bài hoặc public lên GitHub.

## 1. Cấu trúc thư mục dữ liệu chuẩn
Đặt bộ dữ liệu `rumor_detection_acl2017` (Ma et al., ACL 2017) vào một trong các đường dẫn sau (script tự động dò tìm):
- Local: `data/rumdetect2017/rumor_detection_acl2017/{twitter15, twitter16}`
- Google Drive (Colab): `/content/drive/MyDrive/NghiepVuTM/dataset/rumdetect2017/rumor_detection_acl2017/{twitter15, twitter16}`
- Hoặc truyền trực tiếp qua tham số `--data_root <duong_dan>`

Mỗi tập con (`twitter15/`, `twitter16/`) gồm:
- `label.txt`: nhãn 4 lớp (`non-rumor`, `false`, `true`, `unverified`)
- `source_tweets.txt`: nội dung văn bản bài đăng gốc
- `tree/*.txt`: cấu trúc cây lan truyền `(parent -> child)` kèm độ trễ thời gian (phút) và mã người dùng ẩn danh.

## 2. Lưu ý khoa học quan trọng
Bộ công bố công khai `rumdetect2017` chỉ chứa văn bản của **bài đăng gốc** và **không chứa nội dung tweet phản hồi** (do điều khoản API Twitter). Do đó mọi đối sánh với các công trình quốc tế cần tách rõ giữa nhóm dùng tập công khai (`Source-only + Tree`) và nhóm cào đầy đủ text phản hồi qua API cũ (`Full-replies`).
