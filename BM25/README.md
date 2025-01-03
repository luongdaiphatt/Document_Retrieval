### Ghi chú:
* Một số datafile sẽ thay đổi/xuất hiện bên trong folder data để lưu các data đã xử lý, tiết kiệm thời gian khi chạy lần sau. Xóa các file tương ứng nếu muốn preprocess lại data.

### Đánh giá các model sơ bộ:
1. TF-IDF + Vietnamese SBERT (cho kết quả tốt nhất)
2. BM25 + PhoBERT
3. BM25
4. TF-IDF

### Cách chạy:
* Lưu ý: Chạy ngay trong thư mục BM25 (để không bị lỗi về đường dẫn)
1. Cài đặt các thư viện cần thiết:
```
pip install -r requirements.txt
```
2. Chạy app.py:
```
python app.py
```