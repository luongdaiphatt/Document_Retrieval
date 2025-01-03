### Ghi chú:
* Folder models và VnCoreNLP-1.2.jar được dùng để segment text (hàm segment_text)
* Folder phobert_model sẽ xuất hiện sau khi chạy app.py (để lưu phobert model)
* Folder sbert_model sẽ xuất hiện sau khi chạy app.py (để lưu sbert model)
* Một số datafile sẽ thay đổi/xuất hiện bên trong folder data để lưu các data đã xử lý, tiết kiệm thời gian khi chạy lần sau. Xóa các file tương ứng nếu muốn preprocess lại data.

### Đánh giá các model sơ bộ:
1. TF-IDF + Vietnamese SBERT (cho kết quả tốt nhất)
2. BM25 + PhoBERT
3. BM25
4. TF-IDF

### Cách chạy:
* Lưu ý: Cần cài đặt Java để sử dụng VnCoreNLP, và chạy ngay trong thư mục TF-IDF+Vietnamese-SBERT (để không bị lỗi về đường dẫn)
1. Cài đặt các thư viện cần thiết:
```
pip install -r requirements.txt
```
2. Chạy app.py:
```
python app.py
```

### Đã làm:
* Hiển thị giao diện web cho việc nhập câu hỏi và chọn một trong 4 loại model để dự đoán
* Tất cả các model đều được lưu lại để sử dụng lại mà không cần train lại
* Có thể chọn model để dự đoán kết quả (xem code trong app.py trong route '/search')

### Cần làm:
* API để trả về kết quả dự đoán của loại model đã chọn (tf-idf, tf-idf + sbert, bm25, bm25 + phobert)
* So sánh kết quả giữa các model