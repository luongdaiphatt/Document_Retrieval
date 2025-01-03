# Document Retrieval
### Ghi chú:
* Folder pdf chứa file báo cáo
* Folder data chứa dữ liệu đã được tiền xử lý
* Folder web_crawler chứa code crawler dữ liệu và dữ liệu đã crawl
* Folder sbert_model lưu model SBERT đã được huấn luyện (tạo sau khi tự động tải về)
* Folder phobert_model chứa model PhoBERT đã được huấn luyện (tạo sau khi tự động tải về)
* Folder BM25, BM25+PhoBERT, TF-IDF, TF-IDF+Vietnamese-SBERT chứa code (app.py) chạy model tương ứng
* Folder all_model_nDCG chứa code chạy đánh giá nDCG của tất cả các model
* Folder nDCG chứa điểm đánh giá nDCG của các model
* Folder models và VnCoreNLP-1.2.jar được dùng để segment text (hàm segment_text)
* Folder TF_IDF_old chứa code TF-IDF cũ (không sử dụng VnCoreNLP)

### Đánh giá các model sơ bộ:
1. TF-IDF + Vietnamese SBERT (cho kết quả tốt nhất)
2. BM25 + PhoBERT
3. BM25
4. TF-IDF

### Cách chạy:
* Khuyến khích chạy flask app trong môi trường ảo (virtual environment)

#### 1. Cài đặt các thư viện cần thiết:
```
pip install -r requirements.txt
```
#### 2. Chạy app.py (trong thư mục code model muốn chạy):
```
python path/to/app.py
```
* Truy cập localhost:5000 để sử dụng app.

#### 3 (Optional). Chạy đánh giá nDCG (trong thư mục muốn chạy):
```
python path/to/all_model_nDCG/nDCG.py
```
* Truy cập localhost:5001 để đánh giá.
* Điểm đánh giá nDCG được lưu trong nDCG/scores.json