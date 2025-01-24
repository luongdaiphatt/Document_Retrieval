![UIT](https://img.shields.io/badge/from-UIT%20VNUHCM-blue?style=for-the-badge&link=https%3A%2F%2Fwww.uit.edu.vn%2F)
<h2 align="center"> Document Retrieval For Vietnamese News </h2>

<p align="center">
  <img src="https://en.uit.edu.vn/sites/vi/files/banner_en.png" alt="Alt text">
</p>

<h3>Contributors</h3>

-Member: Lường Đại Phát - 21522443
-Member: Hồ Trung Kiên - 22520704

<h3>Description</h3>
Nghiên cứu này đề xuất hệ thống truy vấn bài báo tiếng Việt, kết hợp TF-IDF với SBERT- Vietnamese và BM25 với PhoBERT để nâng
cao hiệu quả tìm kiếm. TF-IDF và BM25 đảm bảo xử lý từ khóa nhanh, trong khi SBERT và PhoBERT khai thác ngữ nghĩa sâu hơn. Kết quả thực nghiệm trên tập dữ liệu bài báo lớn cho thấy mô hình kết hợp đạt độ chính xác và hiệu suất vượt trội, phù hợp cho các hệ thống tìm kiếm tiếng Việt thực tế.


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

<h3>How to use</h3>
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

<h3>Additional information</h3>
Cảm ơn bạn đã sử dụng ứng dụng. Chúng tôi rất vui khi biết rằng bạn hài lòng với sản phẩm của chúng tôi. Chúng tôi luôn nỗ lực cải thiện ứng dụng của mình và phản hồi của bạn rất quan trọng đối với chúng tôi.

<h3>License</h3>

Copyright © 2024 [luongdaiphat, KienHoSD](https://github.com/luongdaiphatt/Document_Retrieval)

This project is [MIT](https://github.com/luongdaiphatt/Document_Retrieval/blob/main/LICENSE) licensed
