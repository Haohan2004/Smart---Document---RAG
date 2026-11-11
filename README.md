# Embedding, Indexing và Retriever

- `src/embedder.py`: tạo model multilingual MPNet, CPU, vector 768 chiều được chuẩn hóa.
- `src/vectorstore.py`: nhận các chunk từ loader của Danh, tạo FAISS và giữ metadata.
- `src/retriever.py`: kết hợp FAISS và BM25 bằng EnsembleRetriever (mỗi bên 50%), lọc theo `file_name` trước khi tìm.
- `src/reranker.py`: Cross-Encoder đa ngôn ngữ chấm lại 5 ứng viên, giữ tối đa 3 đoạn.

Gọi `create_retriever(vectorstore)` để dùng hybrid + reranker. Model reranker chỉ nạp một lần trong tiến trình. Lần đầu cần mạng tải MPNet và `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`; MPNet giới hạn 128 token/đoạn.

FAISS được dùng bản 1.13.2 trong `requirements.txt` để xử lý lỗi OpenMP gặp trên máy Windows hiện tại.

Đã ghép phần upload nhiều file và bảng metadata từ nhánh Danh (`be4b917`). Gọi `create_retriever(vectorstore, selected_source="A.pdf")` để chỉ tìm trong A.pdf; `None` hoặc `"Tất cả tài liệu"` để tìm toàn bộ. Không có file khớp thì trả danh sách rỗng, không tìm sang file khác.

Nguồn hiển thị dưới từng câu trả lời lấy từ các đoạn đã đưa vào context và được lưu cùng tin nhắn. Đổi phạm vi tìm kiếm không làm đổi nguồn của câu trả lời cũ. Hai đoạn cùng chữ nhưng thuộc hai file được giữ riêng.
