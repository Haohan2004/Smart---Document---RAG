# Embedding, Indexing và Retriever

- `embedder.py`: tạo model multilingual MPNet, CPU, vector 768 chiều được chuẩn hóa.
- `vectorstore.py`: nhận các chunk từ loader của Danh, tạo FAISS và giữ metadata.
- `retriever.py`: tìm tối đa 3 chunk liên quan đến câu hỏi bằng similarity search.

Lần đầu cần mạng để tải model. Model giới hạn 128 token/đoạn.
