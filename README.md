# Embedding và Indexing

Cài thư viện (Python 3.12):

```bash
pip install langchain==0.3.16 langchain-community==0.3.16 langchain-core==0.3.33 langchain-huggingface==0.1.2 faiss-cpu==1.9.0 sentence-transformers==5.1.1 torch==2.8.0 transformers==4.57.1 numpy==1.26.4
```

- `embedder.py`: tạo model multilingual MPNet, CPU, vector 768 chiều được chuẩn hóa.
- `vectorstore.py`: nhận các chunk từ loader của Danh, tạo FAISS và giữ metadata. Chưa gồm retriever/UI.

Lần đầu cần mạng để tải model. Model giới hạn 128 token/đoạn; code cảnh báo đoạn quá dài để nhóm điều chỉnh chunking.
