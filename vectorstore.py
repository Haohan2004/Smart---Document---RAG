"""Tạo kho vector FAISS từ các đoạn tài liệu đã chia nhỏ."""

from langchain_community.vectorstores import FAISS


def create_vectorstore(documents, embedder):
    """Nhận danh sách Document và model embedding, trả về kho FAISS trong RAM."""
    chunks = []
    for document in documents:
        if document.page_content.strip():
            chunks.append(document)

    if not chunks:
        raise ValueError("Không có nội dung để tạo vector store.")

    # FAISS tự gọi embedder để tạo vector, đồng thời giữ nội dung và metadata.
    return FAISS.from_documents(chunks, embedder)
