"""Tạo kho vector FAISS từ các đoạn tài liệu đã chia nhỏ."""

import logging

from langchain_community.vectorstores import FAISS


logger = logging.getLogger(__name__)


def create_vectorstore(documents, embedder):
    """Nhận các chunks từ loader và model embedding, trả về kho FAISS trong RAM."""
    chunks = []
    for document in documents:
        if document.page_content.strip():
            chunks.append(document)

    if not chunks:
        raise ValueError("Không có nội dung để tạo vector store.")

    logger.info("Đang tạo FAISS từ %d chunk.", len(chunks))

    # Theo mục 3.3.4: FAISS tự tạo vector và giữ nội dung, metadata.
    vectorstore = FAISS.from_documents(chunks, embedder)

    logger.info("Đã tạo FAISS với %d vector.", vectorstore.index.ntotal)
    return vectorstore
