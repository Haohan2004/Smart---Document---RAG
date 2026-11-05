"""Tạo kho vector FAISS từ các đoạn tài liệu đã chia nhỏ."""

import logging

from langchain.vectorstores import faiss
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

def get_uploaded_vectorstore(vectorstore):
    if vectorstore is None:
        return []
    try:
        sources = set()
        for doc_id in vectorstore.index_to_docstore_id.values():
            doc = vectorstore.docstore.search(doc_id)
            # Trong loader.py ông lưu là "file_name" thay vì "source"
            src = doc.metadata.get("file_name", "")
            if src:
                sources.add(src)
        return sorted(sources)
    except Exception:
        return []