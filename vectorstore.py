"""Mục 3.3.4: FAISS lưu vector cùng nội dung và metadata của Document."""

import logging
from time import perf_counter

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


logger = logging.getLogger(__name__)


def _warn_about_long_chunks(chunks: list[Document], embedder: Embeddings) -> None:
    """Cảnh báo giới hạn model, không tự sửa chunk/metadata của loader."""
    client = getattr(embedder, "_client", None)
    if client is None:
        client = getattr(embedder, "client", None)
    tokenizer = getattr(client, "tokenizer", None)
    token_limit = getattr(client, "max_seq_length", None)
    if not callable(tokenizer) or not isinstance(token_limit, int):
        return
    lengths = tokenizer(
        [chunk.page_content.replace("\n", " ") for chunk in chunks],
        truncation=False, padding=False, return_length=True, verbose=False,
    )["length"]
    too_long = sum(length > token_limit for length in lengths)
    if too_long:
        logger.warning(
            "index.chunks.too_long | count=%d/%d | token_limit=%d | "
            "Model sẽ cắt phần vượt giới hạn; cần thống nhất chunking với loader.",
            too_long, len(chunks), token_limit,
        )


def create_vectorstore(documents: list[Document], embedder: Embeddings) -> FAISS:
    """Tạo FAISS trong RAM từ chunk của Danh, không tự nạp model/chia đoạn.

    Giữ nguyên văn bản và metadata. Bỏ chunk chỉ có khoảng trắng, kèm
    WARNING; báo lỗi khi không còn nội dung hoặc dữ liệu sai kiểu.
    Trả FAISS chuẩn để Hào có thể tích hợp persistence về sau.
    """
    if not isinstance(documents, list):
        logger.error("index.input.invalid | documents phải là list[Document]")
        raise TypeError("documents phải là danh sách Document từ loader.")
    chunks = []
    for position, document in enumerate(documents):
        if not isinstance(document, Document):
            logger.error("index.input.invalid | position=%d | expected=Document", position)
            raise TypeError(f"Phần tử ở vị trí {position} phải là Document.")
        if document.page_content.strip():
            chunks.append(document)
    logger.info("index.input | received=%d | valid=%d", len(documents), len(chunks))
    if len(chunks) < len(documents):
        logger.warning("index.blank_chunks.skipped | count=%d", len(documents) - len(chunks))
    if not chunks:
        logger.error("index.input.empty | không có đoạn có nội dung")
        raise ValueError("Không có đoạn có nội dung để tạo FAISS.")
    if not isinstance(embedder, Embeddings):
        logger.error("index.embedder.invalid | expected=Embeddings")
        raise TypeError("embedder phải là đối tượng Embeddings của LangChain.")

    started = perf_counter()
    logger.info("index.start | chunks=%d | steps=embedding,indexing", len(chunks))
    try:
        _warn_about_long_chunks(chunks, embedder)
        # Lõi mã mục 3.3.4: FAISS tự gọi embed_documents, không nhúng hai lần.
        store = FAISS.from_documents(chunks, embedder)
    except Exception:
        logger.exception("index.failed | chunks=%d", len(chunks))
        raise
    logger.info(
        "index.done | vectors=%d | dimension=%d | elapsed_s=%.3f",
        store.index.ntotal, store.index.d, perf_counter() - started,
    )
    return store
