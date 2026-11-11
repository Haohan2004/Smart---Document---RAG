import re

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.reranker import create_reranker


def tokenize(text):
    """Chuyển chữ thường và bỏ dấu câu, giữ nguyên dấu tiếng Việt."""
    return re.findall(r"\w+", text.casefold()) or [text.casefold()]


def tag_chunks(documents):
    """Thẻ tạm giúp Ensemble phân biệt hai đoạn cùng chữ nhưng khác file."""
    return [
        Document(page_content=doc.page_content, metadata={**doc.metadata, "_retrieval_id": doc.id})
        for doc in documents
    ]


def create_retriever(vectorstore, k=3, use_hybrid=True, use_reranker=True, candidate_k=5, selected_source=None):
    """Lọc theo tên file trước khi tìm; None hoặc 'Tất cả tài liệu' tìm toàn bộ."""
    if k <= 0 or (use_reranker and candidate_k < k):
        raise ValueError("k phải lớn hơn 0; candidate_k phải >= k khi bật reranker.")

    if vectorstore is None:
        return RunnableLambda(lambda question: [])
    documents = [
        vectorstore.docstore.search(doc_id)
        for doc_id in vectorstore.index_to_docstore_id.values()
    ]
    search_kwargs = {"k": candidate_k if use_reranker else k}
    if selected_source not in (None, "Tất cả tài liệu"):
        documents = [doc for doc in documents if doc.metadata.get("file_name") == selected_source]
        search_kwargs.update(filter={"file_name": selected_source}, fetch_k=vectorstore.index.ntotal)
    if not documents:
        return RunnableLambda(lambda question: [])

    count = candidate_k if use_reranker else k
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
    if use_hybrid:
        bm25 = BM25Retriever.from_documents(tag_chunks(documents), k=count, preprocess_func=tokenize)
        retriever = EnsembleRetriever(
            retrievers=[retriever | RunnableLambda(tag_chunks), bm25],
            weights=[0.5, 0.5], id_key="_retrieval_id",
        )
        # Trả Document gốc để thẻ tạm không đi vào metadata của người dùng.
        retriever = retriever | RunnableLambda(
            lambda docs: [vectorstore.docstore.search(doc.metadata["_retrieval_id"]) for doc in docs]
        )

    # Ensemble có thể trả nhiều đoạn hơn k, nên giới hạn trước khi chấm lại.
    retriever = retriever | RunnableLambda(lambda docs: docs[:count])
    if use_reranker:
        return create_reranker(retriever, k)
    return retriever
