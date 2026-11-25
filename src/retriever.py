"""Tạo bộ tìm các đoạn tài liệu liên quan đến câu hỏi."""
import re

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableLambda

def tokenize(text):
    return re.findall(r"\w+", text.casefold())


def create_retriever(vectorstore,k,selected_source=None,mode="faiss",faiss_weight=0.7):
    if vectorstore is None:
        return None
        # Metadata Filtering
    search_kwargs = {"k": k}
    if selected_source not in (None, "Tất cả tài liệu"):
        search_kwargs.update(
            filter={"file_name": selected_source},
            fetch_k=vectorstore.index.ntotal
        )

    """Lấy tối đa 3 đoạn gần với câu hỏi nhất, theo mục 3.3.4 của assignment."""
    faiss_retriever= vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
    # 3. Nếu chỉ sử dụng Semantic Search
    if mode == "faiss":
        return faiss_retriever
    # 4. Lấy toàn bộ chunks để xây dựng BM25
    documents = [
        vectorstore.docstore.search(doc_id)
        for doc_id in vectorstore.index_to_docstore_id.values()
    ]
    # 5. Metadata Filtering cho BM25
    if selected_source not in (None, "Tất cả tài liệu"):
        documents = [
            doc for doc in documents
            if doc.metadata.get("file_name") == selected_source
        ]
    if not documents:
        return RunnableLambda(lambda question: [])
        # 6. Khởi tạo BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(
        documents,
        k=k,
        preprocess_func=tokenize
    )
    # 7. Nếu chỉ sử dụng Keyword Search
    if mode == "bm25":
        return bm25_retriever

        # 8. Hybrid Search
    if mode == "hybrid":

        if faiss_weight == 1:
            return faiss_retriever

        if faiss_weight == 0:
            return bm25_retriever

        hybrid_retriever = EnsembleRetriever(
            retrievers=[
                faiss_retriever,
                bm25_retriever
            ],
            weights=[
                faiss_weight,
                1 - faiss_weight
            ]
        )

        return hybrid_retriever | RunnableLambda(
            lambda docs: docs[:k]
        )

    raise ValueError("Retrieval mode không hợp lệ.")