from langchain_huggingface import HuggingFaceEmbeddings


def get_embedder():
    """Tạo model biến văn bản thành vector 768 chiều; gọi một lần khi khởi tạo app."""
    embedder= HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embedder
