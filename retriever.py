"""Tạo bộ tìm các đoạn tài liệu liên quan đến câu hỏi."""


def create_retriever(vectorstore):
    """Lấy tối đa 3 đoạn gần với câu hỏi nhất, theo mục 3.3.4 của assignment."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
