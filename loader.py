import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from langchain_community.document_loaders import (
    PDFPlumberLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def load_and_chunk_document(
    filename: str,
    content: bytes,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
):
    """
    Nhận file PDF/DOCX (tên file + bytes nội dung), kiểm tra tính hợp lệ,
    trích xuất nội dung, gắn metadata và cắt thành các chunks.

    Trả về: danh sách các chunks (list[Document]).
    """
    # 1. Kiểm tra tham số chunking
    if chunk_size <= 0:
        raise ValueError("chunk_size phải lớn hơn 0.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap phải >= 0 và nhỏ hơn chunk_size.")

    # 2. Kiểm tra tính hợp lệ của file
    if not filename:
        raise ValueError("Vui lòng chọn một file.")

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Chỉ hỗ trợ file PDF hoặc DOCX.")

    if not content:
        raise ValueError("File đang trống.")

    if len(content) > MAX_FILE_SIZE:
        raise ValueError("Dung lượng file không được vượt quá 50 MB.")

    if extension == ".pdf" and not content.startswith(b"%PDF-"):
        raise ValueError("File không có định dạng PDF hợp lệ.")

    if extension == ".docx" and not content.startswith(b"PK"):
        raise ValueError("File không có định dạng DOCX hợp lệ.")

    # 3. Trích xuất văn bản qua file tạm
    temp_path = None
    try:
        with NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        if extension == ".pdf":
            loader = PDFPlumberLoader(temp_path)
        else:
            loader = Docx2txtLoader(temp_path)

        documents = loader.load()

        documents = [
            doc
            for doc in documents
            if doc.page_content.strip()
        ]

        if not documents:
            raise ValueError(
                "Không trích xuất được văn bản từ file. "
                "Tài liệu có thể rỗng hoặc không chứa văn bản đọc được."
            )

        file_name = Path(filename).name
        uploaded_at = datetime.now(timezone.utc).isoformat()

        for doc in documents:
            doc.metadata["file_type"] = extension.lstrip(".")
            doc.metadata["file_name"] = file_name
            doc.metadata["uploaded_at"] = uploaded_at

        # 4. Cắt thành các chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index

        return chunks

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
