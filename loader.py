from pathlib import Path
from tempfile import NamedTemporaryFile
import os

from langchain_community.document_loaders import (
    PDFPlumberLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def validate_file(filename: str, content: bytes) -> None:
    """Kiểm tra tên file, định dạng, dung lượng và nội dung."""

    if not filename:
        raise ValueError("Vui lòng chọn một file.")

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Chỉ hỗ trợ file PDF hoặc DOCX.")

    if not content:
        raise ValueError("File đang trống.")

    if len(content) > MAX_FILE_SIZE:
        raise ValueError("Dung lượng file không được vượt quá 50 MB.")

    # Kiểm tra chữ ký cơ bản của file.
    if extension == ".pdf" and not content.startswith(b"%PDF-"):
        raise ValueError("File không có định dạng PDF hợp lệ.")

    # DOCX thực chất là một gói ZIP chứa các file XML.
    if extension == ".docx" and not content.startswith(b"PK"):
        raise ValueError("File không có định dạng DOCX hợp lệ.")


def extract_document_text(filename: str, content: bytes):
    """Trích xuất văn bản từ PDF hoặc DOCX."""

    validate_file(filename, content)

    extension = Path(filename).suffix.lower()
    temp_path = None

    try:
        with NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        if extension == ".pdf":
            loader = PDFPlumberLoader(temp_path)
        else:
            loader = Docx2txtLoader(temp_path)

        documents = loader.load()

        documents = [
            document
            for document in documents
            if document.page_content.strip()
        ]

        if not documents:
            raise ValueError(
                "Không trích xuất được văn bản từ file. "
                "Tài liệu có thể rỗng hoặc không chứa văn bản đọc được."
            )

        for document in documents:
            document.metadata["source"] = Path(filename).name
            document.metadata["file_type"] = extension.lstrip(".")

        return documents

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def chunk_documents(
    documents,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
):
    """Chia tài liệu thành các chunks và giữ metadata nguồn."""

    if chunk_size <= 0:
        raise ValueError("chunk_size phải lớn hơn 0.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap phải >= 0 và nhỏ hơn chunk_size."
        )

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


def process_document(
    filename: str,
    content: bytes,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
):
    """Pipeline chung: validate -> extract -> chunk."""

    documents = extract_document_text(filename, content)

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return documents, chunks
