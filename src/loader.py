import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Tuple, Union, Any
from langchain_core.documents import Document
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
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> List[Document]:
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
        file_type = extension.lstrip(".")
        uploaded_at = datetime.now(timezone.utc).isoformat()

        # Gắn metadata cho từng Document
        for doc in documents:
            doc.metadata["source"] = file_name
            doc.metadata.pop("file_path", None)
            doc.metadata["file_type"] = extension.lstrip(".")
            doc.metadata["file_name"] = file_name
            doc.metadata["uploaded_at"] = uploaded_at

            if extension == ".pdf":
                page_index = doc.metadata.get("page")
                if isinstance(page_index, int):
                    doc.metadata["page_label"] = str(page_index + 1)
            # DOCX không có thông tin phân trang cố định.
        # 4. Cắt thành các chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(documents)
        per_page_counts = {}
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["file_name"] = file_name
            chunk.metadata["file_type"] = file_type
            chunk.metadata["uploaded_at"] = uploaded_at
            chunk.metadata["source"] = file_name
            if extension == ".pdf" and isinstance(chunk.metadata.get("page"), int):
                page_index = chunk.metadata["page"]
                chunk.metadata["page_chunk_index"] = per_page_counts.get(page_index, 0)
                per_page_counts[page_index] = chunk.metadata["page_chunk_index"] + 1

        return chunks

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

def load_and_chunk_documents(
    files: List[Union[Tuple[str, bytes], Any]],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Load và chunk nhiều file cùng 1 lúc.

    Tham số:
      files: Danh sách các file. Hỗ trợ 2 định dạng:
        1. Tuple (filename, content_bytes)
        2. Streamlit UploadedFile object (có thuộc tính .name và .getvalue())
      chunk_size: Kích thước chunk (None = dùng mặc định)
      chunk_overlap: Độ gối đầu chunk (None = dùng mặc định)

    Gắn metadata cho từng chunk gồm:
      - file_name: Tên file
      - file_type: Loại file (pdf hoặc docx)
      - uploaded_at: Ngày giờ upload (ISO format)
      - chunk_index: Thứ tự chunk trong từng file

    Trả về: danh sách gộp tất cả chunks (List[Document]).
    """
    if not files:
        return []

    all_chunks = []

    for item in files:
        # Hỗ trợ cả Streamlit UploadedFile và tuple (filename, content)
        if isinstance(item, tuple) and len(item) == 2:
            fname, fbytes = item
        elif hasattr(item, "name") and hasattr(item, "getvalue"):
            fname = item.name
            fbytes = item.getvalue()
        elif hasattr(item, "name") and hasattr(item, "read"):
            fname = item.name
            fbytes = item.read()
        else:
            raise ValueError(
                "Định dạng phần tử trong 'files' không hợp lệ. "
                "Cần tuple (filename, bytes) hoặc đối tượng UploadedFile."
            )

        chunks = load_and_chunk_document(
            filename=fname,
            content=fbytes,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)

    return all_chunks