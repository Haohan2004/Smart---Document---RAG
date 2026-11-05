
import streamlit as st

from vectorstore import create_vectorstore
from rag_chain import create_rag_chain
from loader import load_and_chunk_document
from embedder import get_embedder

@st.cache_resource
FILE_SIZE = 100
# Tên file lấy từ giao diện streamlit, khi up có tempPath được xử lý trong loader (Danh) nên bên này kêu lại thôi
def initialize_rag():
    documents = load_and_chunk_document("Tên file", FILE_SIZE)
    embedder = get_embedder()

    vectorstore = create_vectorstore(
        documents, embedder
    )

    rag_chain = create_rag_chain(vectorstore)

    return rag_chain

# Viết file test.py để kiểm thử mà không có frontend, đảm bảo các components được bao gồm trong initialize_rag được khởi tạo, gọi rag_chain.invoke("Câu hỏi") lấy kết quả cuối