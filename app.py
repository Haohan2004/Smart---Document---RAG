import os
import tempfile

import streamlit as st
from datetime import datetime
import time

from src import retriever
from src.embedder import get_embedder
from src.llm import get_llm
from src.loader import load_and_chunk_document
from src.logger import get_logger
from src.rag_chain import create_rag_chain
from src.vectorstore import create_vectorstore, get_uploaded_vectorstore
from src.retriever import create_retriever
st.set_page_config(page_title="SmartDoc AI", layout="wide", initial_sidebar_state="expanded")

logger = get_logger()
# ── Load CSS từ file riêng ───────────────────────────────────────────────────
def load_css(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css("style.css")

@st.cache_resource
def load_embedder():
    return get_embedder()
@st.cache_resource
def load_llm():
    return get_llm()

embedder = load_embedder()

llm = load_llm()

# --- KHỞI TẠO STATE ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None  # nơi chứa FAISS index của tài liệu.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None


# 2 biến cờ hiệu để mang Spinner ra giữa màn hình
if "is_processing_file" not in st.session_state:
    st.session_state.is_processing_file = False
if "pending_file" not in st.session_state:
    st.session_state.pending_file = None
if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False

# --- SIDEBAR (THEO ĐÚNG ẢNH MẪU) ---
with st.sidebar:
    st.markdown("<h2 class='sidebar-title' style='text-align: center; color: white; margin-top: 0; padding-top: 0; margin-bottom: 20px; font-weight: 700; font-size: 26px;'>SmartDoc AI</h2>", unsafe_allow_html=True)
    
    # 4 Expander tính năng
    with st.expander("Hướng dẫn sử dụng"):
        st.write("1. Tải lên tài liệu PDF hoặc DOCX.\n2. Đặt câu hỏi trong khung chat.\n3. Nhận câu trả lời kèm trích dẫn chính xác.")
        
    with st.expander("Cấu hình mô hình"):
        st.info("LLM: Qwen2.5:7b\n\nEmbedding: Multilingual MPNet")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 50, 200, 100)
        
    with st.expander("Thiết lập & Tùy chọn"):
        st.checkbox("Bật chế độ Streaming", value=True)
        st.checkbox("Hiển thị Citations", value=True)
        
    with st.expander("Tài liệu đã đăng",expanded=True):
        uploaded_sources = get_uploaded_vectorstore(st.session_state.vectorstore)
        if uploaded_sources:
            st.markdown(
                f"<div style='background-color:#E6F0FF; color:#31333F; border-radius:8px; padding:8px 12px; font-size:0.875rem;'>"
                f"Đang index: <b>{len(uploaded_sources)}</b> tài liệu"
                f"</div>",
                unsafe_allow_html=True
            )

            for src in uploaded_sources:
                st.caption(f"• {src}")

            source_options = ["Tất cả tài liệu"] + uploaded_sources
            st.selectbox(
                "Tìm kiếm trong:",
                options=source_options,
                index=0,
                key="source_filter_select"
            )
        else:
            st.info("Chưa có tài liệu nào được upload.")
    st.markdown("<div style='margin-top: 25px; margin-bottom: 12px; text-align: center; font-weight: 600; color: white; font-size: 15px;'>Lịch sử cuộc trò chuyện</div>", unsafe_allow_html=True)

    # 2 Nút Đặt lại lịch sử & Hộp thoại mới
    if st.button("🔄 Đặt lại lịch sử", use_container_width=True, key="btn_reset_history"):
        st.session_state.messages = []
        st.rerun()

    if st.button("✨ Hộp thoại mới", use_container_width=True, key="btn_new_chat"):
        st.session_state.messages = []
        st.session_state.file_name = None
        st.rerun()

    # Danh sách các đoạn chat trước đó


# --- MAIN AREA: TIÊU ĐỀ & CẢNH BÁO ---
st.title("📄 Hỏi đáp thông minh với tài liệu")
st.markdown(
    "**SmartDoc AI** - Upload PDF hoặc DOCX → hỏi bất kỳ câu gì liên quan đến nội dung tài liệu. Hệ thống chạy cục bộ, bảo mật cao.")
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

if st.session_state.is_processing_file and st.session_state.pending_file:
        uploaded_file = st.session_state.pending_file
        with st.spinner("Đang xử lý tài liệu..."):
            file_bytes = uploaded_file.getvalue()
            file_size_mb = len(file_bytes) / (1024 * 1024)

            # Kiểm tra kích thước file
            if len(file_bytes) > MAX_FILE_SIZE_BYTES:
                st.error(f"File quá lớn ({file_size_mb:.1f}MB). Giới hạn tối đa là {MAX_FILE_SIZE_MB}MB.")
                st.stop()  # Dừng app ngay lập tức thay cho continue

            # Lấy đuôi file (.pdf, .docx)

            tmp_path = None

            try:

                logger.info(f"Upload nhận file: {uploaded_file.name} | Size: {file_size_mb:.2f}MB")  # ← LOG 1

                # 2. Toàn bộ khu vực này PHẢI nằm thụt lề ngang bằng với with... ở trên
                # ── Splitting — Chia nhỏ văn bản thành chunks ─────────────────
                chunks = load_and_chunk_document(filename=uploaded_file.name,content=file_bytes, chunk_size=1500,
                                                 chunk_overlap=200)

                st.session_state.vectorstore = create_vectorstore(chunks, embedder)

                # Cập nhật thông tin sau khi xử lý thành công
                st.session_state.current_file = uploaded_file.name

                logger.info(f"Vectorstore cập nhật xong: {uploaded_file.name} | chunks={len(chunks)}")  # ← LOG 2

                # 3. Các chữ except phải thẳng hàng dọc với chữ try
            except ValueError as ve:
                # Bắt các lỗi do chính mình chủ động ném ra
                st.error(f"⚠️ Lỗi dữ liệu: {str(ve)}")
                logger.warning(f"File không hợp lệ ({uploaded_file.name}): {str(ve)}")

            except Exception as e:
                # Bắt tất cả các lỗi hệ thống không lường trước
                st.error("Có lỗi hệ thống xảy ra khi đọc file. Vui lòng xem chi tiết ở Terminal hoặc thử file khác.")
                # Dòng log này BẮT BUỘC phải thụt lề vào trong except
                logger.error(f"Lỗi hệ thống khi xử lý {uploaded_file.name}: {str(e)}", exc_info=True)
        st.session_state.is_processing_file = False
        st.session_state.pending_file = None
        st.rerun()

# Banner thông báo trạng thái file
if st.session_state.current_file:
    st.success(f"✅ Đang làm việc với tài liệu: **{st.session_state.current_file}**")
else:
    st.info("Vui lòng tải file PDF hoặc DOCX lên ở ô bên trái để bắt đầu hỏi đáp.")
if st.session_state.is_thinking:
    with st.chat_message("assistant", avatar="🟧"):
        with st.spinner("Đang tìm kiếm thông tin và suy nghĩ"):
            current_question = st.session_state.messages[-1]["content"]
        retriever = create_retriever(st.session_state.vectorstore)
    if retriever:
        chain = create_rag_chain(retriever, llm)
        answer = chain.invoke(current_question)
    st.markdown(answer)
    st.session_state.messages.append({
        "role": "assistant",
        "mode": "RAG",
        "content": answer,

    })
    st.session_state.is_thinking = False
    st.rerun()

# Khung chứa lịch sử chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🔴"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🟧"):
         if "mode" in msg:
            st.caption(f"Chế độ: {msg['mode']}")
            st.write(msg["content"])
            # if "citation" in msg:
            #     with st.expander("Xem nguồn trích dẫn (Citations) & Highlight"):
            #         st.info(msg["citation"])

# --- BOTTOM AREA: UPLOAD & CHAT INLINE (THẲNG HÀNG VỚI NỘI DUNG CHAT PHÍA TRÊN) ---
col1, col2 = st.columns([1, 9], gap="small")

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
error_messages = []  # Danh sách lỗi
with col1:
    uploaded_file = st.file_uploader("Tải tài liệu", type=["pdf", "docx"], label_visibility="collapsed")
    if uploaded_file and uploaded_file.name != st.session_state.current_file:
        if not st.session_state.is_processing_file:
            st.session_state.pending_file = uploaded_file
            st.session_state.is_processing_file = True
            st.rerun()


with col2:
    prompt = st.chat_input("Nhập câu hỏi về tài liệu...")
    if prompt:
        if not st.session_state.current_file:
            st.warning("Vui lòng tải tài liệu lên trước khi đặt câu hỏi!")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.is_thinking = True
        st.rerun()
