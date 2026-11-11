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
from src.vectorstore import create_vectorstore, get_uploaded_vectorstore, get_documents_metadata
from src.retriever import create_retriever
from langchain_core.runnables import RunnableLambda
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
if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []  # Danh sách tên các file đã upload

# Cờ hiệu xử lý
if "is_processing_file" not in st.session_state:
    st.session_state.is_processing_file = False
if "pending_files" not in st.session_state:
    st.session_state.pending_files = []
if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="brand-icon">⚡</span>
        <div class="brand-text">
            <h2>SmartDoc AI</h2>
            <span class="brand-badge">Qwen 2.5 (3B)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Expander tính năng
    with st.expander("Hướng dẫn sử dụng"):
        st.write("1. Tải lên 1 hoặc nhiều tài liệu PDF/DOCX.\n2. Chọn phạm vi tìm kiếm (Tất cả hoặc từng file).\n3. Đặt câu hỏi trong khung chat để nhận câu trả lời kèm trích dẫn.")
        
    with st.expander("⚙️ Cấu hình Chunking (Size & Overlap)", expanded=True):
        st.info("LLM: Qwen2.5:3b | Embedding: Multilingual MPNet")
        chunk_size = st.number_input(
            "Chunk Size (độ dài ký tự mỗi đoạn):",
            min_value=100,
            max_value=3000,
            value=1000,
            step=50,
            help="Số ký tự tối đa của một đoạn văn bản sau khi chia nhỏ."
        )
        chunk_overlap = st.number_input(
            "Chunk Overlap (độ gối đầu giữa các đoạn):",
            min_value=0,
            max_value=1000,
            value=100,
            step=25,
            help="Số ký tự trùng lặp giữa 2 đoạn liên tiếp để bảo toàn ngữ cảnh."
        )
        if chunk_overlap >= chunk_size:
            st.error("⚠️ Chunk Overlap phải nhỏ hơn Chunk Size!")
        
    with st.expander("Thiết lập & Tùy chọn"):
        st.checkbox("Bật chế độ Streaming", value=True)
        st.checkbox("Hiển thị Citations", value=True)
        
    with st.expander("📚 Tài liệu đã lưu trữ (Metadata)", expanded=True):
        docs_meta = get_documents_metadata(st.session_state.vectorstore)
        if docs_meta:
            st.markdown(
                f"<div style='background-color:#E6F0FF; color:#1e3a8a; border-radius:8px; padding:8px 12px; font-size:0.875rem; font-weight:600; margin-bottom:10px;'>"
                f"Đã lưu trữ: <b>{len(docs_meta)}</b> tài liệu"
                f"</div>",
                unsafe_allow_html=True
            )
            for meta in docs_meta:
                # Format ngày upload cho dễ nhìn
                upload_time = meta['uploaded_at']
                if upload_time != "N/A":
                    try:
                        dt = datetime.fromisoformat(upload_time)
                        upload_time = dt.strftime("%d/%m/%Y %H:%M")
                    except Exception:
                        pass
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; padding: 8px; margin-bottom: 6px; font-size: 12px;'>
                    <b>📄 {meta['file_name']}</b><br>
                    <span style='color: #94a3b8;'>Loại:</span> <code>{meta['file_type']}</code> | 
                    <span style='color: #94a3b8;'>Chunks:</span> <code>{meta['chunks_count']}</code><br>
                    <span style='color: #94a3b8;'>Ngày:</span> {upload_time}
                </div>
                """, unsafe_allow_html=True)

            source_options = ["Tất cả tài liệu"] + [m["file_name"] for m in docs_meta]
            selected_source = st.selectbox(
                "🔍 Tìm kiếm trong phạm vi:",
                options=source_options,
                index=0,
                key="source_filter_select"
            )
        else:
            selected_source = "Tất cả tài liệu"
            st.info("Chưa có tài liệu nào trong kho lưu trữ.")

    st.markdown("<div style='margin-top: 25px; margin-bottom: 12px; text-align: center; font-weight: 600; color: white; font-size: 15px;'>Thao tác</div>", unsafe_allow_html=True)

    # 2 Nút Đặt lại lịch sử & Hộp thoại mới
    if st.button("🔄 Đặt lại lịch sử chat", use_container_width=True, key="btn_reset_history"):
        st.session_state.messages = []
        st.rerun()

    if st.button("✨ Xóa toàn bộ dữ liệu & làm mới", use_container_width=True, key="btn_new_chat"):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.uploaded_files_list = []
        st.rerun()


# --- MAIN AREA: TIÊU ĐỀ & CẢNH BÁO ---
st.markdown("""
<div class="main-header">
    <h1>📄 SmartDoc AI — Hỏi đáp thông minh với đa tài liệu</h1>
    <p>Hỗ trợ tải lên nhiều file PDF / DOCX cùng lúc, lưu trữ metadata (tên file, ngày tải, loại file) và truy vấn bằng mô hình Qwen 2.5 (3B).</p>
</div>
""", unsafe_allow_html=True)

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# --- KHỐI UPLOAD ĐA TÀI LIỆU ---
st.markdown("### 📤 Tải tài liệu lên hệ thống")
uploaded_files = st.file_uploader(
    "Chọn một hoặc nhiều file PDF / DOCX:",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    help="Bạn có thể chọn cùng lúc nhiều file PDF hoặc Word (.docx) để đưa vào cùng một kho tri thức."
)

if uploaded_files:
    # Lọc ra những file mới chưa được xử lý
    new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_files_list]
    if new_files and not st.session_state.is_processing_file:
        st.session_state.pending_files = new_files
        st.session_state.is_processing_file = True
        st.rerun()

# Xử lý các file đang chờ xử lý
if st.session_state.is_processing_file and st.session_state.pending_files:
    files_to_process = st.session_state.pending_files
    with st.spinner(f"Đang xử lý và trích xuất {len(files_to_process)} tài liệu..."):
        all_new_chunks = []
        success_files = []
        
        for up_file in files_to_process:
            file_bytes = up_file.getvalue()
            file_size_mb = len(file_bytes) / (1024 * 1024)

            if len(file_bytes) > MAX_FILE_SIZE_BYTES:
                st.error(f"File {up_file.name} quá lớn ({file_size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB), đã bỏ qua.")
                continue

            try:
                logger.info(f"Đang chunk file: {up_file.name} | Size: {file_size_mb:.2f}MB")
                chunks = load_and_chunk_document(
                    filename=up_file.name,
                    content=file_bytes,
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap)
                )
                all_new_chunks.extend(chunks)
                success_files.append(up_file.name)
            except Exception as e:
                st.error(f"Lỗi khi xử lý file {up_file.name}: {str(e)}")
                logger.error(f"Lỗi file {up_file.name}: {str(e)}", exc_info=True)

        if all_new_chunks:
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = create_vectorstore(all_new_chunks, embedder)
            else:
                # Bổ sung tài liệu mới vào VectorStore hiện tại
                st.session_state.vectorstore.add_documents(all_new_chunks)
            
            st.session_state.uploaded_files_list.extend(success_files)
            logger.info(f"Đã cập nhật Vectorstore: thêm {len(all_new_chunks)} chunks từ {len(success_files)} files.")

    st.session_state.is_processing_file = False
    st.session_state.pending_files = []
    st.rerun()

# Banner thông báo trạng thái tài liệu hiện tại
if st.session_state.uploaded_files_list:
    files_badge = ", ".join([f"`{f}`" for f in st.session_state.uploaded_files_list])
    st.success(f"✅ Đã nạp thành công **{len(st.session_state.uploaded_files_list)}** tài liệu: {files_badge}")
else:
    st.info("💡 Vui lòng tải một hoặc nhiều file PDF / DOCX ở trên để bắt đầu hỏi đáp.")

# Xử lý câu trả lời của AI
if st.session_state.is_thinking:
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Đang tìm kiếm thông tin và suy nghĩ..."):
            current_question = st.session_state.messages[-1]["content"]
            retriever = create_retriever(
                st.session_state.vectorstore,
                selected_source=st.session_state.get("source_filter_select", "Tất cả tài liệu"),
                k=3
            )
        documents = retriever.invoke(current_question)
        sources = sorted({doc.metadata["file_name"] for doc in documents if doc.metadata.get("file_name")})
        if documents:
            # Dùng lại đúng các đoạn đã tìm để câu trả lời và nguồn không lệch nhau.
            chain = create_rag_chain(RunnableLambda(lambda question: documents), llm)
            answer = chain.invoke(current_question)
        else:
            answer = "Không tìm thấy tài liệu trong phạm vi đã chọn."
            
        st.markdown(answer)
        st.session_state.messages.append({
            "role": "assistant",
            "mode": f"RAG ({st.session_state.get('source_filter_select', 'Tất cả tài liệu')})",
            "content": answer,
            "sources": sources,
        })
    st.session_state.is_thinking = False
    st.rerun()

# Khung hiển thị lịch sử chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            if "mode" in msg:
                st.caption(f"Chế độ: {msg['mode']}")
            st.write(msg["content"])
            if msg.get("sources"):
                st.caption("Tài liệu dùng làm ngữ cảnh: " + ", ".join(msg["sources"]))

# Ô chat input
prompt = st.chat_input("Nhập câu hỏi về tài liệu...")
if prompt:
    if not st.session_state.uploaded_files_list:
        st.warning("⚠️ Vui lòng tải ít nhất một tài liệu lên trước khi đặt câu hỏi!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.is_thinking = True
    st.rerun()

