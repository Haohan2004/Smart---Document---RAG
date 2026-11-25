import os
import json
import tempfile
import time
from datetime import datetime

import streamlit as st

from src import retriever
from src.embedder import get_embedder
from src.llm import get_llm
from src.loader import load_and_chunk_document, load_and_chunk_documents
from src.logger import get_logger
from src.rag_chain import create_rag_chain
from src.vectorstore import create_vectorstore, get_uploaded_vectorstore,add_to_vectorstore
from src.retriever import create_retriever
from src.persistence import (
    load_chat_history,
    save_chat_history,
    save_vectorstore,
    load_vectorstore
)
st.set_page_config(page_title="SmartDoc AI", layout="wide", initial_sidebar_state="locked")
# đặt tên file lưu trữ lịch sử hội thoại

# Khởi tạo logger
logger = get_logger()




# ── Load CSS từ file riêng ───────────────────────────────────────────────────
def load_css(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def auto_scroll():
    target = st.session_state.pop("scroll_target", None)

    if target not in ("top", "query"):
        return

    target_id = (
        "rag-page-top"
        if target == "top"
        else "rag-latest-query"
    )

    st.html(
        f"""
        <script>
        (() => {{
            const targetId = "{target_id}";
            let attempts = 0;

            function scrollToTarget() {{
                const element = document.getElementById(targetId);

                if (element) {{
                    element.scrollIntoView({{
                        behavior: "instant",
                        block: "start"
                    }});
                }} else if (attempts++ < 30) {{
                    setTimeout(scrollToTarget, 50);
                }}
            }}

            requestAnimationFrame(scrollToTarget);
        }})();
        </script>
        """,
        unsafe_allow_javascript=True
    )

def create_new_chat_session(title="Cuoc tro chuyen moi", keep_current_context=False):
    """Tạo session chat mới và cập nhật state"""
    new_id = max([s["id"] for s in st.session_state.chat_sessions], default=-1) + 1
    st.session_state.chat_sessions.append({
        "id": new_id,
        "title": title,
        "messages": [],
        "vectorstore": None,
        "file": None,
        "files": []
    })
    st.session_state.current_chat_id = new_id
    if not keep_current_context:
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.current_file = None
    return new_id
def save_current_session_to_disk():
    if st.session_state.current_chat_id is None:
        return
    uploaded_sources = get_uploaded_vectorstore(st.session_state.vectorstore)

    for s in st.session_state.chat_sessions:
         if s["id"] == st.session_state.current_chat_id:
            s["messages"] = st.session_state.messages.copy()
            s["file"] = st.session_state.current_file
            s["files"] = uploaded_sources
            if st.session_state.vectorstore is not None:
                save_vectorstore(s["id"], st.session_state.vectorstore)
            break
    st.session_state.chat_sessions = [
        s for s in st.session_state.chat_sessions
        if not (s["title"] == "Cuộc trò chuyện mới" and not s["messages"] and not s.get("file") and not s.get("files"))
    ]
    save_chat_history(st.session_state.chat_sessions)

def load_session_to_state(target_id):
    """Load đúng chat mà người dùng vừa bấm vào từ disk"""
    fresh_sessions = load_chat_history()
    st.session_state.chat_sessions = fresh_sessions
    target = next((s for s in fresh_sessions if s["id"] == target_id), None)

    if target:
        st.session_state.current_chat_id = target_id
        st.session_state.messages = target["messages"].copy()
        files = target.get("files", [])
        st.session_state.current_file = target.get("file") or (files[-1] if files else None)
    else:
        # Nếu không có trên disk → giữ nguyên state hiện tại hoặc reset
        st.session_state.current_chat_id = target_id
        st.session_state.messages = []
        st.session_state.current_file = None
        logger.warning(f"Không tìm thấy chat {target_id} trên disk, reset state")

    # Luôn load vectorstore từ disk (quan trọng nhất để hỏi đáp đúng tài liệu)
    st.session_state.vectorstore = load_vectorstore(target_id, get_embedder())


load_css("style.css")
st.html('<div id="rag-page-top"></div>')

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

if "chat_sessions" not in st.session_state:
    sessions = load_chat_history()

    sessions = [
        s for s in sessions
        if (
            isinstance(s, dict)
            and isinstance(s.get("id"), int)
            and isinstance(s.get("messages"), list)
            and not (
                s.get("title") == "Cuộc trò chuyện mới"
                and not s.get("messages", [])
            )
        )
    ]

    st.session_state.chat_sessions = sessions
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None  # lưu id của đoạn chat đang hoạt động

# 2 biến cờ hiệu để mang Spinner ra giữa màn hình
if "is_processing_file" not in st.session_state:
    st.session_state.is_processing_file = False
if "pending_file" not in st.session_state:
    st.session_state.pending_file = None
if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False
if "upload_summary" not in st.session_state:
    st.session_state.upload_summary = None
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0
if "confirm_reset_history" not in st.session_state:
    st.session_state.confirm_reset_history = False


def build_citations(retrieved_docs):
    citations = []

    for doc in retrieved_docs:
        metadata = doc.metadata or {}

        # Ưu tiên tên file gốc
        filename = metadata.get("file_name")

        if not filename:
            filename = os.path.basename(
                str(metadata.get("source", "Không rõ tài liệu"))
            )

        # Lấy vị trí trong tài liệu
        page_label = metadata.get("page_label")
        page_index = metadata.get("page")
        chunk_index = metadata.get("chunk_index")

        if page_label is not None:
            location = f"Trang {page_label}"

        elif isinstance(page_index, int):
            location = f"Trang {page_index + 1}"

        elif isinstance(chunk_index, int):
            location = f"Đoạn {chunk_index + 1}"

        else:
            location = "Không xác định vị trí"

        citations.append({
            "filename": filename,
            "location": location,
            "content": doc.page_content
        })

    return citations

def show_self_rag(info, key):
    if info is None:
        return

    with st.expander("🟠 Thông tin Self-RAG", expanded=False, key=key):

        attempts = info.get("attempts", 1)
        score = info.get("score")
        feedback = info.get("feedback", "")
        hops = info.get("hops", [])

        score_text = f"{score}/100" if score is not None else "Chưa đánh giá"

        st.markdown(
            f"**Số lần tạo/sửa câu trả lời:** {attempts} | "
            f"**Mức hỗ trợ tài liệu:** {score_text}"
        )

        if hops:
            st.markdown("**Các truy vấn tìm kiếm bổ sung:**")
            for hop in hops:
                st.markdown(f"- {hop}")
        else:
            st.caption("Không cần truy xuất bổ sung.")

        st.markdown(f"**Lý do đánh giá:** {feedback}")

def show_citations(citations, key):
    if not citations:
        return

    with st.expander(
               "Xem nguồn trích dẫn (Citations) & Highlight",
              expanded=False,
                key=key
        ):
        st.write("Hệ thống đã dựa vào các đoạn sau để trả lời:")

        for i, citation in enumerate(citations, start=1):
            st.markdown(
                f"**Nguồn {i}:** File `{citation['filename']}`"
                f" — {citation['location']}"
            )

            st.info(citation["content"])

    def show_citations(citations, key):
        if not citations:
            return

        with st.expander(
                "Xem nguồn trích dẫn (Citations) & Highlight",
                expanded=False,
                key=key
        ):
            st.write("Hệ thống đã dựa vào các đoạn sau để trả lời:")

            for i, citation in enumerate(citations, start=1):
                filename = citation["filename"]
                page = citation["page"]

                page_text = (
                    f"Trang {page}" if page else "Chưa xác định trang"
                )

                st.markdown(
                    f"**Nguồn {i}:** File `{filename}` — {page_text}"
                )

                # Highlight context được truy xuất
                st.info(citation["content"])

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(
        "<h2 class='sidebar-title' style='text-align: center; color: white; margin-top: 0; padding-top: 0; margin-bottom: 20px; font-weight: 700; font-size: 26px;'>SmartDoc AI</h2>",
        unsafe_allow_html=True)

    with st.expander("Hướng dẫn sử dụng"):
        st.write(
            "1. Tải lên tài liệu PDF hoặc DOCX.\n2. Đặt câu hỏi trong khung chat.\n3. Nhận câu trả lời kèm trích dẫn chính xác.")

    with st.expander("Cấu hình mô hình"):
        st.info("LLM: Qwen2.5:7b\n\nEmbedding: Multilingual MPNet")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 50, 200, 100)

    with st.expander("Thiết lập & Tùy chọn"):

        st.selectbox(
            "Self-RAG Mode",
            options=[False, True],
            format_func=lambda x: "On | Self-RAG" if x else "Off | Normal RAG",
            key="enable_self_rag"
        )

        # Chọn phương pháp truy xuất
        retrieval_mode = st.selectbox(
            "Retrieval Mode",
            options=[
                "FAISS (Vector Search)",
                "Hybrid (FAISS + BM25)",
                "BM25 (Keyword Search)"
            ],
            key="retrieval_mode"
        )

        # Chỉ hiện điều chỉnh tỷ lệ khi chọn Hybrid
        if retrieval_mode == "Hybrid (FAISS + BM25)":
            st.markdown("**Điều chỉnh tỷ lệ Hybrid Search:**")

            faiss_ratio = st.slider(
                "Tỷ lệ FAISS (Semantic) / BM25",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key="hybrid_faiss_ratio"
            )

            bm25_ratio = round(1.0 - faiss_ratio, 2)

            st.html(f"""
            <style>
            .hybrid-ratio-box {{
                background: #ffffff !important;
                border-radius: 8px;
                padding: 12px;
                text-align: center;
                margin-top: 10px;
            }}

            .hybrid-ratio-box span {{
                color: #5264c9 !important;
                font-size: 14px;
                font-weight: 700;
            }}
            </style>

            <div class="hybrid-ratio-box">
                <span>
                    FAISS: {faiss_ratio:.2f} &nbsp; | &nbsp;
                    BM25: {bm25_ratio:.2f}
                </span>
            </div>
            """)

    with st.expander("Tài liệu đã đăng", expanded=True):
        uploaded_sources = get_uploaded_vectorstore(
            st.session_state.vectorstore
        )

        if "confirm_delete_document" not in st.session_state:
            st.session_state.confirm_delete_document = False

        if uploaded_sources:
            st.markdown(
                f"""
                <div style="
                    color: white;
                    font-size: 15px;
                    font-weight: 600;
                    margin: 10px 0 15px 0;
                ">
                    📁 Hệ thống đang xử lý: {len(uploaded_sources)} tài liệu
                </div>
                """,
                unsafe_allow_html=True
            )

            for src in uploaded_sources:
                st.markdown(
                    f"""
                    <div style="
                        display: flex;
                        align-items: center;
                        gap: 12px;
                        padding: 8px 0 8px 20px;
                        font-size: 15px;
                        color: white;
                    ">
                        📄 <span>{src}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            source_options = ["Tất cả tài liệu"] + uploaded_sources

            selected_source = st.selectbox(
                "Tìm kiếm trong:",
                options=source_options,
                index=0,
                key="source_filter_select"
            )

            # Nút xóa tài liệu
            if st.button(
                    "🗑️ Xóa tài liệu hiện tại",
                    use_container_width=True,
                    key="btn_delete_document"
            ):
                st.session_state.confirm_delete_document = True

            # Xác nhận xóa tài liệu
            if st.session_state.confirm_delete_document:
                with st.container(border=True):

                    st.markdown(
                        f"""
                        <p style="
                            color: #e74c3c;
                            font-size: 13px;
                            text-align: center;
                        ">
                            Bạn có chắc chắn muốn xóa
                            {selected_source.lower() if selected_source == "Tất cả tài liệu" else selected_source}
                            không?
                        </p>
                        """,
                        unsafe_allow_html=True
                    )

                    col_yes, col_no = st.columns(2)

                    with col_yes:
                        if st.button(
                                "✅ Đồng ý",
                                key="confirm_delete_doc_yes",
                                use_container_width=True

                        ):
                            vectorstore = st.session_state.vectorstore

                            # Tìm ID các chunks cần xóa
                            ids_to_delete = []

                            for doc_id in vectorstore.index_to_docstore_id.values():
                                doc = vectorstore.docstore.search(doc_id)

                                if (
                                        selected_source == "Tất cả tài liệu"
                                        or doc.metadata.get("file_name") == selected_source
                                ):
                                    ids_to_delete.append(doc_id)

                            # Xóa khỏi FAISS
                            if ids_to_delete:
                                vectorstore.delete(ids_to_delete)

                            # Cập nhật danh sách tài liệu
                            remaining_sources = get_uploaded_vectorstore(
                                vectorstore
                            )

                            st.session_state.current_file = (
                                ", ".join(remaining_sources)
                                if remaining_sources else None
                            )

                            st.session_state.upload_summary = None
                            st.session_state.confirm_delete_document = False
                            st.session_state.uploader_version += 1

                            save_current_session_to_disk()

                            st.rerun()

                    with col_no:
                        if st.button(
                                "❌ Hủy",
                                key="confirm_delete_doc_no",
                                use_container_width=True

                        ):
                            st.session_state.confirm_delete_document = False
                            st.rerun()

        else:
            st.info("Chưa có tài liệu nào được upload.")
    # --- LỊCH SỬ CUỘC TRÒ CHUYỆN ---
    st.markdown(
        "<div style='margin-top: -15px; margin-bottom: 25px; "
        "text-align: center; font-weight: 600; color: white; "
        "font-size: 15px;'>Lịch sử cuộc trò chuyện</div>",
        unsafe_allow_html=True
    )

    # 1. Nút đặt lại lịch sử
    if st.button(
            "🔄 Đặt lại lịch sử",
            use_container_width=True,
            key="btn_reset_history"
    ):
        st.session_state.confirm_reset_history = True

    # 2. Hộp xác nhận xóa
    if st.session_state.confirm_reset_history:

        with st.container(border=True):
            st.markdown(
                "<p style='color:#e74c3c; font-size:13px;'>"
                "Bạn có chắc chắn muốn xóa toàn bộ lịch sử không?"
                "</p>",
                unsafe_allow_html=True
            )

            col_yes, col_no = st.columns(2)

            with col_yes:
                if st.button("✅ Đồng ý", key="confirm_yes"):
                    logger.info("User xác nhận xóa lịch sử")

                    st.session_state.chat_sessions = []
                    st.session_state.current_chat_id = None
                    st.session_state.messages = []

                    st.session_state.pending_prompt = None
                    st.session_state.is_thinking = False

                    st.session_state.current_file = None
                    st.session_state.vectorstore = None
                    st.session_state.upload_summary = None
                    st.session_state.pending_file = None
                    st.session_state.is_processing_file = False

                    save_chat_history([])

                    st.session_state.uploader_version += 1
                    st.session_state.confirm_reset_history = False
                    st.rerun()

            with col_no:
                if st.button("❌ Hủy", key="confirm_no"):
                    st.session_state.confirm_reset_history = False
                    st.rerun()

    # 3. Nút hộp thoại mới
    if st.button(
            "✨ Hộp thoại mới",
            use_container_width=True,
            key="btn_new_chat"
    ):
        logger.info("User tạo hộp thoại mới")

        st.session_state.current_chat_id = None
        st.session_state.messages = []
        st.session_state.current_file = None
        st.session_state.vectorstore = None

        st.session_state.pending_prompt = None
        st.session_state.is_thinking = False

        st.session_state.pending_file = None
        st.session_state.is_processing_file = False
        st.session_state.upload_summary = None

        st.session_state.confirm_reset_history = False
        st.session_state.uploader_version += 1

        st.rerun()

    # Khoảng cách trước danh sách chat
    # 4. Danh sách chat history
    st.markdown("---")

    for session in st.session_state.chat_sessions:
        title = session.get("title", "Cuộc trò chuyện")

        if st.button(
                f"📄 {title[:25]}",
                key=f"btn_session_{session['id']}",
                use_container_width=True
        ):
            load_session_to_state(session["id"])
            st.session_state.pending_prompt = None
            st.session_state.is_thinking = False
            st.rerun()


# --- MAIN AREA ---
st.title("📄 Hỏi đáp thông minh với tài liệu")
st.markdown(
    "**SmartDoc AI** - Upload PDF hoặc DOCX → hỏi bất kỳ câu gì liên quan đến nội dung tài liệu. Hệ thống chạy cục bộ, bảo mật cao.")
upload_notification = st.empty()
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

if st.session_state.is_processing_file and st.session_state.pending_file:
    uploaded_file = st.session_state.pending_file
    filenames = [file.name for file in uploaded_file]
    filenames_text = ", ".join(filenames)
    total_size_mb = sum(
        file.size for file in uploaded_file
    ) / (1024 * 1024)
    with st.spinner("Đang xử lý tài liệu..."):

        try:
            logger.info(f"[QUY TRÌNH 1] Bắt đầu xử lý file: {filenames_text} | Kích thước: {len(uploaded_file):.2f}MB")

            start_chunk_time = time.time()
            chunks = load_and_chunk_documents(
                files=uploaded_file,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            logger.info(
                f"Hoàn thành Chunking ({time.time() - start_chunk_time:.2f}s): Tạo ra {len(chunks)} chunks | size={chunk_size}, overlap={chunk_overlap}")
            print("METADATA:", chunks[0].metadata)
            logger.info("[QUY TRÌNH 2] Bắt đầu embedding và lưu vào FAISS vectorstore...")
            start_embed_time = time.time()
            st.session_state.vectorstore = add_to_vectorstore(st.session_state.vectorstore, chunks, embedder)
            st.session_state.current_file = filenames_text
            st.session_state.upload_summary = {
                "filename": filenames_text,
                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_size": total_size_mb,
                "chunks": len(chunks),
                "total_files": len(
                    get_uploaded_vectorstore(st.session_state.vectorstore)
                )
            }
            logger.info(
                f"Hoàn thành Vectorstore ({time.time() - start_embed_time:.2f}s): Đã lưu index cho {filenames_text}.")

            for s in st.session_state.chat_sessions:
                if s["id"] == st.session_state.current_chat_id:
                    if s["title"] == "Cuộc trò chuyện mới":
                        s["title"] = filenames_text
                    break
            save_current_session_to_disk()

        except ValueError as ve:
            logger.warning(f"Lỗi dữ liệu từ file ({filenames_text}): {str(ve)}")
            st.error(f"⚠️ Lỗi dữ liệu: {str(ve)}")

        except Exception as e:
            logger.error(f"Lỗi hệ thống nghiêm trọng khi xử lý file {filenames_text}: {str(e)}", exc_info=True)
            st.error("Có lỗi hệ thống xảy ra khi đọc file. Vui lòng xem chi tiết ở Terminal.")

    st.session_state.is_processing_file = False
    st.session_state.pending_file = None
    st.session_state.uploader_version += 1
    st.rerun()

# Banner thông báo trạng thái file

# Banner thông báo trạng thái file
if st.session_state.current_file:

    st.success(
        f"✅ Đang làm việc với {len(uploaded_sources)} tài liệu: "
        f"**{', '.join(uploaded_sources)}**"
    )

    if st.session_state.upload_summary:
        info = st.session_state.upload_summary

        st.success(
            f"Hoàn tất! Đã xử lý {info['total_files']} tài liệu "
            f"({info['chunks']} chunks)."
        )

        st.info(
            f"Tài liệu mới: {info['filename']}  \n"
            f"Ngày upload: {info['uploaded_at']}  \n"
            f"Tổng dung lượng: {info['total_size']:.2f} MB  \n"
            f"Số đoạn (chunks): {info['chunks']}  \n"
            f"Tổng tài liệu đang index: {info['total_files']}"
        )

else:
    st.info(
        "Vui lòng tải file PDF hoặc DOCX lên "
        "ở ô bên trái để bắt đầu hỏi đáp."
    )
    chat_notification = st.empty()

# Khung chứa lịch sử chat
for index, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        if index == len(st.session_state.messages) - 1:
            st.html('<div id="rag-latest-query"></div>')

        with st.chat_message("user", avatar="🔴"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🟧"):
            if "mode" in msg:
                st.caption(f"Chế độ: {msg['mode']}")

            st.markdown(msg["content"])

            if msg.get("citations"):
                show_citations(
                    msg["citations"],
                    key=f"citations_history_{index}"
                )
            if msg.get("self_rag") is not None:
                show_self_rag(
                    msg["self_rag"],
                    key=f"selfrag_history_{index}"
                )
    auto_scroll()
# --- BOTTOM AREA: UPLOAD & CHAT INLINE ---
col1, col2 = st.columns([1, 9], gap="small")
with col1:
    uploaded_files = st.file_uploader(
        "Tải tài liệu",
        type=None,
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_version}"
    )

    if uploaded_files is not None:
        valid_files = []

        for file in uploaded_files:
            filename = file.name
            extension = os.path.splitext(filename)[1].lower()
            file_size_mb = file.size / (1024 * 1024)

            # Kiểm tra định dạng file
            if extension not in [".pdf", ".docx"]:
                upload_notification.warning(
                    f"{filename}: Định dạng không được phép. "
                    "Chỉ chấp nhận PDF (.pdf) hoặc DOCX (.docx).",
                    icon="⚠️"
                )
                logger.warning(f"File sai định dạng: {filename}")

            # Kiểm tra kích thước file
            elif file.size > MAX_FILE_SIZE_BYTES:
                upload_notification.warning(
                    f"{filename}: quá lớn ({file_size_mb:.1f}MB). "
                    f"Giới hạn tối đa là {MAX_FILE_SIZE_MB}MB.",
                    icon="⚠️"
                )
                logger.warning(f"File quá lớn: {filename}")

            else:
                valid_files.append(file)

        # File hợp lệ -> xử lý như code cũ
        if valid_files and not st.session_state.is_processing_file:
            logger.info(
                f"UI Event: User vừa chọn upload file mới: {filename}"
            )
            st.session_state.pending_file = valid_files
            st.session_state.is_processing_file = True
            st.session_state.scroll_target = "top"
            st.rerun()

with col2:
    prompt = st.chat_input("Nhập câu hỏi về tài liệu...", disabled=st.session_state.is_thinking)
    if prompt:
        mode = "self-rag" if st.session_state.get("enable_self_rag", False) else "rag"
        logger.info(f"[QUY TRÌNH 3] | Mode: '{mode}' | User nhập câu hỏi: '{prompt}'")
        if not st.session_state.current_file:
            logger.warning("User đặt câu hỏi nhưng chưa có file. Chặn request.")
            chat_notification.warning(
                "Vui lòng tải tài liệu lên trước khi đặt câu hỏi!",
                icon="⚠️"
            )
            st.stop()

        if st.session_state.current_chat_id is None:
            create_new_chat_session(
                prompt[:40] + ("..." if len(prompt) > 40 else ""),
                keep_current_context=True
            )

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.scroll_target = "query"
        for s in st.session_state.chat_sessions:
            if s["id"] == st.session_state.current_chat_id:
                if s["title"] == "Cuộc trò chuyện mới":
                    s["title"] = prompt[:40] + ("..." if len(prompt) > 40 else "")
                break

        st.session_state.pending_prompt = prompt
        st.session_state.is_thinking = True
        save_current_session_to_disk()
        st.rerun()
# --- XỬ LÝ KHI AI ĐANG SUY NGHĨ (RAG CHAIN) ---
if st.session_state.is_thinking and st.session_state.pending_prompt:
    current_question = st.session_state.pending_prompt
    with st.chat_message("assistant", avatar="🟧"):
         with st.spinner("Đang tìm kiếm thông tin và suy nghĩ..."):

            logger.info(f"[QUY TRÌNH 4 & 5] Bắt đầu RAG. Câu hỏi: '{current_question}'")
            start_rag_time = time.time()

            # Tạo retriever từ vectorstore hiện tại
            mode_mapping = {
                "FAISS (Vector Search)": "faiss",
                "Hybrid (FAISS + BM25)": "hybrid",
                "BM25 (Keyword Search)": "bm25"
            }
            selected_mode = st.session_state.get(
                "retrieval_mode",
                "FAISS (Vector Search)"
            )
            mode = mode_mapping.get(selected_mode, "faiss")

            mode_name = {
                "faiss": "FAISS",
                "hybrid": "Hybrid",
                "bm25": "BM25"
            }[mode]

            logger.info(
                f"Retrieval mode: {mode} | "
                f"FAISS Weight: {st.session_state.get('hybrid_faiss_ratio', 0.7)} | "
                f"search_type=similarity | top_k=3"
                f"Source: {st.session_state.get('source_filter_select', 'Tất cả tài liệu')}"
            )

            selected_source = st.session_state.get(
                "source_filter_select",
                "Tất cả tài liệu"
            )

            retriever = create_retriever(
                vectorstore=st.session_state.vectorstore,
                k=3,selected_source=st.session_state.get("source_filter_select", "Tất cả tài liệu"),
                mode=mode,
                faiss_weight=st.session_state.get("hybrid_faiss_ratio", 0.7)
            )
            result = ""
            citations = []
            self_rag_info = None
            if retriever:
                citations = []
                try:
                    logger.info("Khởi tạo Retriever thành công. Tiến hành gọi LLM...")
                    self_rag_enabled = st.session_state.get("enable_self_rag", False)
                    logger.info(
                        f"[Settings] Self-RAG {'ON' if self_rag_enabled else 'OFF'}"
                    )
                    chain = create_rag_chain(retriever, llm,enable_self_rag=self_rag_enabled)
                    logger.info(f"Gọi LLM thành công, bắt đầu phản hồi")
                    chat_history = st.session_state.messages.copy()
                    #Loại câu hỏi hiện tại khỏi history
                    if (
                            chat_history
                            and chat_history[-1]["role"] == "user"
                            and chat_history[-1]["content"] == current_question
                    ):
                        chat_history.pop()

                    # 3. Giới hạn 8 tin nhắn gần nhất (4 lượt hội thoại)
                    chat_history = chat_history[-8:]

                    answer = chain.invoke({
                              "question": current_question,
                              "chat_history": chat_history
                    })
                    # Lấy câu trả lời từ LLM
                    result = answer["answer"]

                    # Lấy những chunks đã truy xuất
                    retrieved_docs = answer["source_documents"]
                    if self_rag_enabled:
                        self_rag_info = {
                            "attempts": answer.get("self_rag_attempts", 1),
                            "score": answer.get("support_score"),
                            "feedback": answer.get("self_rag_feedback", ""),
                            "hops": answer.get("self_rag_hops", [])
                        }

                    # Chuyển thành citations để hiển thị và lưu JSON
                    citations = build_citations(retrieved_docs)
                    elapsed_time = time.time() - start_rag_time


                    logger.info(f"Hoàn thành RAG Chain. Thời gian phản hồi: {elapsed_time:.2f}s")
                    logger.debug(f"Output từ LLM: {answer}")

                except Exception as e:
                    logger.error(f"Lỗi kết nối hoặc xử lý LLM (Ollama): {str(e)}", exc_info=True)
                    answer = "⚠️ Lỗi: Không thể nhận câu trả lời từ AI. Vui lòng kiểm tra xem Ollama đã được bật chưa (Xem log terminal)."
            else:
                logger.warning("Không thể tạo retriever (Vectorstore trống).")
                answer = "⚠️ Vui lòng tải tài liệu lên trước khi đặt câu hỏi!"

            # In câu trả lời ra giao diện
            st.markdown(result)
            if citations:
                show_citations(
                    citations,
                    key="current_citations"
                )
            if self_rag_info is not None:
                show_self_rag(
                    self_rag_info,
                    key="current_selfrag"
                )
            # Lưu vào lịch sử chat
            st.session_state.messages.append({
                "role": "assistant",
                "mode": "RAG",
                "content": result,
                "citations": citations,
                "self_rag": self_rag_info
            })
            st.session_state.is_thinking = False
            st.session_state.pending_prompt = None
            save_current_session_to_disk()
            st.rerun()


