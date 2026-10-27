import streamlit as st
from datetime import datetime
import time

st.set_page_config(page_title="SmartDoc AI", layout="wide", initial_sidebar_state="expanded")

# CSS Custom để tái tạo giao diện chính xác 100% theo ảnh mẫu
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&family=Poppins:wght@400;500;600&display=swap');

    /* Ép font lên các thẻ chứa text, chừa icon ra */
    h1, h2, h3, h4, h5, h6, p, label, li, .stMarkdown div {
        font-family: 'Poppins', 'Montserrat', sans-serif !important;
    }

    /* Đảm bảo các thành phần icon của Streamlit giữ nguyên font Material */
    span[class*="material-symbols"], span[class*="icon"], [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* Căn lề chuẩn cho Main Block Container */
    .block-container, [data-testid="stMainBlockContainer"] {
        padding-top: 2rem !important;
        padding-left: 3.5rem !important;
        padding-right: 3.5rem !important;
        padding-bottom: 110px !important;
    }

    /* Thiết kế Sidebar & Cố định kích thước, cấm kéo giãn */
    [data-testid="stSidebar"] {
        background-color: #262730 !important;
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
        padding-top: 0.2rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }

    /* Kéo toàn bộ nội dung Sidebar lên sát đỉnh trên cùng (triệt tiêu padding 6rem mặc định) */
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    [data-testid="stSidebarUserContent"],
    [data-testid="stSidebar"] .block-container,
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding-top: 0.5rem !important;
        margin-top: 0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        display: none !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Tiêu đề SmartDoc AI nằm sát mép trên */
    .sidebar-title {
        text-align: center !important;
        color: white !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 20px !important;
        font-weight: 700 !important;
        font-size: 26px !important;
    }

    /* Vô hiệu hóa và ẩn hoàn toàn thanh kéo co giãn Sidebar */
    [data-testid="stSidebarResizeHandle"] {
        display: none !important;
        pointer-events: none !important;
        width: 0 !important;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {
        color: #FFFFFF !important;
    }

    /* Custom các Expander trong Sidebar - Triệt tiêu nền trắng khi click, focus hoặc mở */
    [data-testid="stSidebar"] [data-testid="stExpander"],
    [data-testid="stSidebar"] details {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 6px !important;
        margin-bottom: 4px !important;
    }

    /* Tiêu đề tóm tắt của Expander: luôn giữ nền trong suốt, chữ trắng rõ ràng */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] details summary,
    [data-testid="stSidebar"] details summary:hover,
    [data-testid="stSidebar"] details summary:focus,
    [data-testid="stSidebar"] details summary:active,
    [data-testid="stSidebar"] details summary:focus-visible,
    [data-testid="stSidebar"] details[open] > summary {
        background-color: transparent !important;
        color: #ffffff !important;
        font-size: 14px !important;
        padding: 8px 6px !important;
        border-radius: 6px !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Hiệu ứng khi rê chuột vào: sáng nhẹ trên nền tối (không bao giờ bị trắng) */
    [data-testid="stSidebar"] details summary:hover {
        background-color: rgba(255, 255, 255, 0.06) !important;
        color: #ffffff !important;
    }

    /* Chữ và icon mũi tên bên trong summary */
    [data-testid="stSidebar"] details summary *,
    [data-testid="stSidebar"] details summary svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        background-color: transparent !important;
    }

    /* Nội dung chi tiết mở ra bên trong Expander */
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"],
    [data-testid="stSidebar"] details > div {
        background-color: transparent !important;
        color: #e0e0e0 !important;
        padding: 6px 6px !important;
        border: none !important;
    }

    /* Nút Đặt lại lịch sử (Màu tím) */
    div[data-testid="stSidebar"] button[key="btn_reset_history"],
    div[data-testid="stSidebar"] div.stButton:nth-of-type(1) button {
        background-color: #8e24aa !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        height: 38px !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stSidebar"] div.stButton:nth-of-type(1) button:hover {
        background-color: #7b1fa2 !important;
    }

    /* Nút Hộp thoại mới (Màu xanh dương) */
    div[data-testid="stSidebar"] button[key="btn_new_chat"],
    div[data-testid="stSidebar"] div.stButton:nth-of-type(2) button {
        background-color: #007bff !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        height: 38px !important;
        margin-bottom: 15px !important;
    }
    div[data-testid="stSidebar"] div.stButton:nth-of-type(2) button:hover {
        background-color: #0069d9 !important;
    }

    /* Danh sách các item lịch sử cuộc trò chuyện */
    .history-item {
        background-color: #383b40;
        color: #e0e0e0;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 7px;
        font-size: 13px;
        cursor: pointer;
        transition: background-color 0.2s ease;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .history-item:hover {
        background-color: #4a4e54;
        color: #ffffff;
    }

    /* Banner thông báo trạng thái file */
    .success-banner {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 5px;
        font-size: 14.5px;
    }

    .date-text {
        color: #6c757d;
        font-size: 12px;
        margin-bottom: 20px;
    }

    /* 1. Ẩn logo upload, dòng 200MB, chữ Drag and drop và nhãn label phía trên */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
    [data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"],
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploadDropzone"] [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploadDropzone"] svg,
    [data-testid="stFileUploadDropzone"] [data-testid="stIconMaterial"],
    [data-testid="stFileUploadDropzone"] span[class*="material-symbols"],
    [data-testid="stFileUploadDropzone"] small,
    [data-testid="stFileUploaderDropzone"] svg,
    [data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"],
    [data-testid="stFileUploaderDropzone"] span[class*="material-symbols"],
    [data-testid="stFileUploaderDropzone"] small {
        display: none !important;
    }

    /* 2. Khung nét đứt bọc gọn nút Browse files */
    [data-testid="stFileUploader"] {
        width: fit-content !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stFileUploader"] section, 
    [data-testid="stFileUploadDropzone"],
    [data-testid="stFileUploaderDropzone"] {
        padding: 2.5px 3px !important;
        border: 1.5px dashed #b0b3b8 !important;
        border-radius: 7px !important;
        background-color: transparent !important;
        min-height: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: fit-content !important;
    }

    /* 3. Nút Browse files xanh lá chuẩn kích thước theo mẫu */
    [data-testid="stFileUploader"] button {
        background-color: #28a745 !important;
        color: white !important;
        border-radius: 5px !important;
        width: auto !important;
        min-width: 105px !important;
        max-width: 125px !important;
        height: 38px !important;
        border: none !important;
        padding: 0 12px !important;
        margin: 0 !important;
        font-size: 0px !important;
        white-space: nowrap !important;
        cursor: pointer !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* Ẩn mọi phần tử con bên trong nút */
    [data-testid="stFileUploader"] button * {
        display: none !important;
        font-size: 0px !important;
    }

    /* Chữ hiển thị trên nút (Browse files giống ảnh mẫu) */
    [data-testid="stFileUploader"] button::after {
        content: "Browse files" !important;
        display: inline-block !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        font-family: 'Poppins', 'Montserrat', sans-serif !important;
    }

    /* Đổi màu khi hover chuột vào nút xanh */
    [data-testid="stFileUploader"] button:hover {
        background-color: #218838 !important;
    }

    /* 4. Ghim cố định thanh upload + chat input ở sát đáy màn hình (Bottom Bar) */
    div[data-testid="stHorizontalBlock"]:has([data-testid="stFileUploader"]) {
        position: fixed !important;
        bottom: 0 !important;
        left: 300px !important; /* Bắt đầu sau Sidebar 300px */
        right: 0 !important;
        background-color: #ffffff !important;
        padding: 10px 3.5rem 14px 3.5rem !important; /* Khớp chuẩn xác lề trái & phải của nội dung phía trên */
        z-index: 99 !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        border-top: 1px solid #eaecef !important;
        box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.04) !important;
        transition: left 0.3s ease !important;
    }

    /* Cột 1 (chứa nút Browse files): vừa khít kích thước nút */
    div[data-testid="stHorizontalBlock"]:has([data-testid="stFileUploader"]) > div[data-testid="column"]:first-child,
    div[data-testid="stHorizontalBlock"]:has([data-testid="stFileUploader"]) [data-testid="column"]:has([data-testid="stFileUploader"]) {
        width: auto !important;
        min-width: unset !important;
        flex: 0 0 auto !important;
        margin-right: 12px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
    }

    /* Cột 2 (chứa chat input): Dài thoáng rộng (~920px), kéo gần tới mép phải nhưng không tràn hết toàn màn hình */
    div[data-testid="stHorizontalBlock"]:has([data-testid="stFileUploader"]) > div[data-testid="column"]:last-child,
    div[data-testid="stHorizontalBlock"]:has([data-testid="stFileUploader"]) [data-testid="column"]:has([data-testid="stChatInput"]) {
        width: 100% !important;
        max-width: 920px !important;
        min-width: 350px !important;
        flex: 1 1 920px !important;
        display: flex !important;
        align-items: center !important;
    }

    /* Khung chat input dài vừa vặn (max-width 920px) */
    [data-testid="stChatInput"] {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        max-width: 920px !important;
    }

    /* NÚT GỬI (Send Button): Luôn hiển thị nổi bật, rõ ràng hình tròn có icon mũi tên */
    [data-testid="stChatInput"] button {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        opacity: 1 !important;
        visibility: visible !important;
        cursor: pointer !important;
        background-color: #e2e8f0 !important;
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        border: none !important;
        margin-right: 4px !important;
        transition: background-color 0.2s ease !important;
    }

    [data-testid="stChatInput"] button:hover {
        background-color: #007bff !important;
    }

    [data-testid="stChatInput"] button svg {
        display: block !important;
        width: 18px !important;
        height: 18px !important;
        fill: #333333 !important;
    }

    [data-testid="stChatInput"] button:hover svg {
        fill: #ffffff !important;
    }

    /* Khi Sidebar đóng (nếu bấm nút thu gọn), thanh bottom bar tự mở rộng sang trái */
    [data-testid="stSidebar"][aria-expanded="false"] ~ * div[data-testid="stHorizontalBlock"]:has([data-testid="stFileUploader"]),
    [data-testid="stSidebar"][aria-expanded="false"] ~ section div[data-testid="stHorizontalBlock"]:has([data-testid="stFileUploader"]),
    [data-testid="stSidebar"][aria-expanded="false"] ~ div div[data-testid="stHorizontalBlock"]:has([data-testid="stFileUploader"]) {
        left: 0 !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- KHỞI TẠO STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Cho tôi doanh thu quý 1"},
        {
            "role": "assistant",
            "mode": "RAG",
            "content": "Doanh thu quý 1 năm 2025 là 450 tỷ VND.",
            "citation": "Trích dẫn từ trang 3, Báo cáo tài chính quý 1: 'Tổng doanh thu thuần đạt mức 450.000.000.000 VNĐ'."
        }
    ]
if "file_name" not in st.session_state:
    st.session_state.file_name = "test_docx_table.pdf"

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
        
    with st.expander("Tài liệu đã đăng"):
        st.write("• test_docx_table.docx")
        st.write("• baocao_taichinh_q1.pdf")

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
    st.markdown("""
    <div class="history-item">📄 test_docx_table.docx</div>
    <div class="history-item">💬 Cho tôi doanh thu quý 1</div>
    <div class="history-item">💬 cho tôi doanh thu quý 1</div>
    """, unsafe_allow_html=True)

# --- MAIN AREA: TIÊU ĐỀ & CẢNH BÁO ---
st.title("📄 Hỏi đáp thông minh với tài liệu")
st.markdown(
    "**SmartDoc AI** - Upload PDF hoặc DOCX → hỏi bất kỳ câu gì liên quan đến nội dung tài liệu. Hệ thống chạy cục bộ, bảo mật cao.")

# Banner thông báo trạng thái file
if st.session_state.file_name:
    st.markdown(f'<div class="success-banner">Đang làm việc với tài liệu: <b>{st.session_state.file_name}</b></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="date-text">Ngày upload: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>',
                unsafe_allow_html=True)
else:
    st.info("Vui lòng tải file PDF hoặc DOCX lên ở ô bên trái để bắt đầu hỏi đáp.")

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
            if "citation" in msg:
                with st.expander("Xem nguồn trích dẫn (Citations) & Highlight"):
                    st.info(msg["citation"])

# --- BOTTOM AREA: UPLOAD & CHAT INLINE (THẲNG HÀNG VỚI NỘI DUNG CHAT PHÍA TRÊN) ---
col1, col2 = st.columns([1, 9], gap="small")

with col1:
    uploaded_file = st.file_uploader("", type=["pdf", "docx"], label_visibility="collapsed")
    if uploaded_file:
        st.session_state.file_name = uploaded_file.name
        st.rerun()

with col2:
    prompt = st.chat_input("Nhập câu hỏi về tài liệu...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()