import logging
import sys


def get_logger(name="SmartDocAI"):
    logger = logging.getLogger(name)

    # Thiết lập mức độ cảnh báo (chỉ ghi nhận từ INFO trở lên, bỏ qua DEBUG lặt vặt)
    logger.setLevel(logging.INFO)

    # Kỹ thuật chống nhân bản log khi Streamlit rerun
    if not logger.handlers:
        # Định dạng câu log: [Thời gian] Mức độ [Tên_file:Dòng_code] Nội dung
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Xuất log ra màn hình Terminal
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # (Tùy chọn) Lưu log vào file văn bản để xem lại khi app sập
        file_handler = logging.FileHandler("app.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger