"""Mục 3.3.3 và 4.4.1: nạp MPNet để nhúng các đoạn tài liệu."""

import logging
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter

from langchain_huggingface import HuggingFaceEmbeddings


logger = logging.getLogger(__name__)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIMENSION = 768
DEFAULT_CACHE_FOLDER = Path(__file__).resolve().parent / ".cache" / "models"
_model_lock = Lock()


@lru_cache(maxsize=1)
def _load_embedder(cache_folder: str, local_files_only: bool) -> HuggingFaceEmbeddings:
    started = perf_counter()
    logger.info("model.load.start | model=%s | device=cpu | offline=%s", MODEL_NAME, local_files_only)
    try:
        # Lõi mã ở mục 3.3.3: đúng model, CPU và chuẩn hóa vector.
        model = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": "cpu", "local_files_only": local_files_only},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder=cache_folder,
        )
    except Exception:
        logger.exception("model.load.failed | model=%s", MODEL_NAME)
        raise
    logger.info("model.load.done | elapsed_s=%.3f", perf_counter() - started)
    return model


def get_embedder(
    cache_folder: str | Path = DEFAULT_CACHE_FOLDER,
    local_files_only: bool = False,
) -> HuggingFaceEmbeddings:
    """Nạp khi cần; dùng lại một cấu hình gần nhất trong cùng tiến trình.

    Mặc định CPU, 768 chiều và normalize_embeddings=True. Lần đầu cần
    tải model; đặt local_files_only=True khi cache đã có đầy đủ. Không
    tự cấu hình logger của ứng dụng và không tải model khi import file.
    """
    if not isinstance(cache_folder, (str, Path)) or not str(cache_folder).strip():
        logger.error("model.config.invalid | cache_folder phải là đường dẫn có nội dung")
        raise ValueError("cache_folder phải là đường dẫn có nội dung.")
    if not isinstance(local_files_only, bool):
        logger.error("model.config.invalid | local_files_only phải là bool")
        raise ValueError("local_files_only phải là True hoặc False.")
    resolved = str(Path(cache_folder).expanduser().resolve())
    # Tránh hai lần nạp model cùng lúc nếu app gọi từ hai luồng.
    with _model_lock:
        hits = _load_embedder.cache_info().hits
        model = _load_embedder(resolved, local_files_only)
        if _load_embedder.cache_info().hits > hits:
            logger.debug("model.reuse | model=%s", MODEL_NAME)
        return model
