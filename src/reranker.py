import logging
from functools import lru_cache
from pathlib import Path

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_reranker():
    """Nạp model một lần, dùng lại cho các câu hỏi sau để giảm thời gian chờ."""
    logger.info("Đang nạp reranker: %s", MODEL_NAME)
    return HuggingFaceCrossEncoder(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu", "cache_folder": str(Path(__file__).resolve().parents[1] / ".cache/models")},
    )


def create_reranker(retriever, k=3):
    """Chấm từng cặp (câu hỏi, đoạn tài liệu), giữ k đoạn điểm cao nhất."""
    compressor = CrossEncoderReranker(model=get_reranker(), top_n=k)
    return ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)
