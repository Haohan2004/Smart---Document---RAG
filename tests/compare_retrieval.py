"""So sánh retrieval bằng MPNet và Cross-Encoder thật, không cần Ollama."""

import argparse
import json
import logging
import os
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".cache/models"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(ROOT / ".cache/models"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from langchain_core.documents import Document
from src.embedder import get_embedder
from src.retriever import create_retriever
from src.reranker import get_reranker, MODEL_NAME
from src.vectorstore import create_vectorstore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats phải >= 1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    texts = [
        "Học phí học kỳ này là 5 triệu đồng. Hạn cuối đóng học phí là ngày 30 tháng 9.",
        "Thư viện mở cửa từ 7 giờ sáng đến 17 giờ chiều, từ thứ Hai đến thứ Sáu.",
        "Python dùng từ khóa def để định nghĩa hàm.",
        "Phòng y tế nằm ở tầng trệt tòa nhà B và hỗ trợ sơ cứu sinh viên.",
        "Sinh viên đăng ký môn học trên cổng thông tin đào tạo trực tuyến.",
        "Mã học phần CS101 là môn Nhập môn lập trình, sử dụng ngôn ngữ Python.",
        "Mã học phần CS102 là môn Cấu trúc dữ liệu, sử dụng ngôn ngữ C++.",
        "Sinh viên quên mật khẩu đăng nhập cần liên hệ phòng Công nghệ thông tin.",
        "Học bổng được xét theo điểm học tập và kết quả rèn luyện.",
        "Nhà ăn phục vụ bữa trưa từ 11 giờ đến 13 giờ.",
        "Giấy xác nhận sinh viên được cấp tại phòng Công tác sinh viên.",
        "Sinh viên nghỉ học vì ốm cần nộp giấy xác nhận của cơ sở y tế.",
    ]
    questions = [
        ("Hạn cuối đóng học phí là khi nào?", 0),
        ("Thư viện mở cửa lúc mấy giờ?", 1),
        ("Trong Python, từ khóa nào dùng để định nghĩa hàm?", 2),
        ("Muốn được sơ cứu thì tới đâu?", 3),
        ("Đăng ký môn học bằng cách nào?", 4),
        ("CS101 học ngôn ngữ lập trình gì?", 5),
        ("CS102 là môn gì?", 6),
        ("Quên mật khẩu thì liên hệ ai?", 7),
        ("Học bổng dựa vào tiêu chí nào?", 8),
        ("Nhà ăn phục vụ bữa trưa khi nào?", 9),
    ]
    documents = [Document(page_content=text, metadata={"file_name": "mau.docx", "chunk_index": i}) for i, text in enumerate(texts)]
    start = perf_counter()
    embedder = get_embedder()
    store = create_vectorstore(documents, embedder)
    index_seconds = perf_counter() - start
    start = perf_counter()
    get_reranker()
    model_seconds = perf_counter() - start
    report = {
        "documents": len(documents), "questions": len(questions), "repeats": args.repeats,
        "environment": {"python": sys.version.split()[0], "device": "cpu", "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
                        "versions": {name: version(name) for name in ("faiss-cpu", "langchain", "sentence-transformers", "torch", "numpy")}},
        "model": MODEL_NAME, "index_seconds": index_seconds,
        "reranker_load_seconds": model_seconds, "results": [],
    }
    modes = [
        ("vector", False, False, 10), ("hybrid", True, False, 10),
        ("vector_rerank", False, True, 10), ("hybrid_rerank", True, True, 10),
        ("hybrid_rerank_5", True, True, 5),
    ]
    for name, hybrid, rerank, candidates in modes:
        start = perf_counter()
        retriever = create_retriever(store, use_hybrid=hybrid, use_reranker=rerank, candidate_k=candidates)
        build_ms = (perf_counter() - start) * 1000
        retriever.invoke(questions[0][0])  # Warm-up, không đưa vào thời gian truy vấn.
        elapsed, ranks = [], []
        for _ in range(args.repeats):
            for question, expected in questions:
                start = perf_counter()
                results = retriever.invoke(question)
                elapsed.append((perf_counter() - start) * 1000)
                assert len(results) == 3
                for doc in results:
                    original = documents[doc.metadata["chunk_index"]]
                    assert doc.page_content == original.page_content and doc.metadata == original.metadata
                ranks.append(next((i for i, doc in enumerate(results, 1) if doc.metadata["chunk_index"] == expected), 0))
        row = {
            "mode": name, "build_ms": round(build_ms, 2),
            "median_query_ms": round(median(elapsed), 2),
            "top1": round(sum(rank == 1 for rank in ranks) / len(ranks), 3),
            "hit_at_3": round(sum(rank > 0 for rank in ranks) / len(ranks), 3),
            "mrr_at_3": round(sum(1 / rank if rank else 0 for rank in ranks) / len(ranks), 3),
        }
        report["results"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report["note"] = "Bộ thử nhỏ có đáp án định trước, không đại diện độ chính xác trên mọi tài liệu. Không làm filter 8.2.8."
    output = ROOT / "logs/retrieval_comparison.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã lưu", output)


if __name__ == "__main__":
    main()
