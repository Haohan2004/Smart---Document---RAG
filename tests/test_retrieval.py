"""Kiểm tra logic retrieval bằng vector/model giả; benchmark dùng model thật."""

import copy
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_community.cross_encoders.base import BaseCrossEncoder

from src.retriever import create_retriever, tokenize
from src.reranker import create_reranker, get_reranker
from src.vectorstore import create_vectorstore


class SampleScorer(BaseCrossEncoder):
    def score(self, pairs):
        return [10 if "đúng đoạn" in text else 0 for question, text in pairs]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document(page_content=text, metadata={"file_name": "mau.docx", "chunk_index": i})
            for i, text in enumerate([
                "Học phí là 5 triệu đồng.", "Thư viện mở cửa lúc 7 giờ.",
                "Python định nghĩa hàm bằng def.", "Mã học phần CS101 học về lập trình.",
                "Phòng y tế ở tòa B.",
            ])
        ]
        self.store = create_vectorstore(self.docs, FakeEmbeddings(size=8))

    def test_tokenize_vietnamese(self):
        self.assertEqual(tokenize("HỌC phí, CS101?"), ["học", "phí", "cs101"])

    def test_vector_and_hybrid_return_top_k_and_preserve_metadata(self):
        before = copy.deepcopy(self.docs)
        for hybrid in (False, True):
            retriever = create_retriever(self.store, use_hybrid=hybrid, use_reranker=False)
            results = retriever.invoke("CS101")
            self.assertEqual(len(results), 3)
            for result in results:
                self.assertIn(result.metadata, [doc.metadata for doc in self.docs])
        self.assertEqual(self.docs, before)

    def test_small_store_and_blank_query_do_not_crash(self):
        store = create_vectorstore(self.docs[:1], FakeEmbeddings(size=8))
        results = create_retriever(store, use_reranker=False).invoke(" ")
        self.assertEqual(len(results), 1)

    def test_reranker_sorts_scores_without_changing_metadata(self):
        docs = self.docs[:2] + [Document(page_content="đúng đoạn", metadata={"file_name": "goc.pdf"})]
        before = copy.deepcopy(docs)
        with patch("src.reranker.get_reranker", return_value=SampleScorer()):
            retriever = create_reranker(RunnableLambda(lambda query: docs), k=1)
            self.assertEqual(retriever.invoke("câu hỏi"), [docs[2]])
        self.assertEqual(docs, before)

    def test_candidates_are_limited_before_reranking(self):
        with patch("src.reranker.get_reranker", return_value=SampleScorer()):
            retriever = create_retriever(self.store, k=1, candidate_k=2)
            self.assertLessEqual(len(retriever.base_retriever.invoke("CS101")), 2)
            self.assertEqual(len(retriever.invoke("CS101")), 1)

    def test_bad_k_is_rejected(self):
        for kwargs in ({"k": 0}, {"k": 3, "candidate_k": 2}):
            with self.assertRaises(ValueError):
                create_retriever(self.store, **kwargs)

    def test_model_is_loaded_once(self):
        get_reranker.cache_clear()
        try:
            with patch("src.reranker.HuggingFaceCrossEncoder", return_value=SampleScorer()) as constructor:
                self.assertIs(get_reranker(), get_reranker())
                self.assertEqual(constructor.call_count, 1)
        finally:
            get_reranker.cache_clear()


if __name__ == "__main__":
    unittest.main()
