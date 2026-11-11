"""Kiểm tra filter trước retrieval, không cần tải model."""

import copy
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings, FakeEmbeddings

from src.retriever import create_retriever
from src.vectorstore import create_vectorstore, get_documents_metadata
from test_retrieval import SampleScorer


class FixedEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.0, 1.0] if "xa" in text else [1.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class MetadataFilterTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document(page_content="Gần câu hỏi nhưng thuộc B", metadata={"file_name": "B.pdf", "chunk_index": i})
            for i in range(25)
        ] + [
            Document(page_content="Đoạn xa thuộc A", metadata={
                "file_name": "A.pdf", "file_type": "pdf", "uploaded_at": "2026-11-11", "chunk_index": i,
            }) for i in range(3)
        ]
        self.store = create_vectorstore(self.docs, FixedEmbeddings())

    def test_filter_before_top_k_in_all_modes(self):
        before = copy.deepcopy(self.docs)
        with patch("src.reranker.get_reranker", return_value=SampleScorer()):
            for hybrid in (False, True):
                for rerank in (False, True):
                    with self.subTest(hybrid=hybrid, rerank=rerank):
                        retriever = create_retriever(self.store, use_hybrid=hybrid, use_reranker=rerank, selected_source="A.pdf")
                        results = retriever.invoke("Gần câu hỏi nhưng thuộc B")
                        self.assertEqual(len(results), 3)
                        self.assertTrue(all(doc.metadata["file_name"] == "A.pdf" for doc in results))
                        self.assertEqual({doc.metadata["chunk_index"] for doc in results}, {0, 1, 2})
                        self.assertTrue(all("_retrieval_id" not in doc.metadata for doc in results))
        self.assertEqual(self.docs, before)

    def test_no_match_never_falls_back_or_loads_model(self):
        with patch("src.reranker.get_reranker") as load:
            results = create_retriever(self.store, selected_source="khong_co.pdf").invoke("câu hỏi")
            self.assertEqual(results, [])
            load.assert_not_called()
        self.assertEqual(create_retriever(None).invoke("câu hỏi"), [])

    def test_identical_text_in_different_files_remains_separate(self):
        docs = [Document(page_content="Nội dung giống nhau", metadata={"file_name": name, "chunk_index": 0})
                for name in ("A.pdf", "B.pdf")]
        store = create_vectorstore(docs, FakeEmbeddings(size=8))
        results = create_retriever(store, selected_source="Tất cả tài liệu", use_reranker=False).invoke("nội dung")
        self.assertEqual(len(results), 2)
        self.assertEqual({doc.metadata["file_name"] for doc in results}, {"A.pdf", "B.pdf"})
        self.assertTrue(all("_retrieval_id" not in doc.metadata for doc in results))

    def test_rebuild_retriever_sees_added_file(self):
        self.store.add_documents([Document(page_content="Môn Python", metadata={"file_name": "C.docx"})])
        results = create_retriever(self.store, selected_source="C.docx", use_reranker=False).invoke("Python")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata, {"file_name": "C.docx"})

    def test_metadata_summary_from_danh(self):
        summary = get_documents_metadata(self.store)
        a = next(item for item in summary if item["file_name"] == "A.pdf")
        self.assertEqual(a["file_type"], "PDF")
        self.assertEqual(a["uploaded_at"], "2026-11-11")
        self.assertEqual(a["chunks_count"], 3)


if __name__ == "__main__":
    unittest.main()
