"""Kiểm tra upload nhiều file, đổi filter và hiển thị nguồn; không gọi model thật."""

import io
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import FakeEmbeddings
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest
import streamlit as st

from test_retrieval import SampleScorer


def sample_docx(name, text):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>' + text + '</w:t></w:r></w:p></w:body></w:document>')
    data.name = name
    return data


class AppMetadataTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()  # Mỗi bài thử dùng model giả riêng.

    def tearDown(self):
        st.cache_resource.clear()

    def test_empty_result_does_not_call_llm(self):
        def forbidden(prompt):
            raise AssertionError("Không được gọi LLM khi không có tài liệu khớp")

        with patch("src.embedder.get_embedder", return_value=FakeEmbeddings(size=8)), patch(
            "src.llm.get_llm", return_value=RunnableLambda(forbidden)
        ), patch("src.retriever.create_retriever", return_value=RunnableLambda(lambda question: [])), patch(
            "streamlit.file_uploader", return_value=[]
        ):
            app = AppTest.from_file(str(Path("app.py").resolve()), default_timeout=30)
            app.session_state["uploaded_files_list"] = ["A.docx"]
            app.session_state["messages"] = [{"role": "user", "content": "Học phí bao nhiêu?"}]
            app.session_state["is_thinking"] = True
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["messages"][-1]["sources"], [])
            self.assertIn("Không tìm thấy", app.session_state["messages"][-1]["content"])

    def test_multi_upload_filter_and_historical_sources(self):
        files = [sample_docx("A.docx", "Học phí là 5 triệu đồng."), sample_docx("B.docx", "Học phí là 9 triệu đồng.")]
        contexts = []

        def answer(prompt):
            context = prompt.to_string().split("Context:\n", 1)[1].split("Câu hỏi:", 1)[0]
            contexts.append(context)
            return "Trả lời thử từ tài liệu đã chọn."

        with patch("src.embedder.get_embedder", return_value=FakeEmbeddings(size=8)), patch(
            "src.llm.get_llm", return_value=RunnableLambda(answer)
        ), patch("src.reranker.get_reranker", return_value=SampleScorer()), patch(
            "streamlit.file_uploader", return_value=files
        ) as uploader:
            app = AppTest.from_file(str(Path("app.py").resolve()), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertTrue(uploader.call_args.kwargs["accept_multiple_files"])
            self.assertEqual(app.session_state["uploaded_files_list"], ["A.docx", "B.docx"])

            app.selectbox[0].select("A.docx").run()
            app.chat_input[0].set_value("Học phí bao nhiêu?").run()
            self.assertFalse(app.exception)
            self.assertIn("5 triệu", contexts[-1])
            self.assertNotIn("9 triệu", contexts[-1])
            self.assertEqual(app.session_state["messages"][-1]["sources"], ["A.docx"])
            self.assertTrue(any("Tài liệu dùng làm ngữ cảnh: A.docx" in item.value for item in app.caption))

            app.selectbox[0].select("B.docx").run()
            app.chat_input[0].set_value("Học phí bao nhiêu?").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(contexts), 2)  # Mỗi câu chỉ gọi chain một lần.
            self.assertIn("9 triệu", contexts[-1])
            self.assertNotIn("5 triệu", contexts[-1])
            answers = [message for message in app.session_state["messages"] if message["role"] == "assistant"]
            self.assertEqual([message["sources"] for message in answers], [["A.docx"], ["B.docx"]])


if __name__ == "__main__":
    unittest.main()
