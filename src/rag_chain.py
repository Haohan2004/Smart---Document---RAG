import json
import os
import re

from langchain_core.prompts import PromptTemplate

from src.conversational import rewrite_with_history, build_history
from src.prompt import VIETNAMESE_PROMPT

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.logger import get_logger

logger = get_logger()
# 1. PROMPT ĐÁNH GIÁ CÂU TRẢ LỜI
EVALUATION_PROMPT = PromptTemplate.from_template("""
Bạn là bộ tự kiểm tra câu trả lời của hệ thống RAG.

Chỉ dựa vào Context để đánh giá câu trả lời.

Yêu cầu:
- Chấm support_score từ 0 đến 100.
- Kiểm tra các khẳng định có được Context hỗ trợ không.
- Kiểm tra câu trả lời có giải quyết đúng câu hỏi không.
- Nếu điểm dưới 75, ghi rõ thông tin thiếu hoặc không được hỗ trợ.

Chỉ trả về JSON hợp lệ theo cấu trúc:
{{"support_score": 0, "feedback": "..."}}

Câu hỏi:
{question}

Context:
{context}

Câu trả lời:
{answer}
""")
# 2. PROMPT SỬA CÂU TRẢ LỜI
CORRECTION_PROMPT = PromptTemplate.from_template("""
Bạn là trợ lý RAG đang sửa câu trả lời dựa trên kết quả tự đánh giá.

Yêu cầu:
- Sửa những nội dung không được Context hỗ trợ.
- Không tự bịa thông tin ngoài tài liệu.
- Nếu Context không đủ, hãy nói rõ thông tin còn thiếu.
- Trả lời bằng tiếng Việt.
- Chỉ trả về câu trả lời đã sửa.

Câu hỏi:
{question}

Context:
{context}

Câu trả lời trước:
{answer}

Phản hồi đánh giá:
{feedback}

Câu trả lời đã sửa:
""")
HOP_PROMPT = PromptTemplate.from_template("""
Phân tích câu hỏi để tìm thông tin trong tài liệu.

Tách thành tối đa 3 truy vấn con,
mỗi truy vấn tìm một dữ kiện cần thiết.

Dựa vào lịch sử hội thoại để hiểu ngữ cảnh câu hỏi.
Dựa vào phản hồi đánh giá để ưu tiên tìm kiếm
những thông tin còn thiếu hoặc chưa được hỗ trợ.

Nếu chỉ cần một lượt tìm, trả về đúng một truy vấn.

Mỗi truy vấn trên một dòng, không đánh số.
Lịch sử hội thoại:
{chat_history}
Câu hỏi: {question}
Phản hồi đánh giá:
{feedback}
Truy vấn con:
""")

def parse_evaluation(raw):
    """Lấy support_score và feedback từ JSON của LLM."""
    try:
        # Cho phép LLM trả thêm markdown trước JSON
        start = raw.find("{")

        if start == -1:
            raise ValueError("Không tìm thấy JSON.")

        result, _ = json.JSONDecoder().raw_decode(raw[start:])

        score = int(float(result["support_score"]))
        score = max(0, min(100, score))

        feedback = str(result.get("feedback", ""))

        return score, feedback

    except (ValueError, TypeError, KeyError, AttributeError):
        return None, "Không phân tích được kết quả tự đánh giá."

def create_rag_chain(retriever,llm,enable_self_rag=False):
    parser = StrOutputParser()
    answer_chain = (
            {
                "context": lambda data: format_docs(
                    data["source_documents"]
                ),
                "question": lambda data: data["question"],
                "chat_history": lambda data: build_history(
                    data["chat_history"]
                )
            }
            | VIETNAMESE_PROMPT
            | llm
            | StrOutputParser()
    )
    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs, start=1):
            metadata = doc.metadata or {}
            filename = metadata.get("file_name") or os.path.basename(
                str(metadata.get("source", "Không rõ"))
            )

            page = metadata.get("page_label")

            if page is None:
                page_index = metadata.get("page")
                page = (
                    page_index + 1
                    if isinstance(page_index, int)
                    else "Không rõ"
                )

            formatted.append(
                f"[Nguồn {i} | File: {filename} | Trang: {page}]\n"
                f"{doc.page_content}"
            )

        return "\n\n".join(formatted)

    rag_chain = (
            {
                 "question": lambda data: data["question"],
                "chat_history": lambda data: data.get("chat_history", ""),
                "source_documents": lambda data: retriever.invoke(
                    rewrite_with_history(
                        data["question"],
                        data.get("chat_history", []),
                        llm
                )
            )
            }
            | RunnablePassthrough.assign(
                answer=answer_chain

    )
    )
    # 6. NẾU SELF-RAG TẮT -> TRẢ NORMAL RAG
    if not enable_self_rag:
        return rag_chain

    # 7. NẾU SELF-RAG BẬT -> TẠO CHAIN ĐÁNH GIÁ
    evaluation_chain = EVALUATION_PROMPT | llm | parser
    correction_chain = CORRECTION_PROMPT | llm | parser
    hop_chain = HOP_PROMPT | llm | parser
    def evaluate_answer(question, context, answer):

        evaluation_result = evaluation_chain.invoke({
            "question": question,
            "context": context,
            "answer": answer
        })
        return parse_evaluation(evaluation_result)

    def document_key(doc):
        metadata = doc.metadata or {}

        return (
            metadata.get("file_name") or metadata.get("source"),
            metadata.get("page"),
            metadata.get("chunk_index"),
            doc.page_content
        )

    def self_rag_check(data):

        question = data["question"]
        chat_history = data.get("chat_history", [])
        answer = data["answer"]

        documents = list(data["source_documents"])
        context = format_docs(documents)

        attempts = 1  # Normal RAG đã tạo câu trả lời đầu tiên
        hops = []
        # Không tìm thấy tài liệu
        if not documents:
            return {
                **data,
                "answer": "Không tìm thấy thông tin liên quan trong tài liệu.",
                "source_documents": [],
                "support_score": None,
                "self_rag_feedback": "Không có tài liệu được truy xuất.",
                "self_rag_attempts": attempts,
                "self_rag_hops": hops
            }

        # BƯỚC 1: ĐÁNH GIÁ CÂU TRẢ LỜI NORMAL RAG
        score, feedback = evaluate_answer(
            question, context, answer
        )
        # BƯỚC 2: NẾU ĐIỂM THẤP -> MULTI-HOP RETRIEVAL
        if score is None or score < 75:

            raw_hops = hop_chain.invoke({
                "question": question,
                "chat_history": build_history(chat_history),
                "feedback": feedback
            })


            for line in raw_hops.splitlines():

                line = re.sub(
                    r"^\s*(?:[-*]|\d+[.)])\s*",
                    "",
                    line
                ).strip()

                if line and line not in hops:
                    hops.append(line)

            # Giới hạn tối đa 3 truy vấn con
            hops = hops[:3]
            logger.info(f"[Multi-hop] Decomposed steps: {hops}")
            seen = {
                document_key(doc)
                for doc in documents
            }

            added_documents = False

            # Truy xuất thêm thông tin
            for query in hops:

                retrieved_docs = retriever.invoke(query)

                for doc in retrieved_docs:

                    key = document_key(doc)

                    if key not in seen:
                        seen.add(key)
                        documents.append(doc)
                        added_documents = True

                    # Giới hạn tổng số chunks
                    if len(documents) >= 10:
                        break

                if len(documents) >= 10:
                    break

            # BƯỚC 3: TẠO LẠI CÂU TRẢ LỜI
            # Chỉ tạo lại khi tìm được tài liệu mới
            if added_documents:
                context = format_docs(documents)

                answer = answer_chain.invoke({
                    "question": question,
                    "chat_history": chat_history,
                    "source_documents": documents
                }).strip()
                attempts +=1
                # Đánh giá lại câu trả lời
                score, feedback = evaluate_answer(
                    question, context, answer
                )

        # BƯỚC 4: NẾU VẪN DƯỚI 75 -> SỬA CÂU TRẢ LỜI
        if score is None or score < 75:

            corrected_answer = correction_chain.invoke({
                "question": question,
                "context": context,
                "answer": answer,
                "feedback": feedback
            })

            if corrected_answer.strip():
                answer = corrected_answer.strip()
                attempts +=1
            # Đánh giá lại lần cuối
            score, feedback = evaluate_answer(
                question, context, answer
            )

        # BƯỚC 5: TRẢ KẾT QUẢ CUỐI CÙNG
        return {
            **data,
            "answer": answer,
            "source_documents": documents,
            "support_score": score,
            "self_rag_feedback": feedback,
            "self_rag_attempts": attempts,
            "self_rag_hops":hops
        }

        # Chạy NORMAL RAG trước, SELF-RAG sau

    return rag_chain | RunnableLambda(self_rag_check)

