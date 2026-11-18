"""Conversational advanced RAG with query planning and answer self-checking."""

import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from src.prompt import VIETNAMESE_PROMPT


QUESTION_REWRITE_PROMPT = PromptTemplate.from_template("""
Dựa vào lịch sử hội thoại, viết lại câu hỏi thành câu độc lập, rõ ràng để tìm trong tài liệu.
Giữ nguyên ngôn ngữ và ý định. Nếu câu đã rõ, giữ nguyên. Chỉ trả về câu hỏi.

Lịch sử:
{chat_history}
Câu hỏi: {question}
Câu hỏi tìm kiếm:
""")

HOP_PROMPT = PromptTemplate.from_template("""
Phân tích câu hỏi để tìm thông tin trong tài liệu. Tách thành tối đa 3 truy vấn con,
mỗi truy vấn tìm một dữ kiện cần thiết cho câu trả lời. Nếu chỉ cần một lượt tìm,
trả về đúng một truy vấn. Mỗi truy vấn trên một dòng, không đánh số.

Câu hỏi: {question}
Truy vấn con:
""")

EVALUATION_PROMPT = PromptTemplate.from_template("""
Bạn là bộ tự kiểm tra câu trả lời RAG. Chỉ dùng Context để đánh giá câu trả lời.
Chấm support_score từ 0 đến 100 theo mức độ các khẳng định được Context hỗ trợ.
Nếu điểm dưới 75, ghi rõ phần thiếu hoặc không được hỗ trợ. Trả về duy nhất JSON:
{{"support_score": 0, "feedback": "..."}}

Câu hỏi: {question}
Context: {context}
Câu trả lời: {answer}
""")


def _history_text(chat_history):
    if isinstance(chat_history, str):
        return chat_history.strip() or "Chưa có lịch sử hội thoại."
    lines = []
    for message in (chat_history or [])[-15:]:
        role = "Người dùng" if message.get("role") == "user" else "Trợ lý"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "Chưa có lịch sử hội thoại."


def _parse_evaluation(raw):
    """Parse evaluator JSON, with a conservative fallback for malformed output."""
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        return 50, "Không phân tích được kết quả tự đánh giá."
    try:
        result = json.loads(match.group(0))
        score = max(0, min(100, int(result.get("support_score", 50))))
        return score, str(result.get("feedback", ""))
    except (ValueError, TypeError, json.JSONDecodeError):
        return 50, "Không phân tích được kết quả tự đánh giá."


def create_rag_chain(retriever, llm):
    """Runnable input: {question, chat_history}; output includes support confidence."""
    parser = StrOutputParser()
    rewrite_chain = QUESTION_REWRITE_PROMPT | llm | parser
    hop_chain = HOP_PROMPT | llm | parser
    answer_chain = VIETNAMESE_PROMPT | llm | parser
    evaluation_chain = EVALUATION_PROMPT | llm | parser

    def run(inputs):
        if isinstance(inputs, str):
            question, chat_history = inputs, []
        else:
            question = inputs["question"]
            chat_history = inputs.get("chat_history", [])

        history = _history_text(chat_history)
        rewritten = rewrite_chain.invoke({"chat_history": history, "question": question}).strip()
        search_question = rewritten or question

        # Multi-hop: decompose a complex query and gather evidence for each hop.
        raw_hops = hop_chain.invoke({"question": search_question})
        hops = [line.strip(" -*\t") for line in raw_hops.splitlines() if line.strip()]
        queries = list(dict.fromkeys([search_question] + hops[:3]))
        documents, seen = [], set()
        for query in queries:
            for doc in retriever.invoke(query):
                key = (doc.metadata.get("file_name"), doc.metadata.get("page"), doc.page_content)
                if key not in seen:
                    seen.add(key)
                    documents.append(doc)

        context = "\n\n".join(doc.page_content for doc in documents)
        answer = answer_chain.invoke({
            "context": context,
            "question": question,
            "chat_history": history,
        }).strip()

        # Self-RAG: judge grounding, then make one evidence-constrained correction if needed.
        score, feedback = _parse_evaluation(evaluation_chain.invoke({
            "question": question, "context": context, "answer": answer,
        }))
        if score < 75:
            correction_prompt = PromptTemplate.from_template("""
Sửa câu trả lời bên dưới theo phản hồi tự kiểm tra. Chỉ đưa ra khẳng định được Context hỗ trợ.
Nếu Context không đủ, hãy nói rõ không tìm thấy thông tin. Trả lời bằng tiếng Việt.

Câu hỏi: {question}
Context: {context}
Câu trả lời trước: {answer}
Phản hồi kiểm tra: {feedback}
Câu trả lời đã sửa:
""")
            answer = (correction_prompt | llm | parser).invoke({
                "question": question, "context": context, "answer": answer,
                "feedback": feedback,
            }).strip()
            score, _ = _parse_evaluation(evaluation_chain.invoke({
                "question": question, "context": context, "answer": answer,
            }))

        return f"{answer}\n\n**Độ tin cậy theo mức hỗ trợ tài liệu: {score}%**"

    return RunnableLambda(run)
