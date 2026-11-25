from langchain_core.output_parsers import StrOutputParser
from src.prompt import CONDENSE_PROMPT
def build_history(messages, max_turns=4):
    history = []

    for m in messages[-max_turns * 2:]:
        role = "User" if m["role"] == "user" else "Assistant"
        history.append(f"{role}: {m['content']}")

    return "\n".join(history)

def rewrite_with_history(question, messages, llm):
    # Sao chép lịch sử để không thay đổi dữ liệu gốc
    messages = messages.copy()

    # Loại câu hỏi hiện tại nếu đã được thêm vào lịch sử
    if (
        messages
        and messages[-1]["role"] == "user"
        and messages[-1]["content"] == question
    ):
        messages.pop()

    history = build_history(messages)

    # Không có lịch sử thì giữ nguyên câu hỏi
    if not history:
        return question

    # Dùng LLM để viết lại câu hỏi
    chain = CONDENSE_PROMPT | llm | StrOutputParser()

    rewritten_question = chain.invoke({
        "history": history,
        "question": question
    }).strip()

    print("Original:", question)
    print("Rewritten:", rewritten_question)
    return rewritten_question or question
