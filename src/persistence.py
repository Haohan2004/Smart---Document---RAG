import os
import json

from langchain_community.vectorstores import FAISS

HISTORY_FILE = "chat_history.json"
FAISS_DIR    = "faiss_store"
def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
         return json.load(f)
    return []

def save_chat_history(messages):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)


def save_vectorstore(session_id: int, vectorstore) -> None:
    if vectorstore is None:
        return
    folder = os.path.join(FAISS_DIR, f"chat_{session_id}")
    os.makedirs(folder, exist_ok=True)
    vectorstore.save_local(folder)

def load_vectorstore(session_id: int, embedder):
    from langchain_community.vectorstores import FAISS

    folder = os.path.join(FAISS_DIR, f"chat_{session_id}")

    if not os.path.isfile(os.path.join(folder, "index.faiss")):
        return None

    return FAISS.load_local(
        folder,
        embedder,
        allow_dangerous_deserialization=True
    )

