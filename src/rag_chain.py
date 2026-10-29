from src.prompt import VIETNAMESE_PROMPT

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain(retriever,llm):




    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | VIETNAMESE_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain