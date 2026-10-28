from prompt import VIETNAMESE_PROMPT
from llm import get_llm
from retriever import create_retriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain(vectorstore):
    retriever = create_retriever(vectorstore)

    llm = get_llm()

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain