import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, get_retriever
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from core.config import validate_setting

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=validate_setting("GOOGLE_API_KEY"),
        temperature=0.2,
    )


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript: str):

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k=4)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert meeting assistant.

Answer the user's question ONLY using the transcript context below.

If answer is unavailable, say:
"I could not find this information in the transcript."

Context:
{context}
"""
        ),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    return rag_chain.invoke(question)