import os
import asyncio
import fitz
import streamlit as st
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
PDF_DIR = "pdfs"
FAISS_INDEX_PATH = "faiss_index"
os.makedirs(PDF_DIR, exist_ok=True)

# LLM + Embedding Models
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Load documents from PDF
def load_documents():
    docs = []
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            with fitz.open(os.path.join(PDF_DIR, filename)) as pdf:
                text = "".join(page.get_text() for page in pdf)
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": filename}))
    return docs

# Build or load vector DB
def setup_retriever():
    docs = load_documents()
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    if os.path.exists(FAISS_INDEX_PATH):
        db = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
        db.add_documents(split_docs)
    else:
        db = FAISS.from_documents(split_docs, embedding_model)
    db.save_local(FAISS_INDEX_PATH)
    return db.as_retriever(search_kwargs={"k": 5})

retriever = setup_retriever()

# LangGraph state
class GraphState(TypedDict):
    question: str
    docs: List[Document]
    answer: str

# LangGraph Nodes
def retrieve_node(state):
    if retriever:
        docs = retriever.invoke(state["question"])
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"question": state["question"], "docs": docs, "answer": context}
    return {"question": state["question"], "docs": [], "answer": "No documents found."}

def generate_node(state):
    context = "\n\n".join(doc.page_content for doc in state["docs"])
    system_prompt = """
You are a senior Indian family law advocate.
If context is provided, use it to answer the user question clearly.
Do not hallucinate legal facts. Refer to laws such as the Hindu Marriage Act, Guardians and Wards Act, etc., when relevant.
"""
    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{state['question']}"
    response = llm.invoke(prompt).content
    return {"answer": response}

def llm_only_node(state):
    prompt = f"You are a family lawyer. Answer this: {state['question']}"
    response = llm.invoke(prompt).content
    return {"answer": response}

# Build LangGraph
def build_graph(mode: str):
    graph = StateGraph(GraphState)
    if mode == "RAG + LLM":
        graph.add_node("retrieve", RunnableLambda(retrieve_node))
        graph.add_node("generate", RunnableLambda(generate_node))
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
    elif mode == "RAG only":
        graph.add_node("retrieve", RunnableLambda(retrieve_node))
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", END)
    else:  # LLM only
        graph.add_node("llm_only", RunnableLambda(llm_only_node))
        graph.set_entry_point("llm_only")
        graph.add_edge("llm_only", END)

    return graph.compile()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config("👩‍⚖️ Family Law Legal Bot")
st.title("👩‍⚖️ Family Law Legal Assistant")

mode = st.radio("Choose Mode:", ["LLM only", "RAG only", "RAG + LLM"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask your family law question...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("🔍 Thinking..."):
        graph = build_graph(mode)
        result = graph.invoke({"question": user_input})
        answer = result.get("answer") or "Relevant documents retrieved."
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
