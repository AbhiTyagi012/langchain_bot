import os
import asyncio
import fitz  # PyMuPDF
import streamlit as st
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# ----------------- Setup -----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

PDF_DIR = "pdfs"
FAISS_INDEX_PATH = "faiss_index"
os.makedirs(PDF_DIR, exist_ok=True)

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GOOGLE_API_KEY
)

# ----------------- PDF Loader -----------------
def load_documents():
    docs = []
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            try:
                with fitz.open(os.path.join(PDF_DIR, filename)) as pdf:
                    text = "".join([page.get_text() for page in pdf])
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": filename}))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return docs

# ----------------- Vector DB Setup -----------------
def setup_retriever():
    docs = load_documents()
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    if os.path.exists(FAISS_INDEX_PATH):
        db = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
        db.add_documents(split_docs)
        db.save_local(FAISS_INDEX_PATH)
    else:
        db = FAISS.from_documents(split_docs, embedding_model)
        db.save_local(FAISS_INDEX_PATH)

    return db.as_retriever(search_kwargs={"k": 5})

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="👨‍👩‍👧‍👦 Family Law Legal Bot", layout="centered")
st.title("👨‍👩‍👧‍👦 Family Law Legal Bot (India)")
st.markdown("Ask your questions related to family issues like divorce, maintenance, custody, domestic violence, etc.")

# Upload PDF interface (optional)
uploaded_file = st.file_uploader("Upload legal documents or policies (PDF only)", type="pdf")
if uploaded_file:
    with open(os.path.join(PDF_DIR, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Uploaded {uploaded_file.name}. Refresh to use in retrieval.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

chat_input = st.chat_input("Type your family law question...")

if chat_input:
    st.session_state.chat_history.append({"role": "user", "content": chat_input})
    with st.chat_message("user"):
        st.markdown(chat_input)

    with st.spinner("⚖️ Thinking like a lawyer..."):
        print()
        retriever = setup_retriever()
        context = ""
        if retriever:
            docs = retriever.invoke(chat_input)
            context = "\n\n".join([doc.page_content for doc in docs])
            print(context)

        # Updated System Prompt
        system_prompt = """You are a senior Indian family law advocate. Your expertise includes:
- Divorce laws (Hindu, Muslim, Christian, etc.)
- Child custody
- Domestic violence protection (e.g. under PWDVA)
- Maintenance and alimony (Section 125 CrPC, etc.)
- Property rights in family disputes
- Marriage legality, guardianship, and adoption

You must:
- Respond like a real lawyer.
- Use formal yet simple Indian legal language.
- Reference relevant acts or sections where possible.
- Ask clarifying questions if user input is vague.
- Generate a legal letter if the user requests one.

Answer based on context if provided; otherwise, use your legal expertise."""

        full_prompt = f"{system_prompt}\n\nContext from uploaded documents:\n{context}\n\nUser Query:\n{chat_input}\n\nAnswer:"
        response = llm.invoke(full_prompt).content

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
