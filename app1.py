import os
import asyncio
import fitz  # PyMuPDF
import streamlit as st
from typing import List, Optional, Dict
from dotenv import load_dotenv
from functools import lru_cache

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ---- Providers ----
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ----------------- Setup -----------------
load_dotenv()

PDF_DIR = "pdfs"
FAISS_INDEX_DIR = "faiss_index"  # we'll suffix with provider+embedding to avoid clashes
os.makedirs(PDF_DIR, exist_ok=True)

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ----------------- Provider registry -----------------
DEFAULT_MODELS: Dict[str, Dict[str, str]] = {
    "gemini":     {"chat": "gemini-1.5-flash",           "embed": "models/embedding-001"},
    "openai":     {"chat": "gpt-4o-mini",                "embed": "text-embedding-3-large"},
    "anthropic":  {"chat": "claude-3-5-sonnet-latest",   "embed": ""},  # use OpenAI/HF for embeds if you want
    "groq":       {"chat": "llama-3.1-70b-versatile",    "embed": ""},  # same note
    "mistral":    {"chat": "mistral-large-latest",       "embed": ""},  # same note
    "ollama":     {"chat": "llama3.1",                   "embed": "nomic-embed-text"},
    # add more (azure_openai, together, etc.) as needed
}

# ---- UI: let the user choose provider/model ----
st.set_page_config(page_title="👨‍👩‍👧‍👦 Family Law Legal Bot", layout="centered")
with st.sidebar:
    st.header("⚙️ Model Settings")

    provider = st.selectbox(
        "Provider",
        list(DEFAULT_MODELS.keys()),
        index=list(DEFAULT_MODELS.keys()).index(os.getenv("PROVIDER", "gemini"))
        if os.getenv("PROVIDER", "gemini") in DEFAULT_MODELS else 0
    )

    chat_model = st.text_input("Chat model", DEFAULT_MODELS[provider]["chat"])
    embed_model = st.text_input(
        "Embedding model (leave blank to reuse provider default / OpenAI TE3)",
        DEFAULT_MODELS[provider].get("embed", "")
    )

    k_retrieval = st.number_input("Top-K retrieved chunks", min_value=1, max_value=20, value=5)

# ----------------- LLM / Embedding factories -----------------
@lru_cache(maxsize=32)
def get_llm(provider: str, model: str):
    if provider == "gemini":
        return ChatGoogleGenerativeAI(model=model, google_api_key=os.getenv("GEMINI_API_KEY"))
    if provider == "openai":
        return ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))
    if provider == "anthropic":
        return ChatAnthropic(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"))
    if provider == "groq":
        return ChatGroq(model=model, api_key=os.getenv("GROQ_API_KEY"))
    if provider == "mistral":
        return ChatMistralAI(model=model, api_key=os.getenv("MISTRAL_API_KEY"))
    if provider == "ollama":
        # local
        return ChatOllama(model=model)

    raise ValueError(f"Unsupported provider: {provider}")

@lru_cache(maxsize=32)
def get_embeddings(provider: str, embed_model: Optional[str]):
    # Sensible fallbacks: if provider doesn't have embedders, use OpenAI (if key present) else Ollama else Gemini
    if provider == "gemini":
        model = embed_model or DEFAULT_MODELS["gemini"]["embed"]
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=os.getenv("GEMINI_API_KEY"))

    if provider == "openai" or (not embed_model and os.getenv("OPENAI_API_KEY")):
        model = embed_model or "text-embedding-3-large"
        return OpenAIEmbeddings(model=model, api_key=os.getenv("OPENAI_API_KEY"))

    if provider == "ollama":
        model = embed_model or DEFAULT_MODELS["ollama"]["embed"]
        return OllamaEmbeddings(model=model)

    # If you want HF embeddings as a generic fallback:
    # from langchain_community.embeddings import HuggingFaceEmbeddings
    # return HuggingFaceEmbeddings(model_name=embed_model or "BAAI/bge-base-en-v1.5")

    # Default to OpenAI if available, else raise:
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY"))

    raise ValueError("No compatible embedding backend configured. Provide OPENAI_API_KEY or choose gemini/ollama with embed model.")

# ----------------- PDF Loader -----------------
def load_documents() -> List[Document]:
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
def setup_retriever(embedding_model, provider: str, embed_model: str, k: int):
    docs = load_documents()
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # keep one FAISS index per embedding model to avoid incompatibility
    index_dir = f"{FAISS_INDEX_DIR}_{provider}_{(embed_model or 'default').replace('/', '_')}"
    if os.path.exists(index_dir):
        db = FAISS.load_local(
            index_dir, embedding_model, allow_dangerous_deserialization=True
        )
        # Optionally: add new docs
        db.add_documents(split_docs)
        db.save_local(index_dir)
    else:
        db = FAISS.from_documents(split_docs, embedding_model)
        db.save_local(index_dir)

    return db.as_retriever(search_kwargs={"k": k})

# ----------------- Helpers -----------------
SYSTEM_PROMPT = """You are a senior Indian family law advocate. Your expertise includes:
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
- Keep continuity across turns using conversation history.
"""

def build_messages(system: str, summary: str, history: List, user_prompt: str):
    msgs = [SystemMessage(content=system)]
    if summary:
        msgs.append(SystemMessage(content=f"Conversation summary so far:\n{summary}"))
    msgs.extend(history)  # already HumanMessage / AIMessage
    msgs.append(HumanMessage(content=user_prompt))
    return msgs

def maybe_summarize_history(llm, threshold_msgs: int = 12):
    """Optional: Summarize long histories to keep token usage down."""
    if len(st.session_state.history) <= threshold_msgs:
        return
    convo_text = "\n".join(
        [("User: " + m.content) if isinstance(m, HumanMessage) else ("Assistant: " + m.content)
         for m in st.session_state.history if not isinstance(m, SystemMessage)]
    )
    summary_prompt = f"""Summarize the following conversation between a user and a family law lawyer.
Focus on the key facts, issues, and legal advice already given, so we can carry forward context concisely:

{st.session_state.summary}

New turns:
{convo_text}

Return only the updated concise summary."""
    summary_resp = llm.invoke([
        SystemMessage(content="You are a helpful summarizer."),
        HumanMessage(content=summary_prompt)
    ])
    st.session_state.summary = summary_resp.content
    st.session_state.history = st.session_state.history[-4:]  # keep last 2 user-assistant pairs

# ----------------- Streamlit UI -----------------
st.title("👨‍👩‍👧‍👦 Family Law Legal Bot (India)")
st.markdown("Ask your questions related to family issues like divorce, maintenance, custody, domestic violence, etc.")

# Upload PDF interface (optional)
uploaded_file = st.file_uploader("Upload legal documents or policies (PDF only)", type="pdf")
if uploaded_file:
    with open(os.path.join(PDF_DIR, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Uploaded {uploaded_file.name}. Refresh to use in retrieval.")

# ----------------- Memory init -----------------
if "history" not in st.session_state:
    st.session_state.history = []  # list[BaseMessage]
if "summary" not in st.session_state:
    st.session_state.summary = ""  # optional rolling summary of long chats

# Show history in UI
for msg in st.session_state.history:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    if isinstance(msg, SystemMessage):
        continue
    with st.chat_message(role):
        st.markdown(msg.content)

chat_input = st.chat_input("Type your family law question...")

# --------- Instantiate the chosen models ----------
llm = get_llm(provider, chat_model)
embedding_model = get_embeddings(provider, embed_model or None)

if chat_input:
    with st.chat_message("user"):
        st.markdown(chat_input)

    with st.spinner("⚖️ Thinking like a lawyer..."):
        retriever = setup_retriever(embedding_model, provider, embed_model or "", k_retrieval)
        context_text = ""
        retrieved_docs = []
        if retriever:
            retrieved_docs = retriever.invoke(chat_input)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 1) Legal reasoning / situation understanding
        reasoning_prompt = f"""
You are a senior Indian family law advocate. Based on the user query and (if any) retrieved legal context,
**understand and interpret the situation** clearly. Identify the **core legal issues, applicable statutes/sections**, and
any **procedural or jurisdictional considerations**.

### Context (if any):
{context_text}

### User Query:
{chat_input}

Return a structured explanation with headings like:
- Facts (as understood)
- Legal Issues
- Applicable Law (with sections/Acts)
- Initial Opinion
"""
        reasoning = llm.invoke(build_messages(SYSTEM_PROMPT, st.session_state.summary, st.session_state.history, reasoning_prompt)).content

        # 2) Action plan
        action_plan_prompt = f"""
Based on the above legal reasoning, provide a **step-by-step practical action plan** for the user.
Consider Indian family law procedures (e.g., where to file, which forms/petitions, evidence, interim relief, etc.).

Your output must be numbered steps and clearly mention **which law/section** each step relies on (if relevant).

### Context (if any):
{context_text}

### User Query:
{chat_input}
"""
        action_plan = llm.invoke(build_messages(SYSTEM_PROMPT, st.session_state.summary, st.session_state.history, action_plan_prompt)).content

        # 3) Sample draft petition/notice/representation
        letter_prompt = f"""
Draft a **sample legal document** (petition / application / notice / representation) that best fits the user's issue.
Use formal Indian legal drafting style.

Include (as applicable):
- Court/Authority
- Parties (Petitioner/Applicant vs Respondent)
- Subject
- Facts
- Grounds / Legal Provisions
- Prayer / Reliefs sought
- Verification / Declaration
- Place / Date / Signature

### Context (if any):
{context_text}

### User Query:
{chat_input}
"""
        draft_letter = llm.invoke(build_messages(SYSTEM_PROMPT, st.session_state.summary, st.session_state.history, letter_prompt)).content

        # 4) Additional advice / resources
        advice_prompt = f"""
Provide **additional practical advice** such as:
- Evidence to collect
- Timelines/limitations (if any)
- Which court/forum to approach
- Pro-bono or legal aid options
- Any government portals / helplines (only if you're confident, else say to verify with official sources)

Be concise and actionable.

### Context (if any):
{context_text}

### User Query:
{chat_input}
"""
        extra_advice = llm.invoke(build_messages(SYSTEM_PROMPT, st.session_state.summary, st.session_state.history, advice_prompt)).content

        # Compose final response
        response = f"""
### 1) Situation Understanding & Legal Reasoning
{reasoning}

---

### 2) Step-by-Step Action Plan
{action_plan}

---

### 3) Sample Draft (Petition / Notice / Application)
{draft_letter}

---

### 4) Additional Advice / Things To Keep In Mind
{extra_advice}
"""

    # Update memory
    st.session_state.history.append(HumanMessage(content=chat_input))
    st.session_state.history.append(AIMessage(content=response))

    with st.chat_message("assistant"):
        st.markdown(response)

        # with st.expander("Source Documents (retrieved)"):
        #     if not retrieved_docs:
        #         st.warning("No source documents retrieved.")
        #     for doc in retrieved_docs:
        #         st.markdown(f"- **Source:** {doc.metadata.get('source')}")

    # Optional: summarize long chats
    maybe_summarize_history(llm)
