import os
import re
import json
import time
import random
import tempfile

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# ── Load secrets ──────────────────────────────────────────────────────────────
load_dotenv()

def get_secret(key: str) -> str:
    """Pull a key from st.secrets (cloud) or .env (local), in that order."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  
    return os.getenv(key, "")

GROQ_API_KEY   = get_secret("GROQ_API_KEY")
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"]   = GROQ_API_KEY
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ── Modern LangChain imports ──────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Mind · RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
  --bg:       #0b0e14;
  --surface:  #13171f;
  --border:   #1e2530;
  --accent:   #00e5a0;
  --accent2:  #7b61ff;
  --text:     #e2e8f0;
  --muted:    #64748b;
  --danger:   #ff4d6d;
  --warn:     #ffb347;
}

html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Syne', sans-serif;
}

.dm-header {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 8px;
}
.dm-header h1 {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: -1px;
  color: var(--accent);
  margin: 0;
  font-family: 'Syne', sans-serif;
}

.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-family: 'Space Mono', monospace;
  font-weight: 700;
  letter-spacing: .5px;
}
.badge-ok   { background: #00e5a018; color: var(--accent); border: 1px solid var(--accent); }
.badge-off  { background: #ff4d6d18; color: var(--danger); border: 1px solid var(--danger); }
.badge-warn { background: #ffb34718; color: var(--warn);   border: 1px solid var(--warn); }

.bubble {
  padding: 12px 16px;
  border-radius: 12px;
  margin: 6px 0;
  font-size: 0.93rem;
  line-height: 1.6;
  max-width: 88%;
}
.bubble-user {
  background: #1e2d3d;
  border: 1px solid #2a3d54;
  margin-left: auto;
  text-align: right;
}
.bubble-ai {
  background: var(--surface);
  border: 1px solid var(--border);
}
.bubble-ai .answer-text { margin-bottom: 8px; }
.source-chip {
  display: inline-block;
  background: #00e5a010;
  border: 1px solid #00e5a040;
  color: var(--accent);
  border-radius: 6px;
  padding: 2px 8px;
  font-size: 0.7rem;
  font-family: 'Space Mono', monospace;
  margin: 2px 3px 0 0;
}

.quiz-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 24px 28px;
  margin-bottom: 16px;
}
.quiz-card h3 {
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0 0 18px 0;
  line-height: 1.5;
}

.stRadio label {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px;
  margin: 4px 0;
  cursor: pointer;
  transition: border-color .15s, background .15s;
  font-family: 'Syne', sans-serif;
}
.stRadio label:hover {
  border-color: var(--accent);
  background: #00e5a00a;
}

.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }
.metric-card {
  flex: 1;
  min-width: 120px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  text-align: center;
}
.metric-card .val {
  font-size: 1.8rem;
  font-weight: 800;
  color: var(--accent);
  font-family: 'Space Mono', monospace;
}
.metric-card .lbl {
  font-size: 0.72rem;
  color: var(--muted);
  letter-spacing: .5px;
  text-transform: uppercase;
  margin-top: 2px;
}

.stTextInput input {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
}
.stButton > button {
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  border-radius: 8px !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  color: #0b0e14 !important;
  border: none !important;
}
.stButton > button[kind="secondary"] {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
}

section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
hr { border-color: var(--border) !important; }
[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 10px !important;
}
.stTabs [data-baseweb="tab"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--muted) !important;
}
.stTabs [aria-selected="true"] {
  background: #00e5a015 !important;
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "vectorstore":   None,  
        "qa_chain":      None,  
        "chat_history":  [],    # For UI display
        "raw_history":   [],    # Format required by modern LangChain
        "loaded_pdfs":   [],    
        "quiz_q":        None,  
        "quiz_opts":     [],    
        "quiz_correct":  "",    
        "quiz_answered": False,
        "quiz_score":    {"correct": 0, "total": 0},
        "quiz_feedback": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
S = st.session_state  

# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _build_chain():
    """Build LCEL retrieval chain to match modern standards."""
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
    retriever = S["vectorstore"].as_retriever(search_kwargs={"k": 4})
    
    # 1. Setup History-Aware Context Generator
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, formulate a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)
    
    # 2. Setup QA System Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based ONLY on the context below:\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 3. Final Combined Chain
    S["qa_chain"] = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def process_pdfs(files) -> str:
    if not files:
        return "error:No files provided."
    if not GROQ_API_KEY or not GOOGLE_API_KEY:
        return "error:API keys missing."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_chunks = []
    names = []

    progress = st.progress(0, text="Reading PDFs…")
    for i, f in enumerate(files):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            pages  = loader.load()          
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
            names.append(f.name)
            os.unlink(tmp_path)
        except Exception as e:
            return f"error:Failed on {f.name}: {e}"
        progress.progress((i + 1) / len(files), text=f"Processed: {f.name}")

    progress.progress(1.0, text="Building vector index…")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        S["vectorstore"] = FAISS.from_documents(documents=all_chunks, embedding=embeddings)
        S["loaded_pdfs"] = names
        S["chat_history"] = []
        S["raw_history"] = []
        _build_chain()
        progress.empty()
        return f"ok:{len(names)} PDF(s) · {len(all_chunks)} chunks indexed"
    except Exception as e:
        progress.empty()
        return f"error:{e}"


def ask(question: str):
    if not S["qa_chain"]:
        return "Please upload and initialize a PDF first.", []

    # Run query through LCEL engine
    result = S["qa_chain"].invoke({
        "input": question,
        "chat_history": S["raw_history"]
    })
    
    answer = result.get("answer", "")
    
    # Format source nodes
    sources = []
    for doc in result.get("context", []):
        meta = doc.metadata or {}
        sources.append({
            "file":    os.path.basename(meta.get("source", "unknown")),
            "page":    meta.get("page", "?"),
            "snippet": doc.page_content[:180].replace("\n", " "),
        })

    seen = set()
    unique_sources = []
    for s in sources:
        key = (s["file"], s["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    # Maintain runtime chat memory state
    S["raw_history"].append(("human", question))
    S["raw_history"].append(("ai", answer))
    
    # Trim runtime buffer window locally (keep last 6 transactions)
    if len(S["raw_history"]) > 12:
        S["raw_history"] = S["raw_history"][-12:]

    return answer, unique_sources


def get_quiz_question():
    if not S["vectorstore"] or not S["qa_chain"]:
        return False, "Upload a PDF first."

    all_data  = list(S["vectorstore"].docstore._dict.values())
    documents = [doc.page_content for doc in all_data]
    if not documents:
        return False, "Vector store is empty."

    text = documents[random.randint(0, len(documents) - 1)]

    prompt = f"""CONTEXT: {text}

Based ONLY on the CONTEXT above, generate ONE multiple-choice question.
Return ONLY a JSON object (no markdown, no extra text):
{{"question": "...", "correct": "...", "wrong": ["w1", "w2", "w3"]}}"""

    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
    try:
        raw = llm.invoke(prompt).content
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return False, "Model returned unexpected format. Try again."
        data    = json.loads(match.group())
        options = [data["correct"]] + data["wrong"][:3]
        random.shuffle(options)

        letters = ["A", "B", "C", "D"]
        S["quiz_q"]        = data["question"]
        S["quiz_opts"]     = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
        S["quiz_correct"]  = letters[options.index(data["correct"])]
        S["quiz_answered"] = False
        return True, ""
    except Exception as e:
        return False, f"Parse error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# UI — SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class='dm-header'>
      <h1>⚡ Digital Mind</h1>
    </div>
    <p style='color:#64748b; font-size:.8rem; margin-top:-6px; font-family:Space Mono,monospace;'>
      RAG · v2.0 · senior edition
    </p>
    """, unsafe_allow_html=True)

    keys_ok = bool(GROQ_API_KEY and GOOGLE_API_KEY)
    if keys_ok:
        st.markdown("<span class='badge badge-ok'>🔑 API Keys Loaded</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge badge-off'>⚠ API Keys Missing</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📄 Load Documents")
    uploaded = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("⚡ Initialize Engine", use_container_width=True, type="primary"):
        if not uploaded:
            st.warning("Please upload at least one PDF.")
        elif not keys_ok:
            st.error("API keys are not set. See above.")
        else:
            with st.spinner("Ingesting PDFs…"):
                result = process_pdfs(uploaded)
            if result.startswith("ok:"):
                st.success(f"✅ {result[3:]}")
            else:
                st.error(result[6:])

    if S["loaded_pdfs"]:
        st.markdown("**Loaded:**")
        for name in S["loaded_pdfs"]:
            st.markdown(f"<span class='badge badge-ok'>📄 {name}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔧 Engine Status")
    ready = S["qa_chain"] is not None
    status_badge = "badge-ok" if ready else "badge-off"
    status_text  = "ONLINE" if ready else "OFFLINE"
    st.markdown(f"<span class='badge {status_badge}'>{status_text}</span>", unsafe_allow_html=True)

    if S["vectorstore"]:
        n_chunks = len(S["vectorstore"].docstore._dict)
        st.markdown(f"<span class='badge badge-warn'>📦 {n_chunks} chunks</span>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI — MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
tab_chat, tab_quiz = st.tabs(["💬 Chat", "🎮 Quiz"])

# ─── TAB 1: CHAT ─────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown("#### Conversational RAG with Source Citations")
    st.caption("The AI remembers the last 6 exchanges. Sources are shown beneath each answer.")

    chat_container = st.container()
    with chat_container:
        for role, text, sources in S["chat_history"]:
            if role == "user":
                st.markdown(f"<div class='bubble bubble-user'>🧑 {text}</div>", unsafe_allow_html=True)
            else:
                chips = "".join(
                    f"<span class='source-chip' title='{s['snippet']}'>📄 {s['file']} · p{s['page']}</span>"
                    for s in sources
                )
                st.markdown(
                    f"""<div class='bubble bubble-ai'>
                        <div class='answer-text'>⚡ {text}</div>
                        {chips}
                    </div>""",
                    unsafe_allow_html=True,
                )

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        user_q = st.text_input(
            "Ask anything about your documents",
            key="chat_input",
            label_visibility="collapsed",
            placeholder="Ask anything about your documents…",
        )
    with col_btn:
        send = st.button("Send →", type="primary", use_container_width=True)

    if send and user_q.strip():
        with st.spinner("Thinking…"):
            answer, sources = ask(user_q.strip())
        S["chat_history"].append(("user",      user_q.strip(), []))
        S["chat_history"].append(("assistant", answer,         sources))
        st.rerun()

    if S["chat_history"]:
        if st.button("🗑 Clear conversation", type="secondary"):
            S["chat_history"] = []
            S["raw_history"] = []
            st.rerun()


# ─── TAB 2: QUIZ ─────────────────────────────────────────────────────────────
with tab_quiz:
    sc = S["quiz_score"]
    col_score, col_btn_next = st.columns([3, 1])
    with col_score:
        acc = int(sc["correct"] / sc["total"] * 100) if sc["total"] else 0
        st.markdown(f"""
        <div class='metric-row'>
          <div class='metric-card'><div class='val'>{sc['correct']}</div><div class='lbl'>Correct</div></div>
          <div class='metric-card'><div class='val'>{sc['total']}</div><div class='lbl'>Total</div></div>
          <div class='metric-card'><div class='val'>{acc}%</div><div class='lbl'>Accuracy</div></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_btn_next:
        st.write("")
        if st.button("Next Question ▶", type="primary", use_container_width=True):
            S["quiz_feedback"] = None 
            with st.spinner("Generating question…"):
                ok, err = get_quiz_question()
            if not ok:
                st.error(err)
            st.rerun()

    if S["quiz_q"]:
        st.markdown(f"""
        <div class='quiz-card'>
          <h3>{S['quiz_q']}</h3>
        </div>
        """, unsafe_allow_html=True)

        choice = st.radio("Choose an answer:", S["quiz_opts"],
                          index=None, key="quiz_radio", label_visibility="collapsed", 
                          disabled=S.get("quiz_answered", False))

        col_check, _ = st.columns([1, 3])
        with col_check:
            if st.button("✅ Check Answer", type="primary", disabled=S.get("quiz_answered", False)):
                if not choice:
                    st.warning("Please select an option first.")
                else:
                    chosen_letter = choice[0]
                    S["quiz_answered"] = True
                    S["quiz_score"]["total"] += 1
                    
                    correct_full_text = next((opt for opt in S["quiz_opts"] if opt.startswith(S["quiz_correct"])), S["quiz_correct"])
                    
                    if chosen_letter == S["quiz_correct"]:
                        S["quiz_score"]["correct"] += 1
                        S["quiz_feedback"] = {"success": True, "msg": "🎉 Correct! Great job!"}
                    else:
                        S["quiz_feedback"] = {"success": False, "msg": f"❌ Incorrect. The correct answer is: **{correct_full_text}**"}
                    
                    st.rerun()
        
        if "quiz_feedback" in S and S["quiz_feedback"]:
            if S["quiz_feedback"]["success"]:
                st.success(S["quiz_feedback"]["msg"])
            else:
                st.error(S["quiz_feedback"]["msg"])
                
    else:
        st.info("Click **Next Question** to begin the quiz.")