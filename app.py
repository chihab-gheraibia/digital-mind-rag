
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
# Local dev:  reads from .env file
# Streamlit Cloud / HF Spaces: reads from st.secrets
load_dotenv()

def get_secret(key: str) -> str:
    """Pull a key from st.secrets (cloud) or .env (local), in that order."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  # No secrets.toml — that's fine locally, we'll use .env instead
    return os.getenv(key, "")

GROQ_API_KEY   = get_secret("GROQ_API_KEY")
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"]   = GROQ_API_KEY
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
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

/* ---------- global tokens ---------- */
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

/* ---- base overrides ---- */
html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Syne', sans-serif;
}

/* header */
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
.dm-header span {
  color: var(--muted);
  font-family: 'Space Mono', monospace;
  font-size: 0.78rem;
}

/* status badge */
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

/* chat bubbles */
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
  cursor: default;
}

/* quiz card */
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

/* option buttons */
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

/* metric cards */
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

/* streamlit widget fixes */
.stTextInput input, .stTextArea textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  font-family: 'Syne', sans-serif !important;
}
.stButton > button {
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  border-radius: 8px !important;
  transition: all .15s !important;
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
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 14px #00e5a030; }

/* sidebar */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}

/* divider */
hr { border-color: var(--border) !important; }

/* file uploader */
[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 10px !important;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  gap: 4px;
}
.stTabs [data-baseweb="tab"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--muted) !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
  background: #00e5a015 !important;
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}

/* spinner text */
.stSpinner > div { color: var(--accent) !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — one isolated object per browser session
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "vectorstore":   None,   # Chroma instance
        "memory":        None,   # ConversationBufferWindowMemory
        "qa_chain":      None,   # ConversationalRetrievalChain
        "chat_history":  [],     # [(role, text, sources)]
        "loaded_pdfs":   [],     # list of filenames ingested
        "quiz_q":        None,   # current quiz question text
        "quiz_opts":     [],     # shuffled option labels
        "quiz_correct":  "",     # correct letter
        "quiz_answered": False,
        "quiz_score":    {"correct": 0, "total": 0},
        "db_path":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
S = st.session_state   # short alias


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _build_chain():
    """(Re)build the QA chain from current vectorstore. Called after PDF load."""
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
    S["memory"] = ConversationBufferWindowMemory(
        k=6,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    S["qa_chain"] = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=S["vectorstore"].as_retriever(search_kwargs={"k": 4}),
        memory=S["memory"],
        return_source_documents=True,
        output_key="answer",
    )


def process_pdfs(files) -> str:
    """Ingest one or more PDFs into ChromaDB. Returns status string."""
    if not files:
        return "error:No files provided."
    if not GROQ_API_KEY or not GOOGLE_API_KEY:
        return "error:API keys missing. Add them to .env or Streamlit secrets."

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
            pages  = loader.load()          # no page cap
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
            names.append(f.name)
            os.unlink(tmp_path)
        except Exception as e:
            return f"error:Failed on {f.name}: {e}"
        progress.progress((i + 1) / len(files), text=f"Processed: {f.name}")

    progress.progress(1.0, text="Building vector index…")
    try:
        db_path = f"./chroma_db_{int(time.time())}"
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        S["vectorstore"] = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=db_path,
        )
        S["db_path"]     = db_path
        S["loaded_pdfs"] = names
        S["chat_history"] = []
        _build_chain()
        progress.empty()
        return f"ok:{len(names)} PDF(s) · {len(all_chunks)} chunks indexed"
    except Exception as e:
        progress.empty()
        return f"error:{e}"


def ask(question: str):
    """
    Run a question through the conversational QA chain.
    Returns (answer_text, source_list).
    source_list = list of dicts with keys: page, file, snippet
    """
    if not S["qa_chain"]:
        return "Please upload and initialize a PDF first.", []

    result = S["qa_chain"].invoke({"question": question})
    answer = result.get("answer", "")

    sources = []
    for doc in result.get("source_documents", []):
        meta = doc.metadata or {}
        sources.append({
            "file":    os.path.basename(meta.get("source", "unknown")),
            "page":    meta.get("page", "?"),
            "snippet": doc.page_content[:180].replace("\n", " "),
        })

    # deduplicate sources by (file, page)
    seen = set()
    unique_sources = []
    for s in sources:
        key = (s["file"], s["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return answer, unique_sources


def get_quiz_question():
    """Pick a random chunk and ask Groq to generate a MCQ."""
    if not S["vectorstore"] or not S["qa_chain"]:
        return False, "Upload a PDF first."

    all_data  = S["vectorstore"].get()
    documents = all_data.get("documents", [])
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

    # API key check
    keys_ok = bool(GROQ_API_KEY and GOOGLE_API_KEY)
    if keys_ok:
        st.markdown("<span class='badge badge-ok'>🔑 API Keys Loaded</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge badge-off'>⚠ API Keys Missing</span>", unsafe_allow_html=True)
        with st.expander("How to fix"):
            st.code("""# 1. Create .env in project root
GROQ_API_KEY=your_key
GOOGLE_API_KEY=your_key

# 2. Or on Streamlit Cloud:
# App settings → Secrets → paste keys""")

    st.markdown("---")

    # PDF uploader
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

    # Engine status
    st.markdown("---")
    st.markdown("### 🔧 Engine Status")
    ready = S["qa_chain"] is not None
    status_badge = "badge-ok" if ready else "badge-off"
    status_text  = "ONLINE" if ready else "OFFLINE"
    st.markdown(f"<span class='badge {status_badge}'>{status_text}</span>", unsafe_allow_html=True)

    if S["vectorstore"]:
        n_chunks = len(S["vectorstore"].get()["ids"])
        st.markdown(f"<span class='badge badge-warn'>📦 {n_chunks} chunks</span>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI — MAIN AREA  (3 tabs)
# ══════════════════════════════════════════════════════════════════════════════
tab_chat, tab_quiz = st.tabs(["💬 Chat", "🎮 Quiz"])


# ─── TAB 1: CHAT ─────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown("#### Conversational RAG with Source Citations")
    st.caption("The AI remembers the last 6 exchanges. Sources are shown beneath each answer.")

    # Render history
    chat_container = st.container()
    with chat_container:
        for role, text, sources in S["chat_history"]:
            if role == "user":
                st.markdown(f"<div class='bubble bubble-user'>🧑 {text}</div>",
                            unsafe_allow_html=True)
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

    # Input row
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
            if S["memory"]:
                S["memory"].clear()
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
            # Clear feedback state for the new question
            S["quiz_feedback"] = None 
            with st.spinner("Generating question…"):
                ok, err = get_quiz_question()
            if not ok:
                st.error(err)
            st.rerun()

    # Question card
    if S["quiz_q"]:
        st.markdown(f"""
        <div class='quiz-card'>
          <h3>{S['quiz_q']}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Disable radio choices if they already answered to keep it fair
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
                    
                    # Find the full text of the correct option to show the user
                    correct_full_text = next((opt for opt in S["quiz_opts"] if opt.startswith(S["quiz_correct"])), S["quiz_correct"])
                    
                    if chosen_letter == S["quiz_correct"]:
                        S["quiz_score"]["correct"] += 1
                        S["quiz_feedback"] = {"success": True, "msg": "🎉 Correct! Great job!"}
                    else:
                        S["quiz_feedback"] = {"success": False, "msg": f"❌ Incorrect. The correct answer is: **{correct_full_text}**"}
                    
                    st.rerun()
        
        # Display the persistent feedback message after the rerun
        if "quiz_feedback" in S and S["quiz_feedback"]:
            if S["quiz_feedback"]["success"]:
                st.success(S["quiz_feedback"]["msg"])
            else:
                st.error(S["quiz_feedback"]["msg"])
                
    else:
        st.info("Click **Next Question** to begin the quiz.")