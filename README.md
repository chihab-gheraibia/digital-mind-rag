<div align="center">

# ⚡ Digital Mind RAG

### *Turn any PDF into a conversational AI — with citations, memory & quizzes*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-00e5a0?style=for-the-badge)](LICENSE)

<br/>

**[🚀 Live Demo](https://your-app.streamlit.app)** · **[📖 Docs](#-local-setup)** · **[🐛 Report Bug](https://github.com/chihab-gheraibia/digital-mind-rag/issues)** · **[✨ Request Feature](https://github.com/chihab-gheraibia/digital-mind-rag/issues)**

<br/>

> Upload your PDFs. Ask anything. Get answers with **exact page citations** — powered by Llama 3.3 70B running on Groq's ultra-fast LPU inference.

</div>

---

## 📸 Preview

```
┌─────────────────────────────────────────────────────────────┐
│  ⚡ Digital Mind          💬 Chat    🎮 Quiz                 │
│  RAG · v2.0 · senior edition                                 │
│  ──────────────────────                                      │
│  🔑 API Keys Loaded                                          │
│                            ┌─────────────────────────────┐  │
│  📄 Load Documents         │ 🧑 What is the main topic?  │  │
│  ┌──────────────────────┐  ├─────────────────────────────┤  │
│  │  Drop PDFs here...   │  │ ⚡ The document covers...   │  │
│  └──────────────────────┘  │ 📄 report.pdf · p3          │  │
│                            │ 📄 report.pdf · p7          │  │
│  ⚡ Initialize Engine      └─────────────────────────────┘  │
│                                                              │
│  🔧 Engine Status                                            │
│  ONLINE  📦 342 chunks                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| | Feature | Description |
|---|---|---|
| 📄 | **Multi-PDF Upload** | Load multiple documents at once — no page cap |
| 🧠 | **Conversational Memory** | Remembers the last 6 exchanges for natural dialogue |
| 🔖 | **Source Citations** | Every answer shows the exact file name and page number |
| 🎮 | **Quiz Engine** | Auto-generates multiple-choice questions from your documents |
| 🔒 | **Secure API Keys** | `.env` locally · Streamlit Secrets in the cloud — never hardcoded |
| 👤 | **Session Isolation** | Per-user `st.session_state` — no data leaking between users |
| ⚡ | **Ultra-Fast Inference** | Groq LPU delivers Llama 3.3 70B responses in ~1 second |

---

## 🛠️ Tech Stack

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit  │───▶│    LangChain     │───▶│   Groq (LPU)    │
│  (Frontend)  │    │ RAG Orchestrator │    │ Llama 3.3 70B   │
└──────────────┘    └────────┬─────────┘    └─────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌──────────────────┐         ┌───────────────────┐
   │     ChromaDB     │         │  Google Gemini    │
   │  Vector Store    │         │  Embeddings 768d  │
   └──────────────────┘         └───────────────────┘
```

| Layer | Technology |
|---|---|
| **UI** | Streamlit 1.35+ with custom dark theme |
| **LLM** | Groq · Llama 3.3 70B Versatile |
| **Embeddings** | Google Gemini · `gemini-embedding-001` (768-dim) |
| **Vector DB** | ChromaDB (local persistent store) |
| **RAG Chain** | LangChain `ConversationalRetrievalChain` |
| **Memory** | `ConversationBufferWindowMemory` (k=6) |

---

## 🚀 Local Setup

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com) (free)
- A [Google AI Studio key](https://aistudio.google.com) (free)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/chihab-gheraibia/digital-mind-rag.git
cd digital-mind-rag

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API keys
cp .env.example .env
```

Open `.env` and fill in your keys:
```env
GROQ_API_KEY=gsk_your_key_here
GOOGLE_API_KEY=AIza_your_key_here
```

```bash
# 5. Run the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. 🎉

---

## ☁️ Deploy to Streamlit Cloud

> Free hosting · public URL · takes ~3 minutes

1. **Fork or push** this repo to your GitHub account
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → New app
3. Select: repo `digital-mind-rag` · branch `main` · file `app.py`
4. Click **Advanced settings → Secrets** and paste:

```toml
GROQ_API_KEY = "gsk_your_key_here"
GOOGLE_API_KEY = "AIza_your_key_here"
```

5. Click **Deploy** → your app goes live at `https://your-app.streamlit.app`

> ✅ Your `.gitignore` excludes `.env` and `secrets.toml` — keys never reach GitHub.

---

## 🤗 Deploy to Hugging Face Spaces

> Free hosting · appears on your HF profile

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) → choose **Streamlit** SDK
2. Push this project to the Space repo:

```bash
git remote add hf https://huggingface.co/spaces/chihab-gheraibia/digital-mind-rag
git push hf main
```

3. **Settings → Repository secrets** → add `GROQ_API_KEY` and `GOOGLE_API_KEY`
4. The Space rebuilds automatically and goes live ✅

---

## 📁 Project Structure

```
digital-mind-rag/
│
├── app.py                    ← Main application (UI + logic)
├── requirements.txt          ← Python dependencies
├── .env.example              ← API key template (safe to commit)
├── .gitignore                ← Excludes .env, secrets, chroma DBs
├── README.md
│
└── .streamlit/
    └── secrets.toml.example  ← Streamlit secrets template
```

---

## 🔐 Security

- **Never commit `.env`** — it is gitignored by default
- **Rotate keys immediately** if accidentally pushed to GitHub
- Streamlit Cloud encrypts secrets at rest — they are never exposed in logs
- `chroma_db_*` folders are gitignored — no user data reaches GitHub

---

## 👥 Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/chihab-gheraibia">
        <img src="https://github.com/chihab-gheraibia.png" width="100px;" alt="Chihab Gheraibia"/><br/>
        <sub><b>Chihab Gheraibia</b></sub>
      </a><br/>
      <sub>Creator & Maintainer</sub>
    </td>
  </tr>
</table>

Contributions are welcome! Feel free to open an [issue](https://github.com/chihab-gheraibia/digital-mind-rag/issues) or submit a pull request.

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

Made with ❤️ by [Chihab Gheraibia](https://github.com/chihab-gheraibia)

⭐ **Star this repo if you find it useful!** ⭐

</div>
