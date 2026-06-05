# ⚡ Digital Mind — Senior-Level RAG

> Conversational PDF intelligence · source citations · quiz engine · vector DB monitor

Built with **Streamlit + LangChain + Groq (Llama 3.3) + Google Gemini Embeddings + ChromaDB**

🔗 **[Live Demo →](https://your-app.streamlit.app)** *(update after deploy)*

---

## Features

| Feature | Detail |
|---|---|
| Multi-PDF support | Upload multiple PDFs, no page cap |
| Conversational memory | Remembers last 6 exchanges |
| Source citations | File + page shown on every answer |
| Quiz engine | Auto-generated MCQs from your documents |
| Secure keys | `.env` locally · Streamlit Secrets in cloud |
| Session isolation | Per-user `st.session_state`, no shared globals |

---

## Tech Stack

- **Groq LPU** — ultra-fast Llama 3.3 70B inference
- **Google Gemini Embeddings** — `gemini-embedding-001` (768-dim vectors)
- **ChromaDB** — local persistent vector store
- **LangChain** — `ConversationalRetrievalChain` + `ConversationBufferWindowMemory`
- **Streamlit** — production web UI with session isolation

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/digital-mind-rag.git
cd digital-mind-rag

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys
cp .env.example .env
# Open .env and paste your real keys

# 5. Run
streamlit run app.py
```

### Getting API Keys

| Key | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys (free) |
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com) → Get API Key (free) |

---

## Deploy to Streamlit Cloud (free · public URL)

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo · branch `main` · main file `app.py`
4. Click **Advanced settings → Secrets** and paste:

```toml
GROQ_API_KEY = "gsk_..."
GOOGLE_API_KEY = "AIza..."
```

5. Click **Deploy** — you get a URL like `https://your-app.streamlit.app`

> Your `.gitignore` already excludes `.env` and `secrets.toml`, so keys never reach GitHub.

---

## Deploy to Hugging Face Spaces (free · shows on your HF profile)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **Create new Space**
2. Choose **Streamlit** as the SDK, set visibility to Public
3. Clone the Space repo and push this project into it:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/digital-mind-rag
git push hf main
```

4. In the Space UI: **Settings → Repository secrets** → add:
   - `GROQ_API_KEY`
   - `GOOGLE_API_KEY`

5. The Space will rebuild automatically and go live.

---

## Folder Structure

```
digital_mind_rag/
├── app.py                       ← main application
├── requirements.txt
├── .env.example                 ← copy → .env and add real keys
├── .gitignore                   ← .env and secrets are excluded
├── README.md
└── .streamlit/
    └── secrets.toml.example     ← copy → secrets.toml for local use
```

---

## Security Notes

- **Never commit `.env` or `secrets.toml`** — both are gitignored
- Rotate your API keys immediately if they are ever accidentally pushed to GitHub
- For team use, use Streamlit Cloud's built-in secrets manager or a secrets vault
