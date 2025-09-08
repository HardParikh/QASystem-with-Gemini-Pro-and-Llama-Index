# QA System with LlamaIndex + Google Gemini (RAG)

A lightweight **Question–Answering (QA)** app that lets you ask questions about your own documents.
It uses a Retrieval-Augmented Generation (RAG) pipeline built with **LlamaIndex (0.10+)** and **Google Gemini** (LLM + embeddings), with a simple **Streamlit** UI.

---

## ✨ Features

* **Bring your own data**: drop PDFs/TXT/MD into a folder and query them.
* **Modern LlamaIndex API**: no `ServiceContext`; uses `Settings`/direct args + `SentenceSplitter`.
* **Gemini Embeddings** (`models/embedding-001`) for vector search.
* **Gemini LLM** for answer synthesis.
* **Persistent index** on disk for fast reloads.
* **Clear logging & custom exception handler** for easier debugging.

---

## 🧠 Architecture (RAG at a glance)

1. **Ingestion** – Load documents from `./data` with `SimpleDirectoryReader`.
2. **Chunking** – Split text (e.g., 800 chars with 20 overlap) via `SentenceSplitter`.
3. **Embeddings** – Create embeddings using **Gemini**; store in a `VectorStoreIndex`.
4. **Retrieval** – Search top-k similar chunks for each query.
5. **Generation** – Pass retrieved context to **Gemini** LLM to synthesize the final answer.
6. **Persistence** – Save index and metadata in `./storage` for reuse.

---

## 📁 Project Structure

```
QASystem/
├─ QAWithPDF/
│  ├─ data_ingestion.py         # load documents from ./data
│  ├─ embedding.py              # build/load index (Gemini embeddings) & query engine
│  ├─ model_api.py              # create/load Gemini LLM
│  ├─ __init__.py               # package marker (optional)
├─ StreamlitApp.py              # Streamlit UI entrypoint
├─ data/                        # your PDFs/TXT/MD go here
├─ storage/                     # generated index (created at runtime)
├─ logger.py                    # project logger
├─ exception.py                 # custom exception (with file/line tracing)
├─ requirements.txt
├─ setup.py                     # local package install (optional)
├─ .env.example                 # sample env file (see below)
└─ README.md
```

---

## ✅ Prerequisites

* Python **3.9+** (tested with 3.9/3.10)
* A **Google Gemini API key** (from Google AI Studio)
* (Optional) `conda` or `venv` for a clean environment

---

## 🔧 Setup

```bash
# 1) Clone
git clone <your-repo-url>
cd QASystem

# 2) Create & activate a virtual env (choose one)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4) Configure your API key (create .env)
copy .env.example .env   # Windows
# or
cp .env.example .env     # macOS/Linux

# Edit .env and set:
# GOOGLE_API_KEY=your_key_here

# 5) Add some documents
# Put your PDFs / .txt / .md into the ./data folder

# 6) Run the app
streamlit run StreamlitApp.py
```

> If you prefer, you can install the local package bits with `python setup.py install` (already wired for imports).

---

## 🔐 Environment Variables

Create a `.env` in the repo root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

> The code loads this key to access both the Gemini LLM and the embedding model.

---

## 🧩 Key Modules

### `QAWithPDF/data_ingestion.py`

* Loads files from `./data` using `SimpleDirectoryReader`.
* Returns a `List[Document]`.

### `QAWithPDF/model_api.py`

* Builds/returns a **Gemini** LLM instance (e.g., `models/gemini-1.5-pro` or similar).
* Central place to switch LLMs or tune generation params (temperature, tokens, etc.).

### `QAWithPDF/embedding.py`

Uses the **current LlamaIndex API** (no `ServiceContext`) to build an index and query engine:

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding

def download_gemini_embedding(model, document, persist_dir: str = "storage"):
    embed_model = GeminiEmbedding(model_name="models/embedding-001")
    splitter = SentenceSplitter(chunk_size=800, chunk_overlap=20)

    index = VectorStoreIndex.from_documents(
        document,
        llm=model,                 # pass LLM directly
        embed_model=embed_model,   # pass embedding model directly
        transformations=[splitter],# chunking
    )
    index.storage_context.persist(persist_dir=persist_dir)
    return index.as_query_engine(llm=model)
```

> This replaces the old `ServiceContext.from_defaults(...)` pattern.

### `StreamlitApp.py`

* Simple UI to ask questions, build/load the index, and show answers.
* Typically:

  1. On start: load docs → build or load index → get `query_engine`
  2. On user submit: `query_engine.query("<your question>")`

---

## ⚙️ Configuration

You can tweak these in `embedding.py`:

* **Chunking**: `chunk_size=800`, `chunk_overlap=20`
* **Embedding model**: `models/embedding-001` (Gemini)
* **Persist dir**: default `"storage"`

To **rebuild** the index (e.g., after changing docs), delete the `storage/` folder and run again.

---

## 🚀 Usage Tips

* Add or update files under `data/`, then restart the app to re-index.
* Ask domain-specific questions that your documents can answer.
* For large corpora, consider a vector DB (e.g., FAISS/Chroma); LlamaIndex supports many backends.

---

## 🧪 Example Queries

* “What is machine learning?”
* “Summarize the main differences between supervised and unsupervised learning.”
* “What does the document say about embeddings and chunking?”

---

## 🛠️ Troubleshooting

### 1) `ServiceContext is deprecated`

This repo is already migrated. We:

* Pass `llm` and `embed_model` directly to `VectorStoreIndex.from_documents`
* Use `SentenceSplitter` for chunking
* (Optionally) you can set global defaults via `from llama_index.core import Settings`:

```python
from llama_index.core import Settings
Settings.llm = <gemini_llm>
Settings.embed_model = GeminiEmbedding("models/embedding-001")
```

### 2) SSL errors installing packages (corporate network)

If you see:

```
SSLCertVerificationError: certificate verify failed: unable to get local issuer certificate
```

* Make sure `pip` and `certifi` are up-to-date: `pip install --upgrade pip certifi`
* If behind a proxy, ensure your corporate CA is trusted by Python/Requests (set `REQUESTS_CA_BUNDLE` to your CA bundle), or configure pip to use your cert.

### 3) `503 failed to connect to all addresses` when calling Gemini

* Check internet / firewall / proxy rules (Gemini endpoints must be reachable over HTTPS).
* Ensure `GOOGLE_API_KEY` is valid and not rate-limited.
* Try again; transient 5xx errors can occur.

---

## 🔒 Safety & Data

* Don’t index sensitive data unless you understand where it’s stored (`./storage` by default).
* Gemini may apply safety filters; handle refusals gracefully in the UI.

---

## 🧭 Roadmap / Ideas

* Pluggable vector stores (FAISS/Chroma/PGVector)
* Citations / source highlighting in answers
* File upload from the UI
* Multi-modal support (images)
* Dockerfile for containerized deploys

---

## 🤝 Contributing

PRs and issues are welcome!
Please:

* Keep code idiomatic and type-friendly.
* Add concise docstrings and logging around new functionality.

---

## 📜 License

MIT (or your preferred license—update this section accordingly).

---

## 🙌 Acknowledgements

* [LlamaIndex](https://github.com/run-llama/llama_index) for the indexing & retrieval framework.
* Google **Gemini** for LLM and embeddings.

---

## 📝 Credits

Built as a clean, practical reference for a RAG workflow using **LlamaIndex 0.10+** and **Gemini** with a simple **Streamlit** front-end.
