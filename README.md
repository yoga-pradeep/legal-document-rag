# GovDoc-RAG: e-Governance Legal Document Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions over
government/legal documents with **cited sources** and **role-based access
control (RBAC)**. Built for an academic portfolio project — combines
**FastAPI**, **LangChain**, **ChromaDB**, and a lightweight **local LLM**
(no paid API key required to run it out of the box).

---

## Why this project

Government and legal documents (acts, circulars, notifications) are long,
dense, and hard to search with keywords. This project lets a user ask a
natural-language question and get:

1. A generated answer, grounded only in retrieved document chunks
   (reduces hallucination vs. asking an LLM directly).
2. The exact source document + section the answer came from (citations —
   critical for legal/government trust).
3. Access restricted by role — a `public` user cannot see `restricted` or
   `confidential` circulars, an `officer` can see `restricted`, only
   `admin` sees everything. This reuses the RBAC pattern from a typical
   backend project and maps it onto document retrieval.

---

## Architecture

```
                     ┌─────────────────────┐
   User question ───▶│   FastAPI /api/query │
   + X-Role header    └─────────┬────────────┘
                                 │
                                 ▼
                     ┌─────────────────────┐
                     │  Role → allowed      │
                     │  access_level filter │
                     └─────────┬────────────┘
                                 │
                                 ▼
                     ┌─────────────────────┐
                     │  Chroma vector store │◀── documents embedded at
                     │  (similarity search)  │    ingest time (LangChain)
                     └─────────┬────────────┘
                                 │  top-k chunks + metadata
                                 ▼
                     ┌─────────────────────┐
                     │  Local LLM (flan-t5) │
                     │  answers ONLY from    │
                     │  retrieved context    │
                     └─────────┬────────────┘
                                 │
                                 ▼
                     Answer + cited sources (JSON)
```

**Stack**
- **FastAPI** — async REST API, automatic OpenAPI docs at `/docs`
- **LangChain** — document loading, legal-aware text splitting, RAG chain orchestration
- **ChromaDB** — local persistent vector store (no external DB needed)
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embeddings, runs on CPU
- **flan-t5-base** (HuggingFace, local) — generation, runs on CPU, no API key
- Optional: swap in OpenAI/Anthropic for generation by setting `LLM_PROVIDER` in `.env` (see below)

---

## Project layout

```
legal-rag-assistant/
├── app/
│   ├── main.py          # FastAPI app + routes
│   ├── config.py         # settings (env vars)
│   ├── schemas.py         # Pydantic request/response models
│   ├── auth.py            # RBAC role dependency
│   ├── ingestion.py        # load, chunk, embed documents into Chroma
│   └── rag_chain.py         # retriever + LLM + prompt -> answer
├── data/sample_docs/         # sample synthetic govt/legal documents + manifest.json
├── scripts/ingest.py           # CLI: python scripts/ingest.py
├── static/index.html            # minimal chat UI (no build step)
├── requirements.txt
├── .env.example
└── README.md
```

> **Note on the sample documents:** the files in `data/sample_docs/` are
> **synthetic demo content** written for this project — short mock
> "circulars" and act summaries, not verbatim reproductions of real
> statutes. Swap them for real (public-domain) government PDFs/text files
> once you've verified the pipeline works.



First run will download the embedding model (~90MB) and the flan-t5-base
model (~250MB) from HuggingFace — needs internet access once, then it's
cached locally.

---

## Run

### 1. Ingest the sample documents into the vector store

```bash
python scripts/ingest.py
```

This reads `data/sample_docs/manifest.json`, chunks each document
(section-aware splitting), embeds the chunks, and persists them to
`./chroma_db/`.

### 2. Start the API

```bash
uvicorn app.main:app --reload --port 8000
```

- Swagger UI: http://localhost:8000/docs
- Simple chat UI: http://localhost:8000/

### 3. Ask a question

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -H "X-Role: officer" \
  -d '{"question": "What is the response time for an RTI request?", "top_k": 3}'
```

Response:

```json
{
  "answer": "A public authority must respond to an RTI request within 30 days...",
  "sources": [
    {
      "document": "right_to_information_summary.txt",
      "section": "Section 4 - Response Timelines",
      "access_level": "public",
      "snippet": "...public authorities are required to respond within thirty (30) days..."
    }
  ]
}
```

Try the same query with `-H "X-Role: public"` vs `-H "X-Role: admin"` and
compare `sources` — public-only users won't see chunks tagged
`restricted` or `confidential`.

---

## Roles (for demo purposes — see "Security note" below)

| Role     | Can see access_level          |
|----------|--------------------------------|
| public   | public                         |
| officer  | public, restricted             |
| admin    | public, restricted, confidential |

Pass the role via the `X-Role` header. Default is `public` if omitted.

---

## API endpoints

| Method | Path            | Description                                      |
|--------|-----------------|---------------------------------------------------|
| GET    | `/health`       | Liveness check                                     |
| GET    | `/api/documents`| List documents visible to the caller's role        |
| POST   | `/api/ingest`   | Re-run ingestion (rebuild the vector store)         |
| POST   | `/api/query`    | Ask a question, get a cited answer                  |

---

