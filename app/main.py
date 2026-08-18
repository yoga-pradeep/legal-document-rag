from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.auth import allowed_access_levels, get_current_role
from app.config import settings
from app.ingestion import build_vectorstore, list_documents
from app.rag_chain import answer_question
from app.schemas import (
    DocumentInfo,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)

app = FastAPI(
    title="GovDoc-RAG",
    description=(
        "RAG-based question answering over government/legal documents "
        "with role-based access control."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/documents", response_model=list[DocumentInfo])
def get_documents(role: str = Depends(get_current_role)):
    allowed = set(allowed_access_levels(role))
    docs = list_documents()
    return [d for d in docs if d["access_level"] in allowed]


@app.post("/api/ingest", response_model=IngestResponse)
def ingest():
    try:
        _, n_docs, n_chunks = build_vectorstore()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return IngestResponse(
        documents_ingested=n_docs,
        chunks_created=n_chunks,
        persist_dir=settings.chroma_persist_dir,
    )


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest, role: str = Depends(get_current_role)):
    allowed = allowed_access_levels(role)

    try:
        answer, chunks = answer_question(
            req.question, allowed, top_k=req.top_k
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Query failed: {e}. Did you run `python scripts/ingest.py` "
                "first to build the vector store?"
            ),
        )

    sources = [
        SourceChunk(
            document=c.metadata.get("document", ""),
            title=c.metadata.get("title", ""),
            section=c.metadata.get("section"),
            access_level=c.metadata.get("access_level", ""),
            snippet=c.page_content[:300],
        )
        for c in chunks
    ]

    return QueryResponse(answer=answer, role=role, sources=sources)
