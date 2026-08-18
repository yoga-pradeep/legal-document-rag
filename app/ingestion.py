"""
Loads the sample government/legal documents, splits them into
section-aware chunks, embeds them, and persists them into a local
Chroma vector store.

Run via `python scripts/ingest.py` or POST /api/ingest.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

# Splits preferentially on legal-document structure (Section headers)
# before falling back to paragraph / sentence / word boundaries.
LEGAL_SEPARATORS = [
    "\nSection ",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]

SECTION_HEADER_RE = re.compile(r"Section\s+\d+\s*-\s*[^\n]+")


def _load_manifest(docs_dir: Path) -> list[dict]:
    manifest_path = docs_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)["documents"]


def _extract_section(chunk_text: str) -> str | None:
    """Best-effort extraction of the nearest 'Section N - Title' heading
    that this chunk falls under, so citations can point to a section
    rather than just a filename."""
    matches = SECTION_HEADER_RE.findall(chunk_text)
    if matches:
        return matches[0].strip()
    return None


def load_and_split_documents(docs_dir: str | None = None) -> list[Document]:
    docs_path = Path(docs_dir or settings.sample_docs_dir)
    manifest = _load_manifest(docs_path)

    splitter = RecursiveCharacterTextSplitter(
        separators=LEGAL_SEPARATORS,
        chunk_size=700,
        chunk_overlap=100,
        length_function=len,
    )

    all_chunks: list[Document] = []

    for entry in manifest:
        file_path = docs_path / entry["filename"]
        if not file_path.exists():
            raise FileNotFoundError(f"Manifest references missing file: {file_path}")

        raw_text = file_path.read_text(encoding="utf-8")
        chunks = splitter.split_text(raw_text)

        for i, chunk_text in enumerate(chunks):
            section = _extract_section(chunk_text)
            all_chunks.append(
                Document(
                    page_content=chunk_text.strip(),
                    metadata={
                        "document": entry["filename"],
                        "title": entry["title"],
                        "access_level": entry["access_level"],
                        "section": section or "General",
                        "chunk_index": i,
                    },
                )
            )

    return all_chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def build_vectorstore(docs_dir: str | None = None) -> tuple[Chroma, int, int]:
    """(Re)builds the Chroma collection from the sample documents.

    Returns (vectorstore, num_documents, num_chunks).
    """
    docs_path = Path(docs_dir or settings.sample_docs_dir)
    manifest = _load_manifest(docs_path)
    chunks = load_and_split_documents(str(docs_path))

    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )

    # Reset the collection so re-running ingestion doesn't duplicate chunks.
    existing_ids = vectorstore.get()["ids"]
    if existing_ids:
        vectorstore.delete(ids=existing_ids)

    vectorstore.add_documents(chunks)

    return vectorstore, len(manifest), len(chunks)


def get_vectorstore() -> Chroma:
    """Opens the existing persisted vector store without rebuilding it."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def list_documents(docs_dir: str | None = None) -> list[dict]:
    docs_path = Path(docs_dir or settings.sample_docs_dir)
    return _load_manifest(docs_path)
