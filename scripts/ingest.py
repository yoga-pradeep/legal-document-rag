"""
CLI entry point for (re)building the vector store from the sample
documents.

Usage:
    python scripts/ingest.py
"""
import sys
from pathlib import Path

# Allow running as `python scripts/ingest.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import build_vectorstore  # noqa: E402


def main():
    print("Ingesting documents and building the Chroma vector store...")
    _, n_docs, n_chunks = build_vectorstore()
    print(f"Done. {n_docs} document(s) split into {n_chunks} chunk(s).")
    print("Vector store persisted to ./chroma_db")
    print("\nNow run: uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
