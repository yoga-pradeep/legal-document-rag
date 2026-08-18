"""
Central configuration for GovDoc-RAG.

Values are read from environment variables / .env file, with sensible
defaults so the project runs out of the box with zero paid API keys.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Vector store
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "govdocs"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Generation
    llm_provider: str = "local"  # "local" or "openai"
    local_llm_model: str = "google/flan-t5-base"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Retrieval
    retrieval_top_k: int = 4

    # Source documents
    sample_docs_dir: str = "./data/sample_docs"

    # RBAC: role -> list of access levels that role may see
    role_access_levels: dict[str, list[str]] = {
        "public": ["public"],
        "officer": ["public", "restricted"],
        "admin": ["public", "restricted", "confidential"],
    }


settings = Settings()
