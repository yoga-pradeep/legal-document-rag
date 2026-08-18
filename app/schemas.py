from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=[
        "What is the response time for an RTI request?"
    ])
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Number of chunks to retrieve. Defaults to server config.",
    )


class SourceChunk(BaseModel):
    document: str
    title: str
    section: str | None = None
    access_level: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    role: str
    sources: list[SourceChunk]


class DocumentInfo(BaseModel):
    filename: str
    title: str
    access_level: str


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_created: int
    persist_dir: str
