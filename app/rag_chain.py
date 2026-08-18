"""
The actual RAG pipeline:

1. Retrieve top-k chunks from Chroma, filtered to only the access levels
   the caller's role is permitted to see (RBAC at retrieval time).
2. Build a prompt that instructs the LLM to answer ONLY from the
   retrieved context and to say so explicitly if the answer isn't
   present, to reduce hallucination.
3. Call a local (flan-t5) or OpenAI LLM depending on config.
4. Return the answer plus the source chunks used, for citation.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.ingestion import get_vectorstore

PROMPT_TEMPLATE = """You are an assistant that answers questions about \
government and legal documents STRICTLY using the provided context.

Rules:
- Only use information found in the context below.
- If the context does not contain the answer, say clearly that the \
documents you have access to do not cover this, and do not guess.
- Be concise and factual. Where relevant, mention the section number.

Context:
{context}

Question: {question}

Answer:"""


@lru_cache(maxsize=1)
def _get_local_llm():
    from langchain_huggingface import HuggingFacePipeline
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    model_name = settings.local_llm_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
    )
    return HuggingFacePipeline(pipeline=pipe)


@lru_cache(maxsize=1)
def _get_openai_llm():
    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise RuntimeError(
            "LLM_PROVIDER=openai but OPENAI_API_KEY is not set in .env"
        )
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


def get_llm():
    if settings.llm_provider == "openai":
        return _get_openai_llm()
    return _get_local_llm()


def retrieve(
    question: str,
    allowed_access_levels: list[str],
    vectorstore: Chroma | None = None,
    top_k: int | None = None,
) -> list[Document]:
    vs = vectorstore or get_vectorstore()
    k = top_k or settings.retrieval_top_k

    results = vs.similarity_search(
        question,
        k=k,
        filter={"access_level": {"$in": allowed_access_levels}},
    )
    return results


def _format_context(chunks: list[Document]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        title = c.metadata.get("title", c.metadata.get("document", "Unknown"))
        section = c.metadata.get("section", "General")
        blocks.append(f"[{i}] ({title} — {section})\n{c.page_content}")
    return "\n\n".join(blocks)


def answer_question(
    question: str,
    allowed_access_levels: list[str],
    top_k: int | None = None,
) -> tuple[str, list[Document]]:
    chunks = retrieve(question, allowed_access_levels, top_k=top_k)

    if not chunks:
        return (
            "No documents visible to your role contain information relevant "
            "to this question.",
            [],
        )

    context = _format_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    llm = get_llm()
    raw_output = llm.invoke(prompt)

    # HuggingFacePipeline returns a str; ChatOpenAI returns a message object.
    answer = raw_output.content if hasattr(raw_output, "content") else str(raw_output)

    return answer.strip(), chunks
