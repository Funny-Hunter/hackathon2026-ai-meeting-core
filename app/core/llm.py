from functools import lru_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.core.config import get_settings


@lru_cache()
def get_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.required_openai_api_key,
        temperature=0.2,
    )


@lru_cache()
def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.required_openai_api_key,
    )
