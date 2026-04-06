import os
from typing import Optional

from src.core.gemini_provider import GeminiProvider
from src.core.llm_provider import LLMProvider
from src.core.local_provider import LocalProvider
from src.core.openai_provider import OpenAIProvider
from src.core.openrouter_provider import OpenRouterProvider


def create_provider(provider_name: Optional[str] = None, model_name: Optional[str] = None) -> LLMProvider:
    provider = (provider_name or os.getenv("DEFAULT_PROVIDER", "gemini")).strip().lower()
    model = model_name or os.getenv("DEFAULT_MODEL")

    if provider in {"google", "gemini"}:
        return GeminiProvider(model_name=model or "gemini-3-flash-preview")

    if provider == "openai":
        return OpenAIProvider(model_name=model or "gpt-4o")

    if provider == "openrouter":
        return OpenRouterProvider(model_name=model or "qwen/qwen3.6-plus:free")

    if provider == "local":
        local_model_path = os.getenv("LOCAL_MODEL_PATH")
        if not local_model_path:
            raise ValueError("LOCAL_MODEL_PATH is required when DEFAULT_PROVIDER=local.")
        return LocalProvider(model_path=local_model_path)

    raise ValueError(
        "Unsupported provider. Use one of: google, gemini, openai, openrouter, local."
    )


def create_default_provider() -> LLMProvider:
    return create_provider()