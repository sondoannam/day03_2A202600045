import os
import random
import time
from typing import Any, Dict, Generator, Optional
from openai import APIError, APIStatusError, OpenAI, RateLimitError
from src.core.llm_provider import LLMProvider

class OpenRouterProvider(LLMProvider):
    """
    LLM Provider For OpenRouter
    """
    def __init__(self, model_name: str = "qwen/qwen3.6-plus:free", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)

        router_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not router_key:
            raise ValueError("OPENROUTER_API_KEY is required to initialize OpenRouterProvider.")

        self.min_request_interval = float(os.getenv("OPENROUTER_MIN_REQUEST_INTERVAL", "2.0"))
        self.max_retries = int(os.getenv("OPENROUTER_MAX_RETRIES", "4"))
        self.initial_backoff = float(os.getenv("OPENROUTER_RETRY_BASE_DELAY", "2.0"))
        self.max_backoff = float(os.getenv("OPENROUTER_MAX_DELAY", "20.0"))
        self._last_request_started_at = 0.0
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=router_key,
        )

    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None) -> list[Dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _wait_for_request_slot(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_started_at
        if self._last_request_started_at and elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self._last_request_started_at = time.monotonic()

    def _should_retry(self, error: Exception) -> bool:
        if isinstance(error, RateLimitError):
            return True

        if isinstance(error, APIStatusError) and error.status_code == 429:
            return True

        if isinstance(error, APIError):
            message = str(error).lower()
            retry_markers = (
                "rate limit",
                "too many requests",
                "rate increased too quickly",
                "scale requests more smoothly",
            )
            return any(marker in message for marker in retry_markers)

        return False

    def _sleep_before_retry(self, attempt: int) -> None:
        backoff = min(self.initial_backoff * (2 ** attempt), self.max_backoff)
        jitter = random.uniform(0, 0.5)
        time.sleep(backoff + jitter)

    def _create_completion(self, messages: list[Dict[str, str]], stream: bool = False):
        last_error = None

        for attempt in range(self.max_retries + 1):
            self._wait_for_request_slot()
            try:
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    extra_body={"reasoning": {"enabled": True}},
                    stream=stream,
                )
            except Exception as error:
                last_error = error
                if attempt >= self.max_retries or not self._should_retry(error):
                    raise
                self._sleep_before_retry(attempt)

        raise last_error

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()

        messages = self._build_messages(prompt, system_prompt)
        response = self._create_completion(messages)

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        message = response.choices[0].message
        content = message.content

        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0
        }

        return {
            "content": content,
            "usage": usage,
            "latency_ms": latency_ms,
            "provider": "openrouter"
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        messages = self._build_messages(prompt, system_prompt)

        for attempt in range(self.max_retries + 1):
            yielded_content = False

            try:
                stream = self._create_completion(messages, stream=True)
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                        yielded_content = True
                        yield chunk.choices[0].delta.content
                return
            except Exception as error:
                if yielded_content or attempt >= self.max_retries or not self._should_retry(error):
                    raise
                self._sleep_before_retry(attempt)