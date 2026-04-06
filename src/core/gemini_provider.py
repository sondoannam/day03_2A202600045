from importlib import import_module
import os
import time
from typing import Dict, Any, Optional, Generator

from src.core.llm_provider import LLMProvider

class GeminiProvider(LLMProvider):
    """
    LLM Provider For Google Gemini
    """
    def __init__(self, model_name: str = "gemini-3-flash-preview", api_key: Optional[str] = None, thinking_level: str = "HIGH"):
        super().__init__(model_name, api_key)

        try:
            genai_module = import_module("google.genai")
            types_module = import_module("google.genai.types")
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. Install the package before using GeminiProvider."
            ) from exc
        
        # Get API key from parameter or environment variable
        gemini_key = self.api_key or os.getenv("GEMINI_API_KEY")
        
        # Initialize client with new SDK
        self.client = genai_module.Client(api_key=gemini_key)
        self.types = types_module
        self.thinking_level = thinking_level

    def _get_config(self, system_prompt: Optional[str]):
        """Trợ thủ đắc lực để đúc kết cấu hình linh lực trước khi thi triển."""
        config_args = {}
        
        # Insert System Prompt if provided
        if system_prompt:
            config_args["system_instruction"] = system_prompt
            
        # Enable thinking config
        if self.thinking_level:
            config_args["thinking_config"] = self.types.ThinkingConfig(thinking_level=self.thinking_level)
            
        return self.types.GenerateContentConfig(**config_args)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        config = self._get_config(system_prompt)

        # Execute API call
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        content = response.text
        
        # Collect token usage information
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if response.usage_metadata:
            usage["prompt_tokens"] = response.usage_metadata.prompt_token_count
            usage["completion_tokens"] = response.usage_metadata.candidates_token_count
            usage["total_tokens"] = response.usage_metadata.total_token_count

        return {
            "content": content,
            "usage": usage,
            "latency_ms": latency_ms,
            "provider": "google"
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        config = self._get_config(system_prompt)

        stream_response = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=config
        )

        for chunk in stream_response:
            if chunk.text:
                yield chunk.text