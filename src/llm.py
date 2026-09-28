from __future__ import annotations

from abc import ABC, abstractmethod

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


class LLMError(Exception):
    pass

class BaseLLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        pass


class GroqProvider(BaseLLMProvider):
    def __init__(self, api_key: str | None = None, model: str = GROQ_MODEL) -> None:
        key = api_key or GROQ_API_KEY
        if not key:
            raise LLMError(
                "GROQ_API_KEY is missing. Add it to .env or choose another LLM_PROVIDER."
            )
        from groq import Groq

        self._client = Groq(api_key=key)
        self._model = model

    def complete(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
        except Exception as exc:
            raise LLMError(f"Groq request failed: {exc}") from exc
        content = response.choices[0].message.content
        return (content or "").strip()


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str | None = None, model: str = GEMINI_MODEL) -> None:
        key = api_key or GEMINI_API_KEY
        if not key:
            raise LLMError(
                "GEMINI_API_KEY is missing. Add it to .env or choose another LLM_PROVIDER."
            )
        import google.generativeai as genai

        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(model)

    def complete(self, prompt: str) -> str:
        try:
            response = self._model.generate_content(prompt)
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc
        return (response.text or "").strip()


class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self._model = model or OLLAMA_MODEL

    def complete(self, prompt: str) -> str:
        import httpx

        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            raise LLMError(
                f"Ollama request failed ({self._base_url}): {exc}"
            ) from exc
        message = data.get("message") or {}
        return str(message.get("content", "")).strip()


def get_llm_provider(provider_name: str | None = None) -> BaseLLMProvider:
    name = (provider_name or LLM_PROVIDER).lower().strip()
    if name == "groq":
        return GroqProvider()
    if name == "gemini":
        return GeminiProvider()
    if name == "ollama":
        return OllamaProvider()
    raise LLMError(
        f"Unsupported LLM provider '{name}'. "
        "Set LLM_PROVIDER to groq, gemini, or ollama."
    )
