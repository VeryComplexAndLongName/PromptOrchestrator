from __future__ import annotations

from pydantic import BaseModel

from .base_client import SummaryLLMClient


class OpenAIConfig(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None


class OpenAISummaryClient(SummaryLLMClient):
    def __init__(self, config: OpenAIConfig | None = None) -> None:
        self.config = config or OpenAIConfig()
        self._client = self._build_client()

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI provider requires the 'openai' package. Install dependencies first."
            ) from exc

        return OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            organization=self.config.organization,
        )

    def generate(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        try:
            response = self._client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            text = getattr(response, "output_text", None)
            if text:
                return text
        except Exception:
            pass

        chat = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return chat.choices[0].message.content or ""
