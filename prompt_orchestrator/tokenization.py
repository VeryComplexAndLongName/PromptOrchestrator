from __future__ import annotations

import tiktoken


class TokenCounter:
    def __init__(self, model: str = "gpt-4o-mini", encoding_name: str | None = None) -> None:
        self.model = model
        self.encoding_name = encoding_name
        self._encoding = self._build_encoding()

    def _build_encoding(self):
        if self.encoding_name:
            return tiktoken.get_encoding(self.encoding_name)
        try:
            return tiktoken.encoding_for_model(self.model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))
