from __future__ import annotations

from .base import CacheBackend


class CornetCacheBackend(CacheBackend):
    """
    Adapter for a hypothetical Cornet cache client.
    Expects client methods: get(key), set(key, value, ttl=None), delete(key).
    """

    def __init__(self, client, key_prefix: str = "prompt_ctx:") -> None:
        self._client = client
        self._prefix = key_prefix

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> dict | None:
        return self._client.get(self._key(key))

    def set(self, key: str, value: dict, ttl_seconds: int | None = None) -> None:
        self._client.set(self._key(key), value, ttl=ttl_seconds)

    def delete(self, key: str) -> None:
        self._client.delete(self._key(key))
