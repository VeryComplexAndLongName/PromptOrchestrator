from __future__ import annotations

import json
from typing import Any

from .base import CacheBackend


class RedisCacheBackend(CacheBackend):
    def __init__(self, redis_client: Any, key_prefix: str = "prompt_ctx:") -> None:
        self._redis = redis_client
        self._prefix = key_prefix

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> dict | None:
        raw = self._redis.get(self._key(key))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def set(self, key: str, value: dict, ttl_seconds: int | None = None) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        cache_key = self._key(key)
        if ttl_seconds:
            self._redis.setex(cache_key, ttl_seconds, payload)
            return
        self._redis.set(cache_key, payload)

    def delete(self, key: str) -> None:
        self._redis.delete(self._key(key))
