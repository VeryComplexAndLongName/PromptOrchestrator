from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .base import CacheBackend


class LocalTTLCacheBackend(CacheBackend):
    def __init__(self, default_ttl_seconds: int = 1800) -> None:
        self._default_ttl = default_ttl_seconds
        self._store: dict[str, tuple[dict, datetime]] = {}

    def get(self, key: str) -> dict | None:
        row = self._store.get(key)
        if row is None:
            return None
        value, expires_at = row
        if datetime.now(timezone.utc) > expires_at:
            self.delete(key)
            return None
        return value

    def set(self, key: str, value: dict, ttl_seconds: int | None = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expires = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        self._store[key] = (value, expires)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
