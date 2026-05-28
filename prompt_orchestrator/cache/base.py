from __future__ import annotations

from abc import ABC, abstractmethod


class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> dict | None:
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: dict, ttl_seconds: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        raise NotImplementedError


class NoCacheBackend(CacheBackend):
    def get(self, key: str) -> dict | None:
        return None

    def set(self, key: str, value: dict, ttl_seconds: int | None = None) -> None:
        return None

    def delete(self, key: str) -> None:
        return None
