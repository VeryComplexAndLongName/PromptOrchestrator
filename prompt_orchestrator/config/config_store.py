from __future__ import annotations

from typing import Any

from .module_config import ModuleConfig


class ConfigStore:
    def __init__(self, config: ModuleConfig | dict[str, Any]) -> None:
        self.set_config(config)

    def set_config(self, config: ModuleConfig | dict[str, Any]) -> None:
        if isinstance(config, ModuleConfig):
            self._config = config
            return
        self._config = ModuleConfig.model_validate(config)

    def get_config(self) -> ModuleConfig:
        return self._config

    def get(self, path: str, default: Any | None = None) -> Any:
        value: Any = self._config
        for part in path.split("."):
            if not part:
                continue
            if hasattr(value, part):
                value = getattr(value, part)
                continue
            return default
        return value

    def get_prompt(self):
        return self._config.prompt

    def get_settings(self):
        return self._config.settings

    def get_summary_llm(self):
        return self._config.summary_llm

    def get_safety_llm(self):
        return self._config.safety_llm

    def as_dict(self) -> dict[str, Any]:
        return self._config.model_dump(mode="json")
