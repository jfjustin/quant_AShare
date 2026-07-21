"""Configuration loader.

Reads ``config.yaml`` (falling back to ``config.example.yaml``) and lets
environment variables override secrets so nothing sensitive lives on disk.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


class Config:
    """Dot-and-bracket accessible config with env overrides for secrets."""

    def __init__(self, data: dict[str, Any]):
        self._d = data

    # --- construction -------------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        if path is None:
            cand = ROOT / "config.yaml"
            path = cand if cand.exists() else ROOT / "config.example.yaml"
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cfg = cls(data)
        cfg._apply_env_overrides()
        return cfg

    def _apply_env_overrides(self) -> None:
        ch = self._d.setdefault("choice", {})
        if os.getenv("EM_USERNAME"):
            ch["username"] = os.environ["EM_USERNAME"]
        if os.getenv("EM_PASSWORD"):
            ch["password"] = os.environ["EM_PASSWORD"]

    # --- access -------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def get(self, path: str, default: Any = None) -> Any:
        """Fetch a nested value with a dotted path, e.g. ``cfg.get('signal.weights')``."""
        node: Any = self._d
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def raw(self) -> dict[str, Any]:
        return self._d
