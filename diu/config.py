"""Configuration loading. One YAML file, dot-access, resolved to absolute paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config(dict):
    """dict with attribute access, so cfg.embedding.model reads naturally."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    # Paths in the file are relative to the project root; make them absolute
    # so the CLI works from any working directory.
    for key, value in raw.get("paths", {}).items():
        raw["paths"][key] = str((PROJECT_ROOT / value).resolve())
    ocr = raw.get("ocr", {})
    if ocr.get("tessdata_dir"):
        ocr["tessdata_dir"] = str((PROJECT_ROOT / ocr["tessdata_dir"]).resolve())

    return Config(raw)


def ensure_dirs(cfg: Config) -> None:
    for key in ("registry_db", "vector_dir", "quarantine"):
        target = Path(cfg.paths[key])
        directory = target.parent if key.endswith("_db") else target
        directory.mkdir(parents=True, exist_ok=True)
