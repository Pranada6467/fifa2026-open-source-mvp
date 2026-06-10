"""Tiny .env loader + config access (no extra dependency).

Reads KEY=VALUE lines from the project-root .env on first access and populates
os.environ without overwriting anything already set in the real environment.
"""
from __future__ import annotations

import os
from pathlib import Path

# config.py lives at src/fifapreds/config.py -> project root is parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_loaded = False


def load_env(path: Path | None = None) -> None:
    global _loaded
    if _loaded and path is None:
        return
    path = path or (PROJECT_ROOT / ".env")
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            # Real environment wins over .env, matching standard dotenv behaviour.
            os.environ.setdefault(key.strip(), value.strip())
    if path is None or path == (PROJECT_ROOT / ".env"):
        _loaded = True


def get(key: str, default: str | None = None) -> str | None:
    load_env()
    return os.environ.get(key, default)
