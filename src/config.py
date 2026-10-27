"""Configuration helpers. Secrets come from Streamlit secrets or environment variables, never from source code."""
from __future__ import annotations

import os
from typing import Mapping


def is_truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_data_mode(youtube_key: str | None, force_sample: bool = False) -> str:
    """'live' when a key is configured, else 'sample'. FORCE_SAMPLE/USE_SAMPLE_DATA always wins (demos, tests)."""
    if force_sample:
        return "sample"
    return "live" if youtube_key and youtube_key.strip() else "sample"


def read_setting(name: str, secrets: Mapping | None = None) -> str:
    """Secrets mapping first, then environment."""
    try:
        if secrets is not None:
            v = secrets.get(name)
            if v:
                return str(v)
    except Exception:  # noqa: BLE001 - e.g. no secrets file
        pass
    return os.environ.get(name, "")


YOUTUBE_KEY_NAMES = ("YOUTUBE_API_KEY", "YouTube_Data_API_key", "YOUTUBE_DATA_API_KEY")
