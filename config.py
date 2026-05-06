from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


ENV_FILE = Path(__file__).resolve().parent / ".env"


@dataclass(frozen=True, slots=True)
class Settings:
    fast_api_key: str


def load_env_file(env_path: Path = ENV_FILE) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_env_file()

    fast_api_key = os.environ.get("DESKEW_FAST_API_KEY", "").strip()
    if not fast_api_key:
        raise RuntimeError("DESKEW_FAST_API_KEY is missing. Set it in .env.")

    return Settings(fast_api_key=fast_api_key)
