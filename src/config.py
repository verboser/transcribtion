from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    model: str = "gpt-5.4-mini"
    max_output_tokens: int = 6000

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Put it into .env, not into code."
            )

        return cls(
            openai_api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip(),
            max_output_tokens=int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "6000")),
        )
