from __future__ import annotations

from functools import lru_cache
import os
from typing import Protocol


DEFAULT_SEMANTIC_MODEL = "intfloat/multilingual-e5-base"
SEMANTIC_TASK_THRESHOLD = 0.78
SEMANTIC_EVIDENCE_THRESHOLD = 0.68

_TRUTHY = {"1", "true", "yes", "on", "y", "да"}


class _Matcher(Protocol):
    def similarity(self, left: str, right: str) -> float:
        ...


def semantic_similarity(left: str, right: str) -> float | None:
    """Return optional semantic similarity.

    The semantic backend is deliberately opt-in. It prevents unit tests and
    normal CLI runs from downloading a HuggingFace model unexpectedly.
    """

    matcher = _get_matcher()
    if matcher is None:
        return None
    if not left.strip() or not right.strip():
        return 0.0
    return matcher.similarity(left, right)


def semantic_similarity_enabled() -> bool:
    return _env_truthy("TRANSCRIBTION_USE_SEMANTIC")


@lru_cache(maxsize=1)
def _get_matcher() -> _Matcher | None:
    if not semantic_similarity_enabled():
        return None

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    model_name = os.getenv("TRANSCRIBTION_SEMANTIC_MODEL", DEFAULT_SEMANTIC_MODEL)
    local_files_only = not _env_truthy("TRANSCRIBTION_SEMANTIC_ALLOW_DOWNLOAD")

    try:
        model = SentenceTransformer(model_name, local_files_only=local_files_only)
    except TypeError:
        if local_files_only:
            return None
        try:
            model = SentenceTransformer(model_name)
        except Exception:
            return None
    except Exception:
        return None

    return _SentenceTransformerMatcher(model)


class _SentenceTransformerMatcher:
    def __init__(self, model) -> None:
        self._model = model

    def similarity(self, left: str, right: str) -> float:
        left_embedding = self._embedding(left)
        right_embedding = self._embedding(right)
        return float(left_embedding @ right_embedding)

    @lru_cache(maxsize=4096)
    def _embedding(self, text: str):
        # E5 models expect a prefix. For symmetric short-text similarity we use
        # the query prefix on both sides and normalized embeddings.
        return self._model.encode(
            f"query: {text}",
            normalize_embeddings=True,
            show_progress_bar=False,
        )


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY
