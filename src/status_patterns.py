from __future__ import annotations

import re


DONE_PATTERNS = [
    r"\bвыполнен[аоы]?\b",
    r"\bвыполнил[аи]?\b",
    r"\bсделан[аоы]?\b",
    r"\bсделали\b",
    r"\bсделано\b",
    r"\bготов[аоы]?\b",
    r"\bутвердил[аи]?\b",
    r"\bутвержден[аоы]?\b",
    r"\bутверждён[аоы]?\b",
    r"\bотправил[аи]?\b",
    r"\bотправлен[аоы]?\b",
    r"\bпередал[аи]?\b",
    r"\bпередан[аоы]?\b",
    r"\bпередали\b",
    r"\bразработан[аоы]?\b",
    r"\bсмонтирован[аоы]?\b",
    r"\bпротестирован[аоы]?\b",
    r"\bзалил[аи]?\b",
    r"\bсогласовал[аи]?\b",
    r"\bподготовил[аи]?\b",
    r"\bпроизв[её]л\b",
    r"\bпосчитан[аоы]?\b",
    r"\bзакрыт(?:а|о|ы)?\b",
]

FAILED_PATTERNS = [
    r"\bне\s+успел[аи]?\b",
    r"\bне\s+успели\b",
    r"\bне\s+сделал[аи]?\b",
    r"\bне\s+сделан[аоы]?\b",
    r"\bне\s+сделали\b",
    r"\bне\s+выполн",
    r"\bне\s+закрыл",
    r"\bне\s+закрыт(?:а|о|ы)?\b",
    r"\bне\s+готов[аоы]?\b",
    r"\bне\s+утверж",
    r"\bне\s+подготов",
    r"\bне\s+отправ",
    r"\bпросроч",
    r"\bсрок\s+прош",
    r"\bдедлайн\s+прош",
    r"\bсорвал",
]


def has_done_signal(text: str) -> bool:
    normalized = _normalize(text)
    if has_failed_signal(normalized):
        return False
    return any(re.search(pattern, normalized) for pattern in DONE_PATTERNS)


def has_failed_signal(text: str) -> bool:
    normalized = _normalize(text)
    return any(re.search(pattern, normalized) for pattern in FAILED_PATTERNS)


def detect_status_signals(text: str) -> tuple[bool, bool]:
    has_done = False
    has_failed = False
    for clause in split_status_clauses(text):
        has_done = has_done or has_done_signal(clause)
        has_failed = has_failed or has_failed_signal(clause)
    return has_done, has_failed


def split_status_clauses(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", _normalize(text))
    clauses = re.split(r"[.!?;]+|,\s+|\s+-\s+|\s+\bно\b\s+", normalized)
    return [clause.strip() for clause in clauses if clause.strip()]


def _normalize(text: str) -> str:
    return text.lower().replace("ё", "е")
