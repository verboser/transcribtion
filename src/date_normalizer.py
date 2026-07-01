from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta

from src.date_patterns import (
    END_OF_MONTH_PATTERN,
    END_OF_WEEK_PATTERN,
    MONTH_PATTERN,
    WEEKDAY_PATTERN,
    find_date_phrases,
)

from dateutil.relativedelta import relativedelta


MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}

WEEKDAYS = {
    "понедельник": 0,
    "понедельника": 0,
    "понедельнику": 0,
    "вторник": 1,
    "вторника": 1,
    "вторнику": 1,
    "среду": 2,
    "среда": 2,
    "среды": 2,
    "среде": 2,
    "четверг": 3,
    "четверга": 3,
    "четвергу": 3,
    "пятницу": 4,
    "пятница": 4,
    "пятницы": 4,
    "пятнице": 4,
    "субботу": 5,
    "суббота": 5,
    "субботы": 5,
    "субботе": 5,
    "воскресенье": 6,
    "воскресенья": 6,
    "воскресенью": 6,
}


@dataclass(frozen=True)
class DateReplacement:
    source: str
    normalized: str


def normalize_deadline(
    deadline_raw: str,
    evidence: str,
    meeting_date: str,
    task_text: str = "",
) -> tuple[str, DateReplacement | None]:
    base = date.fromisoformat(meeting_date)
    sources = _deadline_sources(deadline_raw, evidence, task_text)
    for source in sources:
        normalized = _normalize_phrase(source, base)
        if normalized:
            return normalized, DateReplacement(source=source, normalized=normalized)

    return "", None


def print_replacement_table(replacements: list[DateReplacement]) -> None:
    print("\nТаблица нормализации дат:")
    if not replacements:
        print("Нет относительных или явных сроков для нормализации.")
        return

    seen: set[tuple[str, str]] = set()
    for replacement in replacements:
        key = (replacement.source, replacement.normalized)
        if key in seen:
            continue
        seen.add(key)
        print(f"{replacement.source} -> {replacement.normalized}")


def _normalize_phrase(phrase: str, base: date) -> str:
    text = phrase.lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text).strip(" .,:;")

    if re.search(r"\bсегодня\b", text):
        return base.isoformat()
    if re.search(r"\bпослезавтра\b", text):
        return (base + timedelta(days=2)).isoformat()
    if re.search(r"\bзавтра\b", text):
        return (base + timedelta(days=1)).isoformat()
    if re.search(r"\bчерез\s+месяц\b", text):
        return (base + relativedelta(months=1)).isoformat()
    if re.search(r"\bчерез\s+неделю\b", text):
        return (base + timedelta(days=7)).isoformat()

    next_week_phrase = re.search(
        rf"\b(?:на|в)\s+следующей\s+неделе\s*[,.]?\s+(?:(?:в|во)\s+)?"
        rf"(?P<weekday>{WEEKDAY_PATTERN})\b",
        text,
    )
    if next_week_phrase:
        return _weekday_in_next_week(
            base,
            WEEKDAYS[next_week_phrase.group("weekday")],
        ).isoformat()

    next_weekday = re.search(
        rf"\bследующ(?:ий|ую|ая|ее|ей)?\s+"
        rf"(?P<weekday>{WEEKDAY_PATTERN})\b",
        text,
    )
    if next_weekday:
        return _next_weekday_after(base, WEEKDAYS[next_weekday.group("weekday")]).isoformat()

    if "следующей неделе" in text:
        return ""
    if re.search(END_OF_WEEK_PATTERN, text):
        # In business context, end of week is Friday.
        return _next_or_same_weekday(base, 4).isoformat()
    if "этой неделе" in text:
        return _next_or_same_weekday(base, 4).isoformat()
    if re.search(END_OF_MONTH_PATTERN, text):
        return date(base.year, base.month, monthrange(base.year, base.month)[1]).isoformat()

    explicit = re.search(
        rf"\b(?P<day>\d{{1,2}})\s+(?P<month>{MONTH_PATTERN})\b",
        text,
    )
    if explicit:
        day = int(explicit.group("day"))
        month = MONTHS[explicit.group("month")]
        year = base.year
        candidate = date(year, month, day)
        if candidate < base:
            candidate = date(year + 1, month, day)
        return candidate.isoformat()

    day_only = re.search(
        r"\b(?:(?:до|к|на|срок)\s+(?P<prefixed>\d{1,2})|(?P<bare>\d{1,2})\s+числа)\b",
        text,
    )
    if day_only:
        day = int(day_only.group("prefixed") or day_only.group("bare"))
        candidate = _date_with_day_after_base(base, day)
        if candidate:
            return candidate.isoformat()

    for word, weekday in WEEKDAYS.items():
        if re.search(rf"\b{word}\b", text):
            return _next_weekday_after(base, weekday).isoformat()

    return ""


def _find_relevant_date_phrase(text: str, task_text: str) -> str:
    phrases = find_date_phrases(text)
    if not phrases:
        return ""
    if not task_text or len(phrases) == 1:
        return phrases[0]

    task_terms = _date_selection_terms(task_text)
    scored_phrases = [
        (_date_phrase_score(text, phrase, task_terms), -idx, phrase)
        for idx, phrase in enumerate(phrases)
    ]
    best_score, _, best_phrase = max(scored_phrases)
    return best_phrase if best_score > 0 else phrases[0]


def _deadline_sources(deadline_raw: str, evidence: str, task_text: str) -> list[str]:
    evidence_source = _find_relevant_date_phrase(evidence, task_text)
    sources = [_clean_source(deadline_raw), evidence_source]
    return [source for source in _dedupe_keep_order(sources) if source]


def _clean_source(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .,:;")


def _date_selection_terms(value: str) -> list[str]:
    text = value.lower().replace("ё", "е")
    return [
        term
        for term in re.findall(r"[а-яa-z0-9]+", text)
        if len(term) >= 4 and not term.isdigit()
    ]


def _date_phrase_score(text: str, phrase: str, task_terms: list[str]) -> int:
    match = re.search(re.escape(phrase), text.lower().replace("ё", "е"))
    if not match:
        return 0

    clause = _clause_around_span(text, match.start(), match.end())
    clause_terms = _date_selection_terms(clause)
    clause_score = _term_overlap_score(task_terms, clause_terms)
    if clause_score:
        return clause_score * 10

    window_start = max(0, match.start() - 80)
    window_end = min(len(text), match.end() + 80)
    window_terms = _date_selection_terms(text[window_start:window_end])
    return _term_overlap_score(task_terms, window_terms)


def _clause_around_span(text: str, start: int, end: int) -> str:
    left_candidates = [
        text.rfind(separator, 0, start)
        for separator in [".", "!", "?", ";", ",", "\n"]
    ]
    left = max(left_candidates)
    right_candidates = [
        idx for idx in (
            text.find(separator, end)
            for separator in [".", "!", "?", ";", ",", "\n"]
        )
        if idx != -1
    ]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1 : right]


def _term_overlap_score(left_terms: list[str], right_terms: list[str]) -> int:
    return sum(
        1
        for left_term in left_terms
        if any(_same_term_family(left_term, right_term) for right_term in right_terms)
    )


def _same_term_family(left: str, right: str) -> bool:
    if left == right:
        return True
    if len(left) >= 4 and len(right) >= 4:
        return left[:4] == right[:4]
    return False


def _next_weekday_after(base: date, weekday: int) -> date:
    days = (weekday - base.weekday()) % 7
    if days == 0:
        days = 7
    return base + timedelta(days=days)


def _next_or_same_weekday(base: date, weekday: int) -> date:
    days = (weekday - base.weekday()) % 7
    return base + timedelta(days=days)


def _weekday_in_next_week(base: date, weekday: int) -> date:
    monday_this_week = base - timedelta(days=base.weekday())
    return monday_this_week + timedelta(days=7 + weekday)


def _date_with_day_after_base(base: date, day: int) -> date | None:
    candidate = _date_in_month(base.year, base.month, day)
    if candidate and candidate >= base:
        return candidate

    next_month = base.replace(day=1) + relativedelta(months=1)
    return _date_in_month(next_month.year, next_month.month, day)


def _date_in_month(year: int, month: int, day: int) -> date | None:
    if day < 1 or day > monthrange(year, month)[1]:
        return None
    return date(year, month, day)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
