from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

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
    "вторник": 1,
    "вторника": 1,
    "среду": 2,
    "среда": 2,
    "среды": 2,
    "четверг": 3,
    "четверга": 3,
    "пятницу": 4,
    "пятница": 4,
    "пятницы": 4,
    "субботу": 5,
    "суббота": 5,
    "субботы": 5,
    "воскресенье": 6,
}


@dataclass(frozen=True)
class DateReplacement:
    source: str
    normalized: str


def normalize_deadline(
    deadline_raw: str,
    evidence: str,
    meeting_date: str,
) -> tuple[str, DateReplacement | None]:
    base = date.fromisoformat(meeting_date)
    source = _clean_source(deadline_raw) or _find_date_phrase(evidence)
    if not source:
        return "", None

    normalized = _normalize_phrase(source, base)
    if not normalized:
        return "", None

    return normalized, DateReplacement(source=source, normalized=normalized)


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

    if "послезавтра" in text:
        return (base + timedelta(days=2)).isoformat()
    if "завтра" in text:
        return (base + timedelta(days=1)).isoformat()
    if "через месяц" in text:
        return (base + relativedelta(months=1)).isoformat()
    if "через неделю" in text:
        return (base + timedelta(days=7)).isoformat()

    if "конце недели" in text or "к концу недели" in text:
        # In business context, end of week is Friday.
        return _next_or_same_weekday(base, 4).isoformat()

    explicit = re.search(
        r"\b(?P<day>\d{1,2})\s+"
        r"(?P<month>января|февраля|марта|апреля|мая|июня|июля|августа|"
        r"сентября|октября|ноября|декабря)\b",
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

    day_only = re.search(r"\b(?:до|к|на)?\s*(?P<day>\d{1,2})\s*(?:числа)?\b", text)
    if day_only:
        day = int(day_only.group("day"))
        candidate = _date_with_day_after_base(base, day)
        if candidate:
            return candidate.isoformat()

    for word, weekday in WEEKDAYS.items():
        if re.search(rf"\b{word}\b", text):
            return _next_weekday_after(base, weekday).isoformat()

    return ""


def _find_date_phrase(text: str) -> str:
    normalized = text.lower().replace("ё", "е")
    patterns = [
        r"послезавтра",
        r"завтра",
        r"через месяц",
        r"через неделю",
        r"(?:в|во|к|до|на)\s+(?:понедельник|понедельника|вторник|вторника|среду|среды|четверг|четверга|пятницу|пятницы|субботу|субботы|воскресенье)",
        r"(?:до|к|на)?\s*\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)",
        r"(?:до|к|на)\s+\d{1,2}\s+числа",
        r"(?:к\s+)?концу недели",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(0)
    return ""


def _clean_source(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .,:;")


def _next_weekday_after(base: date, weekday: int) -> date:
    days = (weekday - base.weekday()) % 7
    if days == 0:
        days = 7
    return base + timedelta(days=days)


def _next_or_same_weekday(base: date, weekday: int) -> date:
    days = (weekday - base.weekday()) % 7
    return base + timedelta(days=days)


def _date_with_day_after_base(base: date, day: int) -> date | None:
    try:
        candidate = date(base.year, base.month, day)
    except ValueError:
        return None

    if candidate >= base:
        return candidate

    try:
        return date(base.year, base.month, 1) + relativedelta(months=1, day=day)
    except ValueError:
        return None
