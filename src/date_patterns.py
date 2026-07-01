from __future__ import annotations

import re


MONTH_PATTERN = (
    r"января|февраля|марта|апреля|мая|июня|июля|августа|"
    r"сентября|октября|ноября|декабря"
)

WEEKDAY_PATTERN = (
    r"понедельник|понедельника|понедельнику|"
    r"вторник|вторника|вторнику|"
    r"среда|среду|среды|среде|"
    r"четверг|четверга|четвергу|"
    r"пятница|пятницу|пятницы|пятнице|"
    r"суббота|субботу|субботы|субботе|"
    r"воскресенье|воскресенья|воскресенью"
)

END_OF_WEEK_PATTERN = r"(?:(?:в|к|до|на)\s+)?кон(?:ец|це|цу|ца)\s+недели"
END_OF_MONTH_PATTERN = r"(?:(?:в|к|до|на)\s+)?кон(?:ец|це|цу|ца)\s+месяца"

DATE_PHRASE_PATTERNS = [
    r"\bсегодня\b",
    r"\bпослезавтра\b",
    r"\bзавтра\b",
    r"\bчерез\s+месяц\b",
    r"\bчерез\s+неделю\b",
    rf"(?:на|в)\s+следующей\s+неделе\s*[,.]?\s+(?:(?:в|во)\s+)?(?:{WEEKDAY_PATTERN})",
    r"(?:на|в)\s+этой\s+неделе",
    r"(?:на|в)\s+следующей\s+неделе",
    rf"следующ(?:ий|ую|ая|ее|ей)?\s+(?:{WEEKDAY_PATTERN})",
    rf"(?:в|во|к|до|на)\s+(?:{WEEKDAY_PATTERN})",
    rf"(?:до|к|на)?\s*\d{{1,2}}\s+(?:{MONTH_PATTERN})",
    r"(?:до|к|на)\s+\d{1,2}\s+числа",
    r"\d{1,2}\s+числа",
    END_OF_WEEK_PATTERN,
    END_OF_MONTH_PATTERN,
]


def find_date_phrases(text: str) -> list[str]:
    normalized = text.lower().replace("ё", "е")
    phrases: list[str] = []
    for pattern in DATE_PHRASE_PATTERNS:
        phrases.extend(
            match.group(0).strip()
            for match in re.finditer(pattern, normalized)
        )
    return _dedupe_keep_order(phrases)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
