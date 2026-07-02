from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


LexiconCategory = Literal[
    "new_task",
    "recap",
    "ongoing_state",
    "weak_done",
    "done_rejection",
    "new_assignment",
]


@dataclass(frozen=True)
class LexiconPattern:
    category: LexiconCategory
    name: str
    pattern: str
    description: str


NEW_TASK_PATTERNS = [
    r"\bпод\s+протокол\b",
    r"\bв\s+протокол\b",
    r"\bпротокол\b",
    r"\bзадач",
    r"\bнужно\b",
    r"\bнадо\b",
    r"\bдолжн",
    r"\bпрошу\b",
    r"\bпросьба\b",
    r"\bпоруч",
    r"\bдавайте\b",
    r"\bбер[её]м\s+в\s+работу\b",
    r"\bвозьм(?:ем|ём|ите)\b",
    r"\bзакрепим\b",
    r"\bзафиксируем\b",
    r"\bфиксируем\b",
    r"\bставим\s+срок\b",
    r"\bответственн(?:ый|ая|ые|ое|ого|ому|ым|ом|ую|ой|ых|ыми|о)\b",
    r"\bнаправьте\b",
    r"\bнаправ(?:ить|им)\b",
    r"\bпришл(?:ите|ем|ём)\b",
    r"\bназначить\b",
    r"\bназначим\b",
    r"\bсобер(?:ем|ём|емся|итесь)\b",
    r"\bверн(?:ем|ём|емся|ёмся)\b",
    r"\bдоговорил[аи]сь\b",
    r"\bдоговоримся\b",
    r"\bразошлем\b",
    r"\bразошлём\b",
    r"\bсформировать\b",
    r"\bподготов(?:ка|ку|ить|им|ьте|имся)\b",
    r"\bобсуд(?:им|ить)\b",
    r"\bпровести\b",
    r"\bдоработа(?:ть|ем|йте)\b",
    r"\bсогласова(?:ть|ли)\b",
    r"\bсогласу(?:ем|йте)\b",
    r"\bзапланируем\b",
    r"\bпровер(?:им|ить|ьте)\b",
    r"\bвозьм[её]м\s+паузу\b",
    r"\bбудем\s+(?:делать|готовить|обсуждать|согласовывать|"
    r"дорабатывать|проверять|направлять|проводить)\b",
]


RECAP_PATTERNS = [
    r"\bподытожим\b",
    r"\bрезюмируем\b",
    r"\bитак\b",
    r"\bпо\s+итогам\b",
    r"\bфинально\b",
    r"\bтогда\s+фиксируем\b",
    r"\bдоговорились\b",
    r"\bоста[её]тся\s+зафиксировать\b",
    r"\bв\s+протокол\b",
]


ONGOING_STATE_PATTERNS = [
    r"\bпроизводится\b",
    r"\bпроводится\b",
    r"\bид[её]т\b",
    r"\bначали\b",
    r"\bуже\s+проводим\b",
    r"\bвремя\s+ещ[её]\s+есть\b",
    r"\bсейчас\s+(?:это\s+)?(?:в\s+работе|в\s+процессе)\b",
    r"\bв\s+работе\b",
    r"\bв\s+процессе\b",
    r"\bзанима(?:емся|юсь|ются)\b",
    r"\bпрорабатыва(?:ем|ется|ются|ются|ю)\b",
    r"\bготовность\b",
]


WEAK_DONE_PATTERNS = [
    r"\bв\s+целом\s+готов[аоы]?\b",
    r"\bпрактически\s+готов[аоы]?\b",
    r"\bпочти\s+готов[аоы]?\b",
    r"\bпо\s+факту\s+готов[аоы]?\b",
    r"\bвс[её]\s+сделано\b",
    r"\bвс[её]\s+готово\b",
    r"\bвс[её]\s+выполнено\b",
    r"\bнаписано\s+и\s+сделано\b",
    r"\bготов[аоы]?\s+к\b",
    r"\bготовность\s+высок",
]


DONE_REJECTION_PATTERNS = [
    r"\bосталось\b",
    r"\bожидаем\b",
    r"\bожидает\b",
    r"\bожидают\b",
    r"\bожидани",
    r"\bжд[её]м\b",
    r"\bпосмотрю\b",
    r"\bпрорабатыва",
    r"\bбудем\s+(?:делать|заниматься|проводить|подключать|принимать|"
    r"дорабатывать|согласовывать|обсуждать|смотреть|готовить|производить)\b",
    r"\bбудут\b",
    r"\bнадо\b",
    r"\bнужно\b",
    r"\bготовность\b",
    r"\bтолько\s+взять\b",
    r"\bподкладываем\b",
    r"\bзанима[ею]тся\b",
    r"\bв\s+процессе\b",
    r"\bпроизводится\b",
    r"\bпроводится\b",
    r"\bначали\b",
    r"\bсейчас\s+(?:это\s+)?(?:в\s+работе|в\s+процессе)\b",
    r"\bоста[её]тся\b",
    r"\bпрактически\s+готов",
    r"\bпочти\s+готов",
    r"\bпо\s+факту\s+готов",
    r"^\s*есть\b",
]


NEW_ASSIGNMENT_PATTERNS = [
    r"\bзадач",
    r"\bпод\s+протокол\b",
    r"\bв\s+протокол\b",
    r"\bнужно\b",
    r"\bнадо\b",
    r"\bдолжн",
    r"\bпоруч",
    r"\bпрошу\b",
    r"\bпросьба\b",
    r"\bставим\s+срок\b",
    r"\bответственн(?:ый|ая|ые|ое|ого|ому|ым|ом|ую|ой|ых|ыми|о)\b",
]


PATTERNS_BY_CATEGORY: dict[LexiconCategory, list[str]] = {
    "new_task": NEW_TASK_PATTERNS,
    "recap": RECAP_PATTERNS,
    "ongoing_state": ONGOING_STATE_PATTERNS,
    "weak_done": WEAK_DONE_PATTERNS,
    "done_rejection": DONE_REJECTION_PATTERNS,
    "new_assignment": NEW_ASSIGNMENT_PATTERNS,
}


def _description_for(category: LexiconCategory) -> str:
    return {
        "new_task": "маркер нового поручения или будущего действия",
        "recap": "маркер финального recap или фиксации решений",
        "ongoing_state": "маркер текущей работы или состояния, не задачи",
        "weak_done": "слабая формулировка готовности без надежного объекта",
        "done_rejection": "сигнал, из-за которого completed-кандидат рискованный",
        "new_assignment": "явный assignment-сигнал для новых задач",
    }[category]


LEXICON_PATTERNS: tuple[LexiconPattern, ...] = tuple(
    LexiconPattern(
        category=category,
        name=f"{category}_{idx:02d}",
        pattern=pattern,
        description=_description_for(category),
    )
    for category, patterns in PATTERNS_BY_CATEGORY.items()
    for idx, pattern in enumerate(patterns, start=1)
)


def patterns_for(*categories: LexiconCategory) -> list[str]:
    return [
        pattern
        for category in categories
        for pattern in PATTERNS_BY_CATEGORY[category]
    ]


def find_lexicon_matches(
    text: str,
    categories: tuple[LexiconCategory, ...] | None = None,
) -> list[tuple[LexiconPattern, str]]:
    normalized = normalize_for_lexicon(text)
    allowed = set(categories) if categories else None
    matches: list[tuple[LexiconPattern, str]] = []
    for lexicon_pattern in LEXICON_PATTERNS:
        if allowed is not None and lexicon_pattern.category not in allowed:
            continue
        for match in re.finditer(lexicon_pattern.pattern, normalized, flags=re.I):
            matches.append((lexicon_pattern, match.group(0)))
    return matches


def matches_any(text: str, patterns: list[str]) -> bool:
    normalized = normalize_for_lexicon(text)
    return any(re.search(pattern, normalized, flags=re.I) for pattern in patterns)


def normalize_for_lexicon(text: str) -> str:
    text = text.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()
