from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

import pandas as pd

from src.date_normalizer import DateReplacement, normalize_deadline
from src.schemas import BLOCK_ORDER, DATAFRAME_COLUMNS, ExtractedTask, TaskAnchor
from src.status_patterns import has_done_signal, has_failed_signal


MONTH_WORDS = (
    "января|февраля|марта|апреля|мая|июня|июля|"
    "августа|сентября|октября|ноября|декабря"
)
RESPONSIBLE_MARKER_RE = re.compile(
    r"\bответственн(?:ый|ая|ые|ое|ого|ому|ым|ом|ую|ой|ых|ыми|о)\b",
    re.I,
)
RESPONSIBLE_STOP_WORDS = {
    "срок",
    "сегодня",
    "завтра",
    "послезавтра",
    "будем",
    "буду",
    "будет",
    "вдвоем",
    "вдвоём",
    "числа",
    "число",
    "понедельник",
    "вторник",
    "среду",
    "четверг",
    "пятницу",
    "субботу",
    "воскресенье",
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
}
RESPONSIBLE_ORG_WORDS = {
    "м",
    "п",
    "югра",
    "урал",
    "парус",
}
RESPONSIBLE_CONNECTOR_WORDS = {
    "будет",
    "будут",
    "назначен",
    "назначена",
    "назначены",
    "назначить",
}
RESPONSIBLE_SCOPE_WORDS = {
    "за",
    "по",
    "для",
}
RESPONSIBLE_TASK_WORDS = {
    "подготовить",
    "подготовка",
    "подготовку",
    "сформировать",
    "провести",
    "доработать",
    "направить",
    "согласовать",
    "собрать",
    "утвердить",
    "отчет",
    "отчёт",
    "протокол",
    "программа",
    "программу",
    "чек",
    "листы",
    "файл",
    "встреча",
    "встречу",
    "задача",
    "задачу",
    "задачка",
}
DONE_VERB_TERMS = {
    "выполнен",
    "выполнена",
    "выполнено",
    "выполнены",
    "выполнили",
    "сделан",
    "сделана",
    "сделано",
    "сделаны",
    "сделал",
    "сделали",
    "написано",
    "написали",
    "подготовил",
    "подготовила",
    "подготовили",
    "разработан",
    "разработана",
    "разработано",
    "разработаны",
    "передали",
    "передана",
    "передано",
    "переданы",
    "произвел",
    "произвёл",
    "смонтировано",
    "согласовал",
    "согласовали",
    "залили",
    "протестировано",
}
WEAK_DONE_OBJECT_TERMS = {
    "собственно",
    "задача",
    "задачу",
    "написано",
    "сделано",
}
ONGOING_NEW_PATTERNS = [
    r"\bпроизводится\b",
    r"\bначали\b",
    r"\bуже\s+проводим\b",
    r"\bвремя\s+ещ[её]\s+есть\b",
    r"\bсейчас\b",
]
NEW_ASSIGNMENT_PATTERNS = [
    r"\bзадач",
    r"\bпод\s+протокол\b",
    r"\bнужно\b",
    r"\bнадо\b",
    r"\bдолжн",
    r"\bпоруч",
    RESPONSIBLE_MARKER_RE.pattern,
]


@dataclass(frozen=True)
class PostprocessStats:
    raw_rows: int
    valid_rows: int
    final_rows: int
    filtered_rows: int
    dedup_removed_rows: int


def build_dataframe(
    tasks: list[ExtractedTask],
    meeting_date: str,
    anchors: list[TaskAnchor] | None = None,
) -> tuple[pd.DataFrame, list[DateReplacement]]:
    df, replacements, _ = build_dataframe_with_stats(tasks, meeting_date, anchors)
    return df, replacements


def build_dataframe_with_stats(
    tasks: list[ExtractedTask],
    meeting_date: str,
    anchors: list[TaskAnchor] | None = None,
) -> tuple[pd.DataFrame, list[DateReplacement], PostprocessStats]:
    rows: list[dict[str, str]] = []
    replacements: list[DateReplacement] = []
    anchor_map = {anchor.anchor_id: anchor for anchor in anchors or []}

    for task in tasks:
        task_anchors = _resolve_task_anchors(task, anchor_map)
        if anchors is not None and not task_anchors:
            continue
        if not is_valid_task(task):
            continue

        evidence = _resolve_evidence(task, task_anchors)
        if evidence is None:
            continue
        if task.block == "Новые" and not _evidence_supports_task(task.task, evidence):
            continue

        responsible = _resolve_responsible(task, task_anchors, evidence)
        deadline, replacement = _normalize_task_deadline(task, evidence, meeting_date)
        if replacement:
            replacements.append(replacement)

        if task.block == "Новые" and not deadline:
            continue

        rows.append(
            {
                "Блок": task.block,
                "Задача": task.task,
                "Ответственный": responsible,
                "Срок": deadline,
                "Обоснование": evidence,
            }
        )

    valid_rows = len(rows)
    rows = _deduplicate_rows(rows)
    final_rows = len(rows)
    rows.sort(
        key=lambda row: (
            BLOCK_ORDER.get(row["Блок"], 99),
            row["Ответственный"],
            row["Срок"],
            _normalize_key(row["Задача"]),
        )
    )
    stats = PostprocessStats(
        raw_rows=len(tasks),
        valid_rows=valid_rows,
        final_rows=final_rows,
        filtered_rows=len(tasks) - valid_rows,
        dedup_removed_rows=valid_rows - final_rows,
    )
    return pd.DataFrame(rows, columns=DATAFRAME_COLUMNS), replacements, stats


def is_valid_task(task: ExtractedTask) -> bool:
    if task.block == "Выполненные":
        return _is_valid_done_task(task)
    if task.block == "Невыполненные":
        return _has_explicit_failure_signal(task.evidence)
    if task.block == "Новые":
        return _is_valid_new_task(task)
    return True


def _normalize_task_deadline(
    task: ExtractedTask,
    evidence: str,
    meeting_date: str,
) -> tuple[str, DateReplacement | None]:
    if task.block == "Выполненные":
        return "", None

    deadline_raw = "" if task.block == "Новые" else task.deadline_raw
    return normalize_deadline(
        deadline_raw=deadline_raw,
        evidence=evidence,
        meeting_date=meeting_date,
        task_text=task.task if task.block == "Новые" else "",
    )


def _resolve_task_anchors(
    task: ExtractedTask,
    anchor_map: dict[str, TaskAnchor],
) -> list[TaskAnchor]:
    return [anchor_map[anchor_id] for anchor_id in task.anchor_ids if anchor_id in anchor_map]


def _resolve_evidence(task: ExtractedTask, anchors: list[TaskAnchor]) -> str | None:
    if not anchors:
        return task.evidence

    anchor_text = "\n".join(anchor.text() for anchor in anchors)
    if _quote_is_supported(task.evidence, anchor_text):
        return task.evidence

    return None


def _resolve_responsible(
    task: ExtractedTask,
    anchors: list[TaskAnchor],
    evidence: str,
) -> str:
    explicit_responsible = _explicit_responsible_from_evidence(evidence)
    if explicit_responsible:
        return explicit_responsible

    if not anchors:
        return task.responsible

    speakers = {
        utterance.speaker
        for anchor in anchors
        for utterance in anchor.utterances
    }
    evidence_speaker = _speaker_from_evidence(evidence)
    if evidence_speaker in speakers:
        return evidence_speaker
    if task.responsible in speakers:
        return task.responsible

    return anchors[0].utterances[0].speaker


def _explicit_responsible_from_evidence(evidence: str) -> str:
    text = _strip_evidence_prefix(evidence)
    return _responsible_after_marker(text) or _responsible_from_recap(text)


def _strip_evidence_prefix(evidence: str) -> str:
    return re.sub(
        r"^\s*(?:\[\d{1,5}\]\s*)?[^:\n]{1,120}:\s*",
        "",
        evidence,
    ).strip()


def _responsible_after_marker(text: str) -> str:
    for match in RESPONSIBLE_MARKER_RE.finditer(text):
        tail = _truncate_responsible_tail(text[match.end() :])
        if _starts_with_scope_word(tail):
            responsible = _format_responsible_suffix(tail)
            if responsible:
                return responsible

        tail = _drop_responsible_connectors(tail)
        responsible = _format_responsible_names(tail)
        if responsible:
            return responsible
    return ""


def _truncate_responsible_tail(value: str) -> str:
    return re.split(
        r"\b(?:срок|до|к|на|сегодня|завтра|послезавтра|будем|буду|"
        r"вдвоем|вдвоём|числа|число)\b|\d{1,2}",
        value,
        maxsplit=1,
        flags=re.I,
    )[0].strip()


def _starts_with_scope_word(value: str) -> bool:
    tokens = _word_tokens(value)
    return bool(tokens and tokens[0] in RESPONSIBLE_SCOPE_WORDS)


def _drop_responsible_connectors(value: str) -> str:
    tokens = _word_tokens(value)
    while tokens and tokens[0] in RESPONSIBLE_CONNECTOR_WORDS:
        tokens.pop(0)
    return " ".join(tokens)


def _responsible_from_recap(text: str) -> str:
    normalized = text.lower().replace("ё", "е")
    if "протокол" not in normalized or "срок" not in normalized:
        return ""

    date_match = re.search(rf"\b\d{{1,2}}\s+(?:{MONTH_WORDS})\s+срок\b", normalized)
    if not date_match:
        return ""

    prefix = normalized[: date_match.start()]
    tokens = _word_tokens(prefix)
    skip_tokens = {
        "так",
        "протокол",
        "тогда",
        "еще",
        "ещё",
        "раз",
        "обозначу",
        "обозначим",
        "задача",
        "задачка",
    }

    name_tokens: list[str] = []
    for token in reversed(tokens):
        if token in skip_tokens or token in RESPONSIBLE_TASK_WORDS or len(token) < 3:
            if name_tokens:
                break
            continue
        name_tokens.append(token)
        if len(name_tokens) == 2:
            break

    if not name_tokens:
        return ""

    candidate = " ".join(reversed(name_tokens))
    if not _looks_like_person_name(candidate):
        return ""
    return _format_person_name(candidate)


def _format_responsible_names(value: str) -> str:
    value = _strip_evidence_prefixes(value)
    parts = re.split(r"\s*(?:,|/|[.;\n]|\s+и\s+)\s*", value.strip(), flags=re.I)
    names = [_format_person_name(part) for part in parts]
    names = [name for name in names if name]
    return " и ".join(_dedupe_keep_order(names))


def _strip_evidence_prefixes(value: str) -> str:
    return re.sub(
        r"(?:^|\n)\s*(?:\[\d{1,5}\]\s*)?[^:\n]{1,120}:\s*",
        "\n",
        value,
    )


def _format_responsible_suffix(value: str) -> str:
    tokens = _word_tokens(value)
    name_tokens: list[str] = []
    for token in reversed(tokens):
        if token in RESPONSIBLE_STOP_WORDS or token in RESPONSIBLE_CONNECTOR_WORDS:
            continue
        if token in RESPONSIBLE_SCOPE_WORDS or token in RESPONSIBLE_TASK_WORDS:
            if name_tokens:
                break
            continue
        if len(token) < 3:
            continue
        name_tokens.append(token)
        if len(name_tokens) == 2:
            break

    return _format_person_name(" ".join(reversed(name_tokens)))


def _format_person_name(value: str) -> str:
    tokens: list[str] = []
    for token in _word_tokens(value):
        if (
            token in RESPONSIBLE_STOP_WORDS
            or token in RESPONSIBLE_SCOPE_WORDS
            or token in RESPONSIBLE_ORG_WORDS
        ):
            break
        if token in RESPONSIBLE_CONNECTOR_WORDS or token in RESPONSIBLE_TASK_WORDS:
            continue
        if len(token) < 3:
            continue
        tokens.append(token)
        if len(tokens) == 3:
            break

    if not tokens:
        return ""

    return " ".join(token[:1].upper() + token[1:] for token in tokens)


def _looks_like_person_name(value: str) -> bool:
    tokens = _word_tokens(value)
    if not tokens:
        return False
    return all(
        token not in RESPONSIBLE_TASK_WORDS and token not in RESPONSIBLE_ORG_WORDS
        for token in tokens
    )


def _word_tokens(value: str) -> list[str]:
    return re.findall(r"[а-яёa-z-]+", value.lower().replace("ё", "е"))


def _quote_is_supported(evidence: str, anchor_text: str) -> bool:
    normalized_evidence = _normalize_quote(evidence)
    normalized_anchor = _normalize_quote(anchor_text)
    if not normalized_evidence:
        return False
    return normalized_evidence in normalized_anchor


def _has_explicit_failure_signal(evidence: str) -> bool:
    return has_failed_signal(evidence)


def _is_valid_done_task(task: ExtractedTask) -> bool:
    if _is_generic_done_task(task.task):
        return False
    if _has_done_rejection_signal(task.task):
        return False
    if not _has_concrete_done_object(task.task):
        return False
    if not _done_clause_supports_task(task.task, task.evidence):
        return False
    return True


def _is_valid_new_task(task: ExtractedTask) -> bool:
    evidence = _normalize_key(task.evidence)
    if _matches_any(evidence, ONGOING_NEW_PATTERNS) and not _matches_any(
        evidence,
        NEW_ASSIGNMENT_PATTERNS,
    ):
        return False
    return True


def _has_explicit_done_signal(evidence: str) -> bool:
    return has_done_signal(evidence)


def _done_clause_supports_task(task_text: str, evidence: str) -> bool:
    task_terms = _content_terms(task_text)
    if not task_terms:
        return False

    for clause in _split_clauses(evidence):
        if not _has_explicit_done_signal(clause):
            continue
        clause_terms = _content_terms(clause)
        matches = sum(
            1
            for task_term in task_terms
            if any(_same_term_family(task_term, clause_term) for clause_term in clause_terms)
        )
        if matches >= min(2, len(task_terms)):
            return True
    return False


def _split_clauses(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text)
    clauses = re.split(r"[.!?;]+|,\s+|\s+-\s+", normalized)
    return [clause.strip() for clause in clauses if clause.strip()]


def _has_done_rejection_signal(task_text: str) -> bool:
    text = task_text.lower().replace("ё", "е")
    rejection_patterns = [
        r"\bосталось\b",
        r"\bожидаем\b",
        r"\bожидает\b",
        r"\bожидают\b",
        r"\bожидани",
        r"\bпрорабатыва",
        r"\bбудем\b",
        r"\bбудут\b",
        r"\bнадо\b",
        r"\bнужно\b",
        r"\bготовность\b",
        r"\bтолько\s+взять\b",
        r"\bподкладываем\b",
        r"\bзанима[ею]тся\b",
        r"^\s*есть\b",
    ]
    return any(re.search(pattern, text) for pattern in rejection_patterns)


def _is_generic_done_task(task_text: str) -> bool:
    text = _normalize_key(task_text)
    generic_done = {
        "все сделано",
        "все готово",
        "готово",
        "сделано",
        "выполнено",
        "закрыто",
        "написано и сделано",
    }
    if text in generic_done:
        return True

    vague_patterns = [
        r"^(?:он|она|оно|они|это)\s+готов[аоы]?$",
        r"^(?:в\s+целом\s+)?(?:мы\s+)?(?:к\s+\w+\s+)?готов[аоы]?$",
        r"^(?:мы\s+)?готов[аоы]?\s+к\s+.+$",
    ]
    return any(re.search(pattern, text) for pattern in vague_patterns)


def _has_concrete_done_object(task_text: str) -> bool:
    if _looks_like_person_only_done(task_text):
        return False

    object_terms = [
        term
        for term in _content_terms(task_text)
        if term not in DONE_VERB_TERMS and term not in WEAK_DONE_OBJECT_TERMS
    ]
    return bool(object_terms)


def _looks_like_person_only_done(task_text: str) -> bool:
    return bool(
        re.search(
            r"^\s*[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]\.)?\s+"
            r"(?:подготовил|подготовила|сделал|сделала|выполнил|"
            r"выполнила|написал|написала|разработал|разработала)\b",
            task_text,
        )
    )


def _normalize_key(value: str) -> str:
    value = value.lower().replace("ё", "е")
    value = re.sub(r"[^а-яa-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _normalize_quote(value: str) -> str:
    value = value.lower().replace("ё", "е")
    value = re.sub(r"[\[\]\(\)\"«»“”]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _evidence_supports_task(task: str, evidence: str) -> bool:
    task_terms = _content_terms(task)
    if not task_terms:
        return False

    evidence_terms = _content_terms(evidence)
    matches = sum(
        1
        for task_term in task_terms
        if any(_same_term_family(task_term, evidence_term) for evidence_term in evidence_terms)
    )
    required_matches = min(2, len(task_terms))
    return matches >= required_matches


def _content_terms(value: str) -> list[str]:
    stop_words = {
        "для",
        "или",
        "это",
        "что",
        "как",
        "уже",
        "еще",
        "ещё",
        "там",
        "тут",
        "вот",
        "надо",
        "нужно",
        "будем",
    }
    terms = re.findall(r"[а-яa-z0-9]+", _normalize_key(value))
    return [
        term
        for term in terms
        if len(term) >= 4 and term not in stop_words and not term.isdigit()
    ]


def _same_term_family(left: str, right: str) -> bool:
    if left == right:
        return True
    if len(left) >= 5 and len(right) >= 5:
        return left[:5] == right[:5] or SequenceMatcher(None, left, right).ratio() >= 0.66
    return False


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _speaker_from_evidence(evidence: str) -> str:
    match = re.match(
        r"^\s*(?:\[\d{1,5}\]\s*)?(?P<speaker>[^:\n]{1,120}):",
        evidence,
    )
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group("speaker")).strip()


def row_signature(row: pd.Series | dict[str, str]) -> str:
    return "|".join(
        [
            row["Блок"],
            _normalize_key(row["Ответственный"]),
            _normalize_key(row["Задача"]),
            row["Срок"],
        ]
    )


def _task_similarity(left: str, right: str) -> float:
    left_key = _normalize_key(left)
    right_key = _normalize_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key in right_key or right_key in left_key:
        return 1.0
    return SequenceMatcher(None, left_key, right_key).ratio()


def _deduplicate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    unique: list[dict[str, str]] = []
    for row in rows:
        duplicate_idx = _find_duplicate_row_idx(row, unique)
        if duplicate_idx is None:
            unique.append(row)
            continue
        unique[duplicate_idx] = _merge_duplicate_rows(unique[duplicate_idx], row)
    return unique


def _find_duplicate_row_idx(
    row: dict[str, str],
    unique: list[dict[str, str]],
) -> int | None:
    for idx, existing in enumerate(unique):
        if not _same_row_group(existing, row):
            continue
        if _task_similarity(existing["Задача"], row["Задача"]) >= 0.88:
            return idx
    return None


def _same_row_group(left: dict[str, str], right: dict[str, str]) -> bool:
    return (
        left["Блок"] == right["Блок"]
        and _normalize_key(left["Ответственный"]) == _normalize_key(right["Ответственный"])
        and left["Срок"] == right["Срок"]
    )


def _merge_duplicate_rows(
    left: dict[str, str],
    right: dict[str, str],
) -> dict[str, str]:
    merged = dict(left)
    if len(right["Задача"]) > len(left["Задача"]):
        merged["Задача"] = right["Задача"]
    if len(right["Обоснование"]) > len(left["Обоснование"]):
        merged["Обоснование"] = right["Обоснование"]
    return merged
