from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

import pandas as pd

from src.conversation_lexicon import (
    DONE_REJECTION_PATTERNS,
    NEW_ASSIGNMENT_PATTERNS,
    ONGOING_STATE_PATTERNS,
    WEAK_DONE_PATTERNS,
)
from src.date_normalizer import DateReplacement, normalize_deadline
from src.schemas import BLOCK_ORDER, DATAFRAME_COLUMNS, ExtractedTask, TaskAnchor
from src.semantic_similarity import (
    SEMANTIC_EVIDENCE_THRESHOLD,
    SEMANTIC_TASK_THRESHOLD,
    semantic_similarity,
)
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
RESPONSIBLE_ORG_TAIL_WORDS = {
    "м",
    "п",
    "мп",
    "ооо",
    "ао",
    "зао",
    "пао",
    "ип",
    "ано",
    "нко",
    "фгуп",
    "муп",
}
RESPONSIBLE_ORG_CONTEXT_WORDS = {
    "компания",
    "организация",
    "общество",
    "филиал",
    "департамент",
    "отдел",
    "служба",
    "управление",
    "завод",
    "цех",
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
    "подготовительные",
    "работы",
}
GENERIC_ALL_DONE_OBJECT_TERMS = {
    "цели",
    "задачи",
    "работы",
    "вопросы",
    "пункты",
}
ONGOING_NEW_PATTERNS = ONGOING_STATE_PATTERNS + [
    r"\bготовим(?:ся|сь)\b",
]
NEW_PLANNING_DEADLINE_PATTERNS = [
    r"\bкакой\s+срок\b",
    r"\bсрок\s+общий\b",
    r"\bсрок\b.+\bобсудим\b",
]
NEW_ACTION_PATTERNS = [
    r"\bсформир",
    r"\bподготов",
    r"\bсобра(?:ть|ться|лись|ться)\b",
    r"\bсобер(?:ем|ём|емся|етесь|ется|итесь)\b",
    r"\bпровести\b",
    r"\bсогласова",
    r"\bсогласу",
    r"\bдоработа",
    r"\bнаправ",
    r"\bначать\b",
    r"\bутверд",
    r"\bобсуд",
    r"\bназнач",
    r"\bпровер",
    r"\bразосл",
    r"\bпришл",
    r"\bвысл",
    r"\bсдела",
    r"\bпереда",
]
NEW_TASK_INTENT_PATTERNS = [
    r"\bзадач",
    r"\bпод\s+протокол\b",
    r"\bв\s+протокол\b",
    r"\bнужно\b",
    r"\bнадо\b",
    r"\bдолжн",
    r"\bпрошу\b",
    r"\bпросьба\b",
    r"\bпоруч",
    r"\bдавайте\b",
    r"\bпредлагаю\b",
    r"\bот\s+тебя\b",
    r"\bбер[её]м\s+в\s+работу\b",
    r"\bвозьм(?:ем|ём|ите)\b",
    r"\bдоговорил[аи]сь\b",
    r"\bдоговоримся\b",
    r"\bзафиксируем\b",
    r"\bфиксируем\b",
    r"\bставим\s+срок\b",
    r"\bбудем\s+(?:делать|готовить|обсуждать|согласовывать|"
    r"дорабатывать|проверять|направлять|проводить)\b",
]
NEW_CALENDAR_STATE_PATTERNS = [
    r"\bу\s+нас\s+(?:есть\s+|будет\s+)?(?:встреча|совещание|созвон)\b",
]
DONE_EVIDENCE_REJECTION_PATTERNS = [
    r"\bпосмотрю\b.+\bразработан",
    r"\bутвердил[аи]?\b.+\bжд[её]м\b",
    r"\bкак\s+я\s+понимаю\b.+\bзадач\w*\b.+\bвыполнил",
]
SEMANTIC_TERM_GROUP_PREFIXES = (
    ("встреч", "собер", "совещ", "созвон", "обсуд"),
    ("отправ", "направ", "высл", "пересл"),
    ("утверд", "соглас"),
)


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
            or _is_responsible_org_tail(token, has_name=bool(tokens))
        ):
            break
        if token in RESPONSIBLE_CONNECTOR_WORDS or token in RESPONSIBLE_TASK_WORDS:
            continue
        if len(token) < 3:
            if tokens:
                break
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
        token not in RESPONSIBLE_TASK_WORDS
        and not _is_responsible_org_tail(token, has_name=True)
        for token in tokens
    )


def _is_responsible_org_tail(token: str, has_name: bool) -> bool:
    if token in RESPONSIBLE_ORG_CONTEXT_WORDS:
        return True
    return token in RESPONSIBLE_ORG_TAIL_WORDS and has_name


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
    if _matches_any(_normalize_key(task.evidence), DONE_EVIDENCE_REJECTION_PATTERNS):
        return False
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
    if not _new_task_has_action_support(task.task, task.evidence):
        return False
    if _matches_any(evidence, ONGOING_NEW_PATTERNS) and not _matches_any(
        evidence,
        NEW_ASSIGNMENT_PATTERNS,
    ):
        return False
    if _matches_any(evidence, NEW_PLANNING_DEADLINE_PATTERNS) and not _matches_any(
        _normalize_key(task.task),
        NEW_ACTION_PATTERNS,
    ):
        return False
    return True


def _new_task_has_action_support(task_text: str, evidence: str) -> bool:
    evidence_key = _normalize_key(evidence)
    task_key = _normalize_key(task_text)
    if _matches_any(evidence_key, NEW_ACTION_PATTERNS):
        return True
    has_intent = _matches_any(evidence_key, NEW_TASK_INTENT_PATTERNS)
    if _matches_any(evidence_key, NEW_CALENDAR_STATE_PATTERNS) and not has_intent:
        return False
    if has_intent and _matches_any(task_key, NEW_ACTION_PATTERNS):
        return True
    return _has_strong_semantic_action_overlap(
        _content_terms(task_text),
        _content_terms(evidence),
    )


def _has_explicit_done_signal(evidence: str) -> bool:
    return has_done_signal(evidence)


def _done_clause_supports_task(task_text: str, evidence: str) -> bool:
    task_terms = _content_terms(task_text)
    if not task_terms:
        return False

    for clause in _split_clauses(evidence):
        if not _has_explicit_done_signal(clause):
            continue
        if _has_done_rejection_signal(clause):
            continue
        clause_terms = _content_terms(clause)
        matches = sum(
            1
            for task_term in task_terms
            if any(_terms_support_same_meaning(task_term, clause_term) for clause_term in clause_terms)
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
    return any(
        re.search(pattern, text)
        for pattern in DONE_REJECTION_PATTERNS + WEAK_DONE_PATTERNS
    )


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
    if _is_generic_all_done_without_specific_object(text):
        return True

    vague_patterns = [
        r"^(?:он|она|оно|они|это)\s+готов[аоы]?$",
        r"^(?:в\s+целом\s+)?(?:мы\s+)?(?:к\s+\w+\s+)?готов[аоы]?$",
        r"^(?:мы\s+)?готов[аоы]?\s+к\s+.+$",
    ]
    return any(re.search(pattern, text) for pattern in vague_patterns + WEAK_DONE_PATTERNS)


def _is_generic_all_done_without_specific_object(text: str) -> bool:
    match = re.fullmatch(
        r"все\s+(?P<object>[а-яa-z0-9 ]{3,80}?)\s+выполнен[аоы]?",
        text,
    )
    if not match:
        return False
    object_terms = _content_terms(match.group("object"))
    if not object_terms:
        return True
    return all(term in GENERIC_ALL_DONE_OBJECT_TERMS for term in object_terms)


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
    matches = _count_supported_task_terms(task_terms, evidence_terms)
    required_matches = min(2, len(task_terms))
    if matches >= required_matches:
        return True
    if _has_strong_semantic_action_overlap(task_terms, evidence_terms):
        return True

    score = semantic_similarity(task, evidence)
    return score is not None and score >= SEMANTIC_EVIDENCE_THRESHOLD


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


def _count_supported_task_terms(
    task_terms: list[str],
    evidence_terms: list[str],
) -> int:
    return sum(
        1
        for task_term in task_terms
        if any(
            _terms_support_same_meaning(task_term, evidence_term)
            for evidence_term in evidence_terms
        )
    )


def _terms_support_same_meaning(left: str, right: str) -> bool:
    return _same_term_family(left, right) or _same_semantic_term_group(left, right)


def _same_semantic_term_group(left: str, right: str) -> bool:
    for prefixes in SEMANTIC_TERM_GROUP_PREFIXES:
        if _matches_prefix_group(left, prefixes) and _matches_prefix_group(right, prefixes):
            return True
    return False


def _matches_prefix_group(term: str, prefixes: tuple[str, ...]) -> bool:
    return any(term.startswith(prefix) for prefix in prefixes)


def _has_strong_semantic_action_overlap(
    task_terms: list[str],
    evidence_terms: list[str],
) -> bool:
    if len(task_terms) > 3:
        return False
    return any(
        _same_semantic_term_group(task_term, evidence_term)
        for task_term in task_terms
        for evidence_term in evidence_terms
    )


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
    lexical_score = SequenceMatcher(None, left_key, right_key).ratio()
    semantic_score = semantic_similarity(left, right)
    if semantic_score is None:
        return lexical_score
    return max(lexical_score, semantic_score if semantic_score >= SEMANTIC_TASK_THRESHOLD else 0.0)


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
        if _same_source_relaxed_duplicate(existing, row):
            return idx
        if _same_failed_status_duplicate(existing, row):
            return idx
        if _same_generic_done_evidence(existing, row):
            return idx
        if _task_similarity(existing["Задача"], row["Задача"]) >= 0.88:
            return idx
    return None


def _same_row_group(left: dict[str, str], right: dict[str, str]) -> bool:
    return (
        left["Блок"] == right["Блок"]
        and _normalize_key(left["Ответственный"]) == _normalize_key(right["Ответственный"])
        and left["Срок"] == right["Срок"]
    )


def _same_generic_done_evidence(left: dict[str, str], right: dict[str, str]) -> bool:
    return (
        left["Блок"] == "Выполненные"
        and _is_all_done_summary(left["Задача"])
        and _is_all_done_summary(right["Задача"])
        and bool(_evidence_refs(left["Обоснование"]) & _evidence_refs(right["Обоснование"]))
    )


def _same_source_evidence(left: dict[str, str], right: dict[str, str]) -> bool:
    left_refs = _evidence_refs(left["Обоснование"])
    right_refs = _evidence_refs(right["Обоснование"])
    if left_refs and right_refs:
        return bool(left_refs & right_refs)
    return (
        SequenceMatcher(
            None,
            _normalize_quote(left["Обоснование"]),
            _normalize_quote(right["Обоснование"]),
        ).ratio()
        >= 0.82
    )


def _same_source_relaxed_duplicate(
    left: dict[str, str],
    right: dict[str, str],
) -> bool:
    if not _same_source_evidence(left, right):
        return False
    if _task_similarity(left["Задача"], right["Задача"]) >= 0.45:
        return True

    left_terms = _content_terms(left["Задача"])
    right_terms = _content_terms(right["Задача"])
    if not left_terms or not right_terms:
        return False
    matches = sum(
        1
        for left_term in left_terms
        if any(_same_term_family(left_term, right_term) for right_term in right_terms)
    )
    return matches >= min(2, len(left_terms), len(right_terms))


def _same_failed_status_duplicate(
    left: dict[str, str],
    right: dict[str, str],
) -> bool:
    if left["Блок"] != "Невыполненные":
        return False
    left_text = _normalize_key(left["Задача"] + " " + left["Обоснование"])
    right_text = _normalize_key(right["Задача"] + " " + right["Обоснование"])
    if not has_failed_signal(left_text) or not has_failed_signal(right_text):
        return False
    return _same_source_evidence(left, right) or _task_similarity(
        left["Задача"],
        right["Задача"],
    ) >= 0.55


def _is_all_done_summary(task_text: str) -> bool:
    return bool(re.search(r"\bвсе\b.+\bвыполн", _normalize_key(task_text)))


def _evidence_refs(evidence: str) -> set[str]:
    return set(re.findall(r"\[(\d{1,5})\]", evidence))


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
