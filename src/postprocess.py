from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

import pandas as pd

from src.date_normalizer import DateReplacement, normalize_deadline
from src.schemas import BLOCK_ORDER, DATAFRAME_COLUMNS, ExtractedTask, TaskAnchor


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
    if task.block == "Невыполненные":
        return _has_explicit_failure_signal(task.evidence)
    return True


def _normalize_task_deadline(
    task: ExtractedTask,
    evidence: str,
    meeting_date: str,
) -> tuple[str, DateReplacement | None]:
    deadline_raw = "" if task.block == "Новые" else task.deadline_raw
    return normalize_deadline(
        deadline_raw=deadline_raw,
        evidence=evidence,
        meeting_date=meeting_date,
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


def _quote_is_supported(evidence: str, anchor_text: str) -> bool:
    normalized_evidence = _normalize_quote(evidence)
    normalized_anchor = _normalize_quote(anchor_text)
    if not normalized_evidence:
        return False
    return normalized_evidence in normalized_anchor


def _has_explicit_failure_signal(evidence: str) -> bool:
    text = evidence.lower().replace("ё", "е")
    failure_patterns = [
        r"\bне\s+успел[аи]?\b",
        r"\bне\s+успели\b",
        r"\bне\s+сделал[аи]?\b",
        r"\bне\s+сделали\b",
        r"\bне\s+выполн",
        r"\bне\s+закрыл",
        r"\bне\s+закрыт",
        r"\bне\s+готов",
        r"\bне\s+подготов",
        r"\bне\s+отправ",
        r"\bпросроч",
        r"\bсрок\s+прош",
        r"\bдедлайн\s+прош",
        r"\bсорвал",
    ]
    return any(re.search(pattern, text) for pattern in failure_patterns)


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
