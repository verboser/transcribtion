from __future__ import annotations

import re

import pandas as pd

from src.date_normalizer import DateReplacement, normalize_deadline
from src.schemas import BLOCK_ORDER, DATAFRAME_COLUMNS, ExtractedTask, TaskAnchor


def build_dataframe(
    tasks: list[ExtractedTask],
    meeting_date: str,
    anchors: list[TaskAnchor] | None = None,
) -> tuple[pd.DataFrame, list[DateReplacement]]:
    rows: list[dict[str, str]] = []
    replacements: list[DateReplacement] = []
    anchor_map = {anchor.anchor_id: anchor for anchor in anchors or []}

    for task in deduplicate_tasks(tasks):
        task_anchors = _resolve_task_anchors(task, anchor_map)
        if anchors is not None and not task_anchors:
            continue
        if not is_valid_task(task):
            continue

        evidence = _resolve_evidence(task, task_anchors)
        responsible = _resolve_responsible(task, task_anchors)
        deadline_source = task.deadline_raw or _first_anchor_deadline(task_anchors)
        deadline, replacement = normalize_deadline(
            deadline_raw=deadline_source,
            evidence=evidence,
            meeting_date=meeting_date,
        )
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

    rows.sort(
        key=lambda row: (
            BLOCK_ORDER.get(row["Блок"], 99),
            row["Ответственный"],
            row["Срок"],
            _normalize_key(row["Задача"]),
        )
    )
    return pd.DataFrame(rows, columns=DATAFRAME_COLUMNS), replacements


def deduplicate_tasks(tasks: list[ExtractedTask]) -> list[ExtractedTask]:
    unique: dict[tuple[str, str, str, str, tuple[str, ...]], ExtractedTask] = {}
    for task in tasks:
        key = (
            task.block,
            _normalize_key(task.task),
            _normalize_key(task.responsible),
            _normalize_key(task.deadline_raw),
            tuple(sorted(task.anchor_ids)),
        )
        if key not in unique or len(task.evidence) > len(unique[key].evidence):
            unique[key] = task
    return list(unique.values())


def count_by_block(
    tasks: list[ExtractedTask],
    meeting_date: str,
    anchors: list[TaskAnchor] | None = None,
) -> dict[str, int]:
    df, _ = build_dataframe(tasks, meeting_date, anchors)
    counts = df["Блок"].value_counts().to_dict() if not df.empty else {}
    return {
        "Выполненные": counts.get("Выполненные", 0),
        "Невыполненные": counts.get("Невыполненные", 0),
        "Новые": counts.get("Новые", 0),
        "Всего": len(df),
    }


def is_valid_task(task: ExtractedTask) -> bool:
    if task.block == "Невыполненные":
        return _has_explicit_failure_signal(task.evidence)
    return True


def _resolve_task_anchors(
    task: ExtractedTask,
    anchor_map: dict[str, TaskAnchor],
) -> list[TaskAnchor]:
    return [anchor_map[anchor_id] for anchor_id in task.anchor_ids if anchor_id in anchor_map]


def _resolve_evidence(task: ExtractedTask, anchors: list[TaskAnchor]) -> str:
    if not anchors:
        return task.evidence

    anchor_text = "\n".join(anchor.text() for anchor in anchors)
    if _quote_is_supported(task.evidence, anchor_text):
        return task.evidence

    return _compact_anchor_evidence(anchors)


def _resolve_responsible(task: ExtractedTask, anchors: list[TaskAnchor]) -> str:
    if not anchors:
        return task.responsible

    speakers = {anchor.speaker for anchor in anchors}
    if task.responsible in speakers:
        return task.responsible

    return anchors[0].speaker


def _first_anchor_deadline(anchors: list[TaskAnchor]) -> str:
    for anchor in anchors:
        if anchor.deadline_phrases:
            return anchor.deadline_phrases[0]
    return ""


def _quote_is_supported(evidence: str, anchor_text: str) -> bool:
    normalized_evidence = _normalize_quote(evidence)
    normalized_anchor = _normalize_quote(anchor_text)
    if not normalized_evidence:
        return False
    return normalized_evidence in normalized_anchor


def _compact_anchor_evidence(anchors: list[TaskAnchor]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for anchor in anchors:
        for utterance in anchor.utterances:
            line = utterance.as_prompt_line()
            if line not in seen:
                seen.add(line)
                lines.append(line)
    return " ".join(lines[:4])


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
