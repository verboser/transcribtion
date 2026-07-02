from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import pandas as pd

from src.schemas import DATAFRAME_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_DIR = PROJECT_ROOT / "tests" / "fixtures" / "golden"


@dataclass(frozen=True)
class GoldenTask:
    task_id: str
    block: str
    task: str
    responsible: str
    deadline: str
    evidence_contains: tuple[str, ...]


@dataclass(frozen=True)
class GoldenMatch:
    golden: GoldenTask
    row_idx: int
    task_similarity: float
    responsible_match: bool
    deadline_match: bool

    @property
    def full_match(self) -> bool:
        return self.responsible_match and self.deadline_match


@dataclass(frozen=True)
class GoldenEvalReport:
    transcript_name: str
    expected_count: int
    predicted_count: int
    matched_count: int
    full_match_count: int
    precision: float
    recall: float
    full_match_recall: float
    responsible_accuracy: float
    deadline_accuracy: float
    missed: tuple[GoldenTask, ...]
    false_positive_rows: tuple[dict[str, str], ...]
    field_mismatches: tuple[GoldenMatch, ...]

    def format_lines(self) -> list[str]:
        lines = [
            "Golden eval:",
            (
                "expected={expected}, predicted={predicted}, matched={matched}, "
                "full={full}, precision={precision:.2f}, recall={recall:.2f}, "
                "full_recall={full_recall:.2f}"
            ).format(
                expected=self.expected_count,
                predicted=self.predicted_count,
                matched=self.matched_count,
                full=self.full_match_count,
                precision=self.precision,
                recall=self.recall,
                full_recall=self.full_match_recall,
            ),
            (
                "field accuracy: responsible={responsible:.2f}, "
                "deadline={deadline:.2f}"
            ).format(
                responsible=self.responsible_accuracy,
                deadline=self.deadline_accuracy,
            ),
        ]

        if self.missed:
            lines.append("Missed expected:")
            lines.extend(
                f"- [{task.task_id}] {task.block}: {task.task}"
                for task in self.missed
            )

        if self.false_positive_rows:
            lines.append("False positives:")
            lines.extend(
                "- {block}: {task} | {responsible} | {deadline}".format(
                    block=row["Блок"],
                    task=row["Задача"],
                    responsible=row["Ответственный"],
                    deadline=row["Срок"] or "-",
                )
                for row in self.false_positive_rows
            )

        if self.field_mismatches:
            lines.append("Field mismatches:")
            for match in self.field_mismatches:
                problems = []
                if not match.responsible_match:
                    problems.append(
                        f"responsible expected={match.golden.responsible!r}"
                    )
                if not match.deadline_match:
                    problems.append(f"deadline expected={match.golden.deadline!r}")
                lines.append(
                    f"- [{match.golden.task_id}] {match.golden.task}: "
                    + ", ".join(problems)
                )

        return lines


def load_golden_tasks(
    transcript_name: str,
    golden_dir: Path = DEFAULT_GOLDEN_DIR,
) -> list[GoldenTask]:
    path = golden_dir / f"{Path(transcript_name).stem}.json"
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        GoldenTask(
            task_id=item["id"],
            block=item["block"],
            task=item["task"],
            responsible=item.get("responsible", ""),
            deadline=item.get("deadline", ""),
            evidence_contains=tuple(item.get("evidence_contains", [])),
        )
        for item in payload.get("tasks", [])
    ]


def evaluate_dataframe_against_golden(
    df: pd.DataFrame,
    golden_tasks: list[GoldenTask],
    transcript_name: str,
) -> GoldenEvalReport:
    rows = _rows_from_dataframe(df)
    matches: list[GoldenMatch] = []
    used_rows: set[int] = set()

    for golden in golden_tasks:
        match = _find_best_match(golden, rows, used_rows)
        if match is None:
            continue
        matches.append(match)
        used_rows.add(match.row_idx)

    missed_ids = {match.golden.task_id for match in matches}
    missed = tuple(task for task in golden_tasks if task.task_id not in missed_ids)
    false_positive_rows = tuple(
        row for idx, row in enumerate(rows) if idx not in used_rows
    )
    field_mismatches = tuple(match for match in matches if not match.full_match)

    matched_count = len(matches)
    full_match_count = sum(1 for match in matches if match.full_match)
    responsible_matches = sum(1 for match in matches if match.responsible_match)
    deadline_matches = sum(1 for match in matches if match.deadline_match)

    return GoldenEvalReport(
        transcript_name=transcript_name,
        expected_count=len(golden_tasks),
        predicted_count=len(rows),
        matched_count=matched_count,
        full_match_count=full_match_count,
        precision=_safe_div(matched_count, len(rows)),
        recall=_safe_div(matched_count, len(golden_tasks)),
        full_match_recall=_safe_div(full_match_count, len(golden_tasks)),
        responsible_accuracy=_safe_div(responsible_matches, matched_count),
        deadline_accuracy=_safe_div(deadline_matches, matched_count),
        missed=missed,
        false_positive_rows=false_positive_rows,
        field_mismatches=field_mismatches,
    )


def _rows_from_dataframe(df: pd.DataFrame) -> list[dict[str, str]]:
    if df.empty:
        return []
    return [
        {column: str(row[column]) for column in DATAFRAME_COLUMNS}
        for _, row in df.iterrows()
    ]


def _find_best_match(
    golden: GoldenTask,
    rows: list[dict[str, str]],
    used_rows: set[int],
) -> GoldenMatch | None:
    candidates: list[GoldenMatch] = []
    for row_idx, row in enumerate(rows):
        if row_idx in used_rows or row["Блок"] != golden.block:
            continue

        evidence_supported = _evidence_supported(
            row["Обоснование"],
            golden.evidence_contains,
        )
        task_similarity = _task_similarity(row["Задача"], golden.task)
        if not evidence_supported and task_similarity < 0.68:
            continue

        candidates.append(
            GoldenMatch(
                golden=golden,
                row_idx=row_idx,
                task_similarity=task_similarity,
                responsible_match=(
                    not golden.responsible
                    or _normalize_key(row["Ответственный"])
                    == _normalize_key(golden.responsible)
                ),
                deadline_match=(row["Срок"] == golden.deadline),
            )
        )

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda match: (
            match.full_match,
            match.task_similarity,
            match.responsible_match,
            match.deadline_match,
            -match.row_idx,
        ),
    )


def _evidence_supported(evidence: str, expected_snippets: tuple[str, ...]) -> bool:
    if not expected_snippets:
        return False
    normalized_evidence = _normalize_key(evidence)
    return all(
        _normalize_key(snippet) in normalized_evidence
        for snippet in expected_snippets
    )


def _task_similarity(left: str, right: str) -> float:
    left_terms = _content_terms(left)
    right_terms = _content_terms(right)
    if not left_terms or not right_terms:
        return 0.0

    left_families = {_term_family(term) for term in left_terms}
    right_families = {_term_family(term) for term in right_terms}
    return len(left_families & right_families) / len(left_families | right_families)


def _content_terms(value: str) -> list[str]:
    return [
        term
        for term in re.findall(r"[а-яa-z0-9]+", _normalize_key(value))
        if len(term) >= 4 and not term.isdigit()
    ]


def _term_family(term: str) -> str:
    return term[:7] if len(term) >= 7 else term


def _normalize_key(value: str) -> str:
    value = value.lower().replace("ё", "е")
    value = re.sub(r"[^а-яa-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator
