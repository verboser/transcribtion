from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Callable

import pandas as pd

from src.date_normalizer import DateReplacement, normalize_deadline
from src.postprocess import build_dataframe_with_stats, row_signature
from src.schemas import BLOCK_ORDER, DATAFRAME_COLUMNS, ExtractionResult
from src.semantic_similarity import SEMANTIC_TASK_THRESHOLD, semantic_similarity


Extractor = Callable[[Path, str], ExtractionResult]


@dataclass
class StabilityReport:
    extractions: list[ExtractionResult]
    report_lines: list[str]
    selected_run_idx: int | None
    final_df: pd.DataFrame
    final_replacements: list[DateReplacement]
    final_source: str


def run_stability_check(
    transcript_path: Path,
    meeting_date: str,
    runs: int,
    extractor: Extractor,
) -> StabilityReport:
    extractions: list[ExtractionResult] = []
    report_lines: list[str] = []
    count_rows: list[dict[str, int]] = []
    signature_rows: list[set[str]] = []
    raw_counts: list[int] = []
    valid_counts: list[int] = []
    final_counts: list[int] = []
    filtered_counts: list[int] = []
    dedup_removed_counts: list[int] = []
    dataframes: list[pd.DataFrame] = []
    replacements_by_run: list[list[DateReplacement]] = []

    for run_idx in range(1, runs + 1):
        result = extractor(transcript_path, meeting_date)
        extractions.append(result)
        df, replacements, stats = build_dataframe_with_stats(
            result.tasks,
            meeting_date,
            result.anchors,
        )
        dataframes.append(df)
        replacements_by_run.append(replacements)
        counts = _count_dataframe(df)
        count_rows.append(counts)
        signatures = {row_signature(row) for _, row in df.iterrows()}
        signature_rows.append(signatures)
        raw_counts.append(stats.raw_rows)
        valid_counts.append(stats.valid_rows)
        final_counts.append(stats.final_rows)
        filtered_counts.append(stats.filtered_rows)
        dedup_removed_counts.append(stats.dedup_removed_rows)
        report_lines.append(
            "run_{idx}: Выполненные={done}, Невыполненные={failed}, "
            "Новые={new}, Всего={total}".format(
                idx=run_idx,
                done=counts["Выполненные"],
                failed=counts["Невыполненные"],
                new=counts["Новые"],
                total=counts["Всего"],
            )
        )

    baseline = count_rows[0] if count_rows else {}
    unstable_runs = [
        f"run_{idx}"
        for idx, counts in enumerate(count_rows, start=1)
        if counts != baseline
    ]

    stable = not unstable_runs
    report_lines.append(
        f"Стабильность по количеству: {'Да' if stable else 'Нет'}"
    )
    if unstable_runs:
        report_lines.append(
            "Отличаются прогоны: " + ", ".join(unstable_runs)
        )

    if signature_rows:
        baseline_signatures = signature_rows[0]
        jaccards = [
            _jaccard(baseline_signatures, signatures)
            for signatures in signature_rows
        ]
        report_lines.append(
            "Jaccard точных сигнатур к run_1: "
            + ", ".join(
                f"run_{idx}={score:.2f}"
                for idx, score in enumerate(jaccards, start=1)
            )
        )
        one_off_count = _count_one_off_consensus_groups(dataframes)
        report_lines.append(f"Задач, появившихся только в одном прогоне: {one_off_count}")
        centroid_scores = _average_jaccards(signature_rows)
        report_lines.append(
            "Средний Jaccard к остальным: "
            + ", ".join(
                f"run_{idx}={score:.2f}"
                for idx, score in enumerate(centroid_scores, start=1)
            )
        )
        selected_run_idx = _select_centroid_run_idx(centroid_scores)
        report_lines.append(f"Финальный прогон по centroid: run_{selected_run_idx + 1}")
    else:
        selected_run_idx = None

    if raw_counts:
        report_lines.append(
            "Postprocess: raw_avg={raw:.1f}, valid_avg={valid:.1f}, "
            "final_avg={final:.1f}, filtered_avg={filtered:.1f}, "
            "dedup_removed_avg={dedup_removed:.1f}".format(
                raw=sum(raw_counts) / len(raw_counts),
                valid=sum(valid_counts) / len(valid_counts),
                final=sum(final_counts) / len(final_counts),
                filtered=sum(filtered_counts) / len(filtered_counts),
                dedup_removed=sum(dedup_removed_counts) / len(dedup_removed_counts),
            )
        )

    support_threshold = _consensus_support_threshold(dataframes)
    consensus_df = _build_consensus_dataframe(
        dataframes,
        selected_run_idx=selected_run_idx,
        support_threshold=support_threshold,
    )
    union_count = _count_consensus_groups(dataframes)
    report_lines.append(
        f"Consensus support>={support_threshold}: "
        f"{len(consensus_df)} из {union_count} уникальных строк"
    )

    final_df, final_replacements, final_source = _select_final_dataframe(
        dataframes=dataframes,
        replacements_by_run=replacements_by_run,
        selected_run_idx=selected_run_idx,
        meeting_date=meeting_date,
    )
    report_lines.append(f"Источник финального результата: {final_source}")

    return StabilityReport(
        extractions=extractions,
        report_lines=report_lines,
        selected_run_idx=selected_run_idx,
        final_df=final_df,
        final_replacements=final_replacements,
        final_source=final_source,
    )


def _count_dataframe(df) -> dict[str, int]:
    counts = df["Блок"].value_counts().to_dict() if not df.empty else {}
    return {
        "Выполненные": counts.get("Выполненные", 0),
        "Невыполненные": counts.get("Невыполненные", 0),
        "Новые": counts.get("Новые", 0),
        "Всего": len(df),
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _average_jaccards(signature_rows: list[set[str]]) -> list[float]:
    if len(signature_rows) == 1:
        return [1.0]

    scores: list[float] = []
    for idx, signatures in enumerate(signature_rows):
        other_scores = [
            _jaccard(signatures, other)
            for other_idx, other in enumerate(signature_rows)
            if other_idx != idx
        ]
        scores.append(sum(other_scores) / len(other_scores))
    return scores


def _select_centroid_run_idx(centroid_scores: list[float]) -> int:
    return max(
        range(len(centroid_scores)),
        key=lambda idx: (centroid_scores[idx], -idx),
    )


def _count_one_off_consensus_groups(dataframes: list[pd.DataFrame]) -> int:
    return sum(1 for group in _build_consensus_groups(dataframes) if len(group) == 1)


def _select_final_dataframe(
    dataframes: list[pd.DataFrame],
    replacements_by_run: list[list[DateReplacement]],
    selected_run_idx: int | None,
    meeting_date: str,
) -> tuple[pd.DataFrame, list[DateReplacement], str]:
    support_threshold = _consensus_support_threshold(dataframes)
    consensus_df = _build_consensus_dataframe(
        dataframes,
        selected_run_idx=selected_run_idx,
        support_threshold=support_threshold,
    )
    union_count = _count_consensus_groups(dataframes)

    if not consensus_df.empty or union_count == 0:
        return (
            consensus_df,
            _build_consensus_replacements(consensus_df, meeting_date),
            f"consensus support>={support_threshold}",
        )

    fallback_idx = selected_run_idx if selected_run_idx is not None else len(dataframes) - 1
    return (
        dataframes[fallback_idx].copy(),
        replacements_by_run[fallback_idx],
        f"centroid fallback run_{fallback_idx + 1}",
    )


def _consensus_support_threshold(dataframes: list[pd.DataFrame]) -> int:
    return 1 if len(dataframes) == 1 else 2


def _build_consensus_dataframe(
    dataframes: list[pd.DataFrame],
    selected_run_idx: int | None,
    support_threshold: int,
) -> pd.DataFrame:
    groups = _build_consensus_groups(dataframes)

    preferred_run_idx = selected_run_idx if selected_run_idx is not None else 0
    rows: list[dict[str, str]] = []
    for run_rows in groups:
        if len(run_rows) < support_threshold:
            continue
        if preferred_run_idx in run_rows:
            rows.append(dict(run_rows[preferred_run_idx]))
        else:
            rows.append(dict(run_rows[min(run_rows)]))

    rows.sort(
        key=lambda row: (
            BLOCK_ORDER.get(row["Блок"], 99),
            row["Ответственный"],
            row["Срок"],
            row["Задача"],
        )
    )
    return pd.DataFrame(rows, columns=DATAFRAME_COLUMNS)


def _build_consensus_groups(
    dataframes: list[pd.DataFrame],
) -> list[dict[int, dict[str, str]]]:
    groups: list[dict[int, dict[str, str]]] = []
    for run_idx, df in enumerate(dataframes):
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            group_idx = _find_consensus_group_idx(row_dict, groups)
            if group_idx is None:
                groups.append({run_idx: row_dict})
                continue
            groups[group_idx].setdefault(run_idx, row_dict)
    return groups


def _count_consensus_groups(dataframes: list[pd.DataFrame]) -> int:
    return len(_build_consensus_groups(dataframes))


def _find_consensus_group_idx(
    row: dict[str, str],
    groups: list[dict[int, dict[str, str]]],
) -> int | None:
    for idx, group in enumerate(groups):
        representative = next(iter(group.values()))
        if _same_consensus_group(representative, row):
            return idx
    return None


def _same_consensus_group(left: dict[str, str], right: dict[str, str]) -> bool:
    return (
        left["Блок"] == right["Блок"]
        and _normalize_key(left["Ответственный"]) == _normalize_key(right["Ответственный"])
        and left["Срок"] == right["Срок"]
        and _evidence_compatible(left["Обоснование"], right["Обоснование"])
        and _task_similarity(left["Задача"], right["Задача"]) >= 0.82
    )


def _evidence_compatible(left: str, right: str) -> bool:
    left_refs = set(re.findall(r"\[(\d{1,5})\]", left))
    right_refs = set(re.findall(r"\[(\d{1,5})\]", right))
    if left_refs and right_refs:
        return bool(left_refs & right_refs)
    return SequenceMatcher(None, _normalize_key(left), _normalize_key(right)).ratio() >= 0.78


def _task_similarity(left: str, right: str) -> float:
    left_key = _normalize_key(left)
    right_key = _normalize_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key in right_key or right_key in left_key:
        return 1.0
    lexical_score = max(
        SequenceMatcher(None, left_key, right_key).ratio(),
        _term_jaccard(_content_terms(left_key), _content_terms(right_key)),
    )
    semantic_score = semantic_similarity(left, right)
    if semantic_score is None:
        return lexical_score
    return max(
        lexical_score,
        semantic_score if semantic_score >= SEMANTIC_TASK_THRESHOLD else 0.0,
    )


def _term_jaccard(left_terms: list[str], right_terms: list[str]) -> float:
    left = {_term_family(term) for term in left_terms}
    right = {_term_family(term) for term in right_terms}
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


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


def _build_consensus_replacements(
    df: pd.DataFrame,
    meeting_date: str,
) -> list[DateReplacement]:
    replacements: list[DateReplacement] = []
    seen: set[tuple[str, str]] = set()

    for _, row in df.iterrows():
        if not row["Срок"]:
            continue
        deadline, replacement = normalize_deadline(
            deadline_raw="",
            evidence=row["Обоснование"],
            meeting_date=meeting_date,
            task_text=row["Задача"] if row["Блок"] == "Новые" else "",
        )
        if not replacement or deadline != row["Срок"]:
            continue
        key = (replacement.source, replacement.normalized)
        if key in seen:
            continue
        seen.add(key)
        replacements.append(replacement)

    return replacements
