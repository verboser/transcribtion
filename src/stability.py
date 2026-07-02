from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import math
from pathlib import Path
import re
from typing import Callable

import pandas as pd

from src.date_normalizer import DateReplacement, normalize_deadline
from src.postprocess import build_dataframe_with_stats, row_signature
from src.schemas import (
    BLOCK_ORDER,
    DATAFRAME_COLUMNS,
    ExtractionResult,
    VerificationCandidate,
)
from src.semantic_similarity import SEMANTIC_TASK_THRESHOLD, semantic_similarity


Extractor = Callable[[Path, str], ExtractionResult]
ConsensusVerifier = Callable[[list[VerificationCandidate], str], set[str]]


@dataclass(frozen=True)
class FinalTaskSupport:
    block: str
    task: str
    responsible: str
    deadline: str
    evidence: str
    support_count: int
    support_ratio: float
    run_indices: tuple[int, ...]
    verification_status: str = "raw_consensus"


@dataclass
class StabilityReport:
    extractions: list[ExtractionResult]
    report_lines: list[str]
    selected_run_idx: int | None
    final_df: pd.DataFrame
    final_replacements: list[DateReplacement]
    final_source: str
    run_dataframes: list[pd.DataFrame]
    final_task_support: list[FinalTaskSupport]
    support_threshold: int
    min_consensus_share: float
    min_pairwise_consensus_jaccard: float
    max_count_delta: float
    stability_passed: bool
    verified_candidate_count: int
    verifier_candidate_count: int


def run_stability_check(
    transcript_path: Path,
    meeting_date: str,
    runs: int,
    extractor: Extractor,
    min_consensus_share: float = 0.90,
    max_allowed_count_delta: float = 0.10,
    verifier: ConsensusVerifier | None = None,
    min_verifier_support_share: float = 0.60,
) -> StabilityReport:
    if runs < 1:
        raise ValueError("runs must be at least 1.")
    if not 0 < min_consensus_share <= 1:
        raise ValueError("min_consensus_share must be in the (0, 1] interval.")
    if max_allowed_count_delta < 0:
        raise ValueError("max_allowed_count_delta must be non-negative.")
    if not 0 < min_verifier_support_share <= 1:
        raise ValueError("min_verifier_support_share must be in the (0, 1] interval.")

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

    for run_idx in range(1, runs + 1):
        result = extractor(transcript_path, meeting_date)
        extractions.append(result)
        df, _replacements, stats = build_dataframe_with_stats(
            result.tasks,
            meeting_date,
            result.anchors,
        )
        dataframes.append(df)
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

    consensus_groups = _build_consensus_groups(dataframes)
    support_threshold = _consensus_support_threshold(dataframes, min_consensus_share)
    verifier_support_threshold = _consensus_support_threshold(
        dataframes,
        min_verifier_support_share,
    )
    verifier_candidates, candidate_group_by_id = _build_verification_candidates(
        groups=consensus_groups,
        selected_run_idx=selected_run_idx,
        run_count=len(dataframes),
        support_threshold=support_threshold,
        verifier_support_threshold=verifier_support_threshold,
    )
    verified_candidate_ids = (
        verifier(verifier_candidates, meeting_date)
        if verifier is not None and verifier_candidates
        else set()
    )
    verified_group_indices = {
        candidate_group_by_id[candidate_id]
        for candidate_id in verified_candidate_ids
        if candidate_id in candidate_group_by_id
    }
    consensus_df = _build_consensus_dataframe(
        dataframes,
        selected_run_idx=selected_run_idx,
        support_threshold=support_threshold,
        groups=consensus_groups,
        verified_group_indices=verified_group_indices,
    )
    union_count = len(consensus_groups)
    strict_consensus_count = sum(
        1 for group in consensus_groups if len(group) >= support_threshold
    )
    report_lines.append(
        f"Consensus support>={support_threshold}: "
        f"{strict_consensus_count} из {union_count} уникальных строк"
    )
    report_lines.append(
        "Verifier candidates support>={threshold}: {accepted}/{total}".format(
            threshold=verifier_support_threshold,
            accepted=len(verified_group_indices),
            total=len(verifier_candidates),
        )
    )

    count_deltas = _relative_count_deltas(count_rows)
    max_count_delta = max(count_deltas.values(), default=0.0)
    report_lines.append(
        "Макс. расхождение количества: "
        + ", ".join(
            f"{key}={value:.0%}"
            for key, value in count_deltas.items()
        )
    )

    min_pairwise_consensus_jaccard = _min_pairwise_consensus_jaccard(
        consensus_groups,
        run_count=len(dataframes),
    )
    report_lines.append(
        "Минимальный pairwise consensus Jaccard: "
        f"{min_pairwise_consensus_jaccard:.2f}"
    )

    stability_passed = (
        max_count_delta <= max_allowed_count_delta
        and min_pairwise_consensus_jaccard >= min_consensus_share
    )
    report_lines.append(
        "Порог стабильности {share:.0%}: {status}".format(
            share=min_consensus_share,
            status="Да" if stability_passed else "Нет",
        )
    )

    final_df, final_replacements, final_source = _select_final_dataframe(
        dataframes=dataframes,
        selected_run_idx=selected_run_idx,
        meeting_date=meeting_date,
        min_consensus_share=min_consensus_share,
        groups=consensus_groups,
        verified_group_indices=verified_group_indices,
    )
    final_task_support = _build_final_task_support(
        final_df,
        consensus_groups,
        run_count=len(dataframes),
        verified_group_indices=verified_group_indices,
    )
    report_lines.append(f"Источник финального результата: {final_source}")

    return StabilityReport(
        extractions=extractions,
        report_lines=report_lines,
        selected_run_idx=selected_run_idx,
        final_df=final_df,
        final_replacements=final_replacements,
        final_source=final_source,
        run_dataframes=[df.copy() for df in dataframes],
        final_task_support=final_task_support,
        support_threshold=support_threshold,
        min_consensus_share=min_consensus_share,
        min_pairwise_consensus_jaccard=min_pairwise_consensus_jaccard,
        max_count_delta=max_count_delta,
        stability_passed=stability_passed,
        verified_candidate_count=len(verified_group_indices),
        verifier_candidate_count=len(verifier_candidates),
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
    selected_run_idx: int | None,
    meeting_date: str,
    min_consensus_share: float,
    groups: list[dict[int, dict[str, str]]],
    verified_group_indices: set[int],
) -> tuple[pd.DataFrame, list[DateReplacement], str]:
    support_threshold = _consensus_support_threshold(dataframes, min_consensus_share)
    consensus_df = _build_consensus_dataframe(
        dataframes,
        selected_run_idx=selected_run_idx,
        support_threshold=support_threshold,
        groups=groups,
        verified_group_indices=verified_group_indices,
    )
    source = f"consensus support>={support_threshold} ({min_consensus_share:.0%})"
    if verified_group_indices:
        source += f" + verified_consensus={len(verified_group_indices)}"
    return (
        consensus_df,
        _build_consensus_replacements(consensus_df, meeting_date),
        source,
    )


def _consensus_support_threshold(
    dataframes: list[pd.DataFrame],
    min_consensus_share: float,
) -> int:
    if not dataframes:
        return 0
    return max(1, math.ceil(len(dataframes) * min_consensus_share))


def _build_consensus_dataframe(
    dataframes: list[pd.DataFrame],
    selected_run_idx: int | None,
    support_threshold: int,
    groups: list[dict[int, dict[str, str]]] | None = None,
    verified_group_indices: set[int] | None = None,
) -> pd.DataFrame:
    groups = groups if groups is not None else _build_consensus_groups(dataframes)
    verified_group_indices = verified_group_indices or set()

    preferred_run_idx = selected_run_idx if selected_run_idx is not None else 0
    rows: list[dict[str, str]] = []
    for group_idx, run_rows in enumerate(groups):
        if len(run_rows) < support_threshold and group_idx not in verified_group_indices:
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


def _build_final_task_support(
    final_df: pd.DataFrame,
    groups: list[dict[int, dict[str, str]]],
    run_count: int,
    verified_group_indices: set[int] | None = None,
) -> list[FinalTaskSupport]:
    if final_df.empty:
        return []

    verified_group_indices = verified_group_indices or set()
    support_rows: list[FinalTaskSupport] = []
    for _, row in final_df.iterrows():
        row_dict = row.to_dict()
        group_idx = _find_matching_group_idx(row_dict, groups)
        group = groups[group_idx] if group_idx is not None else {}
        run_indices = tuple(idx + 1 for idx in sorted(group))
        support_count = len(run_indices)
        support_rows.append(
            FinalTaskSupport(
                block=row_dict["Блок"],
                task=row_dict["Задача"],
                responsible=row_dict["Ответственный"],
                deadline=row_dict["Срок"],
                evidence=row_dict["Обоснование"],
                support_count=support_count,
                support_ratio=_safe_div(support_count, run_count),
                run_indices=run_indices,
                verification_status=(
                    "verified_consensus"
                    if group_idx in verified_group_indices
                    else "raw_consensus"
                ),
            )
        )
    return support_rows


def _find_matching_group_idx(
    row: dict[str, str],
    groups: list[dict[int, dict[str, str]]],
) -> int | None:
    for group_idx, group in enumerate(groups):
        if any(_same_consensus_group(row, candidate) for candidate in group.values()):
            return group_idx
    return None


def _build_verification_candidates(
    groups: list[dict[int, dict[str, str]]],
    selected_run_idx: int | None,
    run_count: int,
    support_threshold: int,
    verifier_support_threshold: int,
) -> tuple[list[VerificationCandidate], dict[str, int]]:
    preferred_run_idx = selected_run_idx if selected_run_idx is not None else 0
    candidates: list[VerificationCandidate] = []
    group_by_id: dict[str, int] = {}

    for group_idx, group in enumerate(groups):
        support_count = len(group)
        if support_count < verifier_support_threshold or support_count >= support_threshold:
            continue
        row = group.get(preferred_run_idx) or group[min(group)]
        candidate_id = f"C{len(candidates) + 1:03d}"
        group_by_id[candidate_id] = group_idx
        candidates.append(
            VerificationCandidate(
                candidate_id=candidate_id,
                block=row["Блок"],
                task=row["Задача"],
                responsible=row["Ответственный"],
                deadline=row["Срок"],
                evidence=row["Обоснование"],
                support_count=support_count,
                support_ratio=_safe_div(support_count, run_count),
                run_indices=tuple(idx + 1 for idx in sorted(group)),
            )
        )

    return candidates, group_by_id


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


def _relative_count_deltas(count_rows: list[dict[str, int]]) -> dict[str, float]:
    keys = ["Выполненные", "Невыполненные", "Новые", "Всего"]
    deltas: dict[str, float] = {}
    for key in keys:
        values = [counts[key] for counts in count_rows]
        if not values:
            deltas[key] = 0.0
            continue
        max_value = max(values)
        min_value = min(values)
        deltas[key] = 0.0 if max_value == 0 else (max_value - min_value) / max_value
    return deltas


def _min_pairwise_consensus_jaccard(
    groups: list[dict[int, dict[str, str]]],
    run_count: int,
) -> float:
    if run_count <= 1:
        return 1.0

    group_ids_by_run: list[set[int]] = [set() for _ in range(run_count)]
    for group_idx, group in enumerate(groups):
        for run_idx in group:
            group_ids_by_run[run_idx].add(group_idx)

    scores: list[float] = []
    for left_idx in range(run_count):
        for right_idx in range(left_idx + 1, run_count):
            scores.append(_jaccard(group_ids_by_run[left_idx], group_ids_by_run[right_idx]))
    return min(scores) if scores else 1.0


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


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


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
