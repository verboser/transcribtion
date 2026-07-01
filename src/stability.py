from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.postprocess import build_dataframe_with_stats, row_signature
from src.schemas import ExtractionResult


Extractor = Callable[[Path, str], ExtractionResult]


@dataclass
class StabilityReport:
    extractions: list[ExtractionResult]
    report_lines: list[str]


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

    for run_idx in range(1, runs + 1):
        result = extractor(transcript_path, meeting_date)
        extractions.append(result)
        df, _, stats = build_dataframe_with_stats(
            result.tasks,
            meeting_date,
            result.anchors,
        )
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
            "Jaccard сигнатур к run_1: "
            + ", ".join(
                f"run_{idx}={score:.2f}"
                for idx, score in enumerate(jaccards, start=1)
            )
        )
        one_off_count = _count_one_off_signatures(signature_rows)
        report_lines.append(f"Задач, появившихся только в одном прогоне: {one_off_count}")

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

    return StabilityReport(extractions=extractions, report_lines=report_lines)


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


def _count_one_off_signatures(signature_rows: list[set[str]]) -> int:
    counts: dict[str, int] = {}
    for signatures in signature_rows:
        for signature in signatures:
            counts[signature] = counts.get(signature, 0) + 1
    return sum(1 for count in counts.values() if count == 1)
