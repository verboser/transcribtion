from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.postprocess import count_by_block
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

    for run_idx in range(1, runs + 1):
        result = extractor(transcript_path, meeting_date)
        extractions.append(result)
        counts = count_by_block(result.tasks, meeting_date, result.anchors)
        count_rows.append(counts)
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

    return StabilityReport(extractions=extractions, report_lines=report_lines)
