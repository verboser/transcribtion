from pathlib import Path

from src.schemas import (
    DATAFRAME_COLUMNS,
    ExtractedTask,
    ExtractionResult,
    TaskAnchor,
    TranscriptUtterance,
)
from src.stability import run_stability_check


def test_stability_selects_centroid_run_not_last() -> None:
    run_results = [
        _result([_done("Отчет выполнен", "Иван: отчет выполнен.")]),
        _result(
            [
                _done("Отчет выполнен", "Иван: отчет выполнен."),
                _done("Схемы разработаны", "Иван: схемы разработаны."),
            ]
        ),
        _result(
            [
                _done("Отчет выполнен", "Иван: отчет выполнен."),
                _done("Схемы разработаны", "Иван: схемы разработаны."),
            ]
        ),
        _result([]),
    ]

    def extractor(_path: Path, _meeting_date: str) -> ExtractionResult:
        return run_results.pop(0)

    report = run_stability_check(
        transcript_path=Path("trascripts/transcript.txt"),
        meeting_date="2026-04-13",
        runs=4,
        extractor=extractor,
    )

    assert report.selected_run_idx == 1
    assert "Финальный прогон по centroid: run_2" in report.report_lines


def test_stability_final_dataframe_uses_90_percent_consensus_support() -> None:
    run_results = [
        _result(
            [
                _done("Отчет выполнен", "Иван: отчет выполнен."),
                _done("Схемы разработаны", "Иван: схемы разработаны."),
            ]
        ),
        _result(
            [
                _done("Отчет выполнен", "Иван: отчет выполнен."),
                _done("Схемы разработаны", "Иван: схемы разработаны."),
            ]
        ),
        _result(
            [
                _done("Отчет выполнен", "Иван: отчет выполнен."),
                _done("Схемы разработаны", "Иван: схемы разработаны."),
            ]
        ),
        _result(
            [
                _done("Отчет выполнен", "Иван: отчет выполнен."),
                _done("Схемы разработаны", "Иван: схемы разработаны."),
            ]
        ),
        _result([_done("Отчет выполнен", "Иван: отчет выполнен.")]),
    ]

    def extractor(_path: Path, _meeting_date: str) -> ExtractionResult:
        return run_results.pop(0)

    report = run_stability_check(
        transcript_path=Path("trascripts/transcript.txt"),
        meeting_date="2026-04-13",
        runs=5,
        extractor=extractor,
    )

    assert set(report.final_df["Задача"]) == {"Отчет выполнен"}
    assert list(report.final_df.columns) == DATAFRAME_COLUMNS
    assert "support_count" not in report.final_df.columns
    assert "verification_status" not in report.final_df.columns
    assert report.final_source == "consensus support>=5 (90%)"
    assert report.support_threshold == 5
    assert report.final_task_support[0].support_count == 5
    assert not report.stability_passed


def test_stability_consensus_groups_fuzzy_task_variants() -> None:
    run_results = [
        _result([_done("Отчет выполнен", "Иван: отчет выполнен.")]),
        _result([_done("Выполнен отчет", "Иван: отчет выполнен.")]),
        _result([_done("Отчет выполнен", "Иван: отчет выполнен.")]),
        _result([_done("Выполнен отчет", "Иван: отчет выполнен.")]),
        _result([_done("Отчет выполнен", "Иван: отчет выполнен.")]),
    ]

    def extractor(_path: Path, _meeting_date: str) -> ExtractionResult:
        return run_results.pop(0)

    report = run_stability_check(
        transcript_path=Path("trascripts/transcript.txt"),
        meeting_date="2026-04-13",
        runs=5,
        extractor=extractor,
    )

    assert len(report.final_df) == 1
    assert report.final_source == "consensus support>=5 (90%)"
    assert report.stability_passed


def test_stability_verified_consensus_can_include_near_consensus_candidate() -> None:
    run_results = [
        _result(
            [
                _done("Отчет выполнен", "Иван: отчет выполнен."),
                _done("Схемы разработаны", "Иван: схемы разработаны."),
            ]
        ),
        _result(
            [
                _done("Отчет выполнен", "Иван: отчет выполнен."),
                _done("Схемы разработаны", "Иван: схемы разработаны."),
            ]
        ),
        _result(
            [
                _done("Отчет выполнен", "Иван: отчет выполнен."),
                _done("Схемы разработаны", "Иван: схемы разработаны."),
            ]
        ),
        _result(
            [
                _done("Отчет выполнен", "Иван: отчет выполнен."),
                _done("Схемы разработаны", "Иван: схемы разработаны."),
            ]
        ),
        _result([_done("Отчет выполнен", "Иван: отчет выполнен.")]),
    ]

    def extractor(_path: Path, _meeting_date: str) -> ExtractionResult:
        return run_results.pop(0)

    def verifier(candidates, _meeting_date: str) -> set[str]:
        return {
            candidate.candidate_id
            for candidate in candidates
            if candidate.task == "Схемы разработаны"
        }

    report = run_stability_check(
        transcript_path=Path("trascripts/transcript.txt"),
        meeting_date="2026-04-13",
        runs=5,
        extractor=extractor,
        verifier=verifier,
    )

    assert set(report.final_df["Задача"]) == {"Отчет выполнен", "Схемы разработаны"}
    assert report.verified_candidate_count == 1
    assert report.verifier_candidate_count == 1
    support_by_task = {
        support.task: support
        for support in report.final_task_support
    }
    assert support_by_task["Отчет выполнен"].verification_status == "raw_consensus"
    assert support_by_task["Схемы разработаны"].verification_status == "verified_consensus"
    assert support_by_task["Схемы разработаны"].support_count == 4


def test_stability_rejected_near_consensus_candidate_stays_out_of_final() -> None:
    run_results = [
        _result([_done("Отчет выполнен", "Иван: отчет выполнен.")]),
        _result([_done("Отчет выполнен", "Иван: отчет выполнен.")]),
        _result([_done("Отчет выполнен", "Иван: отчет выполнен.")]),
        _result([_done("Отчет выполнен", "Иван: отчет выполнен.")]),
        _result([]),
    ]

    def extractor(_path: Path, _meeting_date: str) -> ExtractionResult:
        return run_results.pop(0)

    report = run_stability_check(
        transcript_path=Path("trascripts/transcript.txt"),
        meeting_date="2026-04-13",
        runs=5,
        extractor=extractor,
        verifier=lambda _candidates, _meeting_date: set(),
    )

    assert report.final_df.empty
    assert report.verifier_candidate_count == 1
    assert report.verified_candidate_count == 0


def _done(task: str, evidence: str) -> ExtractedTask:
    return ExtractedTask(
        block="Выполненные",
        task=task,
        responsible="Иван",
        deadline_raw="",
        evidence=evidence,
        anchor_ids=("A001",),
    )


def _result(tasks: list[ExtractedTask]) -> ExtractionResult:
    utterances = tuple(
        TranscriptUtterance(idx, "Иван", task.evidence.removeprefix("Иван: "))
        for idx, task in enumerate(tasks, start=1)
    )
    anchors = []
    if utterances:
        anchors.append(
            TaskAnchor(
                anchor_id="A001",
                kind="done",
                line_start=utterances[0].line_no,
                line_end=utterances[-1].line_no,
                speaker="Иван",
                utterances=utterances,
                signals=("done_signal",),
                deadline_phrases=(),
            )
        )
    return ExtractionResult(tasks=tasks, anchors=anchors, raw_response={})
