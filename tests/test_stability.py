from pathlib import Path

from src.schemas import ExtractedTask, ExtractionResult, TaskAnchor, TranscriptUtterance
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


def test_stability_final_dataframe_uses_consensus_support() -> None:
    run_results = [
        _result(
            [
                _done("Отчет выполнен", "Иван: отчет выполнен."),
                _done("Лишняя задача выполнена", "Иван: лишняя задача выполнена."),
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
    ]

    def extractor(_path: Path, _meeting_date: str) -> ExtractionResult:
        return run_results.pop(0)

    report = run_stability_check(
        transcript_path=Path("trascripts/transcript.txt"),
        meeting_date="2026-04-13",
        runs=3,
        extractor=extractor,
    )

    assert set(report.final_df["Задача"]) == {"Отчет выполнен", "Схемы разработаны"}
    assert report.final_source == "consensus support>=2"


def test_stability_consensus_groups_fuzzy_task_variants() -> None:
    run_results = [
        _result([_done("Отчет выполнен", "Иван: отчет выполнен.")]),
        _result([_done("Выполнен отчет", "Иван: отчет выполнен.")]),
    ]

    def extractor(_path: Path, _meeting_date: str) -> ExtractionResult:
        return run_results.pop(0)

    report = run_stability_check(
        transcript_path=Path("trascripts/transcript.txt"),
        meeting_date="2026-04-13",
        runs=2,
        extractor=extractor,
    )

    assert len(report.final_df) == 1
    assert report.final_source == "consensus support>=2"


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
