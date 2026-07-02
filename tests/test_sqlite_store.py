from pathlib import Path
import sqlite3

from src.schemas import ExtractedTask, ExtractionResult, TaskAnchor, TranscriptUtterance
from src.sqlite_store import save_tasks
from src.stability import run_stability_check


def test_save_tasks_writes_final_runs_metrics_and_support(tmp_path) -> None:
    run_results = [
        _result([_done("Отчет выполнен", "Иван: отчет выполнен.")])
        for _ in range(5)
    ]

    def extractor(_path: Path, _meeting_date: str) -> ExtractionResult:
        return run_results.pop(0)

    report = run_stability_check(
        transcript_path=Path("trascripts/transcript.txt"),
        meeting_date="2026-04-13",
        runs=5,
        extractor=extractor,
    )
    db_path = tmp_path / "tasks.sqlite"

    save_tasks(
        report.final_df,
        "transcript.txt",
        db_path,
        meeting_date="2026-04-13",
        stability_report=report,
    )

    with sqlite3.connect(db_path) as connection:
        extraction_row = connection.execute(
            """
            SELECT runs, support_threshold, stability_passed
            FROM extraction_runs
            """
        ).fetchone()
        final_row = connection.execute(
            """
            SELECT block, task, support_count, support_ratio, run_indices,
                   verification_status
            FROM meeting_tasks
            """
        ).fetchone()
        run_task_count = connection.execute(
            "SELECT COUNT(*) FROM meeting_run_tasks"
        ).fetchone()[0]
        run_metric_count = connection.execute(
            "SELECT COUNT(*) FROM meeting_run_metrics"
        ).fetchone()[0]
        report_line_count = connection.execute(
            "SELECT COUNT(*) FROM stability_report_lines"
        ).fetchone()[0]

    assert extraction_row == (5, 5, 1)
    assert final_row == (
        "Выполненные",
        "Отчет выполнен",
        5,
        1.0,
        "1,2,3,4,5",
        "raw_consensus",
    )
    assert run_task_count == 5
    assert run_metric_count == 5
    assert report_line_count > 0


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
    return ExtractionResult(tasks=tasks, anchors=anchors, candidates=[], raw_response={})
