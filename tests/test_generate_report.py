from pathlib import Path

from generate_report import generate_report_html
from src.schemas import ExtractedTask, ExtractionResult, TaskAnchor, TranscriptUtterance
from src.sqlite_store import save_tasks
from src.stability import run_stability_check


def test_generate_report_uses_clean_final_dataframe_and_audit_section(tmp_path) -> None:
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

    html = generate_report_html(db_path, ["transcript.txt"])

    assert "Итоговый DataFrame" in html
    assert "Блок" in html
    assert "Задача" in html
    assert "Ответственный" in html
    assert "Срок" in html
    assert "Обоснование" in html
    assert "Поддержка финальных строк" in html
    assert "raw_consensus" in html
    assert "support_count" not in html
    assert "verification_status" not in html


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
