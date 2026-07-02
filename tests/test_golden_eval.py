import pandas as pd

from src.golden_eval import (
    evaluate_dataframe_against_golden,
    load_golden_tasks,
)


def test_load_golden_tasks() -> None:
    tasks = load_golden_tasks("transcript2.txt")

    assert len(tasks) == 4
    assert tasks[0].task_id == "t2_failed_0012"


def test_load_golden_tasks_is_independent_from_cwd(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    tasks = load_golden_tasks("transcript2.txt")

    assert len(tasks) == 4


def test_evaluate_dataframe_against_golden_counts_matches() -> None:
    golden_tasks = load_golden_tasks("transcript2.txt")
    df = pd.DataFrame(
        [
            {
                "Блок": "Невыполненные",
                "Задача": "не успели по плану до 2 числа",
                "Ответственный": "Человек 3",
                "Срок": "2026-05-02",
                "Обоснование": "Человек 3: Ну по плану до 2 числа не успели.",
            },
            {
                "Блок": "Новые",
                "Задача": "лишняя задача",
                "Ответственный": "Человек 1",
                "Срок": "2026-05-10",
                "Обоснование": "Человек 1: лишняя задача до 10 мая.",
            },
        ]
    )

    report = evaluate_dataframe_against_golden(df, golden_tasks, "transcript2.txt")

    assert report.expected_count == 4
    assert report.predicted_count == 2
    assert report.matched_count == 1
    assert report.full_match_count == 1
    assert report.precision == 0.5
    assert report.recall == 0.25
    assert len(report.missed) == 3
    assert len(report.false_positive_rows) == 1


def test_evaluate_dataframe_against_golden_reports_field_mismatch() -> None:
    golden_tasks = load_golden_tasks("transcript2.txt")
    df = pd.DataFrame(
        [
            {
                "Блок": "Выполненные",
                "Задача": "подготовили 2 вариант",
                "Ответственный": "Человек 8",
                "Срок": "",
                "Обоснование": "Человек 7: На модернизацию мы также подготовили 2 вариант.",
            },
        ]
    )

    report = evaluate_dataframe_against_golden(df, golden_tasks, "transcript2.txt")

    assert report.matched_count == 1
    assert report.full_match_count == 0
    assert len(report.field_mismatches) == 1
    assert report.responsible_accuracy == 0.0
