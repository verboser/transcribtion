from src.postprocess import build_dataframe
from src.schemas import ExtractedTask


def test_filters_in_progress_as_not_failed() -> None:
    df, _ = build_dataframe(
        [
            ExtractedTask(
                block="Невыполненные",
                task="Провести обследование",
                responsible="Иван",
                deadline_raw="",
                evidence="Иван: осталось провести обследование на заводе.",
                anchor_ids=(),
            )
        ],
        "2026-04-15",
    )

    assert df.empty


def test_keeps_explicit_failed_task() -> None:
    df, _ = build_dataframe(
        [
            ExtractedTask(
                block="Невыполненные",
                task="Сделать запуск",
                responsible="Иван",
                deadline_raw="",
                evidence="Иван: на данный момент я еще не успел это сделать.",
                anchor_ids=(),
            )
        ],
        "2026-04-15",
    )

    assert len(df) == 1
