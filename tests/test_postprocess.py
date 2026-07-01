from src.postprocess import build_dataframe, build_dataframe_with_stats
from src.schemas import ExtractedTask, TaskAnchor, TranscriptUtterance


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


def test_row_deduplicate_after_validation() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="new",
        line_start=1,
        line_end=1,
        speaker="Иван",
        utterances=(TranscriptUtterance(1, "Иван", "подготовить отчет до пятницы"),),
        signals=("task_with_deadline",),
        deadline_phrases=("до пятницы",),
    )
    tasks = [
        ExtractedTask(
            block="Новые",
            task="Подготовить отчет",
            responsible="Иван",
            deadline_raw="до пятницы",
            evidence="Иван: подготовить отчет до пятницы",
            anchor_ids=("A001",),
        ),
        ExtractedTask(
            block="Новые",
            task="Подготовить отчет",
            responsible="Иван",
            deadline_raw="до пятницы",
            evidence="Иван: подготовить отчет до пятницы",
            anchor_ids=("A001",),
        ),
    ]

    df, _, stats = build_dataframe_with_stats(tasks, "2026-04-13", [anchor])

    assert len(df) == 1
    assert stats.valid_rows == 2
    assert stats.final_rows == 1
    assert stats.dedup_removed_rows == 1


def test_new_task_accepts_end_of_week_deadline_from_evidence() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="new",
        line_start=1,
        line_end=1,
        speaker="Иван",
        utterances=(TranscriptUtterance(1, "Иван", "подготовить отчет в конце недели"),),
        signals=("task_with_deadline",),
        deadline_phrases=("в конце недели",),
    )
    task = ExtractedTask(
        block="Новые",
        task="Подготовить отчет",
        responsible="Иван",
        deadline_raw="в конце недели",
        evidence="Иван: подготовить отчет в конце недели",
        anchor_ids=("A001",),
    )

    df, _ = build_dataframe([task], "2026-04-13", [anchor])

    assert len(df) == 1
    assert df.iloc[0]["Срок"] == "2026-04-17"


def test_drops_evidence_not_supported_by_anchor() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="new",
        line_start=1,
        line_end=1,
        speaker="Иван",
        utterances=(TranscriptUtterance(1, "Иван", "подготовить отчет до пятницы"),),
        signals=("task_with_deadline",),
        deadline_phrases=("до пятницы",),
    )
    task = ExtractedTask(
        block="Новые",
        task="Подготовить отчет",
        responsible="Иван",
        deadline_raw="до пятницы",
        evidence="Иван: придумал лишнюю задачу",
        anchor_ids=("A001",),
    )

    df, _ = build_dataframe([task], "2026-04-13", [anchor])

    assert df.empty


def test_responsible_uses_speaker_from_evidence_in_multi_speaker_anchor() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="new",
        line_start=1,
        line_end=2,
        speaker="Несколько спикеров",
        utterances=(
            TranscriptUtterance(1, "Сидорова Елена", "да"),
            TranscriptUtterance(2, "Иванова Ольга", "подготовить отчет до пятницы"),
        ),
        signals=("task_with_deadline",),
        deadline_phrases=("до пятницы",),
    )
    task = ExtractedTask(
        block="Новые",
        task="Подготовить отчет",
        responsible="Петров Петр",
        deadline_raw="до пятницы",
        evidence="Иванова Ольга: подготовить отчет до пятницы",
        anchor_ids=("A001",),
    )

    df, _ = build_dataframe([task], "2026-04-13", [anchor])

    assert df.iloc[0]["Ответственный"] == "Иванова Ольга"


def test_valid_duplicate_survives_longer_invalid_evidence() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="new",
        line_start=1,
        line_end=1,
        speaker="Иван",
        utterances=(TranscriptUtterance(1, "Иван", "подготовить отчет до пятницы"),),
        signals=("task_with_deadline",),
        deadline_phrases=("до пятницы",),
    )
    valid = ExtractedTask(
        block="Новые",
        task="Подготовить отчет",
        responsible="Иван",
        deadline_raw="до пятницы",
        evidence="Иван: подготовить отчет до пятницы",
        anchor_ids=("A001",),
    )
    invalid = ExtractedTask(
        block="Новые",
        task="Подготовить отчет",
        responsible="Иван",
        deadline_raw="до пятницы",
        evidence="Иван: подготовить отчет до пятницы и придумать несуществующую часть",
        anchor_ids=("A001",),
    )

    df, _ = build_dataframe([valid, invalid], "2026-04-13", [anchor])

    assert len(df) == 1
    assert df.iloc[0]["Обоснование"] == "Иван: подготовить отчет до пятницы"


def test_new_task_does_not_inherit_deadline_from_anchor_context() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="mixed",
        line_start=1,
        line_end=2,
        speaker="Несколько спикеров",
        utterances=(
            TranscriptUtterance(1, "Иван", "завтра обсудим другую тему"),
            TranscriptUtterance(2, "Мария", "подготовить отчет"),
        ),
        signals=("date_first",),
        deadline_phrases=("завтра",),
    )
    task = ExtractedTask(
        block="Новые",
        task="Подготовить отчет",
        responsible="Мария",
        deadline_raw="",
        evidence="Мария: подготовить отчет",
        anchor_ids=("A001",),
    )

    df, _ = build_dataframe([task], "2026-04-13", [anchor])

    assert df.empty


def test_new_task_does_not_accept_deadline_raw_without_evidence_date() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="mixed",
        line_start=1,
        line_end=2,
        speaker="Несколько спикеров",
        utterances=(
            TranscriptUtterance(1, "Иван", "завтра обсудим другую тему"),
            TranscriptUtterance(2, "Мария", "подготовить отчет"),
        ),
        signals=("date_first",),
        deadline_phrases=("завтра",),
    )
    task = ExtractedTask(
        block="Новые",
        task="Подготовить отчет",
        responsible="Мария",
        deadline_raw="завтра",
        evidence="Мария: подготовить отчет",
        anchor_ids=("A001",),
    )

    df, _ = build_dataframe([task], "2026-04-13", [anchor])

    assert df.empty


def test_new_task_drops_evidence_that_only_supports_deadline() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="new",
        line_start=1,
        line_end=2,
        speaker="Несколько спикеров",
        utterances=(
            TranscriptUtterance(1, "Человек 3", "предусмотреть карманы для обезжирки"),
            TranscriptUtterance(2, "Человек 8", "А завтра?"),
        ),
        signals=("date_first",),
        deadline_phrases=("завтра",),
    )
    task = ExtractedTask(
        block="Новые",
        task="Предусмотреть карманы для обезжирки",
        responsible="Человек 8",
        deadline_raw="завтра",
        evidence="Человек 8: А завтра?",
        anchor_ids=("A001",),
    )

    df, _ = build_dataframe([task], "2026-04-29", [anchor])

    assert df.empty


def test_new_task_requires_lexically_supported_task_phrase() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="new",
        line_start=1,
        line_end=1,
        speaker="Иван",
        utterances=(TranscriptUtterance(1, "Иван", "давайте 13 мая соберемся"),),
        signals=("date_first",),
        deadline_phrases=("13 мая",),
    )
    paraphrased = ExtractedTask(
        block="Новые",
        task="Назначить встречу",
        responsible="Иван",
        deadline_raw="13 мая",
        evidence="Иван: давайте 13 мая соберемся",
        anchor_ids=("A001",),
    )
    lexical = ExtractedTask(
        block="Новые",
        task="Собраться 13 мая",
        responsible="Иван",
        deadline_raw="13 мая",
        evidence="Иван: давайте 13 мая соберемся",
        anchor_ids=("A001",),
    )

    paraphrased_df, _ = build_dataframe([paraphrased], "2026-04-29", [anchor])
    lexical_df, _ = build_dataframe([lexical], "2026-04-29", [anchor])

    assert paraphrased_df.empty
    assert len(lexical_df) == 1
