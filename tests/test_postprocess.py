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


def test_keeps_explicit_done_task() -> None:
    df, _ = build_dataframe(
        [
            ExtractedTask(
                block="Выполненные",
                task="Электрические схемы разработаны и переданы",
                responsible="Ленин",
                deadline_raw="",
                evidence="Ленин: электрические схемы разработаны, спецификации переданы.",
                anchor_ids=(),
            )
        ],
        "2026-04-15",
    )

    assert len(df) == 1


def test_done_filters_state_description_without_done_signal() -> None:
    df, _ = build_dataframe(
        [
            ExtractedTask(
                block="Выполненные",
                task="есть накопление перед печь и после печь",
                responsible="Человек 7",
                deadline_raw="",
                evidence="Человек 7: У нас есть накопление перед печь и после печь.",
                anchor_ids=(),
            )
        ],
        "2026-04-29",
    )

    assert df.empty


def test_done_filters_vague_readiness_tasks() -> None:
    tasks = [
        ExtractedTask(
            block="Выполненные",
            task="она готова",
            responsible="Иванова Ольга",
            deadline_raw="",
            evidence="Иванова Ольга: Вот мы сейчас нашли, что она готова.",
            anchor_ids=(),
        ),
        ExtractedTask(
            block="Выполненные",
            task="в целом мы к этому готовы",
            responsible="Черчилль",
            deadline_raw="",
            evidence="Черчилль: В целом мы к этому готовы.",
            anchor_ids=(),
        ),
        ExtractedTask(
            block="Выполненные",
            task="готовы к запуску",
            responsible="Черчилль",
            deadline_raw="",
            evidence="Черчилль: Мы готовы к запуску.",
            anchor_ids=(),
        ),
    ]

    df, _ = build_dataframe(tasks, "2026-04-15")

    assert df.empty


def test_done_filters_remaining_or_future_work() -> None:
    tasks = [
        ExtractedTask(
            block="Выполненные",
            task="провести небольшое предпроектное обследование",
            responsible="Черчилль",
            deadline_raw="",
            evidence="Черчилль: готовность достаточно высокая, осталось провести небольшое предпроектное обследование.",
            anchor_ids=(),
        ),
        ExtractedTask(
            block="Выполненные",
            task="прорабатываются дополнительные настройки на виртуальной машине",
            responsible="Черчилль",
            deadline_raw="",
            evidence="Черчилль: прорабатываются дополнительные настройки. Системный администратор подготовил машину.",
            anchor_ids=(),
        ),
        ExtractedTask(
            block="Выполненные",
            task="разработаны собственно, ожидаем окончания изготовления",
            responsible="Черчилль",
            deadline_raw="",
            evidence="Черчилль: ПО разработано, ожидаем окончания изготовления и дальше будем производить наладочные работы.",
            anchor_ids=(),
        ),
        ExtractedTask(
            block="Выполненные",
            task="всё уже посчитано, только взять надо",
            responsible="Иванова Ольга",
            deadline_raw="",
            evidence="Иванова Ольга: тут всё уже посчитано, только взять надо.",
            anchor_ids=(),
        ),
    ]

    df, _ = build_dataframe(tasks, "2026-04-15")

    assert df.empty


def test_done_requires_task_terms_in_done_clause() -> None:
    df, _ = build_dataframe(
        [
            ExtractedTask(
                block="Выполненные",
                task="Дополнительные настройки подготовлены",
                responsible="Черчилль",
                deadline_raw="",
                evidence=(
                    "Черчилль: прорабатываются дополнительные настройки. "
                    "Системный администратор подготовил машину."
                ),
                anchor_ids=(),
            )
        ],
        "2026-04-15",
    )

    assert df.empty


def test_done_rejects_negated_statuses() -> None:
    tasks = [
        ExtractedTask(
            block="Выполненные",
            task="Отчет сделали",
            responsible="Иван",
            deadline_raw="",
            evidence="Иван: отчет не сделали.",
            anchor_ids=(),
        ),
        ExtractedTask(
            block="Выполненные",
            task="Отчет готов",
            responsible="Иван",
            deadline_raw="",
            evidence="Иван: отчет не готова.",
            anchor_ids=(),
        ),
        ExtractedTask(
            block="Выполненные",
            task="Отчет выполнен",
            responsible="Иван",
            deadline_raw="",
            evidence="Иван: отчет не выполнено.",
            anchor_ids=(),
        ),
        ExtractedTask(
            block="Выполненные",
            task="Заказы закрыты",
            responsible="Иван",
            deadline_raw="",
            evidence="Иван: заказы не закрыты.",
            anchor_ids=(),
        ),
    ]

    df, _ = build_dataframe(tasks, "2026-04-15")

    assert df.empty


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


def test_new_task_chooses_deadline_related_to_task_text() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="new",
        line_start=1,
        line_end=1,
        speaker="Рузвельт",
        utterances=(
            TranscriptUtterance(
                1,
                "Рузвельт",
                "завтра запланировано совещание, мы обсудим и подготовимся к среде",
            ),
        ),
        signals=("date_first",),
        deadline_phrases=("завтра", "к среде"),
    )
    task = ExtractedTask(
        block="Новые",
        task="Подготовиться к среде",
        responsible="Рузвельт",
        deadline_raw="",
        evidence="Рузвельт: завтра запланировано совещание, мы обсудим и подготовимся к среде",
        anchor_ids=("A001",),
    )

    df, _ = build_dataframe([task], "2026-04-15", [anchor])

    assert len(df) == 1
    assert df.iloc[0]["Срок"] == "2026-04-22"


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


def test_responsible_uses_explicit_assignee_marker_before_speaker() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="new",
        line_start=1,
        line_end=2,
        speaker="Иванова Ольга",
        utterances=(
            TranscriptUtterance(
                1,
                "Иванова Ольга",
                "сформировать чек листы срок 15 апреля ответственную кудинова",
            ),
        ),
        signals=("task_with_deadline",),
        deadline_phrases=("15 апреля",),
    )
    task = ExtractedTask(
        block="Новые",
        task="сформировать чек листы",
        responsible="Иванова Ольга",
        deadline_raw="15 апреля",
        evidence=(
            "Иванова Ольга: сформировать чек листы срок 15 апреля "
            "ответственную кудинова"
        ),
        anchor_ids=("A001",),
    )

    df, _ = build_dataframe([task], "2026-04-13", [anchor])

    assert len(df) == 1
    assert df.iloc[0]["Ответственный"] == "Кудинова"


def test_responsible_uses_multiple_explicit_assignees() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="new",
        line_start=1,
        line_end=1,
        speaker="Иванова Ольга",
        utterances=(
            TranscriptUtterance(
                1,
                "Иванова Ольга",
                "доработать файл цели и риски до 17 числа ответственные Смирнова и фарла будем вдвоём",
            ),
        ),
        signals=("task_with_deadline",),
        deadline_phrases=("до 17 числа",),
    )
    task = ExtractedTask(
        block="Новые",
        task="доработать файл цели и риски",
        responsible="Иванова Ольга",
        deadline_raw="до 17 числа",
        evidence=(
            "Иванова Ольга: доработать файл цели и риски до 17 числа "
            "ответственные Смирнова и фарла будем вдвоём"
        ),
        anchor_ids=("A001",),
    )

    df, _ = build_dataframe([task], "2026-04-13", [anchor])

    assert len(df) == 1
    assert df.iloc[0]["Ответственный"] == "Смирнова и Фарла"


def test_responsible_after_marker_skips_assignment_scope() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="new",
        line_start=1,
        line_end=1,
        speaker="Иван",
        utterances=(
            TranscriptUtterance(
                1,
                "Иван",
                "подготовить отчет до пятницы ответственный за подготовку Петров",
            ),
        ),
        signals=("task_with_deadline",),
        deadline_phrases=("до пятницы",),
    )
    task = ExtractedTask(
        block="Новые",
        task="подготовить отчет",
        responsible="Иван",
        deadline_raw="до пятницы",
        evidence="Иван: подготовить отчет до пятницы ответственный за подготовку Петров",
        anchor_ids=("A001",),
    )

    df, _ = build_dataframe([task], "2026-04-13", [anchor])

    assert len(df) == 1
    assert df.iloc[0]["Ответственный"] == "Петров"


def test_responsible_after_marker_supports_will_be_form() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="new",
        line_start=1,
        line_end=1,
        speaker="Иван",
        utterances=(
            TranscriptUtterance(
                1,
                "Иван",
                "подготовить отчет до пятницы ответственным будет Петров",
            ),
        ),
        signals=("task_with_deadline",),
        deadline_phrases=("до пятницы",),
    )
    task = ExtractedTask(
        block="Новые",
        task="подготовить отчет",
        responsible="Иван",
        deadline_raw="до пятницы",
        evidence="Иван: подготовить отчет до пятницы ответственным будет Петров",
        anchor_ids=("A001",),
    )

    df, _ = build_dataframe([task], "2026-04-13", [anchor])

    assert len(df) == 1
    assert df.iloc[0]["Ответственный"] == "Петров"


def test_responsible_recap_does_not_treat_task_words_as_assignee() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="new",
        line_start=1,
        line_end=1,
        speaker="Иван",
        utterances=(
            TranscriptUtterance(
                1,
                "Иван",
                "протокол подготовить отчет 15 апреля срок",
            ),
        ),
        signals=("task_with_deadline",),
        deadline_phrases=("15 апреля",),
    )
    task = ExtractedTask(
        block="Новые",
        task="подготовить отчет",
        responsible="Иван",
        deadline_raw="15 апреля",
        evidence="Иван: протокол подготовить отчет 15 апреля срок",
        anchor_ids=("A001",),
    )

    df, _ = build_dataframe([task], "2026-04-13", [anchor])

    assert len(df) == 1
    assert df.iloc[0]["Ответственный"] == "Иван"


def test_responsibility_word_is_not_explicit_assignee_marker() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="new",
        line_start=1,
        line_end=1,
        speaker="Иванова Ольга",
        utterances=(
            TranscriptUtterance(
                1,
                "Иванова Ольга",
                "завтра с Чайка должны собраться, чтобы закрепить ответственность за внешние аудиты",
            ),
        ),
        signals=("date_first",),
        deadline_phrases=("завтра",),
    )
    task = ExtractedTask(
        block="Новые",
        task="собраться с Чайка",
        responsible="Иванова Ольга",
        deadline_raw="завтра",
        evidence=(
            "Иванова Ольга: завтра с Чайка должны собраться, "
            "чтобы закрепить ответственность за внешние аудиты"
        ),
        anchor_ids=("A001",),
    )

    df, _ = build_dataframe([task], "2026-04-13", [anchor])

    assert len(df) == 1
    assert df.iloc[0]["Ответственный"] == "Иванова Ольга"


def test_done_task_never_gets_deadline() -> None:
    task = ExtractedTask(
        block="Выполненные",
        task="все цели выполнены",
        responsible="Иванова Ольга",
        deadline_raw="к 30 апреля",
        evidence=(
            "Иванова Ольга: все цели выполнены, потом подумала "
            "запросить информацию к 30 апреля"
        ),
        anchor_ids=(),
    )

    df, replacements = build_dataframe([task], "2026-04-13")

    assert len(df) == 1
    assert df.iloc[0]["Срок"] == ""
    assert replacements == []


def test_new_task_rejects_ongoing_state_without_assignment() -> None:
    tasks = [
        ExtractedTask(
            block="Новые",
            task="подготовительные работы уже проводим",
            responsible="Ленин",
            deadline_raw="1 июля",
            evidence=(
                "Ленин: базовый срок окончания 1 июля, время ещё есть. "
                "Черчилль: подготовительные работы уже проводим."
            ),
            anchor_ids=(),
        ),
        ExtractedTask(
            block="Новые",
            task="испытание с нагревом в обеих сушках",
            responsible="Черчилль",
            deadline_raw="сегодня",
            evidence=(
                "Черчилль: Сегодня производится испытание с нагревом "
                "в обеих сушках, сейчас трубы начали выходить."
            ),
            anchor_ids=(),
        ),
    ]

    df, _ = build_dataframe(tasks, "2026-04-15")

    assert df.empty


def test_done_task_requires_concrete_work_object() -> None:
    tasks = [
        ExtractedTask(
            block="Выполненные",
            task="Борис подготовил",
            responsible="Человек 7",
            deadline_raw="",
            evidence="Человек 7: Вот с нашей стороны Борис подготовил.",
            anchor_ids=(),
        ),
        ExtractedTask(
            block="Выполненные",
            task="написано и сделано",
            responsible="Ленин",
            deadline_raw="",
            evidence="Ленин: половина это то, что было инфраструктурно написано и сделано.",
            anchor_ids=(),
        ),
        ExtractedTask(
            block="Выполненные",
            task="разработаны собственно",
            responsible="Черчилль",
            deadline_raw="",
            evidence="Черчилль: Разработанный п о у Вадим разработаны собственно.",
            anchor_ids=(),
        ),
    ]

    df, _ = build_dataframe(tasks, "2026-04-15")

    assert df.empty


def test_responsible_drops_organization_tail_tokens() -> None:
    anchor = TaskAnchor(
        anchor_id="A001",
        kind="new",
        line_start=1,
        line_end=1,
        speaker="Иванова Ольга",
        utterances=(
            TranscriptUtterance(
                1,
                "Иванова Ольга",
                "направить форму до 17 числа ответственные кудинова м п югра",
            ),
        ),
        signals=("task_with_deadline",),
        deadline_phrases=("до 17 числа",),
    )
    task = ExtractedTask(
        block="Новые",
        task="направить форму",
        responsible="Иванова Ольга",
        deadline_raw="до 17 числа",
        evidence=(
            "Иванова Ольга: направить форму до 17 числа "
            "ответственные кудинова м п югра"
        ),
        anchor_ids=("A001",),
    )

    df, _ = build_dataframe([task], "2026-04-13", [anchor])

    assert len(df) == 1
    assert df.iloc[0]["Ответственный"] == "Кудинова"
