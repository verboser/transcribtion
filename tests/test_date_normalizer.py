from src.date_normalizer import normalize_deadline


def test_tomorrow() -> None:
    value, replacement = normalize_deadline("завтра", "", "2026-04-13")

    assert value == "2026-04-14"
    assert replacement is not None


def test_next_friday_after_meeting_date() -> None:
    value, _ = normalize_deadline("в пятницу", "", "2026-04-13")

    assert value == "2026-04-17"


def test_day_number_rolls_to_next_month_when_needed() -> None:
    value, _ = normalize_deadline("до 2 числа", "", "2026-04-29")

    assert value == "2026-05-02"


def test_no_deadline_when_unreliable() -> None:
    value, replacement = normalize_deadline("", "Иван: Нужно обсудить отдельно.", "2026-04-13")

    assert value == ""
    assert replacement is None


def test_today() -> None:
    value, _ = normalize_deadline("сегодня", "", "2026-04-13")

    assert value == "2026-04-13"


def test_short_relative_words_do_not_match_inside_adjectives() -> None:
    for phrase in ["сегодняшний статус", "завтрашний созвон", "послезавтрашний план"]:
        value, replacement = normalize_deadline(phrase, "", "2026-04-13")

        assert value == ""
        assert replacement is None


def test_end_of_month() -> None:
    value, _ = normalize_deadline("к концу месяца", "", "2026-04-13")

    assert value == "2026-04-30"


def test_end_of_week_forms() -> None:
    value, _ = normalize_deadline("в конце недели", "", "2026-04-13")

    assert value == "2026-04-17"


def test_end_of_this_week_form() -> None:
    value, _ = normalize_deadline("до конца этой недели", "", "2026-04-13")

    assert value == "2026-04-17"


def test_end_of_month_forms() -> None:
    value, _ = normalize_deadline("до конца месяца", "", "2026-04-13")

    assert value == "2026-04-30"


def test_bare_number_without_date_context_is_ignored() -> None:
    value, replacement = normalize_deadline("2 вариант", "", "2026-04-13")

    assert value == ""
    assert replacement is None


def test_invalid_deadline_raw_falls_back_to_evidence() -> None:
    value, replacement = normalize_deadline(
        "2 вариант",
        "Иван: подготовим отчет до пятницы.",
        "2026-04-13",
    )

    assert value == "2026-04-17"
    assert replacement is not None
    assert replacement.source == "до пятницы"


def test_next_week_with_weekday() -> None:
    value, _ = normalize_deadline("на следующей неделе в среду", "", "2026-04-13")

    assert value == "2026-04-22"


def test_next_week_with_comma_and_weekday() -> None:
    value, _ = normalize_deadline("на следующей неделе, в среду", "", "2026-04-13")

    assert value == "2026-04-22"


def test_next_weekday_dative_form() -> None:
    value, _ = normalize_deadline("к следующей пятнице", "", "2026-04-13")

    assert value == "2026-04-17"


def test_selects_date_related_to_task_text() -> None:
    value, replacement = normalize_deadline(
        deadline_raw="",
        evidence=(
            "Рузвельт: завтра запланировано совещание, "
            "мы обсудим и подготовимся к среде."
        ),
        meeting_date="2026-04-15",
        task_text="Подготовиться к среде",
    )

    assert value == "2026-04-22"
    assert replacement is not None
    assert replacement.source == "к среде"


def test_selects_date_from_clause_related_to_task_text() -> None:
    value, replacement = normalize_deadline(
        deadline_raw="",
        evidence="Иван: завтра совещание, подготовить отчет до пятницы.",
        meeting_date="2026-04-13",
        task_text="Подготовить отчет",
    )

    assert value == "2026-04-17"
    assert replacement is not None
    assert replacement.source == "до пятницы"
