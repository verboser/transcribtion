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
