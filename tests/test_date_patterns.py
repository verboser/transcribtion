from src.date_patterns import find_date_phrases


def test_tomorrow_is_not_extracted_from_after_tomorrow() -> None:
    phrases = find_date_phrases("Иван: подготовить отчет послезавтра")

    assert phrases == ["послезавтра"]


def test_short_relative_dates_use_word_boundaries() -> None:
    phrases = find_date_phrases("Иван: это сегодняшний статус, не срок")

    assert phrases == []


def test_end_of_this_week_phrase() -> None:
    phrases = find_date_phrases("Иван: подготовить отчет до конца этой недели")

    assert phrases == ["до конца этой недели"]
