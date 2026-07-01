from src.preprocess import build_task_anchors, parse_transcript


def test_parse_speaker_lines() -> None:
    utterances = parse_transcript("Иван: Сделать отчет.\nМария: Готово.")

    assert len(utterances) == 2
    assert utterances[0].speaker == "Иван"
    assert utterances[0].text == "Сделать отчет."
    assert utterances[1].speaker == "Мария"


def test_merge_continuation_line() -> None:
    utterances = parse_transcript("Иван: Сделать отчет\nдо пятницы")

    assert len(utterances) == 1
    assert utterances[0].text == "Сделать отчет до пятницы"


def test_build_task_anchors_for_new_task_with_deadline() -> None:
    utterances = parse_transcript(
        "Иван: обычное обсуждение\n"
        "Мария: под протокол подготовить отчет до пятницы\n"
        "Иван: подтверждаю\n"
    )

    anchors = build_task_anchors(utterances)

    assert len(anchors) == 1
    assert anchors[0].anchor_id == "A001"
    assert anchors[0].kind == "new"
    assert anchors[0].speaker == "Мария"
    assert "до пятницы" in anchors[0].deadline_phrases


def test_build_task_anchor_for_end_of_week_deadline() -> None:
    utterances = parse_transcript("Иван: подготовить отчет в конце недели")

    anchors = build_task_anchors(utterances)

    assert len(anchors) == 1
    assert anchors[0].kind == "new"
    assert "в конце недели" in anchors[0].deadline_phrases


def test_anchor_deadline_does_not_duplicate_after_tomorrow() -> None:
    utterances = parse_transcript("Иван: подготовить отчет послезавтра")

    anchors = build_task_anchors(utterances)

    assert len(anchors) == 1
    assert anchors[0].deadline_phrases == ("послезавтра",)


def test_build_task_anchors_for_done_signal() -> None:
    utterances = parse_transcript("Иван: отчет подготовили и отправили.")

    anchors = build_task_anchors(utterances)

    assert len(anchors) == 1
    assert anchors[0].kind == "done"


def test_date_first_anchor_without_task_keyword() -> None:
    utterances = parse_transcript(
        "Иван: мы возьмем небольшую паузу\n"
        "Мария: 13 мая\n"
        "Иван: вернемся к обсуждению"
    )

    anchors = build_task_anchors(utterances)

    assert anchors
    assert any("date_first" in anchor.signals for anchor in anchors)
    assert any("13 мая" in anchor.deadline_phrases for anchor in anchors)


def test_low_coverage_adds_final_fallback() -> None:
    lines = [f"Иван: обычная реплика {idx}" for idx in range(60)]
    utterances = parse_transcript("\n".join(lines))

    anchors = build_task_anchors(utterances)

    assert anchors
    assert any("final_tail_fallback" in anchor.signals for anchor in anchors)


def test_overlapping_anchors_are_merged() -> None:
    utterances = parse_transcript(
        "Иван: под протокол подготовить отчет до пятницы\n"
        "Иван: срок 17 апреля\n"
        "Иван: ответственный Иван"
    )

    anchors = build_task_anchors(utterances)

    assert len(anchors) == 1
