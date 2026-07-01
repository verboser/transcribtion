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


def test_build_task_anchors_for_done_signal() -> None:
    utterances = parse_transcript("Иван: отчет подготовили и отправили.")

    anchors = build_task_anchors(utterances)

    assert len(anchors) == 1
    assert anchors[0].kind == "done"
