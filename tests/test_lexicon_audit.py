from src.lexicon_audit import audit_transcript, format_audit_report


def test_lexicon_audit_groups_conversation_markers(tmp_path) -> None:
    transcript = tmp_path / "meeting.txt"
    transcript.write_text(
        "Иван: давайте подготовим отчет до пятницы\n"
        "Мария: сейчас это в работе\n"
        "Иван: финально договорились\n",
        encoding="utf-8",
    )

    matches = audit_transcript(transcript)
    categories = {match.category for match in matches}

    assert "new_task" in categories
    assert "ongoing_state" in categories
    assert "recap" in categories


def test_lexicon_audit_report_is_human_readable(tmp_path) -> None:
    transcript = tmp_path / "meeting.txt"
    transcript.write_text("Иван: подытожим, отчет в работе\n", encoding="utf-8")

    report = format_audit_report(transcript, audit_transcript(transcript))

    assert any("Совпадения по категориям" in line for line in report)
    assert any("Иван:" in line for line in report)
