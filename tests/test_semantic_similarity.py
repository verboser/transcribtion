from src.semantic_similarity import semantic_similarity, semantic_similarity_enabled


def test_semantic_similarity_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TRANSCRIBTION_USE_SEMANTIC", raising=False)

    assert semantic_similarity_enabled() is False
    assert semantic_similarity("назначить встречу", "собраться 13 мая") is None
