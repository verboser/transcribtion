from src.llm_client import (
    _candidate_decision_contract_errors,
    _anchor_groups,
    _parse_candidate_decisions_payload,
    OpenAITaskExtractor,
)
from src.schemas import TaskAnchor, TranscriptUtterance, PreLLMCandidate


def test_anchor_groups_overlap_neighbors() -> None:
    anchors = [_anchor(idx) for idx in range(1, 6)]

    groups = list(_anchor_groups(anchors, group_size=3, overlap=1))

    assert [[anchor.anchor_id for anchor in group] for group in groups] == [
        ["A001", "A002", "A003"],
        ["A003", "A004", "A005"],
    ]


def test_parse_candidate_decisions_payload() -> None:
    payload = {
        "candidate_decisions": [
            {
                "candidate_id": "C001",
                "is_task": True,
                "block": "Новые",
                "task": "подготовить отчет",
                "responsible": "Иван",
                "deadline_raw": "до пятницы",
                "evidence": "Иван: подготовить отчет до пятницы",
            },
            {"candidate_id": "C002", "is_task": False},
        ]
    }
    
    candidates = [
        PreLLMCandidate(
            candidate_id="C001",
            anchor_ids=("A001",),
            evidence_span="",
            candidate_kind="new",
            date_phrases=(),
            speakers=(),
            signals=()
        )
    ]

    tasks = _parse_candidate_decisions_payload(payload, candidates)

    assert len(tasks) == 1
    assert tasks[0].anchor_ids == ("A001",)
    assert tasks[0].task == "подготовить отчет"


def test_parse_candidate_decisions_payload_skips_unknown_candidate_id() -> None:
    payload = {
        "candidate_decisions": [
            {
                "candidate_id": "C999",
                "is_task": True,
                "block": "Новые",
                "task": "подготовить отчет",
                "responsible": "Иван",
                "deadline_raw": "до пятницы",
                "evidence": "Иван: подготовить отчет до пятницы",
            }
        ]
    }

    tasks = _parse_candidate_decisions_payload(payload, [_candidate("C001", "A001")])

    assert tasks == []


def test_candidate_decision_contract_errors_report_missing_extra_and_duplicate_ids() -> None:
    payload = {
        "candidate_decisions": [
            {"candidate_id": "C001"},
            {"candidate_id": "C001"},
            {"candidate_id": "C999"},
        ]
    }

    errors = _candidate_decision_contract_errors(payload, ["C001", "C002"])

    assert errors == {
        "missing_candidate_decisions": ["C002"],
        "extra_candidate_decisions": ["C999"],
        "duplicate_candidate_decisions": ["C001"],
    }


def test_candidate_decision_groups_process_overlap_candidate_once() -> None:
    extractor = object.__new__(OpenAITaskExtractor)
    extractor.anchor_group_size = 3
    extractor.anchor_group_overlap = 1
    calls = []

    def fake_batch(_meeting_date, anchors, candidates):
        calls.append(
            (
                [anchor.anchor_id for anchor in anchors],
                [candidate.candidate_id for candidate in candidates],
            )
        )
        return [], {"candidate_decisions": []}

    extractor._extract_from_candidate_decision_batch = fake_batch
    anchors = [_anchor(idx) for idx in range(1, 6)]
    candidates = [
        _candidate("C001", "A001"),
        _candidate("C002", "A003"),
        _candidate("C003", "A005"),
    ]

    _, payload = extractor._extract_from_candidate_decision_groups(
        "2026-04-29",
        anchors,
        candidates,
    )

    assert calls == [
        (["A001", "A002", "A003"], ["C001", "C002"]),
        (["A003", "A004", "A005"], ["C003"]),
    ]
    assert [group["candidate_ids"] for group in payload["groups"]] == [
        ["C001", "C002"],
        ["C003"],
    ]


def _anchor(idx: int) -> TaskAnchor:
    return TaskAnchor(
        anchor_id=f"A{idx:03d}",
        kind="new",
        line_start=idx,
        line_end=idx,
        speaker="Иван",
        utterances=(TranscriptUtterance(idx, "Иван", f"реплика {idx}"),),
        signals=("date_first",),
        deadline_phrases=(),
    )


def _candidate(candidate_id: str, anchor_id: str) -> PreLLMCandidate:
    return PreLLMCandidate(
        candidate_id=candidate_id,
        anchor_ids=(anchor_id,),
        evidence_span="",
        candidate_kind="new",
        date_phrases=(),
        speakers=(),
        signals=(),
    )
