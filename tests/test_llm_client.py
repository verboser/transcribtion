from src.llm_client import (
    _candidate_decision_contract_errors,
    _anchor_groups,
    _parse_candidate_decisions_payload,
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
