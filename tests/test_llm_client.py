from src.llm_client import (
    _anchor_decision_contract_errors,
    _anchor_groups,
    _parse_anchor_decisions_payload,
)
from src.schemas import TaskAnchor, TranscriptUtterance


def test_anchor_groups_overlap_neighbors() -> None:
    anchors = [_anchor(idx) for idx in range(1, 6)]

    groups = list(_anchor_groups(anchors, group_size=3, overlap=1))

    assert [[anchor.anchor_id for anchor in group] for group in groups] == [
        ["A001", "A002", "A003"],
        ["A003", "A004", "A005"],
    ]


def test_parse_anchor_decisions_flattens_tasks_and_falls_back_to_decision_anchor() -> None:
    payload = {
        "anchor_decisions": [
            {
                "anchor_id": "A001",
                "tasks": [
                    {
                        "block": "Новые",
                        "task": "подготовить отчет",
                        "responsible": "Иван",
                        "deadline_raw": "до пятницы",
                        "evidence": "Иван: подготовить отчет до пятницы",
                        "anchor_ids": [],
                    }
                ],
            },
            {"anchor_id": "A002", "tasks": []},
        ]
    }

    tasks = _parse_anchor_decisions_payload(payload)

    assert len(tasks) == 1
    assert tasks[0].anchor_ids == ("A001",)
    assert tasks[0].task == "подготовить отчет"


def test_anchor_decision_contract_errors_report_missing_extra_and_duplicate_ids() -> None:
    payload = {
        "anchor_decisions": [
            {"anchor_id": "A001", "tasks": []},
            {"anchor_id": "A001", "tasks": []},
            {"anchor_id": "A999", "tasks": []},
        ]
    }

    errors = _anchor_decision_contract_errors(payload, ["A001", "A002"])

    assert errors == {
        "missing_anchor_decisions": ["A002"],
        "extra_anchor_decisions": ["A999"],
        "duplicate_anchor_decisions": ["A001"],
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
