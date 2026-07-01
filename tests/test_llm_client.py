from src.llm_client import _anchor_groups
from src.schemas import TaskAnchor, TranscriptUtterance


def test_anchor_groups_overlap_neighbors() -> None:
    anchors = [_anchor(idx) for idx in range(1, 6)]

    groups = list(_anchor_groups(anchors, group_size=3, overlap=1))

    assert [[anchor.anchor_id for anchor in group] for group in groups] == [
        ["A001", "A002", "A003"],
        ["A003", "A004", "A005"],
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
