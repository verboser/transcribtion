from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Block = Literal["Выполненные", "Невыполненные", "Новые"]
AnchorKind = Literal["done", "failed", "new", "mixed"]

DATAFRAME_COLUMNS = [
    "Блок",
    "Задача",
    "Ответственный",
    "Срок",
    "Обоснование",
]

BLOCK_ORDER = {
    "Выполненные": 0,
    "Невыполненные": 1,
    "Новые": 2,
}


@dataclass(frozen=True)
class TranscriptUtterance:
    line_no: int
    speaker: str
    text: str

    def as_prompt_line(self) -> str:
        return f"[{self.line_no:04d}] {self.speaker}: {self.text}"


@dataclass(frozen=True)
class TaskAnchor:
    anchor_id: str
    kind: AnchorKind
    line_start: int
    line_end: int
    speaker: str
    utterances: tuple[TranscriptUtterance, ...]
    signals: tuple[str, ...]
    deadline_phrases: tuple[str, ...]

    def as_prompt_block(self) -> str:
        deadlines = ", ".join(self.deadline_phrases) if self.deadline_phrases else "-"
        signals = ", ".join(self.signals)
        lines = "\n".join(utterance.as_prompt_line() for utterance in self.utterances)
        return (
            f'<anchor id="{self.anchor_id}" kind="{self.kind}" '
            f'speaker="{self.speaker}" lines="{self.line_start}-{self.line_end}" '
            f'signals="{signals}" deadlines="{deadlines}">\n'
            f"{lines}\n"
            "</anchor>"
        )

    def text(self) -> str:
        return "\n".join(utterance.as_prompt_line() for utterance in self.utterances)


@dataclass(frozen=True)
class ExtractedTask:
    block: Block
    task: str
    responsible: str
    deadline_raw: str
    evidence: str
    anchor_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExtractionResult:
    tasks: list[ExtractedTask]
    anchors: list[TaskAnchor]
    raw_response: dict


@dataclass(frozen=True)
class VerificationCandidate:
    candidate_id: str
    block: str
    task: str
    responsible: str
    deadline: str
    evidence: str
    support_count: int
    support_ratio: float
    run_indices: tuple[int, ...]


TASK_ITEM_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "block",
        "task",
        "responsible",
        "deadline_raw",
        "evidence",
        "anchor_ids",
    ],
    "properties": {
        "block": {
            "type": "string",
            "enum": ["Выполненные", "Невыполненные", "Новые"],
        },
        "task": {
            "type": "string",
            "description": "Краткая формулировка задачи.",
        },
        "responsible": {
            "type": "string",
            "description": (
                "Явно назначенный ответственный из реплики-обоснования. "
                "Если такого назначения нет, говорящий из evidence."
            ),
        },
        "deadline_raw": {
            "type": "string",
            "description": (
                "Дословная фраза срока из реплики. Пустая строка, "
                "если срока нет."
            ),
        },
        "evidence": {
            "type": "string",
            "description": (
                "Дословная цитата из транскрипта с именем спикера, "
                "подтверждающая задачу, статус и срок при наличии."
            ),
        },
        "anchor_ids": {
            "type": "array",
            "description": (
                "ID anchor-блоков, из которых извлечена задача. "
                "Можно указать несколько, если задача и срок в соседних anchors."
            ),
            "items": {"type": "string"},
        },
    },
}


TASK_EXTRACTION_JSON_SCHEMA = {
    "name": "meeting_task_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["tasks"],
        "properties": {
            "tasks": {
                "type": "array",
                "items": TASK_ITEM_JSON_SCHEMA,
            }
        },
    },
}


ANCHOR_DECISION_JSON_SCHEMA = {
    "name": "meeting_anchor_task_decisions",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["anchor_decisions"],
        "properties": {
            "anchor_decisions": {
                "type": "array",
                "description": (
                    "Ровно одно решение для каждого anchor из входной группы. "
                    "Если в anchor нет задач, tasks должен быть пустым массивом."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["anchor_id", "tasks"],
                    "properties": {
                        "anchor_id": {
                            "type": "string",
                            "description": "ID anchor-блока из входа.",
                        },
                        "tasks": {
                            "type": "array",
                            "description": "Задачи, подтвержденные этим anchor.",
                            "items": TASK_ITEM_JSON_SCHEMA,
                        },
                    },
                },
            }
        },
    },
}


TASK_VERIFICATION_JSON_SCHEMA = {
    "name": "meeting_task_verification",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "description": "Решение verifier по каждому кандидату.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["candidate_id", "keep", "reason"],
                    "properties": {
                        "candidate_id": {
                            "type": "string",
                            "description": "ID кандидата из входного списка.",
                        },
                        "keep": {
                            "type": "boolean",
                            "description": (
                                "true, только если evidence прямо подтверждает "
                                "блок, задачу, ответственного и срок."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": "Короткая причина решения.",
                        },
                    },
                },
            }
        },
    },
}
