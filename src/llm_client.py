from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import Settings
from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.schemas import (
    ExtractionResult,
    ExtractedTask,
    TASK_EXTRACTION_JSON_SCHEMA,
)
from src.preprocess import (
    build_task_anchors,
    format_anchors_for_prompt,
    load_transcript,
    parse_transcript,
)


DEFAULT_ANCHOR_GROUP_SIZE = 10
DEFAULT_ANCHOR_GROUP_OVERLAP = 2


class OpenAITaskExtractor:
    def __init__(
        self,
        settings: Settings,
        strategy: str = "grouped",
        anchor_group_size: int = DEFAULT_ANCHOR_GROUP_SIZE,
        anchor_group_overlap: int = DEFAULT_ANCHOR_GROUP_OVERLAP,
    ) -> None:
        if strategy not in {"grouped", "global"}:
            raise ValueError("strategy must be 'grouped' or 'global'.")
        if anchor_group_size < 1:
            raise ValueError("anchor_group_size must be at least 1.")
        if anchor_group_overlap < 0:
            raise ValueError("anchor_group_overlap must be at least 0.")
        self.settings = settings
        self.strategy = strategy
        self.anchor_group_size = anchor_group_size
        self.anchor_group_overlap = anchor_group_overlap
        self.client = OpenAI(api_key=settings.openai_api_key)

    def extract(self, transcript_path: Path, meeting_date: str) -> ExtractionResult:
        raw_text = load_transcript(transcript_path)
        utterances = parse_transcript(raw_text)
        anchors = build_task_anchors(utterances)
        if self.strategy == "global":
            tasks, raw_response = self._extract_from_anchor_batch(
                meeting_date=meeting_date,
                anchors=anchors,
            )
        else:
            tasks, raw_response = self._extract_from_anchor_groups(
                meeting_date=meeting_date,
                anchors=anchors,
            )

        return ExtractionResult(tasks=tasks, anchors=anchors, raw_response=raw_response)

    def _extract_from_anchor_groups(
        self,
        meeting_date: str,
        anchors,
    ) -> tuple[list[ExtractedTask], dict]:
        tasks: list[ExtractedTask] = []
        group_payloads: list[dict] = []

        for group_idx, group in enumerate(
            _anchor_groups(
                anchors,
                group_size=self.anchor_group_size,
                overlap=self.anchor_group_overlap,
            ),
            start=1,
        ):
            group_tasks, payload = self._extract_from_anchor_batch(meeting_date, group)
            tasks.extend(group_tasks)
            group_payloads.append(
                {
                    "group": group_idx,
                    "anchor_ids": [anchor.anchor_id for anchor in group],
                    "response": payload,
                }
            )

        return (
            tasks,
            {
                "strategy": "grouped",
                "group_size": self.anchor_group_size,
                "overlap": self.anchor_group_overlap,
                "groups": group_payloads,
            },
        )

    def _extract_from_anchor_batch(
        self,
        meeting_date: str,
        anchors,
    ) -> tuple[list[ExtractedTask], dict]:
        if not anchors:
            return [], {"tasks": []}

        prompt_anchors = format_anchors_for_prompt(anchors)

        response = self.client.responses.create(
            model=self.settings.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(meeting_date, prompt_anchors),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": TASK_EXTRACTION_JSON_SCHEMA["name"],
                    "strict": TASK_EXTRACTION_JSON_SCHEMA["strict"],
                    "schema": TASK_EXTRACTION_JSON_SCHEMA["schema"],
                }
            },
            temperature=0,
            max_output_tokens=self.settings.max_output_tokens,
        )

        payload = json.loads(_response_text(response))
        return _parse_tasks_payload(payload), payload


def _parse_tasks_payload(payload: dict) -> list[ExtractedTask]:
    return [
        ExtractedTask(
            block=item["block"],
            task=item["task"].strip(),
            responsible=item["responsible"].strip(),
            deadline_raw=item["deadline_raw"].strip(),
            evidence=item["evidence"].strip(),
            anchor_ids=tuple(item["anchor_ids"]),
        )
        for item in payload.get("tasks", [])
    ]


def _anchor_groups(anchors, group_size: int, overlap: int = DEFAULT_ANCHOR_GROUP_OVERLAP):
    effective_overlap = min(max(0, overlap), group_size - 1)
    step = max(1, group_size - effective_overlap)
    for start in range(0, len(anchors), step):
        group = anchors[start : start + group_size]
        if not group:
            break
        yield group
        if start + group_size >= len(anchors):
            break


def _response_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    chunks: list[str] = []
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    if chunks:
        return "".join(chunks)

    raise RuntimeError("OpenAI response did not contain output text.")
