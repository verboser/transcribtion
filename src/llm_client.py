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


class OpenAITaskExtractor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)

    def extract(self, transcript_path: Path, meeting_date: str) -> ExtractionResult:
        raw_text = load_transcript(transcript_path)
        utterances = parse_transcript(raw_text)
        anchors = build_task_anchors(utterances)
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
        tasks = [
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
        return ExtractionResult(tasks=tasks, anchors=anchors, raw_response=payload)


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
