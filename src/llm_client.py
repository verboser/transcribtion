from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import Settings
from src.prompts import (
    SYSTEM_PROMPT,
    build_candidate_decision_user_prompt,
    build_user_prompt,
    build_verifier_user_prompt,
)
from src.schemas import (
    CANDIDATE_DECISION_JSON_SCHEMA,
    ExtractionResult,
    ExtractedTask,
    TASK_EXTRACTION_JSON_SCHEMA,
    TASK_VERIFICATION_JSON_SCHEMA,
    VerificationCandidate,
)
from src.preprocess import (
    build_candidates_and_anchors,
    format_anchors_for_prompt,
    format_candidates_for_prompt,
    load_transcript,
    parse_transcript,
)


DEFAULT_ANCHOR_GROUP_SIZE = 10
DEFAULT_ANCHOR_GROUP_OVERLAP = 2


class OpenAITaskExtractor:
    def __init__(
        self,
        settings: Settings,
        strategy: str = "anchored",
        anchor_group_size: int = DEFAULT_ANCHOR_GROUP_SIZE,
        anchor_group_overlap: int = DEFAULT_ANCHOR_GROUP_OVERLAP,
    ) -> None:
        if strategy not in {"anchored", "grouped", "global"}:
            raise ValueError("strategy must be 'anchored', 'grouped' or 'global'.")
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
        candidates, anchors = build_candidates_and_anchors(utterances)
        if self.strategy == "global":
            tasks, raw_response = self._extract_from_anchor_batch(
                meeting_date=meeting_date,
                anchors=anchors,
            )
        elif self.strategy == "anchored":
            tasks, raw_response = self._extract_from_candidate_decision_groups(
                meeting_date=meeting_date,
                anchors=anchors,
                candidates=candidates,
            )
        else:
            tasks, raw_response = self._extract_from_anchor_groups(
                meeting_date=meeting_date,
                anchors=anchors,
            )

        return ExtractionResult(tasks=tasks, anchors=anchors, candidates=candidates, raw_response=raw_response)

    def verify_candidates(
        self,
        candidates: list[VerificationCandidate],
        meeting_date: str,
    ) -> set[str]:
        if not candidates:
            return set()

        response = self.client.responses.create(
            model=self.settings.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_verifier_user_prompt(
                        meeting_date,
                        _format_verification_candidates(candidates),
                    ),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": TASK_VERIFICATION_JSON_SCHEMA["name"],
                    "strict": TASK_VERIFICATION_JSON_SCHEMA["strict"],
                    "schema": TASK_VERIFICATION_JSON_SCHEMA["schema"],
                }
            },
            temperature=0,
            max_output_tokens=self.settings.max_output_tokens,
        )

        payload = json.loads(_response_text(response))
        candidate_ids = {candidate.candidate_id for candidate in candidates}
        return {
            str(item["candidate_id"]).strip()
            for item in payload.get("decisions", [])
            if item.get("keep") is True
            and str(item.get("candidate_id", "")).strip() in candidate_ids
        }

    def _extract_from_candidate_decision_groups(
        self,
        meeting_date: str,
        anchors,
        candidates,
    ) -> tuple[list[ExtractedTask], dict]:
        tasks: list[ExtractedTask] = []
        group_payloads: list[dict] = []
        processed_candidate_ids: set[str] = set()

        for group_idx, group in enumerate(
            _anchor_groups(
                anchors,
                group_size=self.anchor_group_size,
                overlap=self.anchor_group_overlap,
            ),
            start=1,
        ):
            group_anchor_ids = {a.anchor_id for a in group}
            filtered_candidates = []
            for candidate in candidates:
                if candidate.candidate_id in processed_candidate_ids:
                    continue
                if not candidate.anchor_ids:
                    if group_idx != 1:
                        continue
                elif not any(anchor_id in group_anchor_ids for anchor_id in candidate.anchor_ids):
                    continue
                filtered_candidates.append(candidate)
                processed_candidate_ids.add(candidate.candidate_id)

            if not filtered_candidates:
                continue

            group_tasks, payload = self._extract_from_candidate_decision_batch(
                meeting_date,
                group,
                filtered_candidates,
            )
            tasks.extend(group_tasks)
            group_payloads.append(
                {
                    "group": group_idx,
                    "anchor_ids": [anchor.anchor_id for anchor in group],
                    "candidate_ids": [
                        candidate.candidate_id for candidate in filtered_candidates
                    ],
                    "response": payload,
                }
            )

        return (
            tasks,
            {
                "strategy": "anchored",
                "group_size": self.anchor_group_size,
                "overlap": self.anchor_group_overlap,
                "groups": group_payloads,
            },
        )

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

    def _extract_from_candidate_decision_batch(
        self,
        meeting_date: str,
        anchors,
        candidates,
    ) -> tuple[list[ExtractedTask], dict]:
        if not candidates:
            return [], {"candidate_decisions": []}

        prompt_anchors = format_anchors_for_prompt(anchors)
        prompt_candidates = format_candidates_for_prompt(candidates)

        response = self.client.responses.create(
            model=self.settings.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_candidate_decision_user_prompt(
                        meeting_date,
                        prompt_anchors,
                        prompt_candidates,
                    ),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": CANDIDATE_DECISION_JSON_SCHEMA["name"],
                    "strict": CANDIDATE_DECISION_JSON_SCHEMA["strict"],
                    "schema": CANDIDATE_DECISION_JSON_SCHEMA["schema"],
                }
            },
            temperature=0,
            max_output_tokens=self.settings.max_output_tokens,
        )

        payload = json.loads(_response_text(response))
        expected_candidate_ids = [c.candidate_id for c in candidates]
        contract_errors = _candidate_decision_contract_errors(
            payload,
            expected_candidate_ids,
        )
        if contract_errors:
            payload["contract_errors"] = contract_errors
        return _parse_candidate_decisions_payload(payload, candidates), payload


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


def _parse_candidate_decisions_payload(payload: dict, candidates) -> list[ExtractedTask]:
    tasks: list[ExtractedTask] = []
    cand_map = {c.candidate_id: c for c in candidates}
    for decision in payload.get("candidate_decisions", []):
        if not decision.get("is_task"):
            continue
            
        candidate_id = str(decision.get("candidate_id", "")).strip()
        cand = cand_map.get(candidate_id)
        if cand is None:
            continue
        anchor_ids = cand.anchor_ids
        
        tasks.append(
            ExtractedTask(
                block=decision.get("block", "Новые"),
                task=decision.get("task", "").strip(),
                responsible=decision.get("responsible", "").strip(),
                deadline_raw=decision.get("deadline_raw", "").strip(),
                evidence=decision.get("evidence", "").strip(),
                anchor_ids=anchor_ids,
            )
        )
    return tasks


def _candidate_decision_contract_errors(
    payload: dict,
    expected_candidate_ids: list[str],
) -> dict[str, list[str]]:
    expected = set(expected_candidate_ids)
    seen = [
        str(decision.get("candidate_id", "")).strip()
        for decision in payload.get("candidate_decisions", [])
    ]
    seen_set = set(seen)
    duplicates = sorted(
        candidate_id
        for candidate_id in seen_set
        if candidate_id and seen.count(candidate_id) > 1
    )
    errors = {
        "missing_candidate_decisions": sorted(expected - seen_set),
        "extra_candidate_decisions": sorted(seen_set - expected),
        "duplicate_candidate_decisions": duplicates,
    }
    return {key: value for key, value in errors.items() if value}


def _format_verification_candidates(candidates: list[VerificationCandidate]) -> str:
    blocks: list[str] = []
    for candidate in candidates:
        run_indices = ", ".join(f"run_{idx}" for idx in candidate.run_indices)
        blocks.append(
            "\n".join(
                [
                    f'<candidate id="{candidate.candidate_id}">',
                    f"Блок: {candidate.block}",
                    f"Задача: {candidate.task}",
                    f"Ответственный: {candidate.responsible}",
                    f"Срок: {candidate.deadline or '-'}",
                    (
                        "Поддержка: "
                        f"{candidate.support_count} прогонов "
                        f"({candidate.support_ratio:.0%}); {run_indices}"
                    ),
                    f"Evidence: {candidate.evidence}",
                    "</candidate>",
                ]
            )
        )
    return "\n\n".join(blocks)


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
