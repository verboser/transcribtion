from __future__ import annotations

import re
from pathlib import Path

from src.date_patterns import find_date_phrases
from src.schemas import AnchorKind, TaskAnchor, TranscriptUtterance


SPEAKER_RE = re.compile(r"^\s*(?P<speaker>[^:]{1,120}):\s*(?P<text>.*)\s*$")

DONE_PATTERNS = [
    r"\bвыполнен[аоы]?\b",
    r"\bвыполнил[аи]?\b",
    r"\bсделан[аоы]?\b",
    r"\bсделали\b",
    r"\bготов[аоы]?\b",
    r"\bутвердил[аи]?\b",
    r"\bутвержден[аоы]?\b",
    r"\bутверждён[аоы]?\b",
    r"\bотправил[аи]?\b",
    r"\bпередан",
    r"\bразработан",
    r"\bсмонтирован",
    r"\bпротестирован",
    r"\bзалил[аи]?\b",
    r"\bсогласовал",
    r"\bподготовил",
    r"\bзакрыт",
]

FAILED_PATTERNS = [
    r"\bне\s+успел[аи]?\b",
    r"\bне\s+успели\b",
    r"\bне\s+сделал[аи]?\b",
    r"\bне\s+сделали\b",
    r"\bне\s+выполн",
    r"\bне\s+готов",
    r"\bне\s+утверж",
    r"\bне\s+подготов",
    r"\bпросроч",
    r"\bсрок\s+прош",
]

TASK_INTENT_PATTERNS = [
    r"\bпод\s+протокол\b",
    r"\bпротокол\b",
    r"\bзадач",
    r"\bнужно\b",
    r"\bнадо\b",
    r"\bдолжн",
    r"\bответствен",
    r"\bпросьба\b",
    r"\bнаправьте\b",
    r"\bназначить\b",
    r"\bназначим\b",
    r"\bдоговорил[аи]сь\b",
    r"\bдоговоримся\b",
    r"\bразошлем\b",
    r"\bразошлём\b",
    r"\bсформировать\b",
    r"\bподготовка\b",
    r"\bподготовить\b",
    r"\bподготовим\b",
    r"\bподготовимся\b",
    r"\bобсудим\b",
    r"\bпровести\b",
    r"\bнаправить\b",
    r"\bдоработать\b",
    r"\bсогласовать\b",
    r"\bзапланируем\b",
    r"\bвозьм[её]м\s+паузу\b",
    r"\bбудем\b",
]

LOW_COVERAGE_MIN_RATIO = 0.08
LOW_COVERAGE_MIN_ANCHORS = 15
LOW_COVERAGE_MIN_UTTERANCES = 30
FALLBACK_TAIL_UTTERANCES = 70


def load_transcript(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def parse_transcript(text: str) -> list[TranscriptUtterance]:
    utterances: list[TranscriptUtterance] = []
    current: TranscriptUtterance | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = SPEAKER_RE.match(line)
        if match:
            current = TranscriptUtterance(
                line_no=len(utterances) + 1,
                speaker=normalize_space(match.group("speaker")),
                text=normalize_space(match.group("text")),
            )
            utterances.append(current)
            continue

        if current is not None:
            merged = TranscriptUtterance(
                line_no=current.line_no,
                speaker=current.speaker,
                text=normalize_space(f"{current.text} {line}"),
            )
            utterances[-1] = merged
            current = merged
        else:
            utterances.append(
                TranscriptUtterance(
                    line_no=len(utterances) + 1,
                    speaker="Неизвестный",
                    text=normalize_space(line),
                )
            )

    return utterances


def build_task_anchors(
    utterances: list[TranscriptUtterance],
    window_before: int = 0,
    window_after: int = 2,
) -> list[TaskAnchor]:
    seeds: list[dict] = []

    for idx, utterance in enumerate(utterances):
        text = _norm(utterance.text)
        nearby_text = _nearby_text(utterances, idx, before=1, after=2)
        wide_nearby_text = _nearby_text(utterances, idx, before=3, after=4)
        signals: list[str] = []
        kinds: set[AnchorKind] = set()

        if _matches_any(text, DONE_PATTERNS):
            signals.append("done_signal")
            kinds.add("done")
        if _matches_any(text, FAILED_PATTERNS):
            signals.append("failed_signal")
            kinds.add("failed")

        deadline_phrases = tuple(find_deadline_phrases(nearby_text))
        wide_deadline_phrases = tuple(find_deadline_phrases(wide_nearby_text))
        has_task_intent = _matches_any(text, TASK_INTENT_PATTERNS)
        has_deadline_nearby = bool(deadline_phrases)
        if has_task_intent and has_deadline_nearby:
            signals.append("task_with_deadline")
            kinds.add("new")

        has_date_first_signal = bool(find_deadline_phrases(text)) or _matches_any(
            text,
            [r"\bсрок\b", r"\bответствен"],
        )
        if has_date_first_signal and wide_deadline_phrases:
            signals.append("date_first")
            kinds.add("new")
            deadline_phrases = tuple(_dedupe_keep_order(list(deadline_phrases) + list(wide_deadline_phrases)))

        if not kinds:
            continue

        kind = _resolve_kind(kinds)
        before = 3 if "date_first" in signals else window_before
        after = 4 if "date_first" in signals else window_after
        seeds.append(
            _make_seed(
                utterances=utterances,
                idx=idx,
                kind=kind,
                signals=tuple(sorted(set(signals))),
                deadline_phrases=deadline_phrases,
                window_before=before,
                window_after=after,
            )
        )

    anchors = _build_anchors_from_seeds(_merge_overlapping_seeds(seeds), utterances)
    if _needs_low_coverage_fallback(anchors, utterances):
        fallback_seeds = _build_tail_fallback_seeds(utterances)
        anchors = _build_anchors_from_seeds(
            _merge_overlapping_seeds(seeds + fallback_seeds),
            utterances,
        )

    return anchors


def format_anchors_for_prompt(anchors: list[TaskAnchor]) -> str:
    return "\n\n".join(anchor.as_prompt_block() for anchor in anchors)


def find_deadline_phrases(text: str) -> list[str]:
    return find_date_phrases(text)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _make_seed(
    utterances: list[TranscriptUtterance],
    idx: int,
    kind: AnchorKind,
    signals: tuple[str, ...],
    deadline_phrases: tuple[str, ...],
    window_before: int,
    window_after: int,
) -> dict:
    start = max(0, idx - window_before)
    end = min(len(utterances) - 1, idx + window_after)
    return {
        "start": start,
        "end": end,
        "kind": kind,
        "speakers": (utterances[idx].speaker,),
        "signals": signals,
        "deadline_phrases": deadline_phrases,
    }


def _merge_overlapping_seeds(seeds: list[dict]) -> list[dict]:
    if not seeds:
        return []

    sorted_seeds = sorted(seeds, key=lambda seed: (seed["start"], seed["end"]))
    merged = [sorted_seeds[0].copy()]
    for seed in sorted_seeds[1:]:
        current = merged[-1]
        if seed["start"] <= current["end"] + 1:
            current["end"] = max(current["end"], seed["end"])
            current["kind"] = _merge_kind(current["kind"], seed["kind"])
            current["signals"] = tuple(
                sorted(set(current["signals"]) | set(seed["signals"]))
            )
            current["speakers"] = tuple(
                _dedupe_keep_order(list(current["speakers"]) + list(seed["speakers"]))
            )
            current["deadline_phrases"] = tuple(
                _dedupe_keep_order(
                    list(current["deadline_phrases"]) + list(seed["deadline_phrases"])
                )
            )
            continue
        merged.append(seed.copy())

    return merged


def _build_anchors_from_seeds(
    seeds: list[dict],
    utterances: list[TranscriptUtterance],
) -> list[TaskAnchor]:
    anchors: list[TaskAnchor] = []
    for seed in seeds:
        window = tuple(utterances[seed["start"] : seed["end"] + 1])
        speakers = seed["speakers"]
        speaker = speakers[0] if len(speakers) == 1 else "Несколько спикеров"
        anchors.append(
            TaskAnchor(
                anchor_id=f"A{len(anchors) + 1:03d}",
                kind=seed["kind"],
                line_start=window[0].line_no,
                line_end=window[-1].line_no,
                speaker=speaker,
                utterances=window,
                signals=seed["signals"],
                deadline_phrases=seed["deadline_phrases"],
            )
        )

    return anchors


def _needs_low_coverage_fallback(
    anchors: list[TaskAnchor],
    utterances: list[TranscriptUtterance],
) -> bool:
    if len(utterances) < LOW_COVERAGE_MIN_UTTERANCES:
        return False
    anchored_lines = {utterance.line_no for anchor in anchors for utterance in anchor.utterances}
    coverage = len(anchored_lines) / len(utterances)
    return len(anchors) < LOW_COVERAGE_MIN_ANCHORS or coverage < LOW_COVERAGE_MIN_RATIO


def _build_tail_fallback_seeds(utterances: list[TranscriptUtterance]) -> list[dict]:
    if not utterances:
        return []

    start = max(0, len(utterances) - FALLBACK_TAIL_UTTERANCES)
    chunk_text = " ".join(utterance.text for utterance in utterances[start:])
    return [
        {
            "start": start,
            "end": len(utterances) - 1,
            "kind": "mixed",
            "speakers": ("Несколько спикеров",),
            "signals": ("final_tail_fallback",),
            "deadline_phrases": tuple(find_deadline_phrases(chunk_text)),
        }
    ]


def _resolve_kind(kinds: set[AnchorKind]) -> AnchorKind:
    if len(kinds) == 1:
        return next(iter(kinds))
    return "mixed"


def _merge_kind(left: AnchorKind, right: AnchorKind) -> AnchorKind:
    if left == right:
        return left
    return "mixed"


def _nearby_text(
    utterances: list[TranscriptUtterance],
    idx: int,
    before: int,
    after: int,
) -> str:
    start = max(0, idx - before)
    end = min(len(utterances), idx + after + 1)
    return " ".join(utterance.text for utterance in utterances[start:end])


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _norm(value: str) -> str:
    return normalize_space(value).lower().replace("ё", "е")


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
