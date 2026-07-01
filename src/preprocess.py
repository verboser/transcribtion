from __future__ import annotations

import re
from pathlib import Path

from src.schemas import AnchorKind, TaskAnchor, TranscriptUtterance


SPEAKER_RE = re.compile(r"^\s*(?P<speaker>[^:]{1,120}):\s*(?P<text>.*)\s*$")

DONE_PATTERNS = [
    r"\bвыполн",
    r"\bготов[аыо]?\b",
    r"\bутверд",
    r"\bотправ",
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
    r"\bзадач",
    r"\bнужно\b",
    r"\bнадо\b",
    r"\bдолжн",
    r"\bпросьба\b",
    r"\bнаправьте\b",
    r"\bназначим\b",
    r"\bразошлем\b",
    r"\bразошлём\b",
    r"\bсформировать\b",
    r"\bподготовить\b",
    r"\bпровести\b",
    r"\bнаправить\b",
    r"\bдоработать\b",
    r"\bсогласовать\b",
]

DEADLINE_PATTERNS = [
    r"\bпослезавтра\b",
    r"\bзавтра\b",
    r"\bчерез\s+(?:неделю|месяц)\b",
    r"\b(?:в|во|к|до|на)\s+(?:понедельник|понедельника|вторник|вторника|среду|среды|четверг|четверга|пятницу|пятницы|субботу|субботы|воскресенье)\b",
    r"\b(?:до|к|на)?\s*\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b",
    r"\b(?:до|к|на)\s+\d{1,2}\s+числа\b",
    r"\b(?:к\s+)?концу недели\b",
]


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
    candidates: list[tuple[int, AnchorKind, tuple[str, ...], tuple[str, ...]]] = []

    for idx, utterance in enumerate(utterances):
        text = _norm(utterance.text)
        nearby_text = _nearby_text(utterances, idx, before=1, after=2)
        signals: list[str] = []
        kinds: set[AnchorKind] = set()

        if _matches_any(text, DONE_PATTERNS):
            signals.append("done_signal")
            kinds.add("done")
        if _matches_any(text, FAILED_PATTERNS):
            signals.append("failed_signal")
            kinds.add("failed")

        deadline_phrases = tuple(find_deadline_phrases(nearby_text))
        has_task_intent = _matches_any(text, TASK_INTENT_PATTERNS)
        has_deadline_nearby = bool(deadline_phrases)
        if has_task_intent and has_deadline_nearby:
            signals.append("task_with_deadline")
            kinds.add("new")

        if not kinds:
            continue

        kind = _resolve_kind(kinds)
        candidates.append((idx, kind, tuple(sorted(set(signals))), deadline_phrases))

    anchors: list[TaskAnchor] = []
    seen_windows: set[tuple[int, int, str, AnchorKind]] = set()
    for anchor_no, (idx, kind, signals, deadline_phrases) in enumerate(candidates, start=1):
        start = max(0, idx - window_before)
        end = min(len(utterances) - 1, idx + window_after)
        window = tuple(utterances[start : end + 1])
        key = (window[0].line_no, window[-1].line_no, utterances[idx].speaker, kind)
        if key in seen_windows:
            continue
        seen_windows.add(key)
        anchors.append(
            TaskAnchor(
                anchor_id=f"A{len(anchors) + 1:03d}",
                kind=kind,
                line_start=window[0].line_no,
                line_end=window[-1].line_no,
                speaker=utterances[idx].speaker,
                utterances=window,
                signals=signals,
                deadline_phrases=deadline_phrases,
            )
        )

    return anchors


def format_anchors_for_prompt(anchors: list[TaskAnchor]) -> str:
    return "\n\n".join(anchor.as_prompt_block() for anchor in anchors)


def find_deadline_phrases(text: str) -> list[str]:
    normalized = _norm(text)
    phrases: list[str] = []
    for pattern in DEADLINE_PATTERNS:
        phrases.extend(match.group(0).strip() for match in re.finditer(pattern, normalized))
    return _dedupe_keep_order(phrases)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _resolve_kind(kinds: set[AnchorKind]) -> AnchorKind:
    if len(kinds) == 1:
        return next(iter(kinds))
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
