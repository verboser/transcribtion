from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.conversation_lexicon import LexiconCategory, find_lexicon_matches
from src.preprocess import load_transcript, parse_transcript


@dataclass(frozen=True)
class LexiconLineMatch:
    line_no: int
    speaker: str
    category: LexiconCategory
    pattern_name: str
    matched_text: str
    text: str


def audit_transcript(
    transcript_path: Path,
    categories: tuple[LexiconCategory, ...] | None = None,
) -> list[LexiconLineMatch]:
    utterances = parse_transcript(load_transcript(transcript_path))
    matches: list[LexiconLineMatch] = []
    for utterance in utterances:
        for pattern, matched_text in find_lexicon_matches(utterance.text, categories):
            matches.append(
                LexiconLineMatch(
                    line_no=utterance.line_no,
                    speaker=utterance.speaker,
                    category=pattern.category,
                    pattern_name=pattern.name,
                    matched_text=matched_text,
                    text=utterance.text,
                )
            )
    return matches


def format_audit_report(
    transcript_path: Path,
    matches: list[LexiconLineMatch],
    max_examples_per_category: int = 12,
) -> list[str]:
    lines = [f"Файл: {transcript_path}"]
    if not matches:
        return lines + ["Совпадений по словарю не найдено."]

    by_category = Counter(match.category for match in matches)
    unique_lines_by_category = {
        category: len({match.line_no for match in matches if match.category == category})
        for category in by_category
    }
    lines.append("Совпадения по категориям:")
    for category, count in sorted(by_category.items()):
        lines.append(
            f"- {category}: {count} matches, "
            f"{unique_lines_by_category[category]} реплик"
        )

    lines.append("")
    lines.append("Примеры:")
    grouped: dict[LexiconCategory, list[LexiconLineMatch]] = defaultdict(list)
    for match in matches:
        grouped[match.category].append(match)

    for category in sorted(grouped):
        lines.append(f"## {category}")
        for match in grouped[category][:max_examples_per_category]:
            lines.append(
                "[{line:04d}] {speaker}: {text} "
                "(match={matched}, pattern={pattern})".format(
                    line=match.line_no,
                    speaker=match.speaker,
                    text=match.text,
                    matched=match.matched_text,
                    pattern=match.pattern_name,
                )
            )
        if len(grouped[category]) > max_examples_per_category:
            lines.append(
                f"... еще {len(grouped[category]) - max_examples_per_category}"
            )
        lines.append("")

    return lines
