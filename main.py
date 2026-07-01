from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Settings
from src.date_normalizer import print_replacement_table
from src.llm_client import OpenAITaskExtractor
from src.postprocess import build_dataframe
from src.sqlite_store import save_tasks
from src.stability import run_stability_check


MEETING_DATES = {
    "transcript.txt": "2026-04-13",
    "transcript2.txt": "2026-04-29",
    "transcript3.txt": "2026-04-15",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract action items from meeting transcripts."
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=Path("trascripts/transcript.txt"),
        help="Path to one transcript file.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all assignment transcripts from the trascripts directory.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of extraction runs for stability check.",
    )
    parser.add_argument(
        "--save-sqlite",
        action="store_true",
        help="Optionally save the final run result to SQLite.",
    )
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.all:
        return [Path("trascripts") / name for name in MEETING_DATES]
    return [args.transcript]


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    extractor = OpenAITaskExtractor(settings)

    for transcript_path in resolve_inputs(args):
        meeting_date = MEETING_DATES.get(transcript_path.name)
        if meeting_date is None:
            raise ValueError(
                f"Unknown meeting date for {transcript_path}. "
                "Add it to MEETING_DATES in main.py."
            )

        print("\n" + "=" * 100)
        print(f"Файл: {transcript_path}")
        print(f"Дата встречи: {meeting_date}")

        results = run_stability_check(
            transcript_path=transcript_path,
            meeting_date=meeting_date,
            runs=args.runs,
            extractor=extractor.extract,
        )

        final_result = results.extractions[-1]
        df, replacements = build_dataframe(
            final_result.tasks,
            meeting_date,
            final_result.anchors,
        )

        print("\nИтоговый DataFrame:")
        print(df.to_string(index=False))

        print_replacement_table(replacements)

        print("\nОтчет о стабильности:")
        for line in results.report_lines:
            print(line)

        if args.save_sqlite:
            save_tasks(df, transcript_path.name, Path("tasks.sqlite"))
            print("\nSQLite: результат сохранен в tasks.sqlite")


if __name__ == "__main__":
    main()
