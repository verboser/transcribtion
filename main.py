from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Settings
from src.date_normalizer import print_replacement_table
from src.llm_client import (
    DEFAULT_ANCHOR_GROUP_OVERLAP,
    DEFAULT_ANCHOR_GROUP_SIZE,
    OpenAITaskExtractor,
)
from src.lexicon_audit import audit_transcript, format_audit_report
from src.sqlite_store import save_tasks
from src.stability import run_stability_check


MEETING_DATES = {
    "transcript.txt": "2026-04-13",
    "transcript2.txt": "2026-04-29",
    "transcript3.txt": "2026-04-15",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Извлечь задачи из транскриптов встреч."
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=Path("trascripts/transcript.txt"),
        help="Путь к одному файлу транскрипта.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Обработать все транскрипты из директории trascripts.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help=(
            "Количество прогонов извлечения. Для production обычно достаточно 1; "
            "для проверки стабильности укажите 5."
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=["grouped", "global"],
        default="grouped",
        help=(
            "Стратегия LLM-извлечения: grouped отправляет компактные группы "
            "anchors, global отправляет все anchors одним запросом."
        ),
    )
    parser.add_argument(
        "--anchor-group-size",
        type=int,
        default=DEFAULT_ANCHOR_GROUP_SIZE,
        help="Количество anchors в одном LLM-запросе для --strategy grouped.",
    )
    parser.add_argument(
        "--anchor-group-overlap",
        type=int,
        default=DEFAULT_ANCHOR_GROUP_OVERLAP,
        help="Перекрытие соседних групп anchors для --strategy grouped.",
    )
    parser.add_argument(
        "--save-sqlite",
        action="store_true",
        help="Дополнительно сохранить финальный результат в SQLite.",
    )
    parser.add_argument(
        "--audit-lexicon",
        action="store_true",
        help=(
            "Показать совпадения по словарю разговорных маркеров без вызова LLM "
            "и без проверки OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--audit-max-examples",
        type=int,
        default=12,
        help="Максимум примеров на категорию для --audit-lexicon.",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs должен быть не меньше 1.")
    if args.anchor_group_size < 1:
        parser.error("--anchor-group-size должен быть не меньше 1.")
    if args.anchor_group_overlap < 0:
        parser.error("--anchor-group-overlap должен быть не меньше 0.")
    return args


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.all:
        return [Path("trascripts") / name for name in MEETING_DATES]
    return [args.transcript]


def main() -> None:
    args = parse_args()
    if args.audit_lexicon:
        for transcript_path in resolve_inputs(args):
            matches = audit_transcript(transcript_path)
            for line in format_audit_report(
                transcript_path,
                matches,
                max_examples_per_category=args.audit_max_examples,
            ):
                print(line)
            print("")
        return

    settings = Settings.from_env()
    extractor = OpenAITaskExtractor(
        settings,
        strategy=args.strategy,
        anchor_group_size=args.anchor_group_size,
        anchor_group_overlap=args.anchor_group_overlap,
    )

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

        selected_run_idx = (
            results.selected_run_idx
            if results.selected_run_idx is not None
            else len(results.extractions) - 1
        )
        df = results.final_df
        replacements = results.final_replacements

        print(f"\nВыбран финальный прогон: run_{selected_run_idx + 1}")
        print(f"Источник финальной таблицы: {results.final_source}")
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
