from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Settings
from src.date_normalizer import print_replacement_table
from src.golden_eval import (
    DEFAULT_GOLDEN_DIR,
    evaluate_dataframe_against_golden,
    load_golden_tasks,
)
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
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Извлечь задачи из транскриптов встреч."
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=PROJECT_ROOT / "trascripts" / "transcript.txt",
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
        default=5,
        help=(
            "Количество прогонов извлечения. По умолчанию 5, как в тестовом задании."
        ),
    )
    parser.add_argument(
        "--min-consensus-share",
        type=float,
        default=0.90,
        help=(
            "Минимальная доля прогонов, где строка должна повториться, чтобы попасть "
            "в финальный DataFrame. Для 5 прогонов и 0.90 нужна поддержка 5/5."
        ),
    )
    parser.add_argument(
        "--max-stability-delta",
        type=float,
        default=0.10,
        help="Максимально допустимое расхождение счетчиков между прогонами.",
    )
    parser.add_argument(
        "--fail-on-unstable",
        action="store_true",
        help="Завершить процесс с кодом 2, если стабильность ниже заданного порога.",
    )
    parser.add_argument(
        "--verify-near-consensus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Проверять через отдельный verifier кандидатов, которые повторились "
            "не во всех 5 прогонах, но имеют достаточную поддержку."
        ),
    )
    parser.add_argument(
        "--verifier-min-support-share",
        type=float,
        default=0.60,
        help=(
            "Минимальная доля прогонов для отправки кандидата в verifier. "
            "Для 5 прогонов и 0.60 это поддержка 3/5."
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=["anchored", "grouped", "global"],
        default="anchored",
        help=(
            "Стратегия LLM-извлечения: anchored требует решение по каждому "
            "anchor в группе, grouped возвращает свободный список задач по "
            "группе, global отправляет все anchors одним запросом."
        ),
    )
    parser.add_argument(
        "--anchor-group-size",
        type=int,
        default=DEFAULT_ANCHOR_GROUP_SIZE,
        help=(
            "Количество anchors в одном LLM-запросе для --strategy anchored/grouped."
        ),
    )
    parser.add_argument(
        "--anchor-group-overlap",
        type=int,
        default=DEFAULT_ANCHOR_GROUP_OVERLAP,
        help="Перекрытие соседних групп anchors для --strategy anchored/grouped.",
    )
    parser.add_argument(
        "--save-sqlite",
        action="store_true",
        help="Дополнительно сохранить финальный результат в SQLite.",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=PROJECT_ROOT / "tasks.sqlite",
        help="Путь к SQLite-файлу для --save-sqlite.",
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
    parser.add_argument(
        "--eval-golden",
        action="store_true",
        help="Сравнить финальный DataFrame с golden-разметкой.",
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=DEFAULT_GOLDEN_DIR,
        help="Директория с golden JSON-файлами.",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs должен быть не меньше 1.")
    if not 0 < args.min_consensus_share <= 1:
        parser.error("--min-consensus-share должен быть в диапазоне (0, 1].")
    if args.max_stability_delta < 0:
        parser.error("--max-stability-delta должен быть неотрицательным.")
    if not 0 < args.verifier_min_support_share <= 1:
        parser.error("--verifier-min-support-share должен быть в диапазоне (0, 1].")
    if args.anchor_group_size < 1:
        parser.error("--anchor-group-size должен быть не меньше 1.")
    if args.anchor_group_overlap < 0:
        parser.error("--anchor-group-overlap должен быть не меньше 0.")
    return args


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.all:
        return [PROJECT_ROOT / "trascripts" / name for name in MEETING_DATES]
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

    unstable_detected = False
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
            min_consensus_share=args.min_consensus_share,
            max_allowed_count_delta=args.max_stability_delta,
            verifier=(
                extractor.verify_candidates
                if args.verify_near_consensus
                else None
            ),
            min_verifier_support_share=args.verifier_min_support_share,
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

        if args.eval_golden:
            golden_tasks = load_golden_tasks(transcript_path.name, args.golden_dir)
            if not golden_tasks:
                print(f"\nGolden eval: нет разметки для {transcript_path.name}")
            else:
                report = evaluate_dataframe_against_golden(
                    df,
                    golden_tasks,
                    transcript_path.name,
                )
                print("")
                for line in report.format_lines():
                    print(line)

        if args.save_sqlite:
            save_tasks(
                df,
                transcript_path.name,
                args.sqlite_path,
                meeting_date=meeting_date,
                stability_report=results,
            )
            print(f"\nSQLite: результат сохранен в {args.sqlite_path}")

        unstable_detected = unstable_detected or not results.stability_passed

    if args.fail_on_unstable and unstable_detected:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
