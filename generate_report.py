from __future__ import annotations

import argparse
import html
import sqlite3
from pathlib import Path

import pandas as pd

from src.schemas import DATAFRAME_COLUMNS


DEFAULT_TRANSCRIPTS = ("transcript.txt", "transcript2.txt", "transcript3.txt")
REQUIRED_TABLES = {
    "extraction_runs",
    "meeting_tasks",
    "meeting_run_metrics",
    "stability_report_lines",
}


def generate_report_html(
    db_path: Path,
    transcript_names: list[str] | None = None,
) -> str:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Файл {db_path} не найден. Сначала запустите main.py --all --save-sqlite."
        )
    if db_path.stat().st_size == 0:
        raise ValueError(
            f"Файл {db_path} пустой. Пересоздайте его командой "
            "main.py --all --save-sqlite."
        )

    with sqlite3.connect(db_path) as connection:
        _validate_schema(connection)
        selected_transcripts = transcript_names or _transcripts_from_database(connection)
        sections = [
            _render_transcript_section(connection, transcript_name)
            for transcript_name in selected_transcripts
        ]

    return _render_page(db_path, sections)


def write_report(
    db_path: Path,
    output_path: Path,
    transcript_names: list[str] | None = None,
) -> None:
    html_content = generate_report_html(db_path, transcript_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")


def _validate_schema(connection: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    missing = sorted(REQUIRED_TABLES - tables)
    if missing:
        raise ValueError(
            "SQLite-файл не похож на базу этого проекта. "
            f"Нет таблиц: {', '.join(missing)}."
        )


def _transcripts_from_database(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT DISTINCT transcript_name
        FROM extraction_runs
        ORDER BY
            CASE transcript_name
                WHEN 'transcript.txt' THEN 0
                WHEN 'transcript2.txt' THEN 1
                WHEN 'transcript3.txt' THEN 2
                ELSE 99
            END,
            transcript_name
        """
    ).fetchall()
    return [str(row[0]) for row in rows] or list(DEFAULT_TRANSCRIPTS)


def _render_transcript_section(
    connection: sqlite3.Connection,
    transcript_name: str,
) -> str:
    extraction = pd.read_sql_query(
        """
        SELECT
            id,
            meeting_date,
            created_at,
            runs,
            final_source,
            support_threshold,
            min_consensus_share,
            min_pairwise_consensus_jaccard,
            max_count_delta,
            stability_passed
        FROM extraction_runs
        WHERE transcript_name = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        connection,
        params=(transcript_name,),
    )
    title = html.escape(transcript_name)
    if extraction.empty:
        return (
            '<section class="transcript-section">'
            f"<h1>{title}</h1>"
            "<p class=\"muted\">Для этого транскрипта нет сохранённого запуска.</p>"
            "</section>"
        )

    extraction_id = int(extraction.iloc[0]["id"])
    meta_df = _format_extraction_meta(extraction)
    metrics_df = pd.read_sql_query(
        """
        SELECT
            run_idx AS "Прогон",
            done_count AS "Выполненные",
            failed_count AS "Невыполненные",
            new_count AS "Новые",
            total_count AS "Всего"
        FROM meeting_run_metrics
        WHERE extraction_id = ?
        ORDER BY run_idx
        """,
        connection,
        params=(extraction_id,),
    )
    final_df = pd.read_sql_query(
        """
        SELECT
            block AS "Блок",
            task AS "Задача",
            responsible AS "Ответственный",
            deadline AS "Срок",
            evidence AS "Обоснование"
        FROM meeting_tasks
        WHERE extraction_id = ?
        ORDER BY
            CASE block
                WHEN 'Выполненные' THEN 0
                WHEN 'Невыполненные' THEN 1
                WHEN 'Новые' THEN 2
                ELSE 99
            END,
            responsible,
            deadline,
            task
        """,
        connection,
        params=(extraction_id,),
    )
    final_df = final_df.reindex(columns=DATAFRAME_COLUMNS)
    support_df = _read_support_dataframe(connection, extraction_id)
    report_lines_df = pd.read_sql_query(
        """
        SELECT line AS "Строка отчёта"
        FROM stability_report_lines
        WHERE extraction_id = ?
        ORDER BY line_idx
        """,
        connection,
        params=(extraction_id,),
    )

    return f"""
    <section class="transcript-section">
        <h1>{title}</h1>

        <h2>Сводка запуска</h2>
        {_render_dataframe(meta_df)}

        <h2>Метрики пяти прогонов</h2>
        <p class="muted">Это counts до финального consensus. По ним видно, насколько плавает модель между прогонами.</p>
        {_render_dataframe(metrics_df)}

        <h2>Итоговый DataFrame</h2>
        <p class="muted">Ниже ровно пять колонок из ТЗ. Audit-поля вынесены отдельно и не смешиваются с итоговой таблицей.</p>
        {_render_dataframe(final_df)}

        <h2>Поддержка финальных строк</h2>
        <p class="muted">Эта таблица не является итоговым DataFrame. Она нужна только для аудита 5-run consensus.</p>
        {_render_dataframe(support_df)}

        <h2>Отчёт стабильности</h2>
        {_render_dataframe(report_lines_df)}
    </section>
    """


def _format_extraction_meta(extraction: pd.DataFrame) -> pd.DataFrame:
    row = extraction.iloc[0]
    return pd.DataFrame(
        [
            {
                "Дата встречи": row["meeting_date"],
                "Создано": row["created_at"],
                "Прогонов": int(row["runs"]),
                "Порог support": int(row["support_threshold"]),
                "Consensus": _format_percent(float(row["min_consensus_share"])),
                "Min Jaccard": _format_percent(
                    float(row["min_pairwise_consensus_jaccard"])
                ),
                "Max delta": _format_percent(float(row["max_count_delta"])),
                "90% пройден": "Да" if int(row["stability_passed"]) else "Нет",
                "Источник финала": row["final_source"],
            }
        ]
    )


def _read_support_dataframe(
    connection: sqlite3.Connection,
    extraction_id: int,
) -> pd.DataFrame:
    columns = _table_columns(connection, "meeting_tasks")
    optional_columns = [
        "support_count",
        "support_ratio",
        "run_indices",
        "verification_status",
    ]
    if not all(column in columns for column in optional_columns):
        return pd.DataFrame(
            [{"Комментарий": "В этой версии SQLite нет support-метаданных."}]
        )

    support_df = pd.read_sql_query(
        """
        SELECT
            block AS "Блок",
            task AS "Задача",
            support_count AS "Поддержка",
            support_ratio AS "Доля",
            run_indices AS "Прогоны",
            verification_status AS "Статус"
        FROM meeting_tasks
        WHERE extraction_id = ?
        ORDER BY
            CASE block
                WHEN 'Выполненные' THEN 0
                WHEN 'Невыполненные' THEN 1
                WHEN 'Новые' THEN 2
                ELSE 99
            END,
            task
        """,
        connection,
        params=(extraction_id,),
    )
    if not support_df.empty:
        support_df["Доля"] = support_df["Доля"].map(lambda value: _format_percent(value))
    return support_df


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})")}


def _render_dataframe(df: pd.DataFrame) -> str:
    if df.empty:
        return '<p class="muted">Нет данных.</p>'
    return df.to_html(index=False, classes="table", border=0, escape=True)


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_page(db_path: Path, sections: list[str]) -> str:
    db_name = html.escape(str(db_path))
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Отчёт по извлечению задач</title>
    <style>
        body {{
            margin: 0;
            padding: 32px;
            font-family: Arial, sans-serif;
            color: #202124;
            background: #f7f7f7;
        }}
        .page {{
            max-width: 1440px;
            margin: 0 auto;
        }}
        .summary,
        .transcript-section {{
            background: #fff;
            border: 1px solid #ddd;
            padding: 24px;
            margin-bottom: 24px;
        }}
        h1 {{
            margin: 0 0 16px;
            font-size: 28px;
        }}
        h2 {{
            margin: 24px 0 10px;
            font-size: 20px;
        }}
        .muted {{
            color: #5f6368;
            margin: 0 0 12px;
        }}
        .table {{
            border-collapse: collapse;
            width: 100%;
            margin: 12px 0 20px;
            font-size: 14px;
        }}
        .table th,
        .table td {{
            border: 1px solid #ddd;
            padding: 8px;
            vertical-align: top;
            text-align: left;
        }}
        .table th {{
            background: #f1f3f4;
            font-weight: 600;
        }}
        .table td {{
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <main class="page">
        <section class="summary">
            <h1>Отчёт по извлечению задач</h1>
            <p class="muted">Источник данных: {db_name}</p>
            <p class="muted">Итоговый DataFrame в каждом блоке показан отдельно от audit-метаданных.</p>
        </section>
        {"".join(sections)}
    </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Генерация HTML-отчёта из SQLite")
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=Path("tasks.sqlite"),
        help="Путь к SQLite-файлу, созданному main.py --save-sqlite.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("presentation_report.html"),
        help="Путь к выходному HTML-файлу.",
    )
    parser.add_argument(
        "--transcript",
        action="append",
        dest="transcripts",
        help="Имя транскрипта для отчёта. Можно передать несколько раз.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        write_report(args.sqlite_path, args.output, args.transcripts)
    except (FileNotFoundError, sqlite3.Error, ValueError) as exc:
        raise SystemExit(f"Ошибка генерации отчёта: {exc}") from exc

    print(f"Файл {args.output} сгенерирован из {args.sqlite_path}.")


if __name__ == "__main__":
    main()
