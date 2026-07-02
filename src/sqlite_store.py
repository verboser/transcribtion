from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
from pathlib import Path

import pandas as pd

from src.stability import FinalTaskSupport, StabilityReport


def save_tasks(
    df: pd.DataFrame,
    transcript_name: str,
    db_path: Path,
    meeting_date: str = "",
    stability_report: StabilityReport | None = None,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _ensure_schema(connection)
        extraction_id = _insert_extraction(
            connection,
            transcript_name=transcript_name,
            meeting_date=meeting_date,
            stability_report=stability_report,
        )
        _insert_final_tasks(connection, extraction_id, transcript_name, df, stability_report)
        if stability_report is not None:
            _insert_run_tasks(connection, extraction_id, stability_report.run_dataframes)
            _insert_run_metrics(connection, extraction_id, stability_report.run_dataframes)
            _insert_report_lines(connection, extraction_id, stability_report.report_lines)


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS extraction_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_name TEXT NOT NULL,
            meeting_date TEXT NOT NULL,
            created_at TEXT NOT NULL,
            runs INTEGER NOT NULL,
            final_source TEXT NOT NULL,
            support_threshold INTEGER NOT NULL,
            min_consensus_share REAL NOT NULL,
            min_pairwise_consensus_jaccard REAL NOT NULL,
            max_count_delta REAL NOT NULL,
            stability_passed INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS meeting_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            extraction_id INTEGER NOT NULL,
            transcript_name TEXT NOT NULL,
            block TEXT NOT NULL,
            task TEXT NOT NULL,
            responsible TEXT NOT NULL,
            deadline TEXT NOT NULL,
            evidence TEXT NOT NULL,
            support_count INTEGER NOT NULL,
            support_ratio REAL NOT NULL,
            run_indices TEXT NOT NULL,
            verification_status TEXT NOT NULL DEFAULT 'raw_consensus',
            FOREIGN KEY(extraction_id) REFERENCES extraction_runs(id)
        );

        CREATE TABLE IF NOT EXISTS meeting_run_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            extraction_id INTEGER NOT NULL,
            run_idx INTEGER NOT NULL,
            block TEXT NOT NULL,
            task TEXT NOT NULL,
            responsible TEXT NOT NULL,
            deadline TEXT NOT NULL,
            evidence TEXT NOT NULL,
            FOREIGN KEY(extraction_id) REFERENCES extraction_runs(id)
        );

        CREATE TABLE IF NOT EXISTS meeting_run_metrics (
            extraction_id INTEGER NOT NULL,
            run_idx INTEGER NOT NULL,
            done_count INTEGER NOT NULL,
            failed_count INTEGER NOT NULL,
            new_count INTEGER NOT NULL,
            total_count INTEGER NOT NULL,
            PRIMARY KEY(extraction_id, run_idx),
            FOREIGN KEY(extraction_id) REFERENCES extraction_runs(id)
        );

        CREATE TABLE IF NOT EXISTS stability_report_lines (
            extraction_id INTEGER NOT NULL,
            line_idx INTEGER NOT NULL,
            line TEXT NOT NULL,
            PRIMARY KEY(extraction_id, line_idx),
            FOREIGN KEY(extraction_id) REFERENCES extraction_runs(id)
        );

        CREATE INDEX IF NOT EXISTS idx_meeting_tasks_extraction
            ON meeting_tasks(extraction_id);
        CREATE INDEX IF NOT EXISTS idx_meeting_run_tasks_extraction
            ON meeting_run_tasks(extraction_id, run_idx);
        """
    )
    _ensure_column(
        connection,
        table_name="meeting_tasks",
        column_name="verification_status",
        column_definition="TEXT NOT NULL DEFAULT 'raw_consensus'",
    )


def _insert_extraction(
    connection: sqlite3.Connection,
    transcript_name: str,
    meeting_date: str,
    stability_report: StabilityReport | None,
) -> int:
    runs = len(stability_report.run_dataframes) if stability_report is not None else 1
    connection.execute(
        """
        INSERT INTO extraction_runs (
            transcript_name,
            meeting_date,
            created_at,
            runs,
            final_source,
            support_threshold,
            min_consensus_share,
            min_pairwise_consensus_jaccard,
            max_count_delta,
            stability_passed
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            transcript_name,
            meeting_date,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            runs,
            stability_report.final_source if stability_report is not None else "single dataframe",
            stability_report.support_threshold if stability_report is not None else 1,
            stability_report.min_consensus_share if stability_report is not None else 1.0,
            (
                stability_report.min_pairwise_consensus_jaccard
                if stability_report is not None
                else 1.0
            ),
            stability_report.max_count_delta if stability_report is not None else 0.0,
            int(stability_report.stability_passed) if stability_report is not None else 1,
        ),
    )
    return int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])


def _insert_final_tasks(
    connection: sqlite3.Connection,
    extraction_id: int,
    transcript_name: str,
    df: pd.DataFrame,
    stability_report: StabilityReport | None,
) -> None:
    support_by_key = _support_by_key(stability_report.final_task_support) if stability_report else {}
    for _, row in df.iterrows():
        key = _task_key(row.to_dict())
        support = support_by_key.get(key)
        support_count = support.support_count if support is not None else 1
        support_ratio = support.support_ratio if support is not None else 1.0
        run_indices = _format_run_indices(support.run_indices if support is not None else (1,))
        connection.execute(
            """
            INSERT INTO meeting_tasks (
                extraction_id,
                transcript_name,
                block,
                task,
                responsible,
                deadline,
                evidence,
                support_count,
                support_ratio,
                run_indices,
                verification_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                extraction_id,
                transcript_name,
                row["Блок"],
                row["Задача"],
                row["Ответственный"],
                row["Срок"],
                row["Обоснование"],
                support_count,
                support_ratio,
                run_indices,
                support.verification_status if support is not None else "raw_consensus",
            ),
        )


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    columns = {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name in columns:
        return
    connection.execute(
        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
    )


def _insert_run_tasks(
    connection: sqlite3.Connection,
    extraction_id: int,
    run_dataframes: list[pd.DataFrame],
) -> None:
    for run_idx, df in enumerate(run_dataframes, start=1):
        for _, row in df.iterrows():
            connection.execute(
                """
                INSERT INTO meeting_run_tasks (
                    extraction_id,
                    run_idx,
                    block,
                    task,
                    responsible,
                    deadline,
                    evidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    extraction_id,
                    run_idx,
                    row["Блок"],
                    row["Задача"],
                    row["Ответственный"],
                    row["Срок"],
                    row["Обоснование"],
                ),
            )


def _insert_run_metrics(
    connection: sqlite3.Connection,
    extraction_id: int,
    run_dataframes: list[pd.DataFrame],
) -> None:
    for run_idx, df in enumerate(run_dataframes, start=1):
        counts = _count_dataframe(df)
        connection.execute(
            """
            INSERT INTO meeting_run_metrics (
                extraction_id,
                run_idx,
                done_count,
                failed_count,
                new_count,
                total_count
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                extraction_id,
                run_idx,
                counts["Выполненные"],
                counts["Невыполненные"],
                counts["Новые"],
                counts["Всего"],
            ),
        )


def _insert_report_lines(
    connection: sqlite3.Connection,
    extraction_id: int,
    report_lines: list[str],
) -> None:
    for line_idx, line in enumerate(report_lines, start=1):
        connection.execute(
            """
            INSERT INTO stability_report_lines (extraction_id, line_idx, line)
            VALUES (?, ?, ?)
            """,
            (extraction_id, line_idx, line),
        )


def _count_dataframe(df: pd.DataFrame) -> dict[str, int]:
    counts = df["Блок"].value_counts().to_dict() if not df.empty else {}
    return {
        "Выполненные": int(counts.get("Выполненные", 0)),
        "Невыполненные": int(counts.get("Невыполненные", 0)),
        "Новые": int(counts.get("Новые", 0)),
        "Всего": len(df),
    }


def _support_by_key(
    support_rows: list[FinalTaskSupport],
) -> dict[tuple[str, str, str, str, str], FinalTaskSupport]:
    return {
        (
            support.block,
            support.task,
            support.responsible,
            support.deadline,
            support.evidence,
        ): support
        for support in support_rows
    }


def _task_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row["Блок"],
        row["Задача"],
        row["Ответственный"],
        row["Срок"],
        row["Обоснование"],
    )


def _format_run_indices(run_indices: tuple[int, ...]) -> str:
    return ",".join(str(idx) for idx in run_indices)
