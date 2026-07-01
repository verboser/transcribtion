from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def save_tasks(df: pd.DataFrame, transcript_name: str, db_path: Path) -> None:
    payload = df.copy()
    payload.insert(0, "Файл", transcript_name)
    with sqlite3.connect(db_path) as connection:
        payload.to_sql("meeting_tasks", connection, if_exists="append", index=False)
