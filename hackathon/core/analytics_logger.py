from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pandas as pd


EVENT_COLUMNS = [
    "timestamp",
    "event_type",
    "channel",
    "city_filter",
    "field",
    "moc",
    "title",
    "city",
    "results_count",
    "direct_count",
    "skill_count",
    "match_score",
]


def _analytics_dir(project_root: Path) -> Path:
    directory = project_root / "data" / "processed" / "analytics"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def analytics_artifact_paths(project_root: Path) -> tuple[Path, Path]:
    analytics_dir = _analytics_dir(project_root)
    return analytics_dir / "usage_analytics.db", analytics_dir / "usage_analytics_events.csv"


def initialize_analytics_logger(project_root: Path) -> None:
    sqlite_path, csv_path = analytics_artifact_paths(project_root)
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                channel TEXT,
                city_filter TEXT,
                field TEXT,
                moc TEXT,
                title TEXT,
                city TEXT,
                results_count INTEGER,
                direct_count INTEGER,
                skill_count INTEGER,
                match_score REAL
            )
            """
        )
        connection.commit()

    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
            writer.writeheader()


def log_analytics_event(project_root: Path, event: dict) -> None:
    initialize_analytics_logger(project_root)
    sqlite_path, csv_path = analytics_artifact_paths(project_root)

    normalized_event = {column: event.get(column, "") for column in EVENT_COLUMNS}
    for numeric_column in ["results_count", "direct_count", "skill_count"]:
        value = normalized_event[numeric_column]
        normalized_event[numeric_column] = int(value) if str(value).strip() else None

    score_value = normalized_event["match_score"]
    normalized_event["match_score"] = float(score_value) if str(score_value).strip() else None

    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            INSERT INTO analytics_events (
                timestamp,
                event_type,
                channel,
                city_filter,
                field,
                moc,
                title,
                city,
                results_count,
                direct_count,
                skill_count,
                match_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized_event["timestamp"],
                normalized_event["event_type"],
                normalized_event["channel"],
                normalized_event["city_filter"],
                normalized_event["field"],
                normalized_event["moc"],
                normalized_event["title"],
                normalized_event["city"],
                normalized_event["results_count"],
                normalized_event["direct_count"],
                normalized_event["skill_count"],
                normalized_event["match_score"],
            ),
        )
        connection.commit()

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
        writer.writerow({column: event.get(column, "") for column in EVENT_COLUMNS})


def load_analytics_events(project_root: Path) -> pd.DataFrame:
    sqlite_path, _ = analytics_artifact_paths(project_root)
    if not sqlite_path.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)

    with sqlite3.connect(sqlite_path) as connection:
        events = pd.read_sql_query(
            """
            SELECT
                timestamp,
                event_type,
                channel,
                city_filter,
                field,
                moc,
                title,
                city,
                results_count,
                direct_count,
                skill_count,
                match_score
            FROM analytics_events
            ORDER BY timestamp ASC, id ASC
            """,
            connection,
        )

    if events.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return events