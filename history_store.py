# -*- coding: utf-8 -*-
"""
历史余额存储与导出模块。
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class HistoryStore:
    """负责持久化余额历史并提供导出/压缩能力"""

    def __init__(self, db_path: Path, archive_dir: Path):
        self.db_path = Path(db_path)
        self.archive_dir = Path(archive_dir)
        self._lock = threading.Lock()
        self._init_db()
        self.last_archive_date = self._load_last_archive_date()

    # ---- 基础设施 ----
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS balance_history (
                    model TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    total_equity REAL,
                    available_balance REAL,
                    unrealized_pnl REAL,
                    currency TEXT,
                    PRIMARY KEY (model, timestamp)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )

    def _load_last_archive_date(self):
        with self._get_conn() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = 'last_archive_date'").fetchone()
            if row and row["value"]:
                return datetime.strptime(row["value"], "%Y-%m-%d").date()
        return None

    def _update_last_archive_date(self, day):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('last_archive_date', ?)",
                (day.strftime("%Y-%m-%d"),),
            )

    # ---- 写入与读取 ----
    def append_balance(self, model: str, snapshot: Dict[str, float]):
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO balance_history(model, timestamp, total_equity, available_balance, unrealized_pnl, currency)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    model,
                    snapshot["timestamp"],
                    snapshot.get("total_equity"),
                    snapshot.get("available_balance"),
                    snapshot.get("unrealized_pnl"),
                    snapshot.get("currency", "USDT"),
                ),
            )

    def load_recent_balance(self, model: str, limit: int = 500) -> List[Dict[str, float]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, total_equity, available_balance, unrealized_pnl, currency
                FROM balance_history
                WHERE model = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (model, limit),
            ).fetchall()
        data = [
            {
                "timestamp": row["timestamp"],
                "total_equity": row["total_equity"],
                "available_balance": row["available_balance"],
                "unrealized_pnl": row["unrealized_pnl"],
                "currency": row["currency"],
            }
            for row in reversed(rows)
        ]
        return data

    def fetch_balance_range(self, model: str, start_ts: str, end_ts: str) -> List[Dict[str, float]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, total_equity, available_balance, unrealized_pnl, currency
                FROM balance_history
                WHERE model = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
                """,
                (model, start_ts, end_ts),
            ).fetchall()
        return [
            {
                "timestamp": row["timestamp"],
                "total_equity": row["total_equity"],
                "available_balance": row["available_balance"],
                "unrealized_pnl": row["unrealized_pnl"],
                "currency": row["currency"],
            }
            for row in rows
        ]

    # ---- 存档与导出 ----
    def compress_day(self, day):
        """将指定日期的数据导出为 Excel"""
        day_str = day.strftime("%Y-%m-%d")
        start = f"{day_str} 00:00:00"
        end = f"{day_str} 23:59:59"

        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT model, timestamp, total_equity, available_balance, unrealized_pnl, currency
                FROM balance_history
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY model, timestamp
                """,
                (start, end),
            ).fetchall()

        if not rows:
            return False

        df = pd.DataFrame([dict(row) for row in rows])
        output_path = self.archive_dir / f"balances-{day.strftime('%Y%m%d')}.xlsx"
        df.to_excel(output_path, index=False)
        self._update_last_archive_date(day)
        self.last_archive_date = day
        return True

    def compress_if_needed(self, current_dt: datetime):
        """每日零点后压缩前一日数据"""
        target_day = current_dt.date() - timedelta(days=1)
        if target_day <= datetime(1970, 1, 1).date():
            return
        if self.last_archive_date and target_day <= self.last_archive_date:
            return
        self.compress_day(target_day)

    def export_range_to_excel(self, start_date: str, end_date: str, output_path: Path, models: Optional[List[str]] = None):
        with self._get_conn() as conn:
            # 未指定 models 时，自动从库中取全部模型，避免依赖外部常量
            if not models:
                rows_models = conn.execute("SELECT DISTINCT model FROM balance_history").fetchall()
                models = [r["model"] for r in rows_models]

            if not models:
                raise ValueError("数据库中没有可导出的模型数据。")

            placeholder = ",".join("?" for _ in models)
            query = f"""
                SELECT model, timestamp, total_equity, available_balance, unrealized_pnl, currency
                FROM balance_history
                WHERE model IN ({placeholder}) AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """
            rows = conn.execute(query, (*models, start_date, end_date)).fetchall()

        if not rows:
            raise ValueError("选定时间范围内没有历史数据可导出。")

        df = pd.DataFrame([dict(row) for row in rows])
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False)

    def get_latest_before(self, model: str, timestamp: str):
        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT timestamp, total_equity, available_balance, unrealized_pnl
                FROM balance_history
                WHERE model = ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (model, timestamp),
            ).fetchone()
        return dict(row) if row else None
