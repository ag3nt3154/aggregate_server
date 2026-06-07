from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS requests (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id       TEXT    NOT NULL,
    timestamp        REAL    NOT NULL,
    inbound_model    TEXT    NOT NULL,
    canonical_model  TEXT    NOT NULL,
    backend_id       TEXT,
    status_code      INTEGER NOT NULL,
    queue_time_ms    REAL    NOT NULL,
    backend_time_ms  REAL,
    total_time_ms    REAL    NOT NULL,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    error_message    TEXT
)
"""

_INSERT = """
INSERT INTO requests (
    request_id, timestamp, inbound_model, canonical_model,
    backend_id, status_code, queue_time_ms, backend_time_ms,
    total_time_ms, input_tokens, output_tokens, error_message
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@dataclass
class LogRecord:
    request_id: str
    timestamp: float
    inbound_model: str
    canonical_model: str
    backend_id: str | None
    status_code: int
    queue_time_ms: float
    backend_time_ms: float | None
    total_time_ms: float
    input_tokens: int | None
    output_tokens: int | None
    error_message: str | None

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.request_id, self.timestamp, self.inbound_model, self.canonical_model,
            self.backend_id, self.status_code, self.queue_time_ms, self.backend_time_ms,
            self.total_time_ms, self.input_tokens, self.output_tokens, self.error_message,
        )


class LogWriter:
    def __init__(
        self,
        db_dir: str | Path = "./data/logs",
        queue_maxsize: int = 1000,
        batch_size: int = 50,
    ) -> None:
        self._db_dir = Path(db_dir)
        self._batch_size = batch_size
        self._queue: asyncio.Queue[LogRecord] = asyncio.Queue(maxsize=queue_maxsize)

    def enqueue(self, record: LogRecord) -> None:
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            logger.warning(
                "Log queue full — dropping record for request %s", record.request_id
            )

    async def run(self) -> None:
        self._db_dir.mkdir(parents=True, exist_ok=True)
        while True:
            batch: list[LogRecord] = []
            record = await self._queue.get()
            batch.append(record)
            while len(batch) < self._batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            await self._write_batch(batch)

    def _date_str(self, timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()

    async def _write_batch(self, batch: list[LogRecord]) -> None:
        by_date: dict[str, list[LogRecord]] = {}
        for rec in batch:
            key = self._date_str(rec.timestamp)
            by_date.setdefault(key, []).append(rec)
        for date_str, records in by_date.items():
            db_path = self._db_dir / f"{date_str}.db"
            async with aiosqlite.connect(db_path) as db:
                await db.execute(_CREATE_TABLE)
                await db.executemany(_INSERT, [r.as_tuple() for r in records])
                await db.commit()
