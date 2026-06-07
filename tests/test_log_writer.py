from __future__ import annotations

import asyncio
import sqlite3

import pytest

from aggregate_server.log_writer import LogRecord, LogWriter


def _make_record(**kwargs) -> LogRecord:
    defaults = dict(
        request_id="req-1",
        timestamp=1_700_000_000.0,
        inbound_model="gpt-4",
        canonical_model="llama3",
        backend_id="b1",
        status_code=200,
        queue_time_ms=10.0,
        backend_time_ms=500.0,
        total_time_ms=510.0,
        input_tokens=100,
        output_tokens=50,
        error_message=None,
    )
    return LogRecord(**{**defaults, **kwargs})


async def test_write_batch_creates_db_file(tmp_path):
    writer = LogWriter(db_dir=tmp_path)
    await writer._write_batch([_make_record(timestamp=1_700_000_000.0)])
    db_files = list(tmp_path.glob("*.db"))
    assert len(db_files) == 1


async def test_write_batch_record_readable(tmp_path):
    writer = LogWriter(db_dir=tmp_path)
    rec = _make_record(request_id="abc", input_tokens=42, output_tokens=7)
    await writer._write_batch([rec])
    db_path = list(tmp_path.glob("*.db"))[0]
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT request_id, input_tokens, output_tokens FROM requests").fetchall()
    conn.close()
    assert rows == [("abc", 42, 7)]


async def test_enqueue_drops_when_full(caplog):
    import logging
    writer = LogWriter(queue_maxsize=1)
    writer.enqueue(_make_record(request_id="first"))
    with caplog.at_level(logging.WARNING, logger="aggregate_server.log_writer"):
        writer.enqueue(_make_record(request_id="second"))
    assert writer._queue.qsize() == 1
    assert "dropping" in caplog.text.lower()


async def test_run_processes_enqueued_records(tmp_path):
    writer = LogWriter(db_dir=tmp_path)
    writer.enqueue(_make_record())
    task = asyncio.create_task(writer.run())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(list(tmp_path.glob("*.db"))) == 1


async def test_run_creates_db_dir(tmp_path):
    nested = tmp_path / "deep" / "logs"
    writer = LogWriter(db_dir=nested)
    writer.enqueue(_make_record())
    task = asyncio.create_task(writer.run())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert nested.exists()


async def test_write_batch_multiple_dates(tmp_path):
    writer = LogWriter(db_dir=tmp_path)
    rec1 = _make_record(timestamp=1_700_000_000.0)  # 2023-11-14 UTC
    rec2 = _make_record(timestamp=1_700_100_000.0)  # 2023-11-15 UTC (27h later)
    await writer._write_batch([rec1, rec2])
    assert len(list(tmp_path.glob("*.db"))) == 2
