"""SQLite event store."""

import json
import os
import sqlite3
import threading
import time

from config import SNAPSHOTS_MAX, UNKNOWN_LABEL


class EventStore:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                names TEXT NOT NULL,
                snapshot BLOB
            )
        """)
        self._conn.commit()

    def add(self, names: list, jpeg_bytes: bytes) -> dict:
        ts = round(time.time(), 2)
        names_json = json.dumps(sorted(names), ensure_ascii=False)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (ts, names, snapshot) VALUES (?, ?, ?)",
                (ts, names_json, jpeg_bytes),
            )
            eid = cur.lastrowid
            self._conn.execute(
                "DELETE FROM events WHERE id NOT IN "
                "(SELECT id FROM events ORDER BY id DESC LIMIT ?)",
                (SNAPSHOTS_MAX,),
            )
            self._conn.commit()
        return {"ts": ts, "names": sorted(names), "img": eid}

    def recent(self, limit: int = SNAPSHOTS_MAX) -> list:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, names FROM events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"ts": r[1], "names": json.loads(r[2]), "img": r[0]}
            for r in reversed(rows)
        ]

    def get_snapshot(self, event_id: int) -> bytes | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT snapshot FROM events WHERE id = ?", (event_id,)
            ).fetchone()
        return row[0] if row else None

    def last_greeted_times(self, window_s: float) -> dict:
        cutoff = time.time() - window_s
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, names FROM events WHERE ts > ? ORDER BY ts",
                (cutoff,),
            ).fetchall()
        result = {}
        for ts, names_json in rows:
            for name in json.loads(names_json):
                if name != UNKNOWN_LABEL:
                    result[name] = ts
        return result

    def clear(self):
        with self._lock:
            self._conn.execute("DELETE FROM events")
            self._conn.commit()
            self._conn.execute("VACUUM")
            self._conn.commit()

    def close(self):
        self._conn.close()
