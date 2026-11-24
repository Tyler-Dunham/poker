"""SQLite persistence layer for player profiles, parsed hands, and trained models."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .parser import ParsedHand, parsed_hand_to_dict

DEFAULT_DB_PATH = Path("/Users/Will/dev/poker/data/poker_mirror.db")


class PokerStorage:
    """SQLite storage for Poker Mirror profiles and hand histories."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    archetype_json TEXT NOT NULL DEFAULT '{}',
                    stats_json TEXT NOT NULL DEFAULT '{}',
                    clone_model_json TEXT NOT NULL DEFAULT '{}',
                    exploiter_json TEXT NOT NULL DEFAULT '{}',
                    exploit_report_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS hands (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id INTEGER NOT NULL,
                    hand_id TEXT NOT NULL,
                    source_format TEXT NOT NULL DEFAULT 'unknown',
                    hero_name TEXT NOT NULL DEFAULT 'Hero',
                    net_bb REAL NOT NULL DEFAULT 0.0,
                    hand_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_hands_profile_id ON hands(profile_id);
                """
            )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "name": row["name"],
            "archetype": json.loads(row["archetype_json"]),
            "stats": json.loads(row["stats_json"]),
            "clone_model": json.loads(row["clone_model_json"]),
            "exploiter": json.loads(row["exploiter_json"]),
            "exploit_report": json.loads(row["exploit_report_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_profile(
        self,
        name: str,
        archetype: dict | None = None,
        stats: dict | None = None,
        clone_model: dict | None = None,
    ) -> dict:
        """Create a new player profile and return it."""
        now = self._now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO profiles (
                    name, archetype_json, stats_json, clone_model_json,
                    exploiter_json, exploit_report_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, '{}', '{}', ?, ?)
                """,
                (
                    name,
                    json.dumps(archetype or {}),
                    json.dumps(stats or {}),
                    json.dumps(clone_model or {}),
                    now,
                    now,
                ),
            )
            profile_id = cur.lastrowid
        return self.get_profile(int(profile_id))  # type: ignore[return-value]

    def list_profiles(self) -> list[dict]:
        """List all profiles ordered by most recently updated."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.*, COUNT(h.id) AS hands_stored
                FROM profiles p
                LEFT JOIN hands h ON h.profile_id = p.id
                GROUP BY p.id
                ORDER BY p.updated_at DESC, p.id DESC
                """
            ).fetchall()
        out = []
        for r in rows:
            prof = self._row_to_profile(r)
            prof["hands_stored"] = r["hands_stored"]
            out.append(prof)
        return out

    def get_profile(self, profile_id: int) -> dict | None:
        """Fetch a single profile by ID, or None if not found."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
            if row is None:
                return None
            cnt = conn.execute("SELECT COUNT(*) FROM hands WHERE profile_id = ?", (profile_id,)).fetchone()[0]
        prof = self._row_to_profile(row)
        prof["hands_stored"] = cnt
        return prof

    def add_hands(self, profile_id: int, hands: list[ParsedHand | dict]) -> int:
        """Store parsed hands for a profile and return number of rows inserted."""
        if not hands:
            return 0
        now = self._now_iso()
        rows_to_insert = []
        for h in hands:
            hd = parsed_hand_to_dict(h) if isinstance(h, ParsedHand) else dict(h)
            rows_to_insert.append((
                profile_id,
                str(hd.get("hand_id", "")),
                str(hd.get("source_format", "unknown")),
                str(hd.get("hero_name", "Hero")),
                float(hd.get("net_bb", 0.0)),
                json.dumps(hd),
                now,
            ))
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO hands (profile_id, hand_id, source_format, hero_name, net_bb, hand_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
            conn.execute("UPDATE profiles SET updated_at = ? WHERE id = ?", (now, profile_id))
        return len(rows_to_insert)

    def get_profile_hands(self, profile_id: int) -> list[dict]:
        """Retrieve all stored hand dicts for a profile."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT hand_json FROM hands WHERE profile_id = ? ORDER BY id ASC",
                (profile_id,),
            ).fetchall()
        return [json.loads(r["hand_json"]) for r in rows]

    def save_training_result(
        self,
        profile_id: int,
        stats: dict | None = None,
        clone_model: dict | None = None,
        exploiter: dict | None = None,
        exploit_report: dict | None = None,
        archetype: dict | None = None,
    ) -> dict | None:
        """Update stored stats, clone model, RL exploiter, and exploit report for a profile."""
        existing = self.get_profile(profile_id)
        if existing is None:
            return None
        now = self._now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE profiles
                SET stats_json = ?,
                    clone_model_json = ?,
                    exploiter_json = ?,
                    exploit_report_json = ?,
                    archetype_json = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(stats if stats is not None else existing["stats"]),
                    json.dumps(clone_model if clone_model is not None else existing["clone_model"]),
                    json.dumps(exploiter if exploiter is not None else existing["exploiter"]),
                    json.dumps(exploit_report if exploit_report is not None else existing["exploit_report"]),
                    json.dumps(archetype if archetype is not None else existing["archetype"]),
                    now,
                    profile_id,
                ),
            )
        return self.get_profile(profile_id)

    def delete_profile(self, profile_id: int) -> bool:
        """Delete a profile and its cascaded hands. Returns True if a row was deleted."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
            return cur.rowcount > 0
