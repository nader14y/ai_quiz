"""جدول امتیازها در SQLite — همان رابط ScoreBoard (is_empty / record / top)،
اما داده در یک جدول واقعی است، نه یک فایل JSON.
"""

import sqlite3
from contextlib import closing
from pathlib import Path


class SqliteScoreBoard:
    def __init__(self, path) -> None:
        self._path = Path(path)
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS players ("
                " name TEXT PRIMARY KEY,"
                " wins INTEGER NOT NULL DEFAULT 0,"
                " points INTEGER NOT NULL DEFAULT 0)"
            )

    def _connect(self):
        # closing: بستن اتصال؛ with db: ثبت (commit) یا برگرداندن (rollback) تراکنش
        return _Connection(self._path)

    def is_empty(self) -> bool:
        with self._connect() as db:
            return db.execute("SELECT COUNT(*) FROM players").fetchone()[0] == 0

    def record(self, winner, player1, score1, player2, score2) -> None:
        try:
            with self._connect() as db:
                for name, points in [(player1, score1), (player2, score2)]:
                    db.execute(
                        "INSERT INTO players (name, points) VALUES (?, ?)"
                        " ON CONFLICT(name) DO UPDATE SET points = points + excluded.points",
                        (name, points),
                    )
                if winner is not None:
                    db.execute("UPDATE players SET wins = wins + 1 WHERE name = ?", (winner,))
        except sqlite3.Error as error:
            # web_game فقط OSError را «ذخیره ناموفق» می‌داند
            raise OSError(f"ذخیره در دیتابیس ناموفق بود: {error}") from error

    def top(self, n: int = 5) -> list[tuple[str, dict[str, int]]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT name, wins, points FROM players"
                " ORDER BY wins DESC, points DESC, name LIMIT ?",
                (n,),
            ).fetchall()
        return [(name, {"wins": wins, "points": points}) for name, wins, points in rows]


class _Connection:
    """اتصال SQLite که با خروج از with، هم commit/rollback می‌کند و هم بسته می‌شود."""

    def __init__(self, path):
        self._db = sqlite3.connect(path)

    def __enter__(self):
        return self._db

    def __exit__(self, exc_type, exc, tb):
        with closing(self._db):
            if exc_type is None:
                self._db.commit()
            else:
                self._db.rollback()
