import sqlite3

import pytest

from scoreboard_db import SqliteScoreBoard


def test_empty_board(tmp_path):
    assert SqliteScoreBoard(tmp_path / "q.db").is_empty()


def test_record_and_rank_by_wins_then_points(tmp_path):
    board = SqliteScoreBoard(tmp_path / "q.db")
    board.record("سارا", "علی", 10, "سارا", 20)
    board.record("علی", "علی", 50, "سارا", 5)
    board.record("سارا", "علی", 1, "سارا", 2)

    assert board.top(5) == [
        ("سارا", {"wins": 2, "points": 27}),
        ("علی", {"wins": 1, "points": 61}),
    ]


def test_draw_gives_no_win(tmp_path):
    board = SqliteScoreBoard(tmp_path / "q.db")
    board.record(None, "علی", 5, "سارا", 5)
    assert all(stat["wins"] == 0 for _, stat in board.top())


def test_survives_restart(tmp_path):
    path = tmp_path / "q.db"
    SqliteScoreBoard(path).record("سارا", "علی", 1, "سارا", 2)
    assert SqliteScoreBoard(path).top(1)[0][0] == "سارا"


def test_top_limits_rows(tmp_path):
    board = SqliteScoreBoard(tmp_path / "q.db")
    for i in range(7):
        board.record(f"p{i}", f"p{i}", i, "x", 0)
    assert len(board.top(5)) == 5


def test_save_failure_is_oserror(tmp_path):
    board = SqliteScoreBoard(tmp_path / "q.db")
    with sqlite3.connect(tmp_path / "q.db") as db:
        db.execute("DROP TABLE players")
    with pytest.raises(OSError):
        board.record("a", "a", 1, "b", 2)
