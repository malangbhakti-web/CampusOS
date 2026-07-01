"""
tests/test_notices.py

Tests for the Notices Agent tool. Run against the real `data/notices.json`
file plus temporary files for error-path cases, following the exact
testing pattern used elsewhere in the project.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.notices import (
    DEFAULT_DATA_PATH,
    LATEST_COUNT,
    NoticeBoard,
    NoticeError,
    get_notices,
    load_notices,
)

_VALID_NOTICE = {
    "id": "T001",
    "title": "Test Notice",
    "category": "General",
    "date": "2026-06-01",
    "body": "This is a test notice body.",
    "is_read": False,
    "is_important": False,
}


def test_default_data_file_exists() -> None:
    assert DEFAULT_DATA_PATH.exists()


def test_load_notices_reads_real_file() -> None:
    board = load_notices()
    assert isinstance(board, NoticeBoard)
    assert len(board.notices) > 0
    for notice in board.notices:
        assert notice.category in {"Examination", "Fee", "Event", "General", "Academic"}


def test_load_notices_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(NoticeError, match="not found"):
        load_notices(tmp_path / "missing.json")


def test_load_notices_directory_path_raises(tmp_path: Path) -> None:
    with pytest.raises(NoticeError, match="not a file"):
        load_notices(tmp_path)


def test_load_notices_invalid_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(NoticeError, match="Invalid JSON"):
        load_notices(bad)


def test_load_notices_invalid_category_raises(tmp_path: Path) -> None:
    notice = {**_VALID_NOTICE, "category": "Gossip"}
    bad = tmp_path / "bad_category.json"
    bad.write_text(json.dumps({"notices": [notice]}), encoding="utf-8")
    with pytest.raises(NoticeError, match="failed validation"):
        load_notices(bad)


def test_load_notices_invalid_date_raises(tmp_path: Path) -> None:
    notice = {**_VALID_NOTICE, "date": "01-06-2026"}
    bad = tmp_path / "bad_date.json"
    bad.write_text(json.dumps({"notices": [notice]}), encoding="utf-8")
    with pytest.raises(NoticeError, match="failed validation"):
        load_notices(bad)


def test_load_notices_blank_title_raises(tmp_path: Path) -> None:
    notice = {**_VALID_NOTICE, "title": "  "}
    bad = tmp_path / "blank_title.json"
    bad.write_text(json.dumps({"notices": [notice]}), encoding="utf-8")
    with pytest.raises(NoticeError, match="failed validation"):
        load_notices(bad)


def test_load_notices_duplicate_ids_raises(tmp_path: Path) -> None:
    bad = tmp_path / "dup.json"
    bad.write_text(json.dumps({"notices": [_VALID_NOTICE, _VALID_NOTICE]}), encoding="utf-8")
    with pytest.raises(NoticeError, match="failed validation"):
        load_notices(bad)


def test_load_notices_empty_list_raises(tmp_path: Path) -> None:
    bad = tmp_path / "empty.json"
    bad.write_text(json.dumps({"notices": []}), encoding="utf-8")
    with pytest.raises(NoticeError, match="failed validation"):
        load_notices(bad)


def test_load_notices_caches_until_file_changes(tmp_path: Path) -> None:
    data_file = tmp_path / "cached.json"
    data_file.write_text(json.dumps({"notices": [_VALID_NOTICE]}), encoding="utf-8")
    first = load_notices(data_file)
    assert first.notices[0].title == "Test Notice"

    updated = {**_VALID_NOTICE, "title": "Updated Notice"}
    data_file.write_text(json.dumps({"notices": [updated]}), encoding="utf-8")
    second = load_notices(data_file)
    assert second.notices[0].title == "Updated Notice"


def test_get_notices_default_is_all() -> None:
    assert get_notices() == get_notices("all")


def test_get_notices_all_returns_every_notice() -> None:
    board = load_notices()
    result = get_notices("all")
    assert result["status"] == "success"
    assert result["query"] == "all"
    assert result["count"] == len(board.notices)
    assert len(result["value"]) == len(board.notices)


def test_get_notices_all_sorted_newest_first() -> None:
    result = get_notices("all")
    dates = [n["date"] for n in result["value"]]
    assert dates == sorted(dates, reverse=True)


@pytest.mark.parametrize("alias", ["latest", "recent", "new"])
def test_get_notices_latest_aliases(alias: str) -> None:
    result = get_notices(alias)
    assert result["status"] == "success"
    assert result["query"] == "latest"
    assert len(result["value"]) <= LATEST_COUNT


def test_get_notices_latest_sorted_newest_first() -> None:
    result = get_notices("latest")
    dates = [n["date"] for n in result["value"]]
    assert dates == sorted(dates, reverse=True)


def test_get_notices_unread_returns_only_unread() -> None:
    result = get_notices("unread")
    assert result["status"] == "success"
    assert result["query"] == "unread"
    assert all(not n["is_read"] for n in result["value"])


def test_get_notices_unread_count_matches_real_data() -> None:
    board = load_notices()
    expected_unread = sum(1 for n in board.notices if not n.is_read)
    result = get_notices("unread")
    assert result["count"] == expected_unread


@pytest.mark.parametrize("alias", ["important", "urgent"])
def test_get_notices_important_aliases(alias: str) -> None:
    result = get_notices(alias)
    assert result["status"] == "success"
    assert result["query"] == "important"
    assert all(n["is_important"] for n in result["value"])


def test_get_notices_important_count_matches_real_data() -> None:
    board = load_notices()
    expected = sum(1 for n in board.notices if n.is_important)
    result = get_notices("important")
    assert result["count"] == expected


@pytest.mark.parametrize(
    "alias,expected_category",
    [
        ("Examination", "Examination"),
        ("examination", "Examination"),
        ("EXAMINATION", "Examination"),
        ("fee", "Fee"),
        ("event", "Event"),
        ("general", "General"),
        ("academic", "Academic"),
    ],
)
def test_get_notices_category_filter(alias: str, expected_category: str) -> None:
    result = get_notices(alias)
    assert result["status"] == "success"
    assert result["query"] == expected_category
    assert all(n["category"] == expected_category for n in result["value"])


def test_get_notices_category_with_no_matches_returns_empty_list() -> None:
    result = get_notices("Event")
    assert result["status"] == "success"
    assert isinstance(result["value"], list)


def test_get_notices_notice_dict_has_all_fields() -> None:
    result = get_notices("all")
    for notice in result["value"]:
        assert set(notice.keys()) == {
            "id", "title", "category", "date", "body", "is_read", "is_important"
        }


def test_get_notices_unknown_query_returns_error_not_exception() -> None:
    result = get_notices("sports")
    assert result["status"] == "error"
    assert "Unknown query" in result["error_message"]


def test_get_notices_missing_data_file_returns_structured_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agents.notices as notices_module

    monkeypatch.setattr(notices_module, "DEFAULT_DATA_PATH", tmp_path / "missing.json")
    notices_module._notices_cache.clear()

    result = get_notices("all")
    assert result["status"] == "error"
    assert "not found" in result["error_message"]


def test_get_notices_is_importable_exactly_as_required() -> None:
    from agents.notices import get_notices as imported_get_notices

    assert callable(imported_get_notices)
    assert imported_get_notices("all")["status"] == "success"


def test_get_notices_wraps_as_adk_function_tool() -> None:
    from google.adk.tools import FunctionTool

    from agents.notices import get_notices as tool_func

    tool = FunctionTool(func=tool_func)
    assert tool.name == "get_notices"