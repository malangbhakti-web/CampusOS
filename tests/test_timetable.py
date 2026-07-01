"""
tests/test_timetable.py

Tests for the Timetable Agent tool. These run against the real
`data/timetable.json` file (no mocking of the data layer) plus temporary
files for error-path cases. Time-dependent queries ("today", "tomorrow",
"current_class", "next_class") are exercised with an explicit
`reference_datetime` so results are deterministic — the resolution logic
itself is real production code, only the clock input is fixed.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from agents.timetable import (
    DEFAULT_DATA_PATH,
    Timetable,
    TimetableError,
    get_timetable,
    load_timetable,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WEEKDAY_INDEX = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


def _next_or_same_weekday(start: date, weekday_name: str) -> date:
    """Returns the next date on/after `start` that falls on `weekday_name`."""
    target_index = _WEEKDAY_INDEX[weekday_name]
    days_ahead = (target_index - start.weekday()) % 7
    return start + timedelta(days=days_ahead)


def _at(weekday_name: str, hour: int, minute: int = 0) -> datetime:
    """Builds a real datetime guaranteed to fall on the given weekday."""
    day = _next_or_same_weekday(date.today(), weekday_name)
    return datetime(day.year, day.month, day.day, hour, minute)


_VALID_ENTRY_TEMPLATE = {
    "day": "Monday",
    "subject": "Test Subject",
    "type": "Lecture",
    "start_time": "09:00",
    "end_time": "10:00",
    "room": "T-100",
    "instructor": "Dr. Test",
}


# ---------------------------------------------------------------------------
# load_timetable
# ---------------------------------------------------------------------------


def test_default_data_file_exists() -> None:
    assert DEFAULT_DATA_PATH.exists(), f"Expected data file at {DEFAULT_DATA_PATH}"


def test_load_timetable_reads_real_file() -> None:
    timetable = load_timetable()
    assert isinstance(timetable, Timetable)
    assert len(timetable.entries) > 0
    for entry in timetable.entries:
        assert entry.day in _WEEKDAY_INDEX
        assert entry.type in {"Lecture", "Lab", "Tutorial", "Practical"}


def test_load_timetable_missing_file_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(TimetableError, match="not found"):
        load_timetable(missing_path)


def test_load_timetable_directory_path_raises(tmp_path: Path) -> None:
    with pytest.raises(TimetableError, match="not a file"):
        load_timetable(tmp_path)


def test_load_timetable_invalid_json_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(TimetableError, match="Invalid JSON"):
        load_timetable(bad_file)


def test_load_timetable_invalid_day_raises(tmp_path: Path) -> None:
    entry = {**_VALID_ENTRY_TEMPLATE, "day": "Funday"}
    bad_file = tmp_path / "bad_day.json"
    bad_file.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")
    with pytest.raises(TimetableError, match="failed validation"):
        load_timetable(bad_file)


def test_load_timetable_invalid_type_raises(tmp_path: Path) -> None:
    entry = {**_VALID_ENTRY_TEMPLATE, "type": "Seminar"}
    bad_file = tmp_path / "bad_type.json"
    bad_file.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")
    with pytest.raises(TimetableError, match="failed validation"):
        load_timetable(bad_file)


def test_load_timetable_invalid_time_format_raises(tmp_path: Path) -> None:
    entry = {**_VALID_ENTRY_TEMPLATE, "start_time": "9 AM"}
    bad_file = tmp_path / "bad_time.json"
    bad_file.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")
    with pytest.raises(TimetableError, match="failed validation"):
        load_timetable(bad_file)


def test_load_timetable_end_before_start_raises(tmp_path: Path) -> None:
    entry = {**_VALID_ENTRY_TEMPLATE, "start_time": "11:00", "end_time": "10:00"}
    bad_file = tmp_path / "bad_order.json"
    bad_file.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")
    with pytest.raises(TimetableError, match="failed validation"):
        load_timetable(bad_file)


def test_load_timetable_empty_entries_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "empty.json"
    bad_file.write_text(json.dumps({"entries": []}), encoding="utf-8")
    with pytest.raises(TimetableError, match="failed validation"):
        load_timetable(bad_file)


def test_load_timetable_caches_until_file_changes(tmp_path: Path) -> None:
    data_file = tmp_path / "cached.json"
    data_file.write_text(json.dumps({"entries": [_VALID_ENTRY_TEMPLATE]}), encoding="utf-8")

    first = load_timetable(data_file)
    assert first.entries[0].subject == "Test Subject"

    updated_entry = {**_VALID_ENTRY_TEMPLATE, "subject": "Updated Subject"}
    data_file.write_text(json.dumps({"entries": [updated_entry]}), encoding="utf-8")

    second = load_timetable(data_file)
    assert second.entries[0].subject == "Updated Subject"


# ---------------------------------------------------------------------------
# get_timetable (the ADK-registerable tool)
# ---------------------------------------------------------------------------


def test_get_timetable_default_is_all() -> None:
    assert get_timetable() == get_timetable("all")


def test_get_timetable_all_returns_every_entry() -> None:
    timetable = load_timetable()
    result = get_timetable("all")
    assert result["status"] == "success"
    assert result["resolved_day"] is None
    assert len(result["value"]) == len(timetable.entries)


def test_get_timetable_all_is_sorted_by_day_then_time() -> None:
    result = get_timetable("all")
    day_order_seen = [_WEEKDAY_INDEX[entry["day"]] for entry in result["value"]]
    assert day_order_seen == sorted(day_order_seen)


@pytest.mark.parametrize(
    "alias,expected_day",
    [
        ("Monday", "Monday"),
        ("monday", "Monday"),
        ("MONDAY", "Monday"),
        ("Tuesday", "Tuesday"),
        ("wednesday", "Wednesday"),
        ("Thursday", "Thursday"),
        ("friday", "Friday"),
        ("Saturday", "Saturday"),
        ("sunday", "Sunday"),
    ],
)
def test_get_timetable_weekday_queries_are_case_insensitive(alias: str, expected_day: str) -> None:
    result = get_timetable(alias)
    assert result["status"] == "success"
    assert result["query"] == expected_day
    assert result["resolved_day"] == expected_day
    for entry in result["value"]:
        assert entry["day"] == expected_day


def test_get_timetable_weekday_entries_sorted_by_start_time() -> None:
    result = get_timetable("Monday")
    start_times = [entry["start_time"] for entry in result["value"]]
    assert start_times == sorted(start_times)


def test_get_timetable_sunday_has_no_classes_in_sample_data() -> None:
    result = get_timetable("Sunday")
    assert result["status"] == "success"
    assert result["value"] == []


def test_get_timetable_unknown_query_returns_error_not_exception() -> None:
    result = get_timetable("someday")
    assert result["status"] == "error"
    assert result["query"] == "someday"
    assert "Unknown query" in result["error_message"]


def test_get_timetable_today_resolves_correct_weekday() -> None:
    reference = _at("Wednesday", 7, 0)  # before any class starts
    result = get_timetable("today", reference_datetime=reference)
    assert result["status"] == "success"
    assert result["query"] == "today"
    assert result["resolved_day"] == "Wednesday"
    assert result["value"] == get_timetable("Wednesday")["value"]


def test_get_timetable_tomorrow_resolves_correct_weekday() -> None:
    reference = _at("Wednesday", 7, 0)
    result = get_timetable("tomorrow", reference_datetime=reference)
    assert result["status"] == "success"
    assert result["query"] == "tomorrow"
    assert result["resolved_day"] == "Thursday"
    assert result["value"] == get_timetable("Thursday")["value"]


def test_get_timetable_current_class_returns_entry_when_in_session() -> None:
    # Monday 09:00-10:00 is "Data Structures & Algorithms" Lecture in the
    # sample data; 09:30 falls inside that window.
    reference = _at("Monday", 9, 30)
    result = get_timetable("current_class", reference_datetime=reference)
    assert result["status"] == "success"
    assert result["resolved_day"] == "Monday"
    assert result["value"] is not None
    assert result["value"]["subject"] == "Data Structures & Algorithms"
    assert result["value"]["start_time"] == "09:00"
    assert result["value"]["end_time"] == "10:00"


def test_get_timetable_current_class_returns_none_when_no_class_in_session() -> None:
    reference = _at("Sunday", 23, 0)  # Sunday has no entries at all
    result = get_timetable("current_class", reference_datetime=reference)
    assert result["status"] == "success"
    assert result["value"] is None


def test_get_timetable_next_class_within_same_day() -> None:
    reference = _at("Monday", 8, 0)  # before Monday's first class (09:00)
    result = get_timetable("next_class", reference_datetime=reference)
    assert result["status"] == "success"
    assert result["value"] is not None
    assert result["resolved_day"] == "Monday"
    assert result["value"]["start_time"] == "09:00"
    assert result["value"]["subject"] == "Data Structures & Algorithms"


def test_get_timetable_next_class_crosses_to_next_day_with_entries() -> None:
    # After Monday's last class (ends 13:00) -> next class is Tuesday's
    # first entry (09:00 Computer Networks).
    reference = _at("Monday", 23, 0)
    result = get_timetable("next_class", reference_datetime=reference)
    assert result["status"] == "success"
    assert result["resolved_day"] == "Tuesday"
    assert result["value"]["subject"] == "Computer Networks"
    assert result["value"]["start_time"] == "09:00"


def test_get_timetable_next_class_wraps_full_week() -> None:
    # After Saturday's last class, Sunday has no entries, so next_class
    # must wrap around to next Monday's first entry.
    reference = _at("Saturday", 23, 0)
    result = get_timetable("next_class", reference_datetime=reference)
    assert result["status"] == "success"
    assert result["resolved_day"] == "Monday"
    assert result["value"]["subject"] == "Data Structures & Algorithms"
    assert result["value"]["start_time"] == "09:00"


def test_get_timetable_includes_validated_type_field() -> None:
    result = get_timetable("all")
    for entry in result["value"]:
        assert entry["type"] in {"Lecture", "Lab", "Tutorial", "Practical"}


def test_get_timetable_query_is_case_and_space_insensitive() -> None:
    assert get_timetable("  MONDAY  ")["query"] == "Monday"
    assert get_timetable("Current Class")["status"] == "success"
    assert get_timetable("Current Class")["query"] == "current_class"