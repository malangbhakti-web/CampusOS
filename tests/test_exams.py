"""
tests/test_exams.py

Tests for the Exams Agent tool. Run against the real `data/exams.json`
file plus temporary files for error-path cases, following the exact
testing pattern used elsewhere in the project.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from agents.exams import (
    DEFAULT_DATA_PATH,
    ExamError,
    ExamRecord,
    get_exams,
    load_exams,
)

_SCHEDULED_TEMPLATE = {
    "subject": "Test Subject",
    "exam_type": "Midterm",
    "date": "2026-09-01",
    "max_marks": 50,
    "obtained_marks": None,
    "status": "Scheduled",
}

_COMPLETED_TEMPLATE = {
    "subject": "Test Subject",
    "exam_type": "Midterm",
    "date": "2026-01-01",
    "max_marks": 50,
    "obtained_marks": 40,
    "status": "Completed",
}


def test_default_data_file_exists() -> None:
    assert DEFAULT_DATA_PATH.exists()


def test_load_exams_reads_real_file() -> None:
    record = load_exams()
    assert isinstance(record, ExamRecord)
    assert len(record.exams) > 0


def test_load_exams_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ExamError, match="not found"):
        load_exams(tmp_path / "missing.json")


def test_load_exams_directory_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ExamError, match="not a file"):
        load_exams(tmp_path)


def test_load_exams_invalid_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ExamError, match="Invalid JSON"):
        load_exams(bad)


def test_load_exams_invalid_date_raises(tmp_path: Path) -> None:
    entry = {**_SCHEDULED_TEMPLATE, "date": "01-09-2026"}
    bad = tmp_path / "bad_date.json"
    bad.write_text(json.dumps({"exams": [entry]}), encoding="utf-8")
    with pytest.raises(ExamError, match="failed validation"):
        load_exams(bad)


def test_load_exams_invalid_status_raises(tmp_path: Path) -> None:
    entry = {**_SCHEDULED_TEMPLATE, "status": "InProgress"}
    bad = tmp_path / "bad_status.json"
    bad.write_text(json.dumps({"exams": [entry]}), encoding="utf-8")
    with pytest.raises(ExamError, match="failed validation"):
        load_exams(bad)


def test_load_exams_obtained_exceeds_max_raises(tmp_path: Path) -> None:
    entry = {**_COMPLETED_TEMPLATE, "obtained_marks": 999}
    bad = tmp_path / "bad_marks.json"
    bad.write_text(json.dumps({"exams": [entry]}), encoding="utf-8")
    with pytest.raises(ExamError, match="failed validation"):
        load_exams(bad)


def test_load_exams_completed_without_marks_raises(tmp_path: Path) -> None:
    entry = {**_COMPLETED_TEMPLATE, "obtained_marks": None}
    bad = tmp_path / "bad_completed.json"
    bad.write_text(json.dumps({"exams": [entry]}), encoding="utf-8")
    with pytest.raises(ExamError, match="failed validation"):
        load_exams(bad)


def test_load_exams_scheduled_with_marks_raises(tmp_path: Path) -> None:
    entry = {**_SCHEDULED_TEMPLATE, "obtained_marks": 10}
    bad = tmp_path / "bad_scheduled.json"
    bad.write_text(json.dumps({"exams": [entry]}), encoding="utf-8")
    with pytest.raises(ExamError, match="failed validation"):
        load_exams(bad)


def test_load_exams_empty_list_raises(tmp_path: Path) -> None:
    bad = tmp_path / "empty.json"
    bad.write_text(json.dumps({"exams": []}), encoding="utf-8")
    with pytest.raises(ExamError, match="failed validation"):
        load_exams(bad)


def test_load_exams_caches_until_file_changes(tmp_path: Path) -> None:
    data_file = tmp_path / "cached.json"
    data_file.write_text(json.dumps({"exams": [_SCHEDULED_TEMPLATE]}), encoding="utf-8")
    first = load_exams(data_file)
    assert first.exams[0].exam_type == "Midterm"

    updated = {**_SCHEDULED_TEMPLATE, "exam_type": "Final"}
    data_file.write_text(json.dumps({"exams": [updated]}), encoding="utf-8")
    second = load_exams(data_file)
    assert second.exams[0].exam_type == "Final"


def test_exam_entry_percentage_for_completed() -> None:
    record = load_exams()
    completed = [e for e in record.exams if e.status == "Completed"][0]
    assert completed.percentage == round((completed.obtained_marks / completed.max_marks) * 100, 2)


def test_exam_entry_percentage_none_for_scheduled() -> None:
    record = load_exams()
    scheduled = [e for e in record.exams if e.status == "Scheduled"][0]
    assert scheduled.percentage is None


def test_get_exams_default_is_all() -> None:
    assert get_exams() == get_exams("all")


def test_get_exams_all_returns_every_entry() -> None:
    record = load_exams()
    result = get_exams("all")
    assert result["status"] == "success"
    assert len(result["value"]) == len(record.exams)


@pytest.mark.parametrize("alias", ["upcoming", "scheduled", "pending"])
def test_get_exams_upcoming_aliases(alias: str) -> None:
    result = get_exams(alias)
    assert result["status"] == "success"
    assert result["query"] == "upcoming"
    assert all(e["status"] == "Scheduled" for e in result["value"])


@pytest.mark.parametrize("alias", ["completed", "past", "results"])
def test_get_exams_completed_aliases(alias: str) -> None:
    result = get_exams(alias)
    assert result["status"] == "success"
    assert result["query"] == "completed"
    assert all(e["status"] == "Completed" for e in result["value"])
    assert all(e["obtained_marks"] is not None for e in result["value"])


def test_get_exams_upcoming_sorted_by_date() -> None:
    result = get_exams("upcoming")
    dates = [e["date"] for e in result["value"]]
    assert dates == sorted(dates)


@pytest.mark.parametrize("alias", ["next_exam", "next"])
def test_get_exams_next_exam_aliases(alias: str) -> None:
    result = get_exams(alias, reference_date=date(2026, 1, 1))
    assert result["status"] == "success"
    assert result["query"] == "next_exam"
    assert result["value"] is not None
    assert result["value"]["status"] == "Scheduled"


def test_get_exams_next_exam_none_when_all_past() -> None:
    result = get_exams("next_exam", reference_date=date(2099, 1, 1))
    assert result["status"] == "success"
    assert result["value"] is None


def test_get_exams_specific_subject_returns_all_its_entries() -> None:
    result = get_exams("Data Structures & Algorithms")
    assert result["status"] == "success"
    assert len(result["value"]) >= 1
    assert all(e["subject"] == "Data Structures & Algorithms" for e in result["value"])


def test_get_exams_specific_subject_case_insensitive() -> None:
    result = get_exams("operating systems")
    assert result["status"] == "success"
    assert result["query"] == "Operating Systems"


def test_get_exams_specific_subject_partial_match() -> None:
    result = get_exams("Software")
    assert result["status"] == "success"
    assert result["query"] == "Software Engineering"


def test_get_exams_unknown_query_returns_error_not_exception() -> None:
    result = get_exams("Astrophysics")
    assert result["status"] == "error"
    assert "Unknown query" in result["error_message"]


def test_get_exams_missing_data_file_returns_structured_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agents.exams as exams_module

    monkeypatch.setattr(exams_module, "DEFAULT_DATA_PATH", tmp_path / "missing.json")
    exams_module._exams_cache.clear()

    result = get_exams("all")
    assert result["status"] == "error"
    assert "not found" in result["error_message"]


def test_get_exams_is_importable_exactly_as_required() -> None:
    from agents.exams import get_exams as imported_get_exams

    assert callable(imported_get_exams)
    assert imported_get_exams("all")["status"] == "success"


def test_get_exams_wraps_as_adk_function_tool() -> None:
    from google.adk.tools import FunctionTool

    from agents.exams import get_exams as tool_func

    tool = FunctionTool(func=tool_func)
    assert tool.name == "get_exams"