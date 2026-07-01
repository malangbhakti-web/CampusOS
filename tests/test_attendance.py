"""
tests/test_attendance.py

Tests for the Attendance Agent tool. These run against the real
`data/attendance.json` file (no mocking of the data layer) plus temporary
files for error-path cases, following the exact testing pattern used in
`tests/test_student_profile.py` and `tests/test_timetable.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.attendance import (
    DEFAULT_DATA_PATH,
    LOW_ATTENDANCE_THRESHOLD,
    AttendanceError,
    AttendanceRecord,
    get_attendance,
    load_attendance,
)

_VALID_SUBJECT_TEMPLATE = {
    "subject": "Test Subject",
    "total_classes": 20,
    "attended_classes": 18,
}


# ---------------------------------------------------------------------------
# load_attendance
# ---------------------------------------------------------------------------


def test_default_data_file_exists() -> None:
    assert DEFAULT_DATA_PATH.exists(), f"Expected data file at {DEFAULT_DATA_PATH}"


def test_load_attendance_reads_real_file() -> None:
    record = load_attendance()
    assert isinstance(record, AttendanceRecord)
    assert len(record.subjects) > 0
    for entry in record.subjects:
        assert entry.total_classes >= 0
        assert entry.attended_classes >= 0
        assert entry.attended_classes <= entry.total_classes


def test_load_attendance_missing_file_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(AttendanceError, match="not found"):
        load_attendance(missing_path)


def test_load_attendance_directory_path_raises(tmp_path: Path) -> None:
    with pytest.raises(AttendanceError, match="not a file"):
        load_attendance(tmp_path)


def test_load_attendance_invalid_json_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(AttendanceError, match="Invalid JSON"):
        load_attendance(bad_file)


def test_load_attendance_negative_total_raises(tmp_path: Path) -> None:
    entry = {**_VALID_SUBJECT_TEMPLATE, "total_classes": -5}
    bad_file = tmp_path / "bad_total.json"
    bad_file.write_text(json.dumps({"subjects": [entry]}), encoding="utf-8")
    with pytest.raises(AttendanceError, match="failed validation"):
        load_attendance(bad_file)


def test_load_attendance_negative_attended_raises(tmp_path: Path) -> None:
    entry = {**_VALID_SUBJECT_TEMPLATE, "attended_classes": -1}
    bad_file = tmp_path / "bad_attended.json"
    bad_file.write_text(json.dumps({"subjects": [entry]}), encoding="utf-8")
    with pytest.raises(AttendanceError, match="failed validation"):
        load_attendance(bad_file)


def test_load_attendance_attended_exceeds_total_raises(tmp_path: Path) -> None:
    entry = {**_VALID_SUBJECT_TEMPLATE, "total_classes": 10, "attended_classes": 15}
    bad_file = tmp_path / "bad_exceeds.json"
    bad_file.write_text(json.dumps({"subjects": [entry]}), encoding="utf-8")
    with pytest.raises(AttendanceError, match="failed validation"):
        load_attendance(bad_file)


def test_load_attendance_blank_subject_raises(tmp_path: Path) -> None:
    entry = {**_VALID_SUBJECT_TEMPLATE, "subject": "   "}
    bad_file = tmp_path / "blank_subject.json"
    bad_file.write_text(json.dumps({"subjects": [entry]}), encoding="utf-8")
    with pytest.raises(AttendanceError, match="failed validation"):
        load_attendance(bad_file)


def test_load_attendance_duplicate_subjects_raises(tmp_path: Path) -> None:
    entries = [_VALID_SUBJECT_TEMPLATE, {**_VALID_SUBJECT_TEMPLATE}]
    bad_file = tmp_path / "duplicate.json"
    bad_file.write_text(json.dumps({"subjects": entries}), encoding="utf-8")
    with pytest.raises(AttendanceError, match="failed validation"):
        load_attendance(bad_file)


def test_load_attendance_empty_subjects_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "empty.json"
    bad_file.write_text(json.dumps({"subjects": []}), encoding="utf-8")
    with pytest.raises(AttendanceError, match="failed validation"):
        load_attendance(bad_file)


def test_load_attendance_caches_until_file_changes(tmp_path: Path) -> None:
    data_file = tmp_path / "cached.json"
    data_file.write_text(json.dumps({"subjects": [_VALID_SUBJECT_TEMPLATE]}), encoding="utf-8")

    first = load_attendance(data_file)
    assert first.subjects[0].attended_classes == 18

    updated_entry = {**_VALID_SUBJECT_TEMPLATE, "attended_classes": 5}
    data_file.write_text(json.dumps({"subjects": [updated_entry]}), encoding="utf-8")

    second = load_attendance(data_file)
    assert second.subjects[0].attended_classes == 5


# ---------------------------------------------------------------------------
# Derived fields on SubjectAttendance
# ---------------------------------------------------------------------------


def test_subject_attendance_derived_fields_are_correct() -> None:
    record = load_attendance()
    entry = record.subjects[0]
    assert entry.absent_classes == entry.total_classes - entry.attended_classes
    expected_percentage = round((entry.attended_classes / entry.total_classes) * 100, 2)
    assert entry.percentage == expected_percentage
    assert entry.is_low_attendance == (entry.percentage < LOW_ATTENDANCE_THRESHOLD)


def test_subject_attendance_zero_total_classes_has_zero_percentage() -> None:
    from agents.attendance import SubjectAttendance

    entry = SubjectAttendance(subject="Empty Subject", total_classes=0, attended_classes=0)
    assert entry.percentage == 0.0
    assert entry.absent_classes == 0


# ---------------------------------------------------------------------------
# get_attendance (the ADK-registerable tool)
# ---------------------------------------------------------------------------


def test_get_attendance_default_is_all() -> None:
    assert get_attendance() == get_attendance(student_id="default", subject="all")


def test_get_attendance_all_returns_every_subject() -> None:
    record = load_attendance()
    result = get_attendance(subject="all")
    assert result["status"] == "success"
    assert result["query"] == "all"
    assert len(result["value"]) == len(record.subjects)
    for entry_dict in result["value"]:
        assert set(entry_dict.keys()) == {
            "subject",
            "total_classes",
            "attended_classes",
            "absent_classes",
            "percentage",
            "is_low_attendance",
        }


@pytest.mark.parametrize("alias", ["overall", "summary", "total", "Overall", "  TOTAL  "])
def test_get_attendance_overall_aliases_resolve(alias: str) -> None:
    record = load_attendance()
    result = get_attendance(subject=alias)
    assert result["status"] == "success"
    assert result["query"] == "overall"

    expected_total = sum(e.total_classes for e in record.subjects)
    expected_attended = sum(e.attended_classes for e in record.subjects)
    expected_percentage = round((expected_attended / expected_total) * 100, 2)

    assert result["value"]["total_classes"] == expected_total
    assert result["value"]["attended_classes"] == expected_attended
    assert result["value"]["absent_classes"] == expected_total - expected_attended
    assert result["value"]["percentage"] == expected_percentage


@pytest.mark.parametrize("alias", ["low_attendance", "low", "shortage", "defaulters", "Defaulter"])
def test_get_attendance_low_attendance_aliases_resolve(alias: str) -> None:
    record = load_attendance()
    result = get_attendance(subject=alias)
    assert result["status"] == "success"
    assert result["query"] == "low_attendance"

    expected_low_subjects = {e.subject for e in record.subjects if e.is_low_attendance}
    returned_subjects = {entry["subject"] for entry in result["value"]}
    assert returned_subjects == expected_low_subjects
    for entry_dict in result["value"]:
        assert entry_dict["is_low_attendance"] is True
        assert entry_dict["percentage"] < LOW_ATTENDANCE_THRESHOLD


def test_get_attendance_low_attendance_detects_known_low_subject() -> None:
    # Computer Networks: 25/36 attended ~= 69.44%, below the 75% threshold
    # in the real data/attendance.json file.
    result = get_attendance(subject="low_attendance")
    subjects = {entry["subject"] for entry in result["value"]}
    assert "Computer Networks" in subjects


def test_get_attendance_specific_subject_exact_match() -> None:
    record = load_attendance()
    target = record.subjects[0]
    result = get_attendance(subject=target.subject)
    assert result["status"] == "success"
    assert result["query"] == target.subject
    assert result["value"]["subject"] == target.subject
    assert result["value"]["total_classes"] == target.total_classes
    assert result["value"]["attended_classes"] == target.attended_classes


def test_get_attendance_specific_subject_case_insensitive() -> None:
    result = get_attendance(subject="operating systems")
    assert result["status"] == "success"
    assert result["query"] == "Operating Systems"


def test_get_attendance_specific_subject_partial_match() -> None:
    result = get_attendance(subject="Database")
    assert result["status"] == "success"
    assert result["query"] == "Database Management Systems"


def test_get_attendance_unknown_subject_returns_error_not_exception() -> None:
    result = get_attendance(subject="Quantum Mechanics")
    assert result["status"] == "error"
    assert result["query"] == "Quantum Mechanics"
    assert "Unknown query" in result["error_message"]


def test_get_attendance_student_id_is_echoed_back() -> None:
    result = get_attendance(student_id="STU12345", subject="overall")
    assert result["student_id"] == "STU12345"


def test_get_attendance_missing_data_file_returns_structured_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agents.attendance as attendance_module

    missing_path = tmp_path / "missing_attendance.json"
    monkeypatch.setattr(attendance_module, "DEFAULT_DATA_PATH", missing_path)
    attendance_module._attendance_cache.clear()

    result = get_attendance(subject="all")
    assert result["status"] == "error"
    assert "error_message" in result
    assert "not found" in result["error_message"]


# ---------------------------------------------------------------------------
# ADK FunctionTool compatibility and import compatibility
# ---------------------------------------------------------------------------


def test_get_attendance_is_importable_exactly_as_required() -> None:
    """Confirms `from agents.attendance import get_attendance` works unchanged."""
    from agents.attendance import get_attendance as imported_get_attendance

    assert callable(imported_get_attendance)
    result = imported_get_attendance(subject="overall")
    assert result["status"] == "success"


def test_get_attendance_wraps_as_adk_function_tool() -> None:
    from google.adk.tools import FunctionTool

    from agents.attendance import get_attendance as tool_func

    tool = FunctionTool(func=tool_func)
    assert tool.name == "get_attendance"