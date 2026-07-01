"""
tests/test_student_profile.py

Tests for the Student Profile Agent tool. These run against the real
`data/student_profile.json` file (no mocking of the data layer) plus
temporary files for the error-path cases, so a passing suite means the
tool actually works end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.student_profile import (
    DEFAULT_DATA_PATH,
    StudentProfile,
    StudentProfileError,
    get_student_profile,
    load_student_profile,
)

# ---------------------------------------------------------------------------
# load_student_profile
# ---------------------------------------------------------------------------


def test_default_data_file_exists() -> None:
    assert DEFAULT_DATA_PATH.exists(), f"Expected data file at {DEFAULT_DATA_PATH}"


def test_load_student_profile_reads_real_file() -> None:
    profile = load_student_profile()
    assert isinstance(profile, StudentProfile)
    assert profile.name
    assert profile.enrollment_number
    assert profile.department
    assert 1 <= profile.semester <= 12
    assert "@" in profile.email
    assert 0.0 <= profile.cgpa <= 10.0
    assert len(profile.subjects) > 0


def test_load_student_profile_missing_file_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(StudentProfileError, match="not found"):
        load_student_profile(missing_path)


def test_load_student_profile_invalid_json_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(StudentProfileError, match="Invalid JSON"):
        load_student_profile(bad_file)


def test_load_student_profile_out_of_range_cgpa_raises(tmp_path: Path) -> None:
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text(
        json.dumps(
            {
                "name": "Test Student",
                "enrollment_number": "ENR0001",
                "department": "Physics",
                "semester": 3,
                "email": "test@campusos.edu",
                "cgpa": 11.5,  # invalid: above 10.0
                "subjects": ["Mechanics"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(StudentProfileError, match="failed validation"):
        load_student_profile(invalid_file)


def test_load_student_profile_empty_subjects_raises(tmp_path: Path) -> None:
    invalid_file = tmp_path / "invalid_subjects.json"
    invalid_file.write_text(
        json.dumps(
            {
                "name": "Test Student",
                "enrollment_number": "ENR0002",
                "department": "Mathematics",
                "semester": 2,
                "email": "test2@campusos.edu",
                "cgpa": 7.0,
                "subjects": [],  # invalid: empty
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(StudentProfileError, match="failed validation"):
        load_student_profile(invalid_file)


def test_load_student_profile_directory_path_raises(tmp_path: Path) -> None:
    with pytest.raises(StudentProfileError, match="not a file"):
        load_student_profile(tmp_path)


def test_load_student_profile_caches_until_file_changes(tmp_path: Path) -> None:
    data_file = tmp_path / "cached.json"
    data_file.write_text(
        json.dumps(
            {
                "name": "Cache Student",
                "enrollment_number": "ENR0003",
                "department": "Chemistry",
                "semester": 1,
                "email": "cache@campusos.edu",
                "cgpa": 6.5,
                "subjects": ["Inorganic Chemistry"],
            }
        ),
        encoding="utf-8",
    )

    first = load_student_profile(data_file)
    assert first.department == "Chemistry"

    # Mutate the file in place and confirm the change is picked up (cache
    # is keyed on mtime, not assumed-immutable).
    data_file.write_text(
        json.dumps(
            {
                "name": "Cache Student",
                "enrollment_number": "ENR0003",
                "department": "Biology",
                "semester": 1,
                "email": "cache@campusos.edu",
                "cgpa": 6.5,
                "subjects": ["Genetics"],
            }
        ),
        encoding="utf-8",
    )
    second = load_student_profile(data_file)
    assert second.department == "Biology"


# ---------------------------------------------------------------------------
# get_student_profile (the ADK-registerable tool)
# ---------------------------------------------------------------------------


def test_get_student_profile_all() -> None:
    result = get_student_profile("all")
    assert result["status"] == "success"
    assert result["field"] == "all"
    assert isinstance(result["value"], dict)
    assert result["value"] == result["profile"]


def test_get_student_profile_default_is_all() -> None:
    assert get_student_profile() == get_student_profile("all")


@pytest.mark.parametrize(
    "alias,expected_field",
    [
        ("who_am_i", "all"),
        ("who", "all"),
        ("profile", "all"),
        ("name", "name"),
        ("full_name", "name"),
        ("department", "department"),
        ("dept", "department"),
        ("semester", "semester"),
        ("sem", "semester"),
        ("enrollment_number", "enrollment_number"),
        ("enrollment", "enrollment_number"),
        ("roll_number", "enrollment_number"),
        ("registration_number", "enrollment_number"),
        ("email", "email"),
        ("email_address", "email"),
        ("cgpa", "cgpa"),
        ("gpa", "cgpa"),
        ("subjects", "subjects"),
        ("courses", "subjects"),
        ("enrolled_subjects", "subjects"),
    ],
)
def test_get_student_profile_field_aliases_resolve(alias: str, expected_field: str) -> None:
    result = get_student_profile(alias)
    assert result["status"] == "success"
    assert result["field"] == expected_field


def test_get_student_profile_name_matches_loaded_profile() -> None:
    profile = load_student_profile()
    result = get_student_profile("name")
    assert result["value"] == profile.name


def test_get_student_profile_department_matches_loaded_profile() -> None:
    profile = load_student_profile()
    result = get_student_profile("department")
    assert result["value"] == profile.department


def test_get_student_profile_semester_matches_loaded_profile() -> None:
    profile = load_student_profile()
    result = get_student_profile("semester")
    assert result["value"] == profile.semester


def test_get_student_profile_enrollment_number_matches_loaded_profile() -> None:
    profile = load_student_profile()
    result = get_student_profile("enrollment_number")
    assert result["value"] == profile.enrollment_number


def test_get_student_profile_email_matches_loaded_profile() -> None:
    profile = load_student_profile()
    result = get_student_profile("email")
    assert result["value"] == profile.email


def test_get_student_profile_cgpa_matches_loaded_profile() -> None:
    profile = load_student_profile()
    result = get_student_profile("cgpa")
    assert result["value"] == profile.cgpa


def test_get_student_profile_subjects_matches_loaded_profile() -> None:
    profile = load_student_profile()
    result = get_student_profile("subjects")
    assert result["value"] == profile.subjects
    assert isinstance(result["value"], list)


def test_get_student_profile_field_is_case_and_space_insensitive() -> None:
    assert get_student_profile("  CGPA  ")["field"] == "cgpa"
    assert get_student_profile("Enrollment Number")["field"] == "enrollment_number"


def test_get_student_profile_unknown_field_returns_error_not_exception() -> None:
    result = get_student_profile("favorite_color")
    assert result["status"] == "error"
    assert result["field"] == "favorite_color"
    assert "Unknown field" in result["error_message"]


def test_get_student_profile_includes_full_profile_alongside_single_field() -> None:
    result = get_student_profile("cgpa")
    assert result["status"] == "success"
    assert "profile" in result
    assert set(result["profile"].keys()) == {
        "name",
        "enrollment_number",
        "department",
        "semester",
        "email",
        "cgpa",
        "subjects",
    }


# ---------------------------------------------------------------------------
# ADK FunctionTool compatibility and import compatibility
# ---------------------------------------------------------------------------


def test_get_student_profile_is_importable_exactly_as_required() -> None:
    """Confirms `from agents.student_profile import get_student_profile` works unchanged."""
    from agents.student_profile import get_student_profile as imported_get_student_profile

    assert callable(imported_get_student_profile)
    result = imported_get_student_profile("all")
    assert result["status"] == "success"


def test_get_student_profile_wraps_as_adk_function_tool() -> None:
    from google.adk.tools import FunctionTool

    from agents.student_profile import get_student_profile as tool_func

    tool = FunctionTool(func=tool_func)
    assert tool.name == "get_student_profile"