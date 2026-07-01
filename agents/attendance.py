"""
agents/attendance.py

Attendance Agent tool for CampusOS.

Exposes `get_attendance`, a plain Python callable that reads per-subject
class attendance from `data/attendance.json`, validates it, and returns a
structured (JSON-serializable) result. It follows the exact pattern
established by `agents/student_profile.py` and `agents/timetable.py`, and
is designed to be registered on an ADK `LlmAgent` later as a
`FunctionTool`, e.g.:

    from google.adk.tools import FunctionTool
    from agents.attendance import get_attendance

    attendance_tool = FunctionTool(func=get_attendance)
    root_agent = LlmAgent(..., tools=[attendance_tool])

No natural-language responses are hardcoded here — this module only ever
returns structured data; turning it into a sentence is the LLM's job once
this tool is wired into an agent's `tools` list.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger("campusos.agents.attendance")

#: Default location of the attendance data file, resolved relative to this
#: module's location so it works regardless of the process's cwd.
DEFAULT_DATA_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "attendance.json"

#: Attendance percentage threshold (inclusive lower bound for "OK"); below
#: this a subject is flagged as low attendance. Matches common academic
#: eligibility cutoffs (e.g. 75%).
LOW_ATTENDANCE_THRESHOLD: float = 75.0

#: Query keyword -> canonical resolution mode.
_QUERY_ALIASES: Dict[str, str] = {
    "all": "all",
    "overall": "overall",
    "summary": "overall",
    "total": "overall",
    "low_attendance": "low_attendance",
    "low": "low_attendance",
    "shortage": "low_attendance",
    "defaulter": "low_attendance",
    "defaulters": "low_attendance",
}

#: Module-level cache of (mtime, validated attendance record), keyed by
#: resolved path string. Mirrors the caching strategy used in
#: `agents/student_profile.py` and `agents/timetable.py`.
_attendance_cache: Dict[str, Tuple[float, "AttendanceRecord"]] = {}


class AttendanceError(RuntimeError):
    """Raised when the attendance data file is missing, unreadable, or invalid."""


class SubjectAttendance(BaseModel):
    """Validated attendance counts for a single subject."""

    subject: str
    total_classes: int
    attended_classes: int

    @field_validator("subject")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("subject must not be blank")
        return value.strip()

    @field_validator("total_classes")
    @classmethod
    def _validate_total_classes(cls, value: int) -> int:
        if value < 0:
            raise ValueError("total_classes must not be negative")
        return value

    @field_validator("attended_classes")
    @classmethod
    def _validate_attended_classes(cls, value: int) -> int:
        if value < 0:
            raise ValueError("attended_classes must not be negative")
        return value

    @model_validator(mode="after")
    def _validate_attended_not_exceeding_total(self) -> "SubjectAttendance":
        if self.attended_classes > self.total_classes:
            raise ValueError(
                f"attended_classes ({self.attended_classes}) cannot exceed "
                f"total_classes ({self.total_classes}) for subject "
                f"'{self.subject}'"
            )
        return self

    @property
    def absent_classes(self) -> int:
        """Number of classes missed for this subject."""
        return self.total_classes - self.attended_classes

    @property
    def percentage(self) -> float:
        """Attendance percentage for this subject, rounded to 2 decimals."""
        if self.total_classes == 0:
            return 0.0
        return round((self.attended_classes / self.total_classes) * 100, 2)

    @property
    def is_low_attendance(self) -> bool:
        """Whether this subject's attendance falls below the eligibility threshold."""
        return self.percentage < LOW_ATTENDANCE_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        """Returns a structured, JSON-serializable view including derived fields."""
        return {
            "subject": self.subject,
            "total_classes": self.total_classes,
            "attended_classes": self.attended_classes,
            "absent_classes": self.absent_classes,
            "percentage": self.percentage,
            "is_low_attendance": self.is_low_attendance,
        }


class AttendanceRecord(BaseModel):
    """Validated representation of a student's full attendance record."""

    subjects: List[SubjectAttendance]

    @field_validator("subjects")
    @classmethod
    def _not_empty(cls, value: List[SubjectAttendance]) -> List[SubjectAttendance]:
        if not value:
            raise ValueError("attendance record must contain at least one subject")
        return value

    @model_validator(mode="after")
    def _validate_unique_subjects(self) -> "AttendanceRecord":
        names = [entry.subject.lower() for entry in self.subjects]
        if len(names) != len(set(names)):
            raise ValueError("attendance record contains duplicate subject entries")
        return self


def load_attendance(path: Optional[Path] = None) -> AttendanceRecord:
    """
    Loads and validates the attendance record from a JSON file.

    Args:
        path: Optional override for the data file location. Defaults to
            `DEFAULT_DATA_PATH` (`data/attendance.json` relative to the
            project root) when omitted.

    Returns:
        A validated `AttendanceRecord` instance.

    Raises:
        AttendanceError: If the file does not exist, contains invalid
            JSON, or contains data that fails schema validation.
    """
    resolved_path = Path(path) if path is not None else DEFAULT_DATA_PATH

    if not resolved_path.exists():
        raise AttendanceError(f"Attendance data file not found at: {resolved_path}")
    if not resolved_path.is_file():
        raise AttendanceError(f"Attendance data path is not a file: {resolved_path}")

    file_mtime = resolved_path.stat().st_mtime
    cache_key = str(resolved_path)
    cached = _attendance_cache.get(cache_key)
    if cached is not None and cached[0] == file_mtime:
        return cached[1]

    try:
        raw_text = resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AttendanceError(f"Could not read {resolved_path}: {exc}") from exc

    try:
        raw_data: Any = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise AttendanceError(f"Invalid JSON in {resolved_path}: {exc}") from exc

    try:
        record = AttendanceRecord.model_validate(raw_data)
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError, re-raised as AttendanceError
        raise AttendanceError(
            f"Attendance data in {resolved_path} failed validation: {exc}"
        ) from exc

    _attendance_cache[cache_key] = (file_mtime, record)
    return record


def _overall_summary(record: AttendanceRecord) -> Dict[str, Any]:
    """Builds the aggregated, all-subjects-combined attendance summary."""
    total_classes = sum(entry.total_classes for entry in record.subjects)
    attended_classes = sum(entry.attended_classes for entry in record.subjects)
    absent_classes = total_classes - attended_classes
    percentage = round((attended_classes / total_classes) * 100, 2) if total_classes else 0.0

    return {
        "total_classes": total_classes,
        "attended_classes": attended_classes,
        "absent_classes": absent_classes,
        "percentage": percentage,
        "is_low_attendance": percentage < LOW_ATTENDANCE_THRESHOLD,
    }


def _find_subject(record: AttendanceRecord, subject_query: str) -> Optional[SubjectAttendance]:
    """Case-insensitive, substring-tolerant lookup of a subject by name."""
    normalized_query = subject_query.strip().lower()

    for entry in record.subjects:
        if entry.subject.lower() == normalized_query:
            return entry

    matches = [entry for entry in record.subjects if normalized_query in entry.subject.lower()]
    if len(matches) == 1:
        return matches[0]
    return None


def get_attendance(student_id: str = "default", subject: str = "all") -> Dict[str, Any]:
    """
    Retrieves attendance information for the current student.

    This is the ADK-compatible tool entry point for the Attendance Agent.
    Wrap it with `google.adk.tools.FunctionTool` and add it to an
    `LlmAgent`'s `tools` list so the LLM can call it to answer questions
    such as "what's my overall attendance", "what's my attendance in
    Operating Systems", "how many classes have I attended", "how many
    classes have I missed", or "do I have low attendance anywhere".

    Args:
        student_id: Identifier for the student whose attendance is being
            requested. The current single-student data source ignores
            this value (always resolves to the one profile in
            `data/attendance.json`), but the parameter is accepted now so
            multi-student lookups can be added later without changing
            this function's signature. Defaults to "default".
        subject: Which slice of the attendance record to retrieve.
            Accepts:
            - "all" (default) — every subject's attendance, individually.
            - "overall" / "summary" / "total" — aggregated totals across
              all subjects (total classes, attended, absent, percentage).
            - "low_attendance" / "low" / "shortage" / "defaulters" —
              only subjects below the eligibility threshold
              (`LOW_ATTENDANCE_THRESHOLD`, 75%).
            - A specific subject name (case-insensitive, exact or
              unambiguous partial match), e.g. "Operating Systems".

    Returns:
        A JSON-serializable dict.

        On success for "all"::

            {
                "status": "success",
                "student_id": "<student_id>",
                "query": "all",
                "value": [<subject attendance dict>, ...],
            }

        On success for "overall"::

            {
                "status": "success",
                "student_id": "<student_id>",
                "query": "overall",
                "value": {
                    "total_classes": int,
                    "attended_classes": int,
                    "absent_classes": int,
                    "percentage": float,
                    "is_low_attendance": bool,
                },
            }

        On success for "low_attendance"::

            {
                "status": "success",
                "student_id": "<student_id>",
                "query": "low_attendance",
                "value": [<subject attendance dict>, ...],  # may be empty
            }

        On success for a specific subject::

            {
                "status": "success",
                "student_id": "<student_id>",
                "query": "<matched subject name>",
                "value": <subject attendance dict>,
            }

        Each `<subject attendance dict>` has the shape::

            {
                "subject": str,
                "total_classes": int,
                "attended_classes": int,
                "absent_classes": int,
                "percentage": float,
                "is_low_attendance": bool,
            }

        On failure (missing/invalid data file, or an unrecognized
        subject)::

            {
                "status": "error",
                "student_id": "<student_id>",
                "query": "<subject as originally given>",
                "error_message": "<human-readable explanation>",
            }
    """
    try:
        record = load_attendance()
    except AttendanceError as exc:
        logger.error("Failed to load attendance: %s", exc)
        return {
            "status": "error",
            "student_id": student_id,
            "query": subject,
            "error_message": str(exc),
        }

    normalized_query = (subject or "all").strip().lower().replace(" ", "_")
    canonical_mode = _QUERY_ALIASES.get(normalized_query)

    if canonical_mode == "all":
        return {
            "status": "success",
            "student_id": student_id,
            "query": "all",
            "value": [entry.to_dict() for entry in record.subjects],
        }

    if canonical_mode == "overall":
        return {
            "status": "success",
            "student_id": student_id,
            "query": "overall",
            "value": _overall_summary(record),
        }

    if canonical_mode == "low_attendance":
        low_entries = [entry for entry in record.subjects if entry.is_low_attendance]
        return {
            "status": "success",
            "student_id": student_id,
            "query": "low_attendance",
            "value": [entry.to_dict() for entry in low_entries],
        }

    matched_entry = _find_subject(record, subject)
    if matched_entry is not None:
        return {
            "status": "success",
            "student_id": student_id,
            "query": matched_entry.subject,
            "value": matched_entry.to_dict(),
        }

    valid_subjects = [entry.subject for entry in record.subjects]
    valid_queries = sorted(set(_QUERY_ALIASES.values())) + valid_subjects
    logger.warning("Unknown attendance query requested: %r", subject)
    return {
        "status": "error",
        "student_id": student_id,
        "query": subject,
        "error_message": f"Unknown query '{subject}'. Valid queries: {valid_queries}",
    }