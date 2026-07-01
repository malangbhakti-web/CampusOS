"""
agents/exams.py

Exams Agent tool for CampusOS.

Exposes `get_exams`, a plain Python callable that reads exam schedule and
result data from `data/exams.json`, validates it, and returns a structured
(JSON-serializable) result. It follows the exact pattern established by
`agents/student_profile.py`, `agents/timetable.py`, and
`agents/attendance.py`, and is designed to be registered on an ADK
`LlmAgent` as a `FunctionTool`, e.g.:

    from google.adk.tools import FunctionTool
    from agents.exams import get_exams

    exams_tool = FunctionTool(func=get_exams)
    root_agent = LlmAgent(..., tools=[exams_tool])

No natural-language responses are hardcoded here — this module only ever
returns structured data; turning it into a sentence is the LLM's job once
this tool is wired into an agent's `tools` list.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger("campusos.agents.exams")

#: Default location of the exams data file, resolved relative to this
#: module's location so it works regardless of the process's cwd.
DEFAULT_DATA_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "exams.json"

#: Canonical exam statuses accepted for the `status` field.
_VALID_STATUSES = {"Scheduled", "Completed", "Cancelled"}

#: Query keyword -> canonical resolution mode.
_QUERY_ALIASES: Dict[str, str] = {
    "all": "all",
    "upcoming": "upcoming",
    "scheduled": "upcoming",
    "pending": "upcoming",
    "completed": "completed",
    "past": "completed",
    "results": "completed",
    "next_exam": "next_exam",
    "next": "next_exam",
}

#: Module-level cache of (mtime, validated exam record), keyed by resolved
#: path string. Mirrors the caching strategy used across the project.
_exams_cache: Dict[str, Tuple[float, "ExamRecord"]] = {}


class ExamError(RuntimeError):
    """Raised when the exams data file is missing, unreadable, or invalid."""


class ExamEntry(BaseModel):
    """Validated representation of a single scheduled or completed exam."""

    subject: str
    exam_type: str
    date: str
    max_marks: int
    obtained_marks: Optional[int] = None
    status: str

    @field_validator("subject", "exam_type")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("date")
    @classmethod
    def _validate_date_format(cls, value: str) -> str:
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"'{value}' is not a valid YYYY-MM-DD date") from exc
        return value

    @field_validator("max_marks")
    @classmethod
    def _validate_max_marks(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_marks must be positive")
        return value

    @field_validator("obtained_marks")
    @classmethod
    def _validate_obtained_marks(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 0:
            raise ValueError("obtained_marks must not be negative")
        return value

    @field_validator("status")
    @classmethod
    def _validate_status(cls, value: str) -> str:
        normalized = value.strip().title()
        if normalized not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(_VALID_STATUSES)}, got {value!r}")
        return normalized

    @model_validator(mode="after")
    def _validate_consistency(self) -> "ExamEntry":
        if self.obtained_marks is not None and self.obtained_marks > self.max_marks:
            raise ValueError(
                f"obtained_marks ({self.obtained_marks}) cannot exceed "
                f"max_marks ({self.max_marks}) for subject '{self.subject}'"
            )
        if self.status == "Completed" and self.obtained_marks is None:
            raise ValueError(
                f"exam for subject '{self.subject}' is marked Completed but has no obtained_marks"
            )
        if self.status == "Scheduled" and self.obtained_marks is not None:
            raise ValueError(
                f"exam for subject '{self.subject}' is marked Scheduled but already has obtained_marks"
            )
        return self

    @property
    def percentage(self) -> Optional[float]:
        """Score percentage for this exam, or None if not yet completed."""
        if self.obtained_marks is None:
            return None
        return round((self.obtained_marks / self.max_marks) * 100, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a structured, JSON-serializable view including derived fields."""
        return {
            "subject": self.subject,
            "exam_type": self.exam_type,
            "date": self.date,
            "max_marks": self.max_marks,
            "obtained_marks": self.obtained_marks,
            "percentage": self.percentage,
            "status": self.status,
        }


class ExamRecord(BaseModel):
    """Validated representation of the full exam schedule and result set."""

    exams: List[ExamEntry]

    @field_validator("exams")
    @classmethod
    def _not_empty(cls, value: List[ExamEntry]) -> List[ExamEntry]:
        if not value:
            raise ValueError("exam record must contain at least one entry")
        return value


def load_exams(path: Optional[Path] = None) -> ExamRecord:
    """
    Loads and validates the exam schedule/results from a JSON file.

    Args:
        path: Optional override for the data file location. Defaults to
            `DEFAULT_DATA_PATH` (`data/exams.json` relative to the project
            root) when omitted.

    Returns:
        A validated `ExamRecord` instance.

    Raises:
        ExamError: If the file does not exist, contains invalid JSON, or
            contains data that fails schema validation.
    """
    resolved_path = Path(path) if path is not None else DEFAULT_DATA_PATH

    if not resolved_path.exists():
        raise ExamError(f"Exams data file not found at: {resolved_path}")
    if not resolved_path.is_file():
        raise ExamError(f"Exams data path is not a file: {resolved_path}")

    file_mtime = resolved_path.stat().st_mtime
    cache_key = str(resolved_path)
    cached = _exams_cache.get(cache_key)
    if cached is not None and cached[0] == file_mtime:
        return cached[1]

    try:
        raw_text = resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExamError(f"Could not read {resolved_path}: {exc}") from exc

    try:
        raw_data: Any = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ExamError(f"Invalid JSON in {resolved_path}: {exc}") from exc

    try:
        record = ExamRecord.model_validate(raw_data)
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError, re-raised as ExamError
        raise ExamError(f"Exam data in {resolved_path} failed validation: {exc}") from exc

    _exams_cache[cache_key] = (file_mtime, record)
    return record


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _find_subject_entries(record: ExamRecord, subject_query: str) -> List[ExamEntry]:
    """Case-insensitive, substring-tolerant lookup of exam entries by subject name."""
    normalized_query = subject_query.strip().lower()

    exact = [entry for entry in record.exams if entry.subject.lower() == normalized_query]
    if exact:
        return exact

    return [entry for entry in record.exams if normalized_query in entry.subject.lower()]


def get_exams(
    query: str = "all", reference_date: Optional[date] = None
) -> Dict[str, Any]:
    """
    Retrieves exam schedule and result information for the current student.

    This is the ADK-compatible tool entry point for the Exams Agent. Wrap
    it with `google.adk.tools.FunctionTool` and add it to an `LlmAgent`'s
    `tools` list so the LLM can call it to answer questions such as "what
    exams do I have coming up", "what was my score in the DSA midterm",
    "what's my next exam", or "show me my completed exam results".

    Args:
        query: Which slice of the exam record to retrieve. Accepts:
            - "all" (default) — every exam entry, scheduled and completed.
            - "upcoming" / "scheduled" / "pending" — only exams not yet
              completed, sorted by date ascending.
            - "completed" / "past" / "results" — only completed exams
              with results, sorted by date ascending.
            - "next_exam" / "next" — the single soonest upcoming exam
              on or after `reference_date`, or None if none remain.
            - A specific subject name (case-insensitive, exact or
              unambiguous partial match) — all exam entries for that
              subject.
        reference_date: Optional fixed date to evaluate "next_exam"
            against. Defaults to `date.today()` when omitted — this
            parameter exists primarily so callers (and tests) can get
            deterministic results.

    Returns:
        A JSON-serializable dict.

        On success for "all" / "upcoming" / "completed" / a subject name::

            {
                "status": "success",
                "query": "<canonical query or matched subject>",
                "value": [<exam entry dict>, ...],
            }

        On success for "next_exam"::

            {
                "status": "success",
                "query": "next_exam",
                "value": <exam entry dict> | None,
            }

        Each `<exam entry dict>` has the shape::

            {
                "subject": str,
                "exam_type": str,
                "date": "YYYY-MM-DD",
                "max_marks": int,
                "obtained_marks": int | None,
                "percentage": float | None,
                "status": "Scheduled" | "Completed" | "Cancelled",
            }

        On failure (missing/invalid data file, or an unrecognized
        subject/query)::

            {
                "status": "error",
                "query": "<query as originally given>",
                "error_message": "<human-readable explanation>",
            }
    """
    try:
        record = load_exams()
    except ExamError as exc:
        logger.error("Failed to load exams: %s", exc)
        return {"status": "error", "query": query, "error_message": str(exc)}

    today = reference_date or date.today()
    normalized_query = (query or "all").strip().lower().replace(" ", "_")
    canonical_mode = _QUERY_ALIASES.get(normalized_query)

    if canonical_mode == "all":
        ordered = sorted(record.exams, key=lambda entry: _parse_date(entry.date))
        return {"status": "success", "query": "all", "value": [e.to_dict() for e in ordered]}

    if canonical_mode == "upcoming":
        upcoming = sorted(
            (e for e in record.exams if e.status == "Scheduled"),
            key=lambda entry: _parse_date(entry.date),
        )
        return {"status": "success", "query": "upcoming", "value": [e.to_dict() for e in upcoming]}

    if canonical_mode == "completed":
        completed = sorted(
            (e for e in record.exams if e.status == "Completed"),
            key=lambda entry: _parse_date(entry.date),
        )
        return {
            "status": "success",
            "query": "completed",
            "value": [e.to_dict() for e in completed],
        }

    if canonical_mode == "next_exam":
        future_scheduled = sorted(
            (e for e in record.exams if e.status == "Scheduled" and _parse_date(e.date) >= today),
            key=lambda entry: _parse_date(entry.date),
        )
        next_entry = future_scheduled[0] if future_scheduled else None
        return {
            "status": "success",
            "query": "next_exam",
            "value": next_entry.to_dict() if next_entry else None,
        }

    matched_entries = _find_subject_entries(record, query)
    if matched_entries:
        ordered = sorted(matched_entries, key=lambda entry: _parse_date(entry.date))
        matched_subject = ordered[0].subject
        return {
            "status": "success",
            "query": matched_subject,
            "value": [e.to_dict() for e in ordered],
        }

    valid_subjects = sorted({entry.subject for entry in record.exams})
    valid_queries = sorted(set(_QUERY_ALIASES.values())) + valid_subjects
    logger.warning("Unknown exams query requested: %r", query)
    return {
        "status": "error",
        "query": query,
        "error_message": f"Unknown query '{query}'. Valid queries: {valid_queries}",
    }