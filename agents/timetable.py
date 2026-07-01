"""
agents/timetable.py

Timetable Agent tool for CampusOS.

Exposes `get_timetable`, a plain Python callable that reads the weekly
class schedule from `data/timetable.json`, validates it, and returns a
structured (JSON-serializable) result. It follows the exact pattern
established by `agents/student_profile.py` and is designed to be
registered on an ADK `LlmAgent` later as a `FunctionTool`, e.g.:

    from google.adk.tools import FunctionTool
    from agents.timetable import get_timetable

    timetable_tool = FunctionTool(func=get_timetable)
    root_agent = LlmAgent(..., tools=[timetable_tool])

No natural-language responses are hardcoded here — this module only ever
returns structured data; turning it into a sentence is the LLM's job once
this tool is wired into an agent's `tools` list.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger("campusos.agents.timetable")

#: Default location of the timetable data file, resolved relative to this
#: module's location so it works regardless of the process's cwd.
DEFAULT_DATA_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "timetable.json"

#: Canonical weekday names, ordered Monday-first to match `datetime.weekday()`.
_VALID_DAYS: List[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

#: Canonical class types accepted for the `type` field.
_VALID_TYPES = {"Lecture", "Lab", "Tutorial", "Practical"}

#: Lowercase weekday name -> canonical weekday name, for case-insensitive lookup.
_DAY_NAME_LOOKUP: Dict[str, str] = {day.lower(): day for day in _VALID_DAYS}

#: Query keywords that are not weekday names, used for error messages.
_RELATIVE_QUERIES: List[str] = ["all", "today", "tomorrow", "current_class", "next_class"]

#: Module-level cache of (mtime, validated timetable), keyed by resolved
#: path string. Mirrors the caching strategy in `agents/student_profile.py`.
_timetable_cache: Dict[str, Tuple[float, "Timetable"]] = {}


class TimetableError(RuntimeError):
    """Raised when the timetable data file is missing, unreadable, or invalid."""


def _parse_time(value: str) -> time:
    """Parses an `HH:MM` 24-hour string into a `datetime.time`."""
    return datetime.strptime(value, "%H:%M").time()


class TimetableEntry(BaseModel):
    """Validated representation of a single scheduled class session."""

    day: str
    subject: str
    type: str
    start_time: str
    end_time: str
    room: str
    instructor: str

    @field_validator("subject", "room", "instructor")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("day")
    @classmethod
    def _validate_day(cls, value: str) -> str:
        normalized = _DAY_NAME_LOOKUP.get(value.strip().lower())
        if normalized is None:
            raise ValueError(f"day must be one of {_VALID_DAYS}, got {value!r}")
        return normalized

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        normalized = value.strip().title()
        if normalized not in _VALID_TYPES:
            raise ValueError(f"type must be one of {sorted(_VALID_TYPES)}, got {value!r}")
        return normalized

    @field_validator("start_time", "end_time")
    @classmethod
    def _validate_time_format(cls, value: str) -> str:
        try:
            _parse_time(value)
        except ValueError as exc:
            raise ValueError(f"'{value}' is not a valid HH:MM 24-hour time") from exc
        return value

    @model_validator(mode="after")
    def _validate_time_order(self) -> "TimetableEntry":
        if _parse_time(self.end_time) <= _parse_time(self.start_time):
            raise ValueError(
                f"end_time ({self.end_time}) must be after start_time ({self.start_time})"
            )
        return self


class Timetable(BaseModel):
    """Validated representation of the full weekly timetable."""

    entries: List[TimetableEntry]

    @field_validator("entries")
    @classmethod
    def _not_empty(cls, value: List[TimetableEntry]) -> List[TimetableEntry]:
        if not value:
            raise ValueError("timetable must contain at least one entry")
        return value


def load_timetable(path: Optional[Path] = None) -> Timetable:
    """
    Loads and validates the weekly timetable from a JSON file.

    Args:
        path: Optional override for the data file location. Defaults to
            `DEFAULT_DATA_PATH` (`data/timetable.json` relative to the
            project root) when omitted.

    Returns:
        A validated `Timetable` instance.

    Raises:
        TimetableError: If the file does not exist, contains invalid JSON,
            or contains data that fails schema validation.
    """
    resolved_path = Path(path) if path is not None else DEFAULT_DATA_PATH

    if not resolved_path.exists():
        raise TimetableError(f"Timetable data file not found at: {resolved_path}")
    if not resolved_path.is_file():
        raise TimetableError(f"Timetable data path is not a file: {resolved_path}")

    file_mtime = resolved_path.stat().st_mtime
    cache_key = str(resolved_path)
    cached = _timetable_cache.get(cache_key)
    if cached is not None and cached[0] == file_mtime:
        return cached[1]

    try:
        raw_text = resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TimetableError(f"Could not read {resolved_path}: {exc}") from exc

    try:
        raw_data: Any = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise TimetableError(f"Invalid JSON in {resolved_path}: {exc}") from exc

    try:
        timetable = Timetable.model_validate(raw_data)
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError, re-raised as TimetableError
        raise TimetableError(f"Timetable data in {resolved_path} failed validation: {exc}") from exc

    _timetable_cache[cache_key] = (file_mtime, timetable)
    return timetable


def _entries_for_day(timetable: Timetable, day_name: str) -> List[TimetableEntry]:
    """Returns a day's entries, sorted chronologically by start time."""
    matching = [entry for entry in timetable.entries if entry.day == day_name]
    return sorted(matching, key=lambda entry: _parse_time(entry.start_time))


def _current_class(timetable: Timetable, now: datetime) -> Optional[TimetableEntry]:
    """Returns the entry whose time window contains `now`, if any."""
    today_name = now.strftime("%A")
    current_time = now.time()
    for entry in _entries_for_day(timetable, today_name):
        if _parse_time(entry.start_time) <= current_time < _parse_time(entry.end_time):
            return entry
    return None


def _next_class(timetable: Timetable, now: datetime) -> Optional[TimetableEntry]:
    """
    Returns the soonest upcoming entry after `now`.

    Checks the remainder of today first, then walks forward day by day
    (wrapping around the full week, since the timetable repeats weekly) and
    returns the first entry of the first day found with any classes.
    """
    today_name = now.strftime("%A")
    current_time = now.time()
    for entry in _entries_for_day(timetable, today_name):
        if _parse_time(entry.start_time) > current_time:
            return entry

    for offset in range(1, 8):
        future_day_name = (now + timedelta(days=offset)).strftime("%A")
        future_entries = _entries_for_day(timetable, future_day_name)
        if future_entries:
            return future_entries[0]

    return None


def get_timetable(
    query: str = "all", reference_datetime: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Retrieves class schedule information from the student's weekly timetable.

    This is the ADK-compatible tool entry point for the Timetable Agent.
    Wrap it with `google.adk.tools.FunctionTool` and add it to an
    `LlmAgent`'s `tools` list so the LLM can call it to answer scheduling
    questions.

    Args:
        query: Which slice of the timetable to retrieve. Accepts:
            - "all" — the full weekly timetable (default).
            - "Monday" .. "Sunday" (case-insensitive) — that day's classes.
            - "today" — the current day's classes.
            - "tomorrow" — the next day's classes.
            - "current_class" — the class in session right now, or `None`
              if no class is currently running.
            - "next_class" — the soonest upcoming class, wrapping to next
              week if nothing remains in the current week.
        reference_datetime: Optional fixed point in time to evaluate
            "today"/"tomorrow"/"current_class"/"next_class" against.
            Defaults to `datetime.now()` when omitted — this parameter
            exists primarily so callers (and tests) can get deterministic
            results; the LLM-facing call simply omits it.

    Returns:
        A JSON-serializable dict.

        On success::

            {
                "status": "success",
                "query": "<canonical resolved query keyword or weekday>",
                "resolved_day": "<weekday name>" | None,
                "value": <list[dict] | dict | None>,
            }

        `value` is a list of entry dicts for weekday/"all"/"today"/"tomorrow"
        queries, a single entry dict (or `None`) for "current_class"/
        "next_class".

        On failure (missing/invalid data file, or an unrecognized query)::

            {
                "status": "error",
                "query": "<query as originally given>",
                "error_message": "<human-readable explanation>",
            }
    """
    try:
        timetable = load_timetable()
    except TimetableError as exc:
        logger.error("Failed to load timetable: %s", exc)
        return {"status": "error", "query": query, "error_message": str(exc)}

    now = reference_datetime or datetime.now()
    normalized = (query or "all").strip().lower().replace(" ", "_")

    if normalized == "all":
        ordered_entries = sorted(
            timetable.entries,
            key=lambda entry: (_VALID_DAYS.index(entry.day), _parse_time(entry.start_time)),
        )
        return {
            "status": "success",
            "query": "all",
            "resolved_day": None,
            "value": [entry.model_dump() for entry in ordered_entries],
        }

    if normalized in _DAY_NAME_LOOKUP:
        day_name = _DAY_NAME_LOOKUP[normalized]
        entries = _entries_for_day(timetable, day_name)
        return {
            "status": "success",
            "query": day_name,
            "resolved_day": day_name,
            "value": [entry.model_dump() for entry in entries],
        }

    if normalized == "today":
        day_name = now.strftime("%A")
        entries = _entries_for_day(timetable, day_name)
        return {
            "status": "success",
            "query": "today",
            "resolved_day": day_name,
            "value": [entry.model_dump() for entry in entries],
        }

    if normalized == "tomorrow":
        day_name = (now + timedelta(days=1)).strftime("%A")
        entries = _entries_for_day(timetable, day_name)
        return {
            "status": "success",
            "query": "tomorrow",
            "resolved_day": day_name,
            "value": [entry.model_dump() for entry in entries],
        }

    if normalized == "current_class":
        entry = _current_class(timetable, now)
        return {
            "status": "success",
            "query": "current_class",
            "resolved_day": now.strftime("%A"),
            "value": entry.model_dump() if entry else None,
        }

    if normalized == "next_class":
        entry = _next_class(timetable, now)
        return {
            "status": "success",
            "query": "next_class",
            "resolved_day": entry.day if entry else None,
            "value": entry.model_dump() if entry else None,
        }

    valid_queries = _RELATIVE_QUERIES + _VALID_DAYS
    logger.warning("Unknown timetable query requested: %r", query)
    return {
        "status": "error",
        "query": query,
        "error_message": f"Unknown query '{query}'. Valid queries: {valid_queries}",
    }