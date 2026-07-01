"""
agents/student_profile.py

Student Profile Agent tool for CampusOS.

Exposes `get_student_profile`, a plain Python callable that reads a
student's profile from `data/student_profile.json`, validates it, and
returns a structured (JSON-serializable) result. It is designed to be
registered on an ADK `LlmAgent` later as a `FunctionTool`, e.g.:

    from google.adk.tools import FunctionTool
    from agents.student_profile import get_student_profile

    student_profile_tool = FunctionTool(func=get_student_profile)
    root_agent = LlmAgent(..., tools=[student_profile_tool])

No natural-language responses are hardcoded here — this module only ever
returns structured data. Turning that data into a sentence (e.g. "Your
department is Computer Science") is the LLM's job once this tool is wired
into an agent's `tools` list.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ValidationError, field_validator

logger = logging.getLogger("campusos.agents.student_profile")

#: Default location of the student profile data file, resolved relative to
#: this module's location so it works regardless of the process's cwd.
DEFAULT_DATA_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "student_profile.json"

#: Canonical field name -> accepted question/alias phrasings.
_FIELD_ALIASES: Dict[str, str] = {
    "all": "all",
    "profile": "all",
    "who_am_i": "all",
    "who": "all",
    "name": "name",
    "full_name": "name",
    "student_name": "name",
    "department": "department",
    "dept": "department",
    "semester": "semester",
    "sem": "semester",
    "enrollment_number": "enrollment_number",
    "enrollment": "enrollment_number",
    "roll_number": "enrollment_number",
    "registration_number": "enrollment_number",
    "email": "email",
    "email_address": "email",
    "cgpa": "cgpa",
    "gpa": "cgpa",
    "subjects": "subjects",
    "courses": "subjects",
    "enrolled_subjects": "subjects",
}

#: Module-level cache of (mtime, validated profile), keyed by resolved path
#: string. Avoids re-reading/re-validating the file on every tool call while
#: still picking up edits to the JSON file (cache is invalidated on mtime
#: change), since the file may be edited externally between agent turns.
_profile_cache: Dict[str, Tuple[float, "StudentProfile"]] = {}


class StudentProfileError(RuntimeError):
    """Raised when the student profile data file is missing, unreadable, or invalid."""


class StudentProfile(BaseModel):
    """Validated representation of a single student's profile record."""

    name: str
    enrollment_number: str
    department: str
    semester: int
    email: str
    cgpa: float
    subjects: List[str]

    @field_validator("name", "enrollment_number", "department", "email")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("semester")
    @classmethod
    def _validate_semester(cls, value: int) -> int:
        if value < 1 or value > 12:
            raise ValueError("semester must be between 1 and 12")
        return value

    @field_validator("cgpa")
    @classmethod
    def _validate_cgpa(cls, value: float) -> float:
        if not 0.0 <= value <= 10.0:
            raise ValueError("cgpa must be between 0.0 and 10.0")
        return value

    @field_validator("email")
    @classmethod
    def _validate_email(cls, value: str) -> str:
        local_part, _, domain_part = value.partition("@")
        if not local_part or "." not in domain_part:
            raise ValueError(f"'{value}' is not a valid email address")
        return value

    @field_validator("subjects")
    @classmethod
    def _validate_subjects(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("subjects must not be an empty list")
        if any(not subject.strip() for subject in value):
            raise ValueError("subjects must not contain blank entries")
        return value


def load_student_profile(path: Optional[Path] = None) -> StudentProfile:
    """
    Loads and validates the student profile from a JSON file.

    Args:
        path: Optional override for the data file location. Defaults to
            `DEFAULT_DATA_PATH` (`data/student_profile.json` relative to
            the project root) when omitted.

    Returns:
        A validated `StudentProfile` instance.

    Raises:
        StudentProfileError: If the file does not exist, contains invalid
            JSON, or contains data that fails schema validation.
    """
    resolved_path = Path(path) if path is not None else DEFAULT_DATA_PATH

    if not resolved_path.exists():
        raise StudentProfileError(f"Student profile data file not found at: {resolved_path}")
    if not resolved_path.is_file():
        raise StudentProfileError(f"Student profile data path is not a file: {resolved_path}")

    file_mtime = resolved_path.stat().st_mtime
    cache_key = str(resolved_path)
    cached = _profile_cache.get(cache_key)
    if cached is not None and cached[0] == file_mtime:
        return cached[1]

    try:
        raw_text = resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StudentProfileError(f"Could not read {resolved_path}: {exc}") from exc

    try:
        raw_data: Any = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise StudentProfileError(f"Invalid JSON in {resolved_path}: {exc}") from exc

    try:
        profile = StudentProfile.model_validate(raw_data)
    except ValidationError as exc:
        raise StudentProfileError(
            f"Student profile data in {resolved_path} failed validation: {exc}"
        ) from exc

    _profile_cache[cache_key] = (file_mtime, profile)
    return profile


def get_student_profile(field: str = "all") -> Dict[str, Any]:
    """
    Retrieves information from the current student's profile.

    This is the ADK-compatible tool entry point for the Student Profile
    Agent. Wrap it with `google.adk.tools.FunctionTool` and add it to an
    `LlmAgent`'s `tools` list so the LLM can call it to answer questions
    such as "who am I", "what is my department", "what is my CGPA", or
    "what subjects am I enrolled in".

    Args:
        field: Which piece of profile information to retrieve. Accepts
            "all" for the full profile, or any of: "name", "department",
            "semester", "enrollment_number", "email", "cgpa", "subjects".
            Common aliases are also accepted, e.g. "who_am_i", "dept",
            "gpa", "roll_number", "courses". Defaults to "all".

    Returns:
        A JSON-serializable dict.

        On success::

            {
                "status": "success",
                "field": "<canonical field name>",
                "value": <str | int | float | list[str] | dict>,
                "profile": {<full profile as a dict>},
            }

        On failure (missing/invalid data file, or an unrecognized field)::

            {
                "status": "error",
                "field": "<field as originally given>",
                "error_message": "<human-readable explanation>",
            }
    """
    try:
        profile = load_student_profile()
    except StudentProfileError as exc:
        logger.error("Failed to load student profile: %s", exc)
        return {"status": "error", "field": field, "error_message": str(exc)}

    normalized_field = (field or "all").strip().lower().replace(" ", "_")
    canonical_field = _FIELD_ALIASES.get(normalized_field)

    if canonical_field is None:
        valid_fields = sorted(set(_FIELD_ALIASES.values()))
        logger.warning("Unknown student profile field requested: %r", field)
        return {
            "status": "error",
            "field": field,
            "error_message": f"Unknown field '{field}'. Valid fields: {valid_fields}",
        }

    profile_dict = profile.model_dump()
    value: Any = profile_dict if canonical_field == "all" else profile_dict[canonical_field]

    return {
        "status": "success",
        "field": canonical_field,
        "value": value,
        "profile": profile_dict,
    }