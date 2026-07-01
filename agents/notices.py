"""
agents/notices.py

Notices Agent tool for CampusOS.

Exposes `get_notices`, a plain Python callable that reads campus notices
from `data/notices.json`, validates it, and returns a structured
(JSON-serializable) result. It follows the exact pattern established by
`agents/student_profile.py`, `agents/timetable.py`, `agents/attendance.py`,
`agents/exams.py`, and `agents/fees.py`, and is designed to be registered
on an ADK `LlmAgent` as a `FunctionTool`, e.g.:

    from google.adk.tools import FunctionTool
    from agents.notices import get_notices

    notices_tool = FunctionTool(func=get_notices)
    root_agent = LlmAgent(..., tools=[notices_tool])

No natural-language responses are hardcoded here — this module only ever
returns structured data; turning it into a sentence is the LLM's job once
this tool is wired into an agent's `tools` list.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, field_validator

logger = logging.getLogger("campusos.agents.notices")

#: Default location of the notices data file, resolved relative to this
#: module's location so it works regardless of the process's cwd.
DEFAULT_DATA_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "notices.json"

#: Valid notice categories.
_VALID_CATEGORIES = {"Examination", "Fee", "Event", "General", "Academic"}

#: Special query keyword -> canonical resolution mode.
#: Anything else is treated as a category name lookup.
_KEYWORD_ALIASES: Dict[str, str] = {
    "all": "all",
    "latest": "latest",
    "recent": "latest",
    "new": "latest",
    "unread": "unread",
    "important": "important",
    "urgent": "important",
}

#: How many notices "latest" returns.
LATEST_COUNT: int = 5

#: Module-level cache of (mtime, validated notice board), keyed by resolved
#: path string. Mirrors the caching strategy used across the project.
_notices_cache: Dict[str, Tuple[float, "NoticeBoard"]] = {}


class NoticeError(RuntimeError):
    """Raised when the notices data file is missing, unreadable, or invalid."""


class Notice(BaseModel):
    """Validated representation of a single campus notice."""

    id: str
    title: str
    category: str
    date: str
    body: str
    is_read: bool
    is_important: bool

    @field_validator("id", "title", "body")
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

    @field_validator("category")
    @classmethod
    def _validate_category(cls, value: str) -> str:
        normalized = value.strip().title()
        if normalized not in _VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(_VALID_CATEGORIES)}, got {value!r}"
            )
        return normalized

    def to_dict(self) -> Dict[str, Any]:
        """Returns a structured, JSON-serializable view of this notice."""
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "date": self.date,
            "body": self.body,
            "is_read": self.is_read,
            "is_important": self.is_important,
        }


class NoticeBoard(BaseModel):
    """Validated representation of all campus notices."""

    notices: List[Notice]

    @field_validator("notices")
    @classmethod
    def _not_empty(cls, value: List[Notice]) -> List[Notice]:
        if not value:
            raise ValueError("notice board must contain at least one notice")
        return value

    @field_validator("notices")
    @classmethod
    def _unique_ids(cls, value: List[Notice]) -> List[Notice]:
        ids = [n.id for n in value]
        if len(ids) != len(set(ids)):
            raise ValueError("notice board contains duplicate notice IDs")
        return value


def load_notices(path: Optional[Path] = None) -> NoticeBoard:
    """
    Loads and validates campus notices from a JSON file.

    Args:
        path: Optional override for the data file location. Defaults to
            `DEFAULT_DATA_PATH` (`data/notices.json` relative to the
            project root) when omitted.

    Returns:
        A validated `NoticeBoard` instance.

    Raises:
        NoticeError: If the file does not exist, contains invalid JSON,
            or contains data that fails schema validation.
    """
    resolved_path = Path(path) if path is not None else DEFAULT_DATA_PATH

    if not resolved_path.exists():
        raise NoticeError(f"Notices data file not found at: {resolved_path}")
    if not resolved_path.is_file():
        raise NoticeError(f"Notices data path is not a file: {resolved_path}")

    file_mtime = resolved_path.stat().st_mtime
    cache_key = str(resolved_path)
    cached = _notices_cache.get(cache_key)
    if cached is not None and cached[0] == file_mtime:
        return cached[1]

    try:
        raw_text = resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise NoticeError(f"Could not read {resolved_path}: {exc}") from exc

    try:
        raw_data: Any = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise NoticeError(f"Invalid JSON in {resolved_path}: {exc}") from exc

    try:
        board = NoticeBoard.model_validate(raw_data)
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError, re-raised as NoticeError
        raise NoticeError(f"Notices data in {resolved_path} failed validation: {exc}") from exc

    _notices_cache[cache_key] = (file_mtime, board)
    return board


def _sorted_by_date_desc(notices: List[Notice]) -> List[Notice]:
    """Returns notices sorted newest-first."""
    return sorted(notices, key=lambda n: n.date, reverse=True)


def get_notices(query: str = "all") -> Dict[str, Any]:
    """
    Retrieves campus notices for the current student.

    This is the ADK-compatible tool entry point for the Notices Agent.
    Wrap it with `google.adk.tools.FunctionTool` and add it to an
    `LlmAgent`'s `tools` list so the LLM can call it to answer questions
    such as "any new notices?", "show me exam notices", "what important
    notices do I have", or "what haven't I read yet".

    Args:
        query: Which slice of notices to retrieve. Accepts:
            - "all" (default) — every notice, newest-first.
            - "latest" / "recent" / "new" — the 5 most recent notices.
            - "unread" — notices where `is_read` is false, newest-first.
            - "important" / "urgent" — notices where `is_important` is
              true, newest-first.
            - A category name (case-insensitive): "Examination", "Fee",
              "Event", "General", "Academic".

    Returns:
        A JSON-serializable dict.

        On success::

            {
                "status": "success",
                "query": "<canonical query or category name>",
                "count": int,
                "value": [<notice dict>, ...],
            }

        Each `<notice dict>` has the shape::

            {
                "id": str,
                "title": str,
                "category": str,
                "date": "YYYY-MM-DD",
                "body": str,
                "is_read": bool,
                "is_important": bool,
            }

        On failure (missing/invalid data file, or an unrecognized
        query)::

            {
                "status": "error",
                "query": "<query as originally given>",
                "error_message": "<human-readable explanation>",
            }
    """
    try:
        board = load_notices()
    except NoticeError as exc:
        logger.error("Failed to load notices: %s", exc)
        return {"status": "error", "query": query, "error_message": str(exc)}

    normalized_query = (query or "all").strip().lower()
    canonical_mode = _KEYWORD_ALIASES.get(normalized_query)

    if canonical_mode == "all":
        notices = _sorted_by_date_desc(board.notices)
        return {
            "status": "success",
            "query": "all",
            "count": len(notices),
            "value": [n.to_dict() for n in notices],
        }

    if canonical_mode == "latest":
        notices = _sorted_by_date_desc(board.notices)[:LATEST_COUNT]
        return {
            "status": "success",
            "query": "latest",
            "count": len(notices),
            "value": [n.to_dict() for n in notices],
        }

    if canonical_mode == "unread":
        notices = _sorted_by_date_desc([n for n in board.notices if not n.is_read])
        return {
            "status": "success",
            "query": "unread",
            "count": len(notices),
            "value": [n.to_dict() for n in notices],
        }

    if canonical_mode == "important":
        notices = _sorted_by_date_desc([n for n in board.notices if n.is_important])
        return {
            "status": "success",
            "query": "important",
            "count": len(notices),
            "value": [n.to_dict() for n in notices],
        }

    # Category lookup — case-insensitive match against valid categories.
    category_match = normalized_query.title()
    if category_match in _VALID_CATEGORIES:
        notices = _sorted_by_date_desc(
            [n for n in board.notices if n.category == category_match]
        )
        return {
            "status": "success",
            "query": category_match,
            "count": len(notices),
            "value": [n.to_dict() for n in notices],
        }

    valid_queries = sorted(_KEYWORD_ALIASES.keys()) + sorted(_VALID_CATEGORIES)
    logger.warning("Unknown notices query requested: %r", query)
    return {
        "status": "error",
        "query": query,
        "error_message": f"Unknown query '{query}'. Valid queries: {valid_queries}",
    }