"""
agents/fees.py

Fees Agent tool for CampusOS.

Exposes `get_fees`, a plain Python callable that reads fee status and
payment history from `data/fees.json`, validates it, and returns a
structured (JSON-serializable) result. It follows the exact pattern
established by `agents/student_profile.py`, `agents/timetable.py`,
`agents/attendance.py`, and `agents/exams.py`, and is designed to be
registered on an ADK `LlmAgent` as a `FunctionTool`, e.g.:

    from google.adk.tools import FunctionTool
    from agents.fees import get_fees

    fees_tool = FunctionTool(func=get_fees)
    root_agent = LlmAgent(..., tools=[fees_tool])

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

from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger("campusos.agents.fees")

#: Default location of the fees data file, resolved relative to this
#: module's location so it works regardless of the process's cwd.
DEFAULT_DATA_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "fees.json"

#: Query keyword -> canonical resolution mode.
_QUERY_ALIASES: Dict[str, str] = {
    "all": "summary",
    "summary": "summary",
    "status": "summary",
    "total": "total_fee",
    "total_fee": "total_fee",
    "paid": "paid_fee",
    "paid_fee": "paid_fee",
    "pending": "pending_fee",
    "pending_fee": "pending_fee",
    "due": "pending_fee",
    "due_fee": "pending_fee",
    "due_date": "due_date",
    "deadline": "due_date",
    "history": "payment_history",
    "payment_history": "payment_history",
    "payments": "payment_history",
    "transactions": "payment_history",
}

#: Module-level cache of (mtime, validated fee record), keyed by resolved
#: path string. Mirrors the caching strategy used across the project.
_fees_cache: Dict[str, Tuple[float, "FeeRecord"]] = {}


class FeeError(RuntimeError):
    """Raised when the fees data file is missing, unreadable, or invalid."""


class PaymentRecord(BaseModel):
    """Validated representation of a single fee payment."""

    date: str
    amount: float
    mode: str
    receipt_no: str

    @field_validator("mode", "receipt_no")
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

    @field_validator("amount")
    @classmethod
    def _validate_amount(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("amount must be positive")
        return value


class FeeRecord(BaseModel):
    """Validated representation of a student's fee status and payment history."""

    total_fee: float
    due_date: str
    payment_history: List[PaymentRecord]

    @field_validator("total_fee")
    @classmethod
    def _validate_total_fee(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("total_fee must be positive")
        return value

    @field_validator("due_date")
    @classmethod
    def _validate_due_date_format(cls, value: str) -> str:
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"'{value}' is not a valid YYYY-MM-DD date") from exc
        return value

    @model_validator(mode="after")
    def _validate_paid_not_exceeding_total(self) -> "FeeRecord":
        paid = sum(payment.amount for payment in self.payment_history)
        if paid > self.total_fee:
            raise ValueError(
                f"sum of payment_history ({paid}) cannot exceed total_fee ({self.total_fee})"
            )
        return self

    @property
    def paid_fee(self) -> float:
        """Total amount paid so far, derived from payment_history."""
        return round(sum(payment.amount for payment in self.payment_history), 2)

    @property
    def pending_fee(self) -> float:
        """Remaining amount owed."""
        return round(self.total_fee - self.paid_fee, 2)

    @property
    def payment_status(self) -> str:
        """One of 'Paid', 'Partially Paid', or 'Unpaid'."""
        if self.pending_fee <= 0:
            return "Paid"
        if self.paid_fee > 0:
            return "Partially Paid"
        return "Unpaid"


def load_fees(path: Optional[Path] = None) -> FeeRecord:
    """
    Loads and validates the fee record from a JSON file.

    Args:
        path: Optional override for the data file location. Defaults to
            `DEFAULT_DATA_PATH` (`data/fees.json` relative to the project
            root) when omitted.

    Returns:
        A validated `FeeRecord` instance.

    Raises:
        FeeError: If the file does not exist, contains invalid JSON, or
            contains data that fails schema validation.
    """
    resolved_path = Path(path) if path is not None else DEFAULT_DATA_PATH

    if not resolved_path.exists():
        raise FeeError(f"Fees data file not found at: {resolved_path}")
    if not resolved_path.is_file():
        raise FeeError(f"Fees data path is not a file: {resolved_path}")

    file_mtime = resolved_path.stat().st_mtime
    cache_key = str(resolved_path)
    cached = _fees_cache.get(cache_key)
    if cached is not None and cached[0] == file_mtime:
        return cached[1]

    try:
        raw_text = resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FeeError(f"Could not read {resolved_path}: {exc}") from exc

    try:
        raw_data: Any = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise FeeError(f"Invalid JSON in {resolved_path}: {exc}") from exc

    try:
        record = FeeRecord.model_validate(raw_data)
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError, re-raised as FeeError
        raise FeeError(f"Fee data in {resolved_path} failed validation: {exc}") from exc

    _fees_cache[cache_key] = (file_mtime, record)
    return record


def _summary_dict(record: FeeRecord) -> Dict[str, Any]:
    return {
        "total_fee": record.total_fee,
        "paid_fee": record.paid_fee,
        "pending_fee": record.pending_fee,
        "due_date": record.due_date,
        "payment_status": record.payment_status,
    }


def get_fees(query: str = "all") -> Dict[str, Any]:
    """
    Retrieves fee status and payment information for the current student.

    This is the ADK-compatible tool entry point for the Fees Agent. Wrap
    it with `google.adk.tools.FunctionTool` and add it to an `LlmAgent`'s
    `tools` list so the LLM can call it to answer questions such as "what
    is my total fee", "how much have I paid", "how much do I still owe",
    "when is my fee due", or "show my payment history".

    Args:
        query: Which slice of the fee record to retrieve. Accepts:
            - "all" / "summary" / "status" (default "all") — total,
              paid, pending, due date, and payment status together.
            - "total" / "total_fee" — the total fee amount.
            - "paid" / "paid_fee" — total amount paid so far.
            - "pending" / "pending_fee" / "due" / "due_fee" — remaining
              amount owed.
            - "due_date" / "deadline" — the fee payment due date.
            - "history" / "payment_history" / "payments" /
              "transactions" — the list of individual payments made.

    Returns:
        A JSON-serializable dict.

        On success for "summary"::

            {
                "status": "success",
                "query": "summary",
                "value": {
                    "total_fee": float,
                    "paid_fee": float,
                    "pending_fee": float,
                    "due_date": "YYYY-MM-DD",
                    "payment_status": "Paid" | "Partially Paid" | "Unpaid",
                },
            }

        On success for a single numeric/date field::

            {"status": "success", "query": "<field>", "value": <float | str>}

        On success for "payment_history"::

            {
                "status": "success",
                "query": "payment_history",
                "value": [
                    {"date": str, "amount": float, "mode": str, "receipt_no": str},
                    ...
                ],
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
        record = load_fees()
    except FeeError as exc:
        logger.error("Failed to load fees: %s", exc)
        return {"status": "error", "query": query, "error_message": str(exc)}

    normalized_query = (query or "all").strip().lower().replace(" ", "_")
    canonical_mode = _QUERY_ALIASES.get(normalized_query)

    if canonical_mode == "summary":
        return {"status": "success", "query": "summary", "value": _summary_dict(record)}

    if canonical_mode == "total_fee":
        return {"status": "success", "query": "total_fee", "value": record.total_fee}

    if canonical_mode == "paid_fee":
        return {"status": "success", "query": "paid_fee", "value": record.paid_fee}

    if canonical_mode == "pending_fee":
        return {"status": "success", "query": "pending_fee", "value": record.pending_fee}

    if canonical_mode == "due_date":
        return {"status": "success", "query": "due_date", "value": record.due_date}

    if canonical_mode == "payment_history":
        ordered = sorted(record.payment_history, key=lambda p: p.date)
        return {
            "status": "success",
            "query": "payment_history",
            "value": [p.model_dump() for p in ordered],
        }

    valid_queries = sorted(set(_QUERY_ALIASES.keys()))
    logger.warning("Unknown fees query requested: %r", query)
    return {
        "status": "error",
        "query": query,
        "error_message": f"Unknown query '{query}'. Valid queries: {valid_queries}",
    }