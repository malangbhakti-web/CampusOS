"""
tests/test_fees.py

Tests for the Fees Agent tool. Run against the real `data/fees.json` file
plus temporary files for error-path cases, following the exact testing
pattern used elsewhere in the project.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.fees import (
    DEFAULT_DATA_PATH,
    FeeError,
    FeeRecord,
    get_fees,
    load_fees,
)

_VALID_RECORD = {
    "total_fee": 100000.0,
    "due_date": "2026-12-01",
    "payment_history": [
        {"date": "2026-01-01", "amount": 40000.0, "mode": "UPI", "receipt_no": "R-1"},
    ],
}


def test_default_data_file_exists() -> None:
    assert DEFAULT_DATA_PATH.exists()


def test_load_fees_reads_real_file() -> None:
    record = load_fees()
    assert isinstance(record, FeeRecord)
    assert record.total_fee > 0
    assert record.paid_fee >= 0
    assert record.pending_fee >= 0


def test_load_fees_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FeeError, match="not found"):
        load_fees(tmp_path / "missing.json")


def test_load_fees_directory_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FeeError, match="not a file"):
        load_fees(tmp_path)


def test_load_fees_invalid_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(FeeError, match="Invalid JSON"):
        load_fees(bad)


def test_load_fees_negative_total_raises(tmp_path: Path) -> None:
    record = {**_VALID_RECORD, "total_fee": -100.0}
    bad = tmp_path / "bad_total.json"
    bad.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(FeeError, match="failed validation"):
        load_fees(bad)


def test_load_fees_invalid_due_date_raises(tmp_path: Path) -> None:
    record = {**_VALID_RECORD, "due_date": "01/12/2026"}
    bad = tmp_path / "bad_date.json"
    bad.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(FeeError, match="failed validation"):
        load_fees(bad)


def test_load_fees_negative_payment_amount_raises(tmp_path: Path) -> None:
    record = {
        **_VALID_RECORD,
        "payment_history": [{"date": "2026-01-01", "amount": -10.0, "mode": "UPI", "receipt_no": "R-1"}],
    }
    bad = tmp_path / "bad_amount.json"
    bad.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(FeeError, match="failed validation"):
        load_fees(bad)


def test_load_fees_payments_exceed_total_raises(tmp_path: Path) -> None:
    record = {
        "total_fee": 1000.0,
        "due_date": "2026-12-01",
        "payment_history": [
            {"date": "2026-01-01", "amount": 2000.0, "mode": "UPI", "receipt_no": "R-1"},
        ],
    }
    bad = tmp_path / "bad_exceeds.json"
    bad.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(FeeError, match="failed validation"):
        load_fees(bad)


def test_load_fees_blank_receipt_no_raises(tmp_path: Path) -> None:
    record = {
        **_VALID_RECORD,
        "payment_history": [{"date": "2026-01-01", "amount": 100.0, "mode": "UPI", "receipt_no": "  "}],
    }
    bad = tmp_path / "bad_receipt.json"
    bad.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(FeeError, match="failed validation"):
        load_fees(bad)


def test_load_fees_caches_until_file_changes(tmp_path: Path) -> None:
    data_file = tmp_path / "cached.json"
    data_file.write_text(json.dumps(_VALID_RECORD), encoding="utf-8")
    first = load_fees(data_file)
    assert first.total_fee == 100000.0

    updated = {**_VALID_RECORD, "total_fee": 200000.0}
    data_file.write_text(json.dumps(updated), encoding="utf-8")
    second = load_fees(data_file)
    assert second.total_fee == 200000.0


def test_fee_record_derived_fields() -> None:
    record = load_fees()
    expected_paid = round(sum(p.amount for p in record.payment_history), 2)
    assert record.paid_fee == expected_paid
    assert record.pending_fee == round(record.total_fee - expected_paid, 2)


def test_fee_record_payment_status_unpaid() -> None:
    record = FeeRecord(total_fee=1000.0, due_date="2026-12-01", payment_history=[])
    assert record.payment_status == "Unpaid"


def test_fee_record_payment_status_partially_paid() -> None:
    record = FeeRecord(
        total_fee=1000.0,
        due_date="2026-12-01",
        payment_history=[
            {"date": "2026-01-01", "amount": 200.0, "mode": "UPI", "receipt_no": "R-1"}
        ],
    )
    assert record.payment_status == "Partially Paid"


def test_fee_record_payment_status_paid() -> None:
    record = FeeRecord(
        total_fee=1000.0,
        due_date="2026-12-01",
        payment_history=[
            {"date": "2026-01-01", "amount": 1000.0, "mode": "UPI", "receipt_no": "R-1"}
        ],
    )
    assert record.payment_status == "Paid"


def test_get_fees_default_is_all() -> None:
    assert get_fees() == get_fees("all")


@pytest.mark.parametrize("alias", ["all", "summary", "status"])
def test_get_fees_summary_aliases(alias: str) -> None:
    result = get_fees(alias)
    assert result["status"] == "success"
    assert result["query"] == "summary"
    assert set(result["value"].keys()) == {
        "total_fee",
        "paid_fee",
        "pending_fee",
        "due_date",
        "payment_status",
    }


@pytest.mark.parametrize("alias", ["total", "total_fee"])
def test_get_fees_total_fee_aliases(alias: str) -> None:
    record = load_fees()
    result = get_fees(alias)
    assert result["status"] == "success"
    assert result["query"] == "total_fee"
    assert result["value"] == record.total_fee


@pytest.mark.parametrize("alias", ["paid", "paid_fee"])
def test_get_fees_paid_fee_aliases(alias: str) -> None:
    record = load_fees()
    result = get_fees(alias)
    assert result["value"] == record.paid_fee


@pytest.mark.parametrize("alias", ["pending", "pending_fee", "due", "due_fee"])
def test_get_fees_pending_fee_aliases(alias: str) -> None:
    record = load_fees()
    result = get_fees(alias)
    assert result["query"] == "pending_fee"
    assert result["value"] == record.pending_fee


@pytest.mark.parametrize("alias", ["due_date", "deadline"])
def test_get_fees_due_date_aliases(alias: str) -> None:
    record = load_fees()
    result = get_fees(alias)
    assert result["query"] == "due_date"
    assert result["value"] == record.due_date


@pytest.mark.parametrize("alias", ["history", "payment_history", "payments", "transactions"])
def test_get_fees_payment_history_aliases(alias: str) -> None:
    record = load_fees()
    result = get_fees(alias)
    assert result["status"] == "success"
    assert result["query"] == "payment_history"
    assert len(result["value"]) == len(record.payment_history)
    for entry in result["value"]:
        assert set(entry.keys()) == {"date", "amount", "mode", "receipt_no"}


def test_get_fees_payment_history_sorted_by_date() -> None:
    result = get_fees("history")
    dates = [p["date"] for p in result["value"]]
    assert dates == sorted(dates)


def test_get_fees_unknown_query_returns_error_not_exception() -> None:
    result = get_fees("scholarship")
    assert result["status"] == "error"
    assert "Unknown query" in result["error_message"]


def test_get_fees_missing_data_file_returns_structured_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agents.fees as fees_module

    monkeypatch.setattr(fees_module, "DEFAULT_DATA_PATH", tmp_path / "missing.json")
    fees_module._fees_cache.clear()

    result = get_fees("all")
    assert result["status"] == "error"
    assert "not found" in result["error_message"]


def test_get_fees_is_importable_exactly_as_required() -> None:
    from agents.fees import get_fees as imported_get_fees

    assert callable(imported_get_fees)
    assert imported_get_fees("all")["status"] == "success"


def test_get_fees_wraps_as_adk_function_tool() -> None:
    from google.adk.tools import FunctionTool

    from agents.fees import get_fees as tool_func

    tool = FunctionTool(func=tool_func)
    assert tool.name == "get_fees"