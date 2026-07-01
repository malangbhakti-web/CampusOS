"""
tests/test_integration.py

Integration tests for CampusOS.

These tests verify that:
- Every tool is independently importable and callable.
- Every tool returns structured JSON (no exceptions to callers).
- The root agent imports correctly, carries all six tools, and is an
  ADK LlmAgent.
- Every tool is individually wrappable as an ADK FunctionTool.
- The CampusOSRuntime initialises without errors.

No live model calls are made — these are structural integration tests,
not end-to-end conversation tests.
"""

from __future__ import annotations

import pytest

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool


# ---------------------------------------------------------------------------
# Tool import and basic call
# ---------------------------------------------------------------------------


def test_get_student_profile_loads_and_returns_success() -> None:
    from agents.student_profile import get_student_profile

    result = get_student_profile("all")
    assert result["status"] == "success"
    assert "value" in result


def test_get_timetable_loads_and_returns_success() -> None:
    from agents.timetable import get_timetable

    result = get_timetable("all")
    assert result["status"] == "success"
    assert isinstance(result["value"], list)


def test_get_attendance_loads_and_returns_success() -> None:
    from agents.attendance import get_attendance

    result = get_attendance(subject="all")
    assert result["status"] == "success"
    assert isinstance(result["value"], list)


def test_get_exams_loads_and_returns_success() -> None:
    from agents.exams import get_exams

    result = get_exams("all")
    assert result["status"] == "success"
    assert isinstance(result["value"], list)


def test_get_fees_loads_and_returns_success() -> None:
    from agents.fees import get_fees

    result = get_fees("summary")
    assert result["status"] == "success"
    assert "total_fee" in result["value"]


def test_get_notices_loads_and_returns_success() -> None:
    from agents.notices import get_notices

    result = get_notices("all")
    assert result["status"] == "success"
    assert isinstance(result["value"], list)


# ---------------------------------------------------------------------------
# All tools importable from agents package
# ---------------------------------------------------------------------------


def test_all_tools_importable_from_agents_package() -> None:
    from agents import (
        get_attendance,
        get_exams,
        get_fees,
        get_notices,
        get_student_profile,
        get_timetable,
    )

    for tool_fn in (
        get_student_profile,
        get_timetable,
        get_attendance,
        get_exams,
        get_fees,
        get_notices,
    ):
        assert callable(tool_fn), f"{tool_fn.__name__} is not callable"


# ---------------------------------------------------------------------------
# Every tool wraps cleanly as a FunctionTool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path,func_name",
    [
        ("agents.student_profile", "get_student_profile"),
        ("agents.timetable", "get_timetable"),
        ("agents.attendance", "get_attendance"),
        ("agents.exams", "get_exams"),
        ("agents.fees", "get_fees"),
        ("agents.notices", "get_notices"),
    ],
)
def test_tool_wraps_as_function_tool(module_path: str, func_name: str) -> None:
    import importlib

    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    tool = FunctionTool(func=func)
    assert tool.name == func_name


# ---------------------------------------------------------------------------
# Root agent structure
# ---------------------------------------------------------------------------


def test_root_agent_importable_from_agents_package() -> None:
    from agents import root_agent

    assert root_agent is not None


def test_root_agent_is_llm_agent() -> None:
    from agents import root_agent

    assert isinstance(root_agent, LlmAgent)


def test_root_agent_has_six_tools() -> None:
    from agents import root_agent

    assert len(root_agent.tools) == 6


def test_root_agent_tool_names_match_expected() -> None:
    from agents import root_agent

    registered_names = {tool.name for tool in root_agent.tools}
    expected_names = {
        "get_student_profile",
        "get_timetable",
        "get_attendance",
        "get_exams",
        "get_fees",
        "get_notices",
    }
    assert registered_names == expected_names


def test_root_agent_uses_gemini_model() -> None:
    from agents import root_agent

    assert "gemini" in root_agent.model.lower()


def test_root_agent_has_instruction() -> None:
    from agents import root_agent

    assert root_agent.instruction
    assert "CampusOS" in root_agent.instruction


def test_build_root_agent_produces_independent_instances() -> None:
    from agents.root_agent import build_root_agent

    a = build_root_agent()
    b = build_root_agent()
    assert a is not b
    assert a.tools is not b.tools


# ---------------------------------------------------------------------------
# CampusOSRuntime initialisation (no live calls)
# ---------------------------------------------------------------------------


def test_campusos_runtime_initialises_without_error() -> None:
    from app import CampusOSRuntime

    runtime = CampusOSRuntime()
    assert runtime.runner is not None
    assert runtime.session_service is not None


def test_campusos_runtime_runner_uses_root_agent() -> None:
    from agents import root_agent
    from app import CampusOSRuntime

    runtime = CampusOSRuntime()
    assert runtime.runner.agent is root_agent


# ---------------------------------------------------------------------------
# Data files all exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative_path",
    [
        "data/student_profile.json",
        "data/timetable.json",
        "data/attendance.json",
        "data/exams.json",
        "data/fees.json",
        "data/notices.json",
    ],
)
def test_data_file_exists(relative_path: str) -> None:
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    assert (project_root / relative_path).exists(), f"Missing: {relative_path}"


# ---------------------------------------------------------------------------
# All tools return structured JSON (never raise) even on unknown queries
# ---------------------------------------------------------------------------


def test_all_tools_return_structured_json_on_unknown_query() -> None:
    from agents.attendance import get_attendance
    from agents.exams import get_exams
    from agents.fees import get_fees
    from agents.notices import get_notices
    from agents.student_profile import get_student_profile
    from agents.timetable import get_timetable

    results = [
        get_student_profile("nonexistent_field"),
        get_timetable("nonexistent_day"),
        get_attendance(subject="nonexistent_subject"),
        get_exams("nonexistent_query"),
        get_fees("nonexistent_query"),
        get_notices("nonexistent_query"),
    ]

    for result in results:
        assert isinstance(result, dict), "Tool must return a dict"
        assert "status" in result, "Tool result must have a 'status' key"
        assert result["status"] in {"success", "error"}