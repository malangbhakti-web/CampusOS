from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from agents.student_profile import get_student_profile
from agents.timetable import get_timetable
from agents.attendance import get_attendance
from agents.exams import get_exams
from agents.fees import get_fees
from agents.notices import get_notices


def build_root_agent():
    return LlmAgent(
        name="CampusOS",
        model="gemini-2.0-flash",
        description="CampusOS Root Agent",
        instruction="""
You are CampusOS.

You help students with:
- Student Profile
- Timetable
- Attendance
- Exams
- Fees
- Notices

Always use the appropriate tool.
""",
        tools=[
            FunctionTool(func=get_student_profile),
            FunctionTool(func=get_timetable),
            FunctionTool(func=get_attendance),
            FunctionTool(func=get_exams),
            FunctionTool(func=get_fees),
            FunctionTool(func=get_notices),
        ],
    )


root_agent = build_root_agent()