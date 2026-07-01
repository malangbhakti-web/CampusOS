from agents.student_profile import get_student_profile
from agents.timetable import get_timetable
from agents.attendance import get_attendance
from agents.exams import get_exams
from agents.fees import get_fees
from agents.notices import get_notices


class OrchestratorAgent:
    def __init__(self):
        self.name = "CampusOS Orchestrator"

    def handle_request(self, user_input):
        print(f"\n🎓 {self.name}")
        print("=" * 40)

        query = user_input.lower().strip()

        # Student Profile
        if any(x in query for x in [
            "who am i",
            "profile",
            "my profile",
            "name",
            "department",
            "semester",
            "cgpa",
            "email",
            "enrollment",
            "roll"
        ]):

            if "department" in query:
                return get_student_profile("department")

            elif "semester" in query:
                return get_student_profile("semester")

            elif "cgpa" in query:
                return get_student_profile("cgpa")

            elif "email" in query:
                return get_student_profile("email")

            elif "enrollment" in query or "roll" in query:
                return get_student_profile("enrollment_number")

            elif "subject" in query or "course" in query:
                return get_student_profile("subjects")

            elif "name" in query:
                return get_student_profile("name")

            else:
                return get_student_profile()

        # Timetable
        elif any(x in query for x in [
            "timetable",
            "time table",
            "schedule",
            "class"
        ]):

            if "today" in query:
                return get_timetable("today")

            elif "tomorrow" in query:
                return get_timetable("tomorrow")

            elif "next" in query:
                return get_timetable("next_class")

            elif "current" in query or "now" in query:
                return get_timetable("current_class")

            else:
                return get_timetable("all")

        # Attendance
        elif "attendance" in query:

            if "os" in query:
                return get_attendance(subject="Operating Systems")

            elif "dbms" in query:
                return get_attendance(subject="Database Management Systems")

            elif "network" in query:
                return get_attendance(subject="Computer Networks")

            elif "software" in query:
                return get_attendance(subject="Software Engineering")

            elif "low" in query:
                return get_attendance(query="low")

            else:
                return get_attendance()

        # Exams
        elif "exam" in query:

            if "today" in query:
                return get_exams("today")

            elif "completed" in query:
                return get_exams("completed")

            elif "upcoming" in query:
                return get_exams("upcoming")

            else:
                return get_exams()

        # Fees
        elif any(x in query for x in [
            "fee",
            "fees",
            "payment",
            "pending"
        ]):

            if "pending" in query:
                return get_fees("pending")

            elif "status" in query:
                return get_fees("status")

            elif "summary" in query:
                return get_fees("summary")

            else:
                return get_fees()

        # Notices
        elif any(x in query for x in [
            "notice",
            "announcement",
            "circular"
        ]):

            if "latest" in query or "recent" in query:
                return get_notices("latest")

            else:
                return get_notices()

        # Unknown
        else:
            return {
                "status": "error",
                "message": "Sorry, I couldn't understand your request."
            }