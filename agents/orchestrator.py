class OrchestratorAgent:
    def __init__(self):
        self.name = "CampusOS Orchestrator"

    def handle_request(self, user_input):
        print(f"\n🎓 {self.name}")
        print("=" * 40)

        if "study" in user_input.lower():
            return "📚 I'll prepare a study plan for you."

        elif "schedule" in user_input.lower():
            return "🗓️ I'll organize your daily schedule."

        elif "health" in user_input.lower():
            return "💧 Don't forget to drink water and take breaks."

        else:
            return "🤖 I understand your request. More agents will handle it soon."