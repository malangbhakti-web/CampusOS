from agents.orchestrator import OrchestratorAgent

agent = OrchestratorAgent()

print("=" * 45)
print("🎓 Welcome to CampusOS")
print("=" * 45)

while True:
    user = input("\nYou : ")

    if user.lower() == "exit":
        print("\n👋 Goodbye!")
        break

    response = agent.handle_request(user)
    print("\nCampusOS :", response)