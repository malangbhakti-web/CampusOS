from agents.orchestrator import OrchestratorAgent
from agents import root_agent


class DummyRunner:
    def __init__(self, agent):
        self.agent = agent


class DummySessionService:
    pass


class CampusOSRuntime:
    def __init__(self):
        self.runner = DummyRunner(root_agent)
        self.session_service = DummySessionService()


def main():
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

        print("\nCampusOS :")

        if isinstance(response, dict):

            if response.get("status") == "success":

                if "value" in response:
                    value = response["value"]

                    if isinstance(value, dict):
                        for k, v in value.items():
                            print(f"{k.title()}: {v}")

                    elif isinstance(value, list):
                        for item in value:
                            print("•", item)

                    else:
                        print(value)

                elif "profile" in response:
                    for k, v in response["profile"].items():
                        print(f"{k.title()}: {v}")

            else:
                print(response.get("message") or response.get("error_message"))

        else:
            print(response)


if __name__ == "__main__":
    main()