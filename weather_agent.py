from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# Define system prompt
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# Define tools
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

# Configure model
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Set up memory
checkpointer = InMemorySaver()

# Create agent (no response_format — Groq/Llama doesn't support nested tool calls)
agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    checkpointer=checkpointer
)

# Run agent interactively
config = {"configurable": {"thread_id": "1"}}

active_user_id = input("Please enter your User ID (e.g., 1 or 2): ").strip()
if not active_user_id:
    active_user_id = "1" # Default fallback
    
print(f"\nWelcome User {active_user_id}!")
print("Weather Agent ready! Type a city name or ask about your local weather.")
print("Examples: 'what is the weather in Tokyo?' or 'what is the weather outside?'")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input or user_input.lower() == "quit":
        break
        
    if user_input.lower() == "memory":
        state = agent.get_state(config)
        print("\n--- Memory Dump ---")
        for msg in state.values.get("messages", []):
            # Print the role and content of each message
            print(f"[{msg.type.upper()}]: {msg.content}")
        print("-------------------\n")
        continue

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        context=Context(user_id=active_user_id)
    )

    print(f"\nAgent: {response['messages'][-1].content}\n")