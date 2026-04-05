from dotenv import load_dotenv
load_dotenv()

from langgraph.prebuilt import create_react_agent
# from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain_groq import ChatGroq

# 1. Define tool with @tool decorator
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    print("getting weather....")
    return f"It's always sunny in {city}!"

@tool
def calculate(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    print("calculating....")
    return a + b

# 2. Create LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# 3. Create agent
agent = create_react_agent(
    model=llm,
    tools=[get_weather,calculate],
    prompt="""You are a helpful assistant. If you need to use a tool, use it. 
    Otherwise, answer the question. and if you need to use both tools, use both tools.
    """
)

# 4. Run it
result = agent.invoke({
    "messages": [{"role": "user", "content": "what is the weather in sf and what is the sum of 2 and 3?"}]
})

print(result["messages"][-1].content)