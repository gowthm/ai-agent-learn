# 🌤️ AI Weather Agent with LangGraph

Welcome to the AI Weather Agent! This application is a fully functional conversational AI that dynamically retrieves the weather based on a user's ID using **LangChain**, **LangGraph**, and **Groq**. 

## 🚀 Setup & Preparation

Follow these steps to get everything running smoothly:

### 1. Install Dependencies
You will need to install the required Python packages:
```bash
pip install langchain langgraph langchain-groq python-dotenv
```

### 2. Environment Variables
Create a `.env` file in the root of your project and add your GROQ API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the Agent
Run the main script using Python. 
```bash
python weather_agent.py
```

---

## 📽️ How It Works (Agent Workflow "Animation")

The diagram below visually animates the step-by-step communication between you, the AI, and its dynamic tools during a conversation.

```mermaid
sequenceDiagram
    actor User
    participant Agent as AI Agent 🧠
    participant Tool as Tool (get_user_location) 🎒
    participant Weather as Tool (get_weather) 🌦️
    participant Memory as InMemorySaver 💾
    
    Note over User, Agent: Step 1: Provide User ID
    User->>Agent: Enter User ID (e.g., '1' or '2')
    Agent->>Memory: Submits Context(user_id) into Memory
    
    Note over User, Agent: Step 2: Ask a Question
    User->>Agent: "What is the weather outside?"
    
    Note over Agent: The AI realizes it needs <br/>to know the user's location
    
    Agent->>Tool: Call `get_user_location`
    Note over Tool: Opens Context Backpack <br/> finds `user_id` inside
    Tool-->>Agent: Returns "Florida" (if user=1) or "SF" (if user=2)
    
    Agent->>Weather: Call `get_weather_for_location` (SF/Florida)
    Weather-->>Agent: Returns "It's always sunny in [City]!"
    
    Agent->>User: "It's always sunny in Florida, so you don't even need to ask!"
    
    Note over User, Agent: Step 3: Accessing Memory
    User->>Agent: Type `memory`
    Memory-->>User: Dumps entire chat history onto terminal!
```

---

## 🎒 The "Backpack" Context Pattern
This project uses a powerful **Context Schema** to pass variables. 

Instead of polluting the AI's prompt with secret or internal system variables (like user IDs), we define a `@dataclass Context`, and safely pass it directly into our Tools behind the scenes! This makes the agent both **safe** and **dynamic**.
