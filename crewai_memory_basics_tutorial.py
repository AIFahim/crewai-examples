"""
CrewAI Memory Layers - Simple Demo for Students
Shows how agents remember information and learn from past interactions
"""

from crewai import Agent, Task, Crew, Process, LLM
from crewai.memory.unified_memory import Memory
from crewai.tools import tool
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Local Ollama model used by every agent in this example
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def ollama_llm(model: str = OLLAMA_MODEL) -> LLM:
    """Build an LLM pointed at the local Ollama server."""
    return LLM(model=f"ollama/{model}", base_url=OLLAMA_BASE_URL)


# Memory needs BOTH an embedder (for vector search) and an LLM (for the
# extraction/recall analysis). Both default to OpenAI, so point each at Ollama
# and share one store across every agent and crew below.
OLLAMA_EMBEDDER = {
    "provider": "ollama",
    "config": {
        "url": f"{OLLAMA_BASE_URL}/api/embeddings",
        "model_name": OLLAMA_EMBED_MODEL,
    },
}

MEMORY = Memory(llm=f"ollama/{OLLAMA_MODEL}", embedder=OLLAMA_EMBEDDER)

# Simple memory storage tool
memory_store = {}

@tool("store_memory")
def store_information(key: str, value: str) -> str:
    """Store information in memory for later use"""
    memory_store[key] = {
        "value": value,
        "timestamp": datetime.now().isoformat()
    }
    return f"Stored '{key}' in memory"

@tool("recall_memory")
def recall_information(key: str) -> str:
    """Recall information from memory"""
    if key in memory_store:
        info = memory_store[key]
        return f"Recalled: {info['value']} (stored at {info['timestamp']})"
    return f"No memory found for '{key}'"

# Agent Classes with Memory
class MemoryAssistantAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Memory Assistant",
            goal="Remember important information and use it in future tasks",
            backstory="You have an excellent memory and never forget important details.",
            tools=[store_information, recall_information],
            llm=ollama_llm(),
            max_iter=1,
            max_retry_limit=1,
            memory=MEMORY,
            verbose=True
        )

class LearningAssistantAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Learning Assistant",
            goal="Learn from past experiences and improve responses",
            backstory="You continuously learn and adapt based on previous interactions.",
            llm=ollama_llm(),
            max_iter=1,
            max_retry_limit=1,
            memory=MEMORY,
            verbose=True
        )

# Initialize agents
memory_agent = MemoryAssistantAgent()
learning_agent = LearningAssistantAgent()

# Example 1: Basic Memory Usage
print("EXAMPLE 1: Basic Memory Usage")
print("-" * 40)

# Task 1: Store information
store_task = Task(
    description="Store the following information: The client's name is John Smith and he prefers email communication.",
    expected_output="Confirmation that information was stored",
    agent=memory_agent
)

# Task 2: Use stored information
recall_task = Task(
    description="How should we contact the client? What is their name?",
    expected_output="Correct client name and communication preference from memory",
    agent=memory_agent
)

# Create crew with memory enabled
basic_crew = Crew(
    agents=[memory_agent],
    tasks=[store_task, recall_task],
    process=Process.sequential,
    memory=MEMORY,  # Enable memory for the crew
    verbose=True
)

# Example 2: Learning from Context
print("\nEXAMPLE 2: Learning from Context")
print("-" * 40)

# Series of related tasks where agent learns
task1 = Task(
    description="Analyze this error: 'TypeError: unsupported operand type(s)'",
    expected_output="Error analysis",
    agent=learning_agent
)

task2 = Task(
    description="The error was in line: result = '5' + 5. Fix it.",
    expected_output="Solution based on previous analysis",
    agent=learning_agent
)

task3 = Task(
    description="What type of error did we just fix and how to prevent it?",
    expected_output="Summary of learning from previous tasks",
    agent=learning_agent
)

learning_crew = Crew(
    agents=[learning_agent],
    tasks=[task1, task2, task3],
    process=Process.sequential,
    memory=MEMORY,  # Enable memory
    verbose=True
)

# Example 3: Persistent Memory (Long-term)
print("\nEXAMPLE 3: Persistent Memory")
print("-" * 40)

class PersistentProjectAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Project Manager",
            goal="Manage project information across multiple sessions",
            backstory="You maintain project continuity across different work sessions.",
            llm=ollama_llm(),
            max_iter=1,
            max_retry_limit=1,
            memory=MEMORY,
            verbose=True
        )

persistent_agent = PersistentProjectAgent()

project_task = Task(
    description="Current project status: Phase 1 completed, Phase 2 starting next week. Budget: $50,000 remaining.",
    expected_output="Project status recorded",
    agent=persistent_agent
)

# This crew uses memory
persistent_crew = Crew(
    agents=[persistent_agent],
    tasks=[project_task],
    process=Process.sequential,
    memory=MEMORY,  # Enable memory
    verbose=True
)

# Run examples (uncomment to execute)
if __name__ == "__main__":
    print("\nRunning Memory Examples...")
    print("=" * 50)
    
    # Run basic memory example
    basic_crew.kickoff()

    # Run learning example
    learning_crew.kickoff()

    # Run persistent memory example
    persistent_crew.kickoff()
    
    print("\nKey Concepts:")
    print("1. Agents with memory=True can remember information")
    print("2. Crews with memory=True enable memory for all agents")
    print("3. Agents can use tools to store and recall information")
    print("4. Agents can build on previous task outcomes")
    print("5. Memory helps maintain context and improve responses")