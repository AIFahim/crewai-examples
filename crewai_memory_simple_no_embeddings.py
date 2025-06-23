"""
CrewAI Memory Example - Without Embeddings
Simple demonstration of agent memory without requiring vector embeddings
"""

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Simple in-memory storage
conversation_history = []

@tool("log_interaction")
def log_interaction(message: str) -> str:
    """Log an interaction to memory"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    conversation_history.append(entry)
    return f"Logged: {entry}"

@tool("recall_history")
def recall_history() -> str:
    """Recall conversation history"""
    if not conversation_history:
        return "No conversation history available"
    return "Conversation History:\n" + "\n".join(conversation_history[-5:])  # Last 5 entries

# Agent class with memory capabilities
class ConversationAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Conversational Assistant",
            goal="Maintain context and remember important information from our conversation",
            backstory="""You are a helpful assistant who pays attention to details
            and remembers important information shared during conversations.""",
            tools=[log_interaction, recall_history],
            llm="azure/gpt-4o-mini",
            max_iter=1,
            max_retry_limit=1,
            verbose=True
        )

# Initialize agent
agent = ConversationAgent()

# Example tasks showing memory in action
print("Memory Demonstration - Simple Conversation")
print("=" * 50)

# Task 1: Introduction
intro_task = Task(
    description="""The user says: 'Hi, I'm Alice and I work as a data scientist. 
    I'm interested in machine learning.' 
    Log this information and greet the user.""",
    expected_output="A greeting acknowledging the user's introduction",
    agent=agent
)

# Task 2: Follow-up using memory
followup_task = Task(
    description="""The user asks: 'Can you recommend some resources for my field?'
    Use the conversation history to remember what field they work in.""",
    expected_output="Recommendations based on the user's field from memory",
    agent=agent
)

# Task 3: Recall conversation
summary_task = Task(
    description="""Summarize what you've learned about the user from this conversation.
    Use the recall history tool.""",
    expected_output="A summary of key information about the user",
    agent=agent
)

# Create crew without memory storage (using agent's tools instead)
crew = Crew(
    agents=[agent],
    tasks=[intro_task, followup_task, summary_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print("\nStarting conversation with memory...\n")
    
    # Run the crew
    result = crew.kickoff()
    
    print("\n" + "=" * 50)
    print("Final Result:")
    print(result)
    
    print("\n" + "=" * 50)
    print("Complete Conversation History:")
    for entry in conversation_history:
        print(entry)