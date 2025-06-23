"""
CrewAI Simple Demo - Running Example for Students
A simplified, ready-to-run example demonstrating CrewAI basics
"""

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Simple tool example
@tool("research_tool")
def research_tool(topic: str) -> str:
    """Simulates researching a topic"""
    knowledge_base = {
        "python": "Python is a versatile programming language created in 1991.",
        "ai": "AI enables machines to mimic human intelligence.",
        "web": "Web development involves creating websites and web applications.",
    }
    
    topic_lower = topic.lower()
    for key, info in knowledge_base.items():
        if key in topic_lower:
            return info
    return f"General information about {topic}"

# Agent Classes following the pattern
class ResearchAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Researcher',
            goal='Research topics thoroughly',
            backstory='You are a curious researcher who loves learning new things.',
            tools=[research_tool],
            # llm="azure/gpt-4o-mini",
            llm=LLM(
                model="ollama/llama3.2:1b",
                base_url="http://localhost:11434"
            ),
            max_iter=1,
            max_retry_limit=1,
            verbose=True
        )

class AnalystAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Analyst',
            goal='Analyze research findings',
            backstory='You excel at finding patterns and insights in information.',
            llm=LLM(
                model="ollama/llama3.2:1b",
                base_url="http://localhost:11434"
            ),
            max_iter=1,
            max_retry_limit=1,
            verbose=True
        )

class WriterAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Writer',
            goal='Create clear summaries',
            backstory='You write concise and easy-to-understand content.',
            llm="azure/gpt-4o-mini",
            max_iter=1,
            max_retry_limit=1,
            verbose=True
        )

# Initialize agents
researcher = ResearchAgent()
analyst = AnalystAgent()
writer = WriterAgent()

# Create tasks for each agent
task1 = Task(
    description='Research Python programming and AI topics',
    expected_output='Key facts about Python and AI',
    agent=researcher
)

task2 = Task(
    description='Analyze how Python relates to AI development',
    expected_output='Analysis of Python\'s role in AI',
    agent=analyst,
    context=[task1]  # Depends on task1
)

task3 = Task(
    description='Write a brief summary about Python and AI for beginners',
    expected_output='A 2-3 sentence beginner-friendly summary',
    agent=writer,
    context=[task1, task2]  # Depends on both previous tasks
)

# Create and run the crew
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[task1, task2, task3],
    process=Process.sequential,  # Run tasks in order
    verbose=True
)

# To run this example:
# 1. Install: pip install crewai langchain-openai
# 2. Set Azure OpenAI environment variables (see top of file)
# 3. Run: python crewai_simple_demo.py

if __name__ == "__main__":
    print("Starting CrewAI Demo...\n")
    result = crew.kickoff()
    print(f"\nFinal Result:\n{result}")