"""
CrewAI Study Example - Comprehensive Agent Features (Azure OpenAI)
This example demonstrates all key features of CrewAI agents using Azure OpenAI.
"""

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI Configuration is loaded from .env file
# No need to set environment variables here

# 1. CUSTOM TOOLS - Define tools that agents can use
@tool("search_tool")
def search_information(query: str) -> str:
    """
    A simple search tool that simulates searching for information.
    In real applications, this could connect to APIs, databases, etc.
    """
    # Simulated search results
    results = {
        "python": "Python is a high-level programming language known for its simplicity.",
        "ai": "Artificial Intelligence is the simulation of human intelligence by machines.",
        "crewai": "CrewAI is a framework for orchestrating role-playing AI agents.",
    }
    
    query_lower = query.lower()
    for key, value in results.items():
        if key in query_lower:
            return f"Search Result: {value}"
    
    return f"No specific information found for: {query}"

@tool("calculator")
def calculate(expression: str) -> str:
    """
    A simple calculator tool for basic math operations.
    """
    try:
        result = eval(expression)
        return f"Calculation result: {result}"
    except:
        return "Invalid mathematical expression"

# 2. AGENT DEFINITIONS - Create agents with different roles and capabilities

# Agent Classes following the production pattern
class ResearchSpecialistAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Research Specialist',
            goal='Find and analyze information on various topics',
            backstory="""You are an experienced researcher with a keen eye for detail.
            You excel at finding relevant information and presenting it clearly.
            You always verify facts and provide comprehensive insights.""",
            tools=[search_information],
            llm="azure/gpt-4o-mini",
            max_iter=3,
            max_retry_limit=2,
            memory=True,
            allow_delegation=True,
            verbose=True
        )

class DataAnalystAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Data Analyst',
            goal='Analyze information and provide meaningful insights',
            backstory="""You are a skilled data analyst who loves working with numbers
            and finding patterns. You can break down complex information into
            understandable insights.""",
            tools=[calculate],
            llm="azure/gpt-4o-mini",
            max_iter=2,
            max_retry_limit=1,
            allow_delegation=False,
            verbose=True
        )

class ContentWriterAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Content Writer',
            goal='Create clear and engaging content based on research and analysis',
            backstory="""You are a talented writer who can transform complex information
            into easy-to-understand content. You focus on clarity and engagement.""",
            tools=[],
            llm="azure/gpt-4o-mini",
            max_iter=2,
            max_retry_limit=1,
            allow_delegation=True,
            verbose=True
        )

class ProjectManagerAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Project Manager',
            goal='Coordinate the team to deliver high-quality results',
            backstory="""You are an experienced project manager who ensures smooth
            collaboration between team members. You delegate tasks effectively
            and ensure project success.""",
            tools=[],
            llm="azure/gpt-4o",
            max_iter=1,
            max_retry_limit=1,
            allow_delegation=True,
            verbose=True
        )

# Initialize agents
research_agent = ResearchSpecialistAgent()
analysis_agent = DataAnalystAgent()
writer_agent = ContentWriterAgent()
manager_agent = ProjectManagerAgent()

# 3. TASK DEFINITIONS - Create tasks for agents to complete

# Task 1: Research Task
research_task = Task(
    description="""Research information about Python programming language
    and artificial intelligence. Find key facts and interesting details
    that would be useful for students.""",
    expected_output="""A comprehensive summary of findings about Python
    and AI, including key features, applications, and relevance.""",
    agent=research_agent,
    tools=[search_information],
)

# Task 2: Analysis Task
analysis_task = Task(
    description="""Analyze the research findings and calculate some metrics:
    1. Calculate the year Python was created (1991) to current year difference
    2. Provide insights on the relationship between Python and AI""",
    expected_output="""An analysis report with calculations and insights
    about Python's age and its relationship with AI development.""",
    agent=analysis_agent,
    context=[research_task],  # This task depends on research_task
    tools=[calculate],
)

# Task 3: Content Creation Task
writing_task = Task(
    description="""Based on the research and analysis, create a brief
    educational article (3-4 paragraphs) about Python and AI for students.
    Make it engaging and easy to understand.""",
    expected_output="""A well-written educational article suitable for
    students learning about Python and AI.""",
    agent=writer_agent,
    context=[research_task, analysis_task],  # Depends on both previous tasks
)

# Task 4: Review Task
review_task = Task(
    description="""Review all the work done by the team and provide
    a final summary of the project outcomes. Ensure quality and completeness.""",
    expected_output="""A final project summary highlighting key deliverables
    and team performance.""",
    agent=manager_agent,
    context=[research_task, analysis_task, writing_task],
)

# 4. CREW ASSEMBLY - Different ways to organize your agents

# Example 1: Sequential Process (tasks run one after another)
sequential_crew = Crew(
    agents=[research_agent, analysis_agent, writer_agent, manager_agent],
    tasks=[research_task, analysis_task, writing_task, review_task],
    process=Process.sequential,  # Tasks run in order
    verbose=True,  # Show detailed execution logs
    max_rpm=10,  # Rate limiting for API calls
    share_crew=True,  # Agents can share context
)

# Example 2: Hierarchical Process (manager delegates tasks)
hierarchical_crew = Crew(
    agents=[manager_agent, research_agent, analysis_agent, writer_agent],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.hierarchical,
    manager_llm="azure/gpt-4o-mini",  # Manager's LLM
    verbose=True,
)

# 5. ADVANCED FEATURES DEMONSTRATION

# Additional specialized agents for advanced examples
class TimeConstrainedAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Fast Worker',
            goal='Complete tasks quickly and efficiently',
            backstory='You work under tight deadlines and must be efficient.',
            llm="azure/gpt-4o-mini",
            max_iter=1,
            max_retry_limit=1,
            execution_timeout=30,  # 30 seconds timeout
            verbose=True
        )

class SpecializedPythonAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Python Specialist',
            goal='Provide expert Python programming advice',
            backstory='You are a Python expert with 10 years of experience.',
            system_template="""You are a Python programming expert.
            Always provide code examples when explaining concepts.
            Focus on best practices and clean code principles.""",
            llm="azure/gpt-4o",
            max_iter=2,
            max_retry_limit=1,
            verbose=True
        )

# Initialize specialized agents
time_sensitive_agent = TimeConstrainedAgent()
specialized_agent = SpecializedPythonAgent()

# 6. EXECUTION EXAMPLES

def run_basic_example():
    """Run the basic sequential crew example"""
    print("=" * 50)
    print("Running Basic Sequential CrewAI Example")
    print("=" * 50)
    
    # Execute the crew
    result = sequential_crew.kickoff()
    
    print("\n" + "=" * 50)
    print("FINAL RESULT:")
    print("=" * 50)
    print(result)
    
    return result

def run_with_custom_inputs():
    """Run crew with custom inputs"""
    print("=" * 50)
    print("Running CrewAI with Custom Inputs")
    print("=" * 50)
    
    # Create a task with input variables
    custom_task = Task(
        description="""Research about {topic} and create a summary
        focusing on {focus_area}.""",
        expected_output="A detailed summary of the research findings",
        agent=research_agent,
    )
    
    custom_crew = Crew(
        agents=[research_agent],
        tasks=[custom_task],
        process=Process.sequential,
    )
    
    # Execute with inputs
    result = custom_crew.kickoff(
        inputs={
            "topic": "Machine Learning",
            "focus_area": "practical applications"
        }
    )
    
    return result

def demonstrate_callbacks():
    """Demonstrate callback functionality"""
    
    def task_callback(task_output):
        """Called when a task completes"""
        print(f"\n[CALLBACK] Task completed: {task_output.task.description[:50]}...")
        print(f"[CALLBACK] Output preview: {task_output.raw[:100]}...")
    
    # Create crew with callbacks
    callback_crew = Crew(
        agents=[research_agent, writer_agent],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        task_callback=task_callback,  # Called after each task
        verbose=True,
    )
    
    return callback_crew.kickoff()

# 7. MAIN EXECUTION

if __name__ == "__main__":
    # Choose which example to run
    print("CrewAI Study Examples")
    print("1. Basic Sequential Example")
    print("2. Custom Inputs Example")
    print("3. Callback Demonstration")
    
    # For study purposes, we'll run the basic example
    # Uncomment the line below to run:
    result = run_basic_example()
    

    
    print("\nThis example demonstrates:")
    print("- Agent creation with roles, goals, and backstories")
    print("- Custom tool creation and assignment")
    print("- Task definition with dependencies")
    print("- Different crew processes (sequential, hierarchical)")
    print("- Advanced features like custom LLMs, timeouts, and callbacks")
    print("- Memory and delegation capabilities")