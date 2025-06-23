"""
CrewAI Memory Layers and Knowledge Base Example (Azure OpenAI)
Demonstrates all memory types and knowledge base integration in CrewAI
"""

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory
from crewai.knowledge import Document, KnowledgeBase
import os
from typing import List
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. KNOWLEDGE BASE SETUP
class ProjectKnowledgeBase:
    """Custom knowledge base for storing project information"""
    
    def __init__(self, base_path: str = "./knowledge"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.documents = []
    
    def add_document(self, title: str, content: str, metadata: dict = None):
        """Add a document to the knowledge base"""
        doc = Document(
            title=title,
            content=content,
            metadata=metadata or {},
            created_at=datetime.now().isoformat()
        )
        self.documents.append(doc)
        
        # Save to file
        doc_path = os.path.join(self.base_path, f"{title.replace(' ', '_')}.json")
        with open(doc_path, 'w') as f:
            json.dump({
                "title": doc.title,
                "content": doc.content,
                "metadata": doc.metadata,
                "created_at": doc.created_at
            }, f, indent=2)
        
        return doc
    
    def search(self, query: str) -> List[Document]:
        """Search documents in the knowledge base"""
        results = []
        query_lower = query.lower()
        
        for doc in self.documents:
            if query_lower in doc.title.lower() or query_lower in doc.content.lower():
                results.append(doc)
        
        return results
    
    def load_all(self):
        """Load all documents from disk"""
        for filename in os.listdir(self.base_path):
            if filename.endswith('.json'):
                with open(os.path.join(self.base_path, filename), 'r') as f:
                    data = json.load(f)
                    doc = Document(
                        title=data['title'],
                        content=data['content'],
                        metadata=data['metadata'],
                        created_at=data['created_at']
                    )
                    self.documents.append(doc)

# Initialize knowledge base
knowledge_base = ProjectKnowledgeBase()

# Add some initial knowledge
knowledge_base.add_document(
    "Python Best Practices",
    "Use type hints, follow PEP 8, write tests, use virtual environments",
    {"category": "programming", "language": "python"}
)

knowledge_base.add_document(
    "AI Project Guidelines",
    "Define clear objectives, prepare quality data, choose appropriate models, validate results",
    {"category": "ai", "domain": "machine learning"}
)

# 2. MEMORY-ENHANCED TOOLS
@tool("knowledge_search")
def search_knowledge(query: str) -> str:
    """Search the knowledge base for relevant information"""
    results = knowledge_base.search(query)
    if results:
        return f"Found {len(results)} documents:\n" + "\n".join([
            f"- {doc.title}: {doc.content[:100]}..." for doc in results
        ])
    return f"No documents found for query: {query}"

@tool("add_to_knowledge")
def add_to_knowledge(title: str, content: str) -> str:
    """Add new information to the knowledge base"""
    doc = knowledge_base.add_document(title, content)
    return f"Added document '{doc.title}' to knowledge base"

@tool("remember_context")
def remember_important_info(info: str) -> str:
    """Store important information for future reference"""
    # This simulates storing in agent's memory
    return f"Remembered: {info}"

# 3. MEMORY-AWARE AGENT CLASSES
class ResearchAgentWithMemory(Agent):
    """Research agent with enhanced memory capabilities"""
    
    def __init__(self):
        super().__init__(
            role="Senior Research Analyst",
            goal="Conduct thorough research using accumulated knowledge and memory",
            backstory="""You are an experienced researcher with perfect recall.
            You remember all previous research findings and build upon past knowledge.
            You actively use the knowledge base to store and retrieve information.""",
            tools=[search_knowledge, add_to_knowledge, remember_context],
            llm="azure/gpt-4o-mini",
            max_iter=3,
            max_retry_limit=2,
            memory=True,
            verbose=True
        )

class LearningAgent(Agent):
    """Agent that learns and improves over time"""
    
    def __init__(self):
        super().__init__(
            role="Learning Specialist",
            goal="Learn from interactions and improve responses over time",
            backstory="""You are an AI that gets smarter with each interaction.
            You remember past conversations, learn from mistakes, and continuously
            improve your knowledge base.""",
            tools=[search_knowledge, remember_context],
            llm="azure/gpt-4o-mini",
            max_iter=2,
            max_retry_limit=1,
            memory=True,
            verbose=True
        )

class KnowledgeManagerAgent(Agent):
    """Agent responsible for managing organizational knowledge"""
    
    def __init__(self):
        super().__init__(
            role="Knowledge Manager",
            goal="Organize, maintain, and enhance the collective knowledge base",
            backstory="""You are responsible for maintaining the organization's
            knowledge repository. You ensure information is properly stored,
            categorized, and easily retrievable.""",
            tools=[search_knowledge, add_to_knowledge],
            llm="azure/gpt-4o-mini",
            max_iter=2,
            max_retry_limit=1,
            memory=True,
            verbose=True
        )

# 4. MEMORY CONFIGURATION FOR CREW
class MemoryEnhancedCrew:
    """Crew with different memory configurations"""
    
    def __init__(self):
        self.research_agent = ResearchAgentWithMemory()
        self.learning_agent = LearningAgent()
        self.knowledge_manager = KnowledgeManagerAgent()
    
    def create_crew_with_short_term_memory(self, tasks: List[Task]):
        """Create a crew with short-term memory (session-based)"""
        return Crew(
            agents=[self.research_agent, self.learning_agent],
            tasks=tasks,
            process=Process.sequential,
            memory=ShortTermMemory(),  # Remembers within session
            verbose=True
        )
    
    def create_crew_with_long_term_memory(self, tasks: List[Task]):
        """Create a crew with long-term memory (persistent)"""
        return Crew(
            agents=[self.research_agent, self.learning_agent, self.knowledge_manager],
            tasks=tasks,
            process=Process.sequential,
            memory=LongTermMemory(
                storage_path="./crew_memory"  # Persists between sessions
            ),
            verbose=True
        )
    
    def create_crew_with_entity_memory(self, tasks: List[Task]):
        """Create a crew with entity memory (tracks specific entities)"""
        return Crew(
            agents=[self.research_agent, self.knowledge_manager],
            tasks=tasks,
            process=Process.sequential,
            memory=EntityMemory(
                storage_path="./entity_memory"  # Tracks entities like people, projects
            ),
            verbose=True
        )

# 5. EXAMPLE TASKS DEMONSTRATING MEMORY
def create_memory_demo_tasks():
    """Create tasks that demonstrate memory capabilities"""
    
    # Task 1: Initial Research
    research_task = Task(
        description="""Research Python programming best practices.
        Search the knowledge base first, then add any new findings.
        Remember key points for future reference.""",
        expected_output="Comprehensive research findings with memory notes",
        agent=ResearchAgentWithMemory()
    )
    
    # Task 2: Learning Task (builds on previous)
    learning_task = Task(
        description="""Based on the previous research, identify patterns
        and create learning guidelines. Reference what was discovered
        in the previous task without repeating the research.""",
        expected_output="Learning guidelines based on accumulated knowledge",
        agent=LearningAgent(),
        context=[research_task]  # Can access previous task's memory
    )
    
    # Task 3: Knowledge Organization
    organization_task = Task(
        description="""Review all findings and organize them in the knowledge base.
        Create a structured summary of everything learned. Update existing
        documents if needed.""",
        expected_output="Organized knowledge base with structured summary",
        agent=KnowledgeManagerAgent(),
        context=[research_task, learning_task]
    )
    
    return [research_task, learning_task, organization_task]

# 6. ADVANCED MEMORY PATTERNS
class ConversationalAgentWithMemory(Agent):
    """Agent that remembers conversation history"""
    
    def __init__(self):
        super().__init__(
            role="Conversational Assistant",
            goal="Maintain context across conversations and remember user preferences",
            backstory="""You are a helpful assistant who remembers past conversations,
            user preferences, and builds relationships over time.""",
            llm="azure/gpt-4o-mini",
            max_iter=1,
            max_retry_limit=1,
            memory=True,
            verbose=True
        )

def demonstrate_conversation_memory():
    """Show how agents remember across multiple interactions"""
    
    agent = ConversationalAgentWithMemory()
    
    # First conversation
    task1 = Task(
        description="The user's name is Alice and she likes Python programming",
        expected_output="Acknowledged user information",
        agent=agent
    )
    
    # Second conversation (agent should remember Alice)
    task2 = Task(
        description="What programming language does the user prefer?",
        expected_output="Correctly recall user's preference from memory",
        agent=agent
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task1, task2],
        process=Process.sequential,
        memory=ShortTermMemory(),
        verbose=True
    )
    
    return crew

# 7. USAGE EXAMPLES
if __name__ == "__main__":
    print("CrewAI Memory and Knowledge Base Demo")
    print("=" * 50)
    
    # Example 1: Short-term memory demo
    print("\n1. SHORT-TERM MEMORY DEMO")
    print("-" * 30)
    memory_crew = MemoryEnhancedCrew()
    tasks = create_memory_demo_tasks()
    
    short_term_crew = memory_crew.create_crew_with_short_term_memory(tasks[:2])
    # Uncomment to run: results = short_term_crew.kickoff()
    
    # Example 2: Long-term memory demo
    print("\n2. LONG-TERM MEMORY DEMO")
    print("-" * 30)
    long_term_crew = memory_crew.create_crew_with_long_term_memory(tasks)
    # Uncomment to run: results = long_term_crew.kickoff()
    
    # Example 3: Conversation memory
    print("\n3. CONVERSATION MEMORY DEMO")
    print("-" * 30)
    conversation_crew = demonstrate_conversation_memory()
    # Uncomment to run: results = conversation_crew.kickoff()
    
    # Example 4: Knowledge base search
    print("\n4. KNOWLEDGE BASE DEMO")
    print("-" * 30)
    print("Current knowledge base contents:")
    for doc in knowledge_base.documents:
        print(f"- {doc.title}")
    
    print("\nSearching for 'Python':")
    results = knowledge_base.search("Python")
    for doc in results:
        print(f"Found: {doc.title}")
    
    print("\n" + "=" * 50)
    print("Memory Types Summary:")
    print("- ShortTermMemory: Remembers within a session")
    print("- LongTermMemory: Persists across sessions")
    print("- EntityMemory: Tracks specific entities (people, projects, etc.)")
    print("\nKnowledge Base Features:")
    print("- Store documents with metadata")
    print("- Search across all stored knowledge")
    print("- Agents can add new knowledge during tasks")