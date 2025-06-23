"""
CrewAI with RAG and Vector Knowledge Base Example
Demonstrates integration with vector databases for enhanced knowledge retrieval
"""

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import os
import numpy as np
from typing import List, Tuple
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Simulated Vector Database (in production, use Pinecone, Weaviate, or Chroma)
class VectorKnowledgeBase:
    """Simulated vector database for RAG operations"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
    
    def add_document(self, text: str, metadata: dict = None):
        """Add document with simulated embedding"""
        # In production, use actual embedding model (e.g., OpenAI embeddings)
        embedding = np.random.rand(384)  # Simulated 384-dim embedding
        
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})
        
        return len(self.documents) - 1
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float, dict]]:
        """Search for similar documents using cosine similarity"""
        # In production, embed the query using the same model
        query_embedding = np.random.rand(384)
        
        if not self.embeddings:
            return []
        
        # Calculate similarities (simplified)
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            # Simplified similarity calculation
            similarity = np.random.random()  # In production, use cosine similarity
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        results = []
        for idx, score in similarities[:top_k]:
            results.append((
                self.documents[idx],
                score,
                self.metadata[idx]
            ))
        
        return results

# Initialize vector knowledge base
vector_kb = VectorKnowledgeBase()

# Add sample documents
vector_kb.add_document(
    "Machine learning models require quality training data for optimal performance.",
    {"topic": "ML", "category": "best_practices"}
)
vector_kb.add_document(
    "Python decorators are functions that modify other functions' behavior.",
    {"topic": "Python", "category": "programming"}
)
vector_kb.add_document(
    "Agile methodology emphasizes iterative development and continuous feedback.",
    {"topic": "Agile", "category": "project_management"}
)
vector_kb.add_document(
    "Neural networks consist of interconnected layers of artificial neurons.",
    {"topic": "AI", "category": "deep_learning"}
)

# RAG-enhanced tools
@tool("vector_search")
def search_vector_knowledge(query: str) -> str:
    """Search the vector knowledge base using semantic similarity"""
    results = vector_kb.search(query, top_k=3)
    
    if not results:
        return "No relevant documents found."
    
    response = f"Found {len(results)} relevant documents:\n\n"
    for i, (doc, score, meta) in enumerate(results, 1):
        response += f"{i}. [Score: {score:.2f}] {doc}\n"
        response += f"   Topic: {meta.get('topic', 'N/A')}, Category: {meta.get('category', 'N/A')}\n\n"
    
    return response

@tool("add_to_vector_kb")
def add_to_vector_knowledge(text: str, topic: str, category: str) -> str:
    """Add new document to vector knowledge base"""
    idx = vector_kb.add_document(
        text,
        {"topic": topic, "category": category, "source": "agent_contribution"}
    )
    return f"Added document to vector KB at index {idx}"

@tool("hybrid_search")
def hybrid_knowledge_search(query: str) -> str:
    """Combine vector search with keyword matching for better results"""
    # Vector search results
    vector_results = vector_kb.search(query, top_k=2)
    
    # Simple keyword search (in production, use BM25 or similar)
    keyword_results = []
    query_lower = query.lower()
    for i, doc in enumerate(vector_kb.documents):
        if query_lower in doc.lower():
            keyword_results.append((doc, vector_kb.metadata[i]))
    
    response = "Hybrid Search Results:\n\n"
    response += "Semantic Matches:\n"
    for doc, score, meta in vector_results:
        response += f"- {doc} (Score: {score:.2f})\n"
    
    response += "\nKeyword Matches:\n"
    for doc, meta in keyword_results[:2]:
        response += f"- {doc}\n"
    
    return response

# RAG-Enhanced Agent Classes
class RAGResearchAgent(Agent):
    """Research agent with RAG capabilities"""
    
    def __init__(self):
        super().__init__(
            role="RAG Research Specialist",
            goal="Conduct research using vector similarity search and retrieval-augmented generation",
            backstory="""You are an advanced research agent equipped with semantic search
            capabilities. You use vector embeddings to find the most relevant information
            and augment your responses with retrieved knowledge.""",
            tools=[vector_search, hybrid_search],
            llm="azure/gpt-4o-mini",
            max_iter=3,
            max_retry_limit=2,
            verbose=True
        )

class KnowledgeContributorAgent(Agent):
    """Agent that contributes to the knowledge base"""
    
    def __init__(self):
        super().__init__(
            role="Knowledge Contributor",
            goal="Expand the knowledge base with new, verified information",
            backstory="""You are responsible for adding high-quality information to the
            vector knowledge base. You ensure all contributions are accurate and well-categorized.""",
            tools=[vector_search, add_to_vector_kb],
            llm="azure/gpt-4o-mini",
            max_iter=2,
            max_retry_limit=1,
            verbose=True
        )

class QAAgent(Agent):
    """Question-answering agent using RAG"""
    
    def __init__(self):
        super().__init__(
            role="Q&A Specialist",
            goal="Answer questions accurately using the knowledge base",
            backstory="""You excel at answering questions by retrieving relevant context
            from the knowledge base and providing comprehensive, accurate answers.""",
            tools=[vector_search, hybrid_search],
            llm="azure/gpt-4o-mini",
            max_iter=2,
            max_retry_limit=1,
            verbose=True
        )

# Advanced RAG Patterns
class ChainOfThoughtRAGAgent(Agent):
    """Agent that uses chain-of-thought with RAG"""
    
    def __init__(self):
        super().__init__(
            role="Analytical Researcher",
            goal="Solve complex problems using step-by-step reasoning with knowledge retrieval",
            backstory="""You break down complex problems into steps, retrieving relevant
            information for each step to build comprehensive solutions.""",
            tools=[vector_search],
            llm="azure/gpt-4o",
            max_iter=3,
            max_retry_limit=2,
            verbose=True
        )

# Example Tasks
def create_rag_demo_tasks():
    """Create tasks demonstrating RAG capabilities"""
    
    # Research task using vector search
    research_task = Task(
        description="""Research machine learning best practices.
        Use vector search to find relevant information, then summarize
        the key points for beginners.""",
        expected_output="Summary of ML best practices based on knowledge base",
        agent=RAGResearchAgent()
    )
    
    # Knowledge expansion task
    contribution_task = Task(
        description="""Based on the research findings, contribute a new document
        about 'Feature Engineering in Machine Learning' to the knowledge base.
        Ensure it's properly categorized.""",
        expected_output="New knowledge base entry created",
        agent=KnowledgeContributorAgent(),
        context=[research_task]
    )
    
    # Q&A task
    qa_task = Task(
        description="""Answer this question using the knowledge base:
        'What are the key principles of neural networks and how do they
        relate to machine learning best practices?'""",
        expected_output="Comprehensive answer based on retrieved knowledge",
        agent=QAAgent()
    )
    
    return [research_task, contribution_task, qa_task]

# RAG Crew Configuration
class RAGKnowledgeCrew:
    """Crew configured for RAG operations"""
    
    def __init__(self):
        self.rag_researcher = RAGResearchAgent()
        self.contributor = KnowledgeContributorAgent()
        self.qa_specialist = QAAgent()
        self.analyst = ChainOfThoughtRAGAgent()
    
    def create_research_crew(self):
        """Create a crew for research tasks"""
        tasks = create_rag_demo_tasks()
        
        return Crew(
            agents=[self.rag_researcher, self.contributor, self.qa_specialist],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
    
    def create_analysis_crew(self, question: str):
        """Create a crew for complex analysis"""
        
        # Break down the problem
        breakdown_task = Task(
            description=f"""Break down this question into sub-questions:
            '{question}'
            
            For each sub-question, search the knowledge base for relevant information.""",
            expected_output="Structured breakdown with relevant knowledge",
            agent=self.analyst
        )
        
        # Synthesize findings
        synthesis_task = Task(
            description="""Based on the breakdown and retrieved information,
            synthesize a comprehensive answer that addresses all aspects
            of the original question.""",
            expected_output="Complete answer with supporting evidence",
            agent=self.qa_specialist,
            context=[breakdown_task]
        )
        
        return Crew(
            agents=[self.analyst, self.qa_specialist],
            tasks=[breakdown_task, synthesis_task],
            process=Process.sequential,
            verbose=True
        )

# Usage Example
if __name__ == "__main__":
    print("RAG-Enhanced CrewAI Demo")
    print("=" * 50)
    
    # Show current knowledge base
    print(f"\nCurrent Knowledge Base: {len(vector_kb.documents)} documents")
    for i, (doc, meta) in enumerate(zip(vector_kb.documents, vector_kb.metadata)):
        print(f"{i+1}. [{meta.get('topic', 'N/A')}] {doc[:60]}...")
    
    # Example 1: Basic RAG search
    print("\n\nExample 1: Vector Search Demo")
    print("-" * 40)
    results = vector_search("machine learning training")
    print(results)
    
    # Example 2: Research crew
    print("\n\nExample 2: RAG Research Crew")
    print("-" * 40)
    rag_crew = RAGKnowledgeCrew()
    research_crew = rag_crew.create_research_crew()
    # Uncomment to run: research_crew.kickoff()
    
    # Example 3: Complex analysis
    print("\n\nExample 3: Complex Analysis with RAG")
    print("-" * 40)
    complex_question = "How do agile methodologies apply to ML projects?"
    analysis_crew = rag_crew.create_analysis_crew(complex_question)
    # Uncomment to run: analysis_crew.kickoff()
    
    print("\n\nRAG Benefits:")
    print("1. Semantic search finds conceptually similar content")
    print("2. Reduces hallucinations by grounding responses in retrieved knowledge")
    print("3. Knowledge base can be continuously expanded")
    print("4. Supports hybrid search (semantic + keyword)")
    print("5. Enables fact-checking and source attribution")