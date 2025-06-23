# CrewAI Examples 🤖

A collection of educational examples demonstrating CrewAI features, from basic concepts to advanced implementations.

## 🌟 Features

- **Progressive Learning Path**: Start with basics and advance to complex patterns
- **Multiple LLM Support**: Examples for both Azure OpenAI and Ollama (local)
- **Memory Systems**: Demonstrations of agent memory and knowledge persistence
- **RAG Implementation**: Vector search and retrieval-augmented generation
- **Production Patterns**: Class-based agent design following best practices

## 📚 Examples Overview

### 1. [Basic Beginner Example](crewai_basic_beginner_example.py)
Simple introduction to CrewAI concepts with three agents working together.

### 2. [Comprehensive Features Tutorial](crewai_comprehensive_features_tutorial.py)
Complete overview of all CrewAI features including tools, delegation, and different crew processes.

### 3. [Memory Basics Tutorial](crewai_memory_basics_tutorial.py)
Introduction to agent memory capabilities and information persistence.

### 4. [Advanced Memory & Knowledge System](crewai_advanced_memory_knowledge_system.py)
Complex memory layers, knowledge base management, and entity tracking.

### 5. [RAG Vector Search Example](crewai_rag_vector_search_example.py)
Implementation of Retrieval-Augmented Generation with vector search capabilities.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda for package management
- (Optional) Ollama for local LLM execution

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/AIFahim/crewai-examples.git
cd crewai-examples
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your API keys or Ollama settings
```

### Configuration Options

#### Option 1: Azure OpenAI
```env
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
```

#### Option 2: Ollama (Local)
```env
OLLAMA_MODEL=llama3.2:1b
OLLAMA_BASE_URL=http://localhost:11434
```

## 🎯 Running Examples

Start with the basic example:
```bash
python crewai_basic_beginner_example.py
```

Progress through more complex examples:
```bash
python crewai_comprehensive_features_tutorial.py
python crewai_memory_basics_tutorial.py
```

## 📝 Code Structure

All examples follow a consistent pattern:

```python
from crewai import Agent, Task, Crew, Process

# Define agent classes
class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Agent Role",
            goal="What the agent aims to achieve",
            backstory="Context that shapes behavior",
            llm="azure/gpt-4o-mini",  # or Ollama config
            max_iter=1,
            max_retry_limit=1,
            verbose=True
        )

# Create tasks and crews
agent = MyAgent()
task = Task(
    description="What needs to be done",
    expected_output="What the result should look like",
    agent=agent
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential,
    verbose=True
)

# Execute
result = crew.kickoff()
```

## 🛠️ Switching Between LLMs

### Using Azure OpenAI
```python
llm="azure/gpt-4o-mini"
```

### Using Ollama
```python
from crewai import LLM

llm=LLM(
    model="ollama/llama3.2:1b",
    base_url="http://localhost:11434"
)
```

## 📖 Learning Path

1. **Start Here**: `crewai_basic_beginner_example.py`
   - Understand agents, tasks, and crews
   - See how agents collaborate

2. **Explore Features**: `crewai_comprehensive_features_tutorial.py`
   - Learn about tools and delegation
   - Understand different process types

3. **Add Memory**: `crewai_memory_basics_tutorial.py`
   - See how agents remember information
   - Understand context persistence

4. **Advanced Concepts**: Memory systems and RAG
   - Build knowledge bases
   - Implement vector search

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- New examples
- Bug fixes
- Documentation improvements
- Feature demonstrations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) - The amazing framework these examples are built on
- [Ollama](https://ollama.ai/) - For local LLM execution
- Community contributors

## 📞 Contact

- GitHub: [@AIFahim](https://github.com/AIFahim)
- Issues: [GitHub Issues](https://github.com/AIFahim/crewai-examples/issues)

---

⭐ If you find these examples helpful, please consider giving the repository a star!