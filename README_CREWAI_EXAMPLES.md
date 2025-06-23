# CrewAI Examples for Students

This repository contains educational examples demonstrating CrewAI features.

## 📚 Examples Overview (In Learning Order)

### 1. 🟢 **crewai_basic_beginner_example.py**
- **Purpose**: Simple introduction to CrewAI concepts
- **Features**: Basic agents, tasks, and sequential workflow
- **Best for**: First-time CrewAI users
- **Run**: `python crewai_basic_beginner_example.py`

### 2. 🔵 **crewai_comprehensive_features_tutorial.py**
- **Purpose**: Complete overview of all CrewAI features
- **Features**: Multiple agent types, tools, delegation, different crew processes
- **Best for**: Understanding full CrewAI capabilities
- **Run**: `python crewai_comprehensive_features_tutorial.py`

### 3. 🧠 **crewai_memory_basics_tutorial.py**
- **Purpose**: Introduction to agent memory capabilities
- **Features**: Short-term memory, long-term memory, learning agents
- **Best for**: Understanding how agents remember information
- **Run**: `python crewai_memory_basics_tutorial.py`

### 4. 🎓 **crewai_advanced_memory_knowledge_system.py**
- **Purpose**: Advanced memory and knowledge management
- **Features**: Knowledge base, entity memory, conversation memory
- **Best for**: Complex applications with persistent knowledge
- **Run**: `python crewai_advanced_memory_knowledge_system.py`

### 5. 🔍 **crewai_rag_vector_search_example.py**
- **Purpose**: RAG (Retrieval-Augmented Generation) implementation
- **Features**: Vector search, semantic similarity, hybrid search
- **Best for**: Building AI with external knowledge sources
- **Run**: `python crewai_rag_vector_search_example.py`

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configure API**:
   - Copy `.env.example` to `.env`
   - Add your API key
   - Update model names if needed

3. **Test Configuration**:
   ```bash
   python test_config.py
   ```

4. **Run Examples**:
   Start with the basic example and progress through the list above.

## 📖 Key Concepts

### Agents
- **Role**: What the agent does (e.g., "Researcher", "Analyst")
- **Goal**: What the agent aims to achieve
- **Backstory**: Context that shapes the agent's behavior
- **Tools**: Functions the agent can use
- **Memory**: Whether the agent remembers past interactions

### Tasks
- **Description**: What needs to be done
- **Expected Output**: What the result should look like
- **Agent**: Which agent performs the task
- **Context**: Dependencies on other tasks

### Crews
- **Sequential**: Tasks run one after another
- **Hierarchical**: Manager agent delegates tasks

## 🛠️ Troubleshooting

### Common Issues:

1. **DeploymentNotFound Error**:
   - Run `python test_azure_config.py` to find your deployment names
   - Update the `llm` parameter in agents to match your deployments

2. **API Key Issues**:
   - Ensure your `.env` file has the correct API credentials
   - Check that environment variables are loaded

3. **Import Errors**:
   - Make sure all dependencies are installed: `pip install -r requirements.txt`

## 📝 Next Steps

1. Start with the basic example
2. Experiment with modifying agent roles and goals
3. Try creating your own tools
4. Build a custom crew for your use case

## 🤝 Contributing

Feel free to create more examples or improve existing ones. Follow the established patterns:
- Use class-based agents inheriting from `Agent`
- Include proper error handling
- Add helpful comments
- Keep examples educational and clear