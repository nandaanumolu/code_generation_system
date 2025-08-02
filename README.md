# Multi-Agent System with ADK Framework

A production-ready multi-agent system built using the Agent Development Kit (ADK) framework, featuring advanced observability, memory management, and orchestration capabilities for intelligent code generation and review.

## 🏗️ Architecture Overview

This system implements a sophisticated multi-agent architecture with the following key components:

### Core Components

1. **Central Agent Hub** - Orchestrates all agent interactions and manages system flow
2. **Memory System** - Persistent context management across agent interactions
3. **Guardrails** - Input/output validation and safety mechanisms
4. **Tools Integration** - External service connections (Google Search, MCP)
5. **Observability Stack** - Comprehensive logging, metrics, and tracing
6. **Retrieval/Fallback System** - Error handling and recovery mechanisms

### Agent Workflow

```
User Input → Guardrails → Parallel Agents → Review Loop → Final Output → Guardrails → Codebase
```

## 🚀 Features

- **Parallel Agent Processing**: Multiple AI agents (Gemini, GPT-4.1, Claude) work simultaneously
- **Iterative Review System**: Automated code review and refinement loop
- **Production-Grade Observability**: Full telemetry with logs, metrics, and distributed tracing
- **Memory Persistence**: Context retention across sessions
- **Safety Guardrails**: Input validation and output filtering
- **Tool Integration**: Extensible tool system with Google Search and custom tools
- **MCP (Model Context Protocol) Support**: Advanced context management

## 📋 Prerequisites

- Python 3.8+
- ADK Framework
- API Keys for:
  - OpenAI (GPT-4.1)
  - Google (Gemini)
  - Anthropic (Claude)
  - Google Search API

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adk-multi-agent-system.git
cd adk-multi-agent-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## ⚙️ Configuration

### Environment Variables

```env
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
GOOGLE_SEARCH_API_KEY=your_search_key
GOOGLE_SEARCH_ENGINE_ID=your_engine_id

# ADK Configuration
ADK_LOG_LEVEL=INFO
ADK_MEMORY_BACKEND=redis
ADK_MEMORY_TTL=3600

# Observability
ENABLE_TRACING=true
JAEGER_ENDPOINT=http://localhost:14268/api/traces
PROMETHEUS_PORT=9090

# Guardrails
MAX_INPUT_LENGTH=10000
MAX_OUTPUT_LENGTH=50000
ENABLE_CONTENT_FILTERING=true
```

### Agent Configuration

Create `config/agents.yaml`:

```yaml
agents:
  gemini:
    model: "gemini-pro"
    temperature: 0.7
    max_tokens: 8000
    role: "initial_code_generation"
    
  gpt4:
    model: "gpt-4-1106-preview"
    temperature: 0.7
    max_tokens: 8000
    role: "alternative_implementation"
    
  claude_reviewer:
    model: "claude-3-opus-20240229"
    temperature: 0.3
    max_tokens: 4000
    role: "code_review_critic"
    
  claude_refactor:
    model: "claude-3-opus-20240229"
    temperature: 0.5
    max_tokens: 8000
    role: "code_refactoring"
```

## 🎯 Usage

### Basic Usage

```python
from adk_multi_agent import MultiAgentSystem

# Initialize the system
system = MultiAgentSystem(config_path="config/agents.yaml")

# Process a request
result = await system.process(
    input_text="Create a REST API for user management with authentication",
    options={
        "enable_review_loop": True,
        "max_iterations": 3,
        "output_format": "python"
    }
)

print(result.final_code)
print(result.review_history)
print(result.metrics)
```

### Advanced Usage with Custom Tools

```python
from adk_multi_agent import MultiAgentSystem, Tool

# Define custom tool
class DatabaseSchemaRetriever(Tool):
    async def execute(self, query: str) -> str:
        # Your implementation here
        return schema_info

# Register tool
system.register_tool("db_schema", DatabaseSchemaRetriever())

# Use in processing
result = await system.process(
    input_text="Generate SQL queries for user analytics",
    tools=["db_schema", "google_search"]
)
```

## 📊 Observability

### Logging

The system uses structured logging with the following levels:

- **INFO**: General operation information
- **DEBUG**: Detailed debugging information
- **WARNING**: Warning messages
- **ERROR**: Error messages

Access logs:
```bash
tail -f logs/adk-multi-agent.log
```

### Metrics

Prometheus metrics available at `http://localhost:9090/metrics`:

- `agent_request_duration_seconds` - Request processing time
- `agent_request_total` - Total number of requests
- `agent_review_iterations_total` - Number of review iterations
- `memory_operations_total` - Memory system operations
- `tool_invocations_total` - Tool usage statistics

### Distributed Tracing

Jaeger UI available at `http://localhost:16686`

Traces include:
- End-to-end request flow
- Individual agent processing times
- Tool invocation spans
- Memory operations
- Review loop iterations

## 🧠 Memory System

### Memory Backends

1. **Redis** (Recommended for production)
```python
memory_config = {
    "backend": "redis",
    "connection_string": "redis://localhost:6379",
    "ttl": 3600
}
```

2. **In-Memory** (Development only)
```python
memory_config = {
    "backend": "memory",
    "max_items": 1000
}
```

### Memory Management

```python
# Store context
await system.memory.store(
    key="user_context",
    value={"preferences": "functional_programming"},
    ttl=3600
)

# Retrieve context
context = await system.memory.retrieve("user_context")
```

## 🛡️ Guardrails

### Input Validation

- Maximum input length enforcement
- Content filtering for harmful inputs
- Format validation
- Language detection

### Output Validation

- Code syntax validation
- Security vulnerability scanning
- License compliance checking
- Output size limits

### Custom Guardrails

```python
from adk_multi_agent.guardrails import Guardrail

class CustomSecurityGuardrail(Guardrail):
    async def validate(self, content: str) -> tuple[bool, str]:
        # Your validation logic
        if has_security_issue(content):
            return False, "Security issue detected"
        return True, ""

system.register_guardrail(CustomSecurityGuardrail())
```

## 🔧 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=adk_multi_agent

# Run specific test suite
pytest tests/test_agents.py
```

### Code Quality

```bash
# Format code
black .

# Lint
flake8 .

# Type checking
mypy .
```

### Local Development Setup

```bash
# Start required services
docker-compose up -d redis jaeger prometheus

# Run in development mode
python -m adk_multi_agent.main --dev
```

## 📈 Performance Optimization

### Caching Strategy

- Agent response caching with Redis
- Tool result caching
- Memory operation optimization

### Parallel Processing

- Concurrent agent execution
- Asynchronous tool invocations
- Batch processing support

### Resource Management

```python
# Configure resource limits
resource_config = {
    "max_concurrent_agents": 5,
    "request_timeout": 300,
    "memory_limit_mb": 1024
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Backend Connection Failed**
   - Check Redis connection string
   - Ensure Redis is running
   - Verify network connectivity

2. **Agent Timeout**
   - Increase timeout in configuration
   - Check API rate limits
   - Monitor agent performance metrics

3. **Review Loop Infinite Iteration**
   - Set maximum iteration limit
   - Implement convergence criteria
   - Add fallback mechanisms

### Debug Mode

```python
# Enable debug mode
system = MultiAgentSystem(debug=True)

# Access debug information
print(system.debug_info)
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ADK Framework team for the excellent foundation
- OpenAI, Anthropic, and Google for their powerful AI models
- The open-source community for invaluable tools and libraries

## 📞 Support

- **Documentation**: [https://docs.example.com](https://docs.example.com)
- **Issues**: [GitHub Issues](https://github.com/yourusername/adk-multi-agent-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/adk-multi-agent-system/discussions)
- **Email**: support@example.com

---

Built with ❤️ using ADK Framework