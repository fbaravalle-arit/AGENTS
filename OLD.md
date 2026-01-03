# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains everything needed to build and manage a multi-agent system that creates professional landing pages. It's designed based on Google's Agent Development Kit (ADK) production best practices and includes both technical implementation and project management frameworks.

## Key Files

- `Process_Architecture.md` - Comprehensive collection of ADK examples and patterns including:
  -     
- `google_adk.py` - Comprehensive collection of ADK examples and patterns including:
  - Agent configurations (LlmAgent, SequentialAgent, ParallelAgent, LoopAgent)
  - Tool integrations (google_search, custom functions, AgentTool, MCP toolsets)
  - Code execution capabilities (BuiltInCodeExecutor)
  - Resumable workflows with human-in-the-loop approval
  - Session management with InMemorySessionService
  - MCP (Model Context Protocol) server integrations

- `initial_setup.py` - Empty setup file (currently unused)

## Architecture Patterns

### Agent Types
The codebase demonstrates multiple agent orchestration patterns:
- **LlmAgent**: Single-purpose agents with specific tools and instructions
- **SequentialAgent**: Runs sub-agents in order (e.g., outline → write → edit)
- **ParallelAgent**: Executes sub-agents simultaneously for concurrent tasks
- **LoopAgent**: Iterative refinement with max_iterations safety limit

### Tool Integration
Three main tool categories are used:
1. **Function Tools**: Python functions with type hints and docstrings (e.g., `get_exchange_rate`, `place_shipping_order`)
2. **Agent Tools**: Using other agents as tools via `AgentTool(agent=...)`
3. **MCP Tools**: External server integrations via `McpToolset` (Everything, Kaggle, GitHub servers)

### Resumable Workflows
Critical pattern for long-running operations requiring human approval:
- Uses `App` with `ResumabilityConfig(is_resumable=True)`
- Tools call `tool_context.request_confirmation()` to pause execution
- Workflow resumes using same `invocation_id` when approval is provided
- See `place_shipping_order` function and `run_shipping_workflow` for implementation

### Session State Management
Agents communicate via session state using `output_key` parameters:
- Each agent stores results in session state (e.g., `output_key="research_findings"`)
- Subsequent agents access via placeholders in instructions (e.g., `{tech_research}`)
- Enables data flow between agents in sequential/parallel workflows

## Development Context

### Environment
- Designed for **Kaggle notebook** execution (Jupyter environment)
- Requires `GOOGLE_API_KEY` from Kaggle secrets
- Uses async/await patterns extensively

### API Configuration
- Retry configuration for Google APIs with exponential backoff:
  - 5 max attempts, 7x delay multiplier, 1s initial delay
  - Retries on HTTP 429, 500, 503, 504 errors
- Model: `gemini-2.5-flash-lite` (or `gemini-2.5-flash` for some agents)

### Key Dependencies
```python
google-adk
google.genai
google.adk.agents
google.adk.tools
google.adk.runners
google.adk.sessions
mcp (Model Context Protocol)
```

## Important Implementation Notes

### Tool Design
- All custom functions must have comprehensive docstrings describing args and return values
- Type hints are required for proper ADK integration
- Tools should return structured dictionaries with `status` field for error handling
- Use `ToolContext` parameter when tools need state or confirmation capabilities

### Agent Instructions
- Be explicit about tool usage order and error checking
- Use placeholders `{key}` to inject session state values
- For code-generating agents, prohibit text output and enforce code-only responses
- Include clear failure modes (e.g., "If status is 'error', explain to user")

### MCP Integration
- MCP servers run via npx with StdioConnectionParams
- Use `tool_filter` to limit exposed tools (e.g., `["getTinyImage"]`)
- Timeout set to 30 seconds for server connections
- HTTP-based MCP servers use `StreamableHTTPServerParams` with auth headers

### Web UI Access
The `get_adk_proxy_url()` helper generates Kaggle-specific proxy URLs for ADK web interface. The pattern extracts kernel/token from Jupyter server base URL to construct the proxy path.

## Common Patterns

### Running Agents
```python
runner = InMemoryRunner(agent=root_agent)
response = await runner.run_debug("user query")
```

### Resumable Apps
```python
app = App(name="app_name", root_agent=agent, resumability_config=ResumabilityConfig(is_resumable=True))
runner = Runner(app=app, session_service=session_service)
```

### Approval Workflow
1. Tool calls `tool_context.request_confirmation(hint="...", payload={...})`
2. Tool returns status "pending"
3. Workflow detects `adk_request_confirmation` event
4. Human provides decision
5. Workflow calls `runner.run_async()` with same `invocation_id` to resume
