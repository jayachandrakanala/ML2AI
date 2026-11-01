> ## Documentation Index
>
> Fetch the complete documentation index at: https://docs.langchain.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Deep Agents overview

> Build agents that can plan, use subagents, and leverage file systems for complex tasks

Deep Agents is the easiest way to start building agents and applications that are powered by LLMs—with built-in capabilities for file systems for context management, subagent-spawning, and long-term memory.
Optional capabilities such as [task planning](#task-planning) and [skills](#skills) extend the harness when your use case needs them.
You can use deep agents for any task, including complex, multi-step tasks.

Deep Agents comes with the following capabilities:

- **Take actions in an environment**: Take actions via tools, read and write files, execute code
- **Connect to your data**: Load memories, skills, and domain knowledge at the right moment
- **Manage growing context**: Summarize history and offload large results across long runs
- **Parallelize tasks**: Delegate to general or specialized subagents running in isolated context windows
- **Stay in the loop**: Pause for human approval at critical decision points
- **Improve over time**: Update memory, skills, and prompts based on real usage

See [Core capabilities](#core-capabilities) for a full breakdown of each component.

## Try it

## Quickstart

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_deep_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_deep_agent(
    model="openai:gpt-5.5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_deep_agent(
    model="openrouter:z-ai/glm-5.2",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_deep_agent(
    model="fireworks:accounts/fireworks/models/glm-5p2",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_deep_agent(
    model="baseten:zai-org/GLM-5.2",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_deep_agent(
    model="ollama:north-mini-code-1.0",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

> **[View example trace](https://smith.langchain.com/public/44780f65-3414-46eb-8239-944c6ee5454f/r)**
>
> Open a public LangSmith run for this example.

See the [Quickstart](https://docs.langchain.com/oss/python/deepagents/quickstart.md) and [Customization guide](https://docs.langchain.com/oss/python/deepagents/customization.md) to get started building your own agents and applications with Deep Agents.

> [!TIP]
>
> Trace requests, debug agent behavior, and evaluate outputs with [LangSmith](https://smith.langchain.com?utm_source=docs&utm_medium=cta&utm_campaign=langsmith-signup&utm_content=oss-deepagents-overview). Follow the [observability quickstart](https://docs.langchain.com/langsmith/observability-quickstart.md) to get set up. When ready for production, see [Going to production](https://docs.langchain.com/oss/python/deepagents/going-to-production) for LangSmith deployment options.

## Core capabilities

![Agent harness capabilities by category](https://mintcdn.com/langchain-5e9cc07a/jtty0O--UJOKG0nK/oss/images/agent_harness_capabilities.svg?fit=max&auto=format&n=jtty0O--UJOKG0nK&q=85&s=0ff671d72badd0844826660dfcb04391)

Deep Agents is an ["agent harness"](https://docs.langchain.com/oss/python/concepts/products.md#agent-harnesses-like-the-deep-agents-sdk). It is the same core tool calling loop as other agent frameworks, but with built-in capabilities that make agents reliable for real tasks:

> **[Execution environment](#execution-environment)**
>
> Tools, virtual filesystem, optional sandbox, and REPL (interpreter)

> **[Context management](#context-management)**
>
> Skills, memory, summarization, context offloading, and prompt caching

> **[Delegation](#delegation)**
>
> Subagent spawning and optional task planning

> **[Steering](#steering)**
>
> Human-in-the-loop approval and interrupts

[`deepagents`](https://pypi.org/project/deepagents/) is a standalone library built on top of [LangChain](https://docs.langchain.com/oss/python/langchain/)'s core building blocks for agents. It uses the [LangGraph](https://docs.langchain.com/oss/python/langgraph/) runtime for durable execution, streaming, human-in-the-loop, and other features.

[LangChain](https://docs.langchain.com/oss/python/langchain/) is the framework that provides the core building blocks for your agents.
To learn more about the differences between LangChain, LangGraph, and Deep Agents, see [Frameworks, runtimes, and harnesses](https://docs.langchain.com/oss/python/concepts/products.md). For a side-by-side comparison with Anthropic's harness, see [Deep Agents vs. Claude Agent SDK](https://docs.langchain.com/oss/python/deepagents/comparison.md).

For building custom agents without these built-in capabilities, consider using LangChain's [`create_agent`](https://docs.langchain.com/oss/python/langchain/agents) or building a custom [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview.md) workflow.

## Execution environment

The execution environment is where an agent acts. It has four layers:

- **[Tools](#tools-and-mcp)**: custom functions, APIs, and databases the agent can call
- **[Virtual filesystem](#virtual-filesystem-access)**: file tools backed by pluggable backends
- **[Filesystem permissions](#filesystem-permissions)**: declarative access control over which paths agents can read or write
- **[Code execution](#code-execution)**: sandboxed shell execution and an in-process JavaScript interpreter

**[Streaming](#streaming)** allows you to keep up with everything happening using typed event streams for messages, tools, values, and delegated tasks.

### Tools and MCP

Pass custom functions, LangChain tools, or tools from any [MCP server](https://docs.langchain.com/oss/python/deepagents/tools.md#mcp-tools) with the `tools=` parameter. Deep Agents fully support the [Model Context Protocol (MCP)](https://docs.langchain.com/oss/python/langchain/mcp.md), letting you connect to databases, APIs, file systems, and more through a standard interface.

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search, fetch_page, run_query],
)
```

For more information on defining custom tools, using MCP servers, and the full list of built-in harness tools, see [Tools](https://docs.langchain.com/oss/python/deepagents/tools.md).

### Virtual filesystem access

The harness provides a configurable virtual filesystem which can be backed by different [pluggable backends](https://docs.langchain.com/oss/python/deepagents/backends.md): in-memory state, local disk, LangGraph store, composite routing, or a custom backend with [permission rules](https://docs.langchain.com/oss/python/deepagents/permissions.md) for read and write access.

The backends support the following file system operations:

| Tool         | Description                                                                                                                                                                                                              |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ls`         | List files in a directory with metadata (size, modified time)                                                                                                                                                            |
| `read_file`  | Read file contents with line numbers, supports offset/limit for large files. Also supports returning multimodal content blocks for non-text files (images, video, audio, and documents). See supported extensions below. |
| `write_file` | Create a new file, or overwrite an existing one                                                                                                                                                                          |
| `edit_file`  | Perform exact string replacements in files (with global replace mode)                                                                                                                                                    |
| `delete`     | Delete a file, or a directory and its contents recursively                                                                                                                                                               |
| `glob`       | Find files matching patterns (e.g., `**/*.py`)                                                                                                                                                                           |
| `grep`       | Search file contents with multiple output modes (files only, content with context, or counts)                                                                                                                            |
| `execute`    | Run shell commands in the environment (available with [sandbox backends](https://docs.langchain.com/oss/python/deepagents/sandboxes.md) only)                                                                            |

> [!NOTE]
>
> The `delete` tool requires `deepagents>=0.7`. Backends that do not support deletion have the tool automatically hidden from the model.

> **Supported multimodal file extensions**
>
> | Type                                                                            | Extensions                                                                |
> | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
> | [Image](https://docs.langchain.com/oss/python/langchain/messages.md#multimodal) | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.heic`, `.heif`                |
> | [Video](https://docs.langchain.com/oss/python/langchain/messages.md#multimodal) | `.mp4`, `.mpeg`, `.mov`, `.avi`, `.flv`, `.mpg`, `.webm`, `.wmv`, `.3gpp` |
> | [Audio](https://docs.langchain.com/oss/python/langchain/messages.md#multimodal) | `.wav`, `.mp3`, `.aiff`, `.aac`, `.ogg`, `.flac`                          |
> | [File](https://docs.langchain.com/oss/python/langchain/messages.md#multimodal)  | `.pdf`, `.ppt`, `.pptx`                                                   |

> **Running without the default filesystem tools**
>
> To hide the filesystem tools listed above from the model, register a [harness profile](https://docs.langchain.com/oss/python/deepagents/profiles.md#harness-profiles) with `excluded_tools`:
>
> ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
> from deepagents import HarnessProfile, register_harness_profile
>
> register_harness_profile(
>     "anthropic:claude-sonnet-4-6",
>     HarnessProfile(
>         excluded_tools=frozenset(
>             {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep"}
>         ),
>     ),
> )
> ```
>
> Removing [`FilesystemMiddleware`](https://reference.langchain.com/python/deepagents/middleware/filesystem/FilesystemMiddleware) itself via `excluded_middleware` is intentionally rejected—it is required scaffolding in the [Deep Agents stack](https://docs.langchain.com/oss/python/deepagents/customization.md#deep-agents-stack). Use `excluded_tools` to hide only the model-visible tool surface and leave the middleware in place. To remove the `task` tool, see [Running without subagents](https://docs.langchain.com/oss/python/deepagents/subagents.md#running-without-subagents).

> **Restricting filesystem tools**
>
> > [!NOTE]
> >
> > The `tools` allowlist on `FilesystemMiddleware` requires `deepagents>=0.7`.
>
> To expose only a subset of the filesystem tools listed above, instead of hiding them all, pass a `tools` allowlist to [`FilesystemMiddleware`](https://reference.langchain.com/python/deepagents/middleware/filesystem/FilesystemMiddleware) and provide the instance through `middleware=`. Any built-in filesystem tool left out of the list is removed from the model's tool list.
>
> ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
> from deepagents import create_deep_agent
> from deepagents.middleware import FilesystemMiddleware
>
> # Read-only agent: write_file, edit_file, delete, and execute are never shown
> agent = create_deep_agent(
>     model="claude-sonnet-4-6",
>     middleware=[
>         FilesystemMiddleware(backend=backend, tools=["read_file", "ls", "glob", "grep"]),
>     ],
> )
> ```
>
> `read_file` must always be included in the list—omitting it raises `ValueError` when the agent is created. The `execute` and `delete` tools are also dropped from the tool surface whenever the configured backend doesn't support them, whether or not you include them in `tools`. Custom tools you add through `create_deep_agent`'s own `tools=` argument are never affected by this allowlist.
>
> Passing your own [`FilesystemMiddleware`](https://reference.langchain.com/python/deepagents/middleware/filesystem/FilesystemMiddleware) instance this way replaces the default one for the main agent and the general-purpose subagent inherits the same restriction. See [Override a default middleware instance](https://docs.langchain.com/oss/python/deepagents/customization.md#override-a-default-middleware-instance) for more information. Declarative subagents don't inherit it: include a `FilesystemMiddleware(tools=...)` instance in that subagent's own `middleware` field to restrict it independently.

The virtual filesystem is used by several other harness capabilities such as skills, memory, code execution, and context management.
You can also use the file system when building custom tools and middleware for Deep Agents.

For more information, see [backends](https://docs.langchain.com/oss/python/deepagents/backends.md). To generate a durable repository wiki that agents can read from the filesystem, see [OpenWiki](https://docs.langchain.com/oss/openwiki/overview.md).

### Filesystem permissions

The harness supports declarative permission rules that control which files and directories the agent can read or write. Permissions apply to the built-in filesystem tools listed above and are evaluated in declaration order with first-match-wins semantics.

Define permissions by passing a list of rules to `permissions=` when creating the agent. Each rule includes:

- `operations`: `"read"` and/or `"write"`
- `paths`: Glob patterns for files or directories
- `mode`: `"allow"` or `"deny"`

Rules are evaluated top to bottom, and the first matching rule wins. If no rule matches, the operation is allowed.

This model lets you restrict agents to specific directories (for example, `/workspace/`), protect sensitive files such as `.env` or credentials, and give subagents narrower access than the parent agent.

Permissions do not apply to [sandbox backends](https://docs.langchain.com/oss/python/deepagents/sandboxes.md), which support arbitrary command execution via the `execute` tool. For custom validation logic, use [backend policy hooks](https://docs.langchain.com/oss/python/deepagents/backends.md#add-policy-hooks).

For the full rule structure, examples, and subagent inheritance, see [Permissions](https://docs.langchain.com/oss/python/deepagents/permissions.md).

### Code execution

Deep Agents supports code execution in two ways:

- [Sandbox backends](https://docs.langchain.com/oss/python/deepagents/sandboxes.md) expose an `execute` tool for shell commands in an isolated environment.
- [Interpreters](https://docs.langchain.com/oss/python/deepagents/interpreters.md) add an `eval` tool that runs JavaScript in a scoped QuickJS runtime.

Use sandbox backends when the agent needs to install dependencies, run tests, call CLIs, or work with an operating-system filesystem. Sandbox backends implement the `SandboxBackendProtocolV2`; when detected, the harness adds the `execute` tool to the agent's available tools.

Use interpreters when the agent needs a lightweight programmable layer for loops, batching, deterministic data transformations, or programmatic tool calling. Interpreters do not provide shell access, package installs, or filesystem and network access.

For sandbox setup, providers, and file transfer APIs, see [Sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes.md). For the QuickJS runtime and programmatic tool calling, see [Interpreters](https://docs.langchain.com/oss/python/deepagents/interpreters.md).

### Streaming

[Event streaming](https://docs.langchain.com/oss/python/deepagents/event-streaming.md) exposes agent runs as typed projections for messages, tool calls, values, and output. Deep Agents add `stream.subagents` so each delegated task gets its own handle with independent message, tool-call, and nested subagent streams.

## Context management

The context management component controls what the agent knows, how long it can operate within token limits, and what it retains across sessions. It has four layers:

- **[Skills](#skills)**: on-demand domain knowledge loaded progressively from skill files
- **[Memory](#memory)**: persistent instructions and preferences loaded at startup from `AGENTS.md` files
- **[Summarization and context offloading](#summarization-and-context-offloading)**: automatic compression of conversation history and large tool results
- **[Prompt caching](#prompt-caching)**: static prompt sections are cache-eligible to speed up inference and reduce cost on supported models

### Skills

Skills package specialized workflows, domain knowledge, and custom instructions for your deep agent.

Each skill follows the [Agent Skills standard](https://agentskills.io/) and lives in a directory with a `SKILL.md` file. Skills can also include scripts, templates, reference docs, and other supporting resources.

Deep Agents load skills with progressive disclosure: the agent reads `SKILL.md` frontmatter at startup, then reads full skill content only when a task needs it. This keeps startup context compact while still making rich capabilities available on demand.

For more information, see [Skills](https://docs.langchain.com/oss/python/deepagents/skills.md).

### Memory

Memory gives your deep agent persistent context across conversations, such as coding style, preferences, conventions, and project guidelines.

Memory uses [`AGENTS.md` files](https://agents.md/) that you pass through the `memory` parameter when creating the agent. Unlike skills, memory files are always loaded, and the content is stored in the configured backend (`StateBackend`, `StoreBackend`, or `FilesystemBackend`).

The agent can also update memory based on interactions and feedback, so preferences and patterns can carry forward without needing to restate them in each thread.

For configuration details and examples, see [Memory](https://docs.langchain.com/oss/python/deepagents/customization.md#memory). To generate a repository wiki that coding agents discover through `AGENTS.md`, see [OpenWiki](https://docs.langchain.com/oss/openwiki/overview.md).

### Summarization and context offloading

The harness manages context so deep agents can handle long-running work within token limits while keeping the most relevant information in scope.

This context flow has four parts:

- **Input context**: System prompt, memory, skills, and tool prompts define what the agent starts with.
- **Compression**: Built-in offloading and summarization compress conversation history and large intermediate results.
- **Isolation**: Subagents quarantine heavy subtasks and return only final results (see [Delegation](#delegation)).
- **Long-term memory**: Persistent storage in the virtual filesystem carries information across threads.

Together, these mechanisms support multi-step tasks that exceed a single context window while reducing manual context trimming and token usage.

For configuration details, see [Context engineering](https://docs.langchain.com/oss/python/deepagents/context-engineering.md). For multimodal inputs and tool outputs, see [Multimodal](https://docs.langchain.com/oss/python/deepagents/multimodal.md).

### Prompt caching

For Anthropic and Amazon Bedrock models, `create_deep_agent` automatically applies prompt caching to static sections of the system prompt—the base agent instructions, memory, and skill content that repeat on every turn. This avoids reprocessing the same tokens across calls, reducing both latency and cost on long-running agents.

Prompt caching is enabled by default when using an Anthropic model, or a Bedrock model (Claude or Nova). No configuration is required.

For other providers, see [Middleware integrations](https://docs.langchain.com/oss/python/integrations/middleware.md) for available provider-specific caching middleware.

## Delegation

The delegation component enables agents to break large problems into smaller, parallelizable units of work. It has two layers:

- **[Task planning](#task-planning)**: an opt-in `write_todos` tool for structured task tracking
- **[Subagents](#subagents)**: ephemeral child agents that handle isolated subtasks

### Task planning

Task planning is an opt-in harness capability that lets agents maintain a structured task list during execution.

Starting in v0.7 task planning is opt-in only. In earlier versions, task planning middleware was included by default.

Planning is often useful for:

- Long or complicated multi-step tasks
- Less capable models that benefit from an explicit accountability tool
- UIs that stream progress from agent state (see [Todo list](https://docs.langchain.com/oss/python/deepagents/frontend/todo-list.md))

Pass [`TodoListMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/todo/TodoListMiddleware) to the middleware parameter to give the agent a `write_todos` tool for maintaining a structured task list during execution.

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_deep_agent(
    model="google_genai:gemini-3.6-flash",
    middleware=[TodoListMiddleware()],
)
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_deep_agent(
    model="openai:gpt-5.5",
    middleware=[TodoListMiddleware()],
)
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    middleware=[TodoListMiddleware()],
)
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_deep_agent(
    model="openrouter:z-ai/glm-5.2",
    middleware=[TodoListMiddleware()],
)
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_deep_agent(
    model="fireworks:accounts/fireworks/models/glm-5p2",
    middleware=[TodoListMiddleware()],
)
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_deep_agent(
    model="baseten:zai-org/GLM-5.2",
    middleware=[TodoListMiddleware()],
)
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_deep_agent(
    model="ollama:north-mini-code-1.0",
    middleware=[TodoListMiddleware()],
)
```

Tasks support status tracking (`'pending'`, `'in_progress'`, `'completed'`) and are persisted in agent state. This gives agents a lightweight planning layer for organizing long-running and multi-step work.

For configuration options and behavior details, see [To-do list](https://docs.langchain.com/oss/python/langchain/middleware/built-in.md#to-do-list).

### Subagents

The harness includes a built-in `task` tool that lets the main agent create ephemeral subagents for isolated, long-running, multi-step, or parallel tasks.
Two subagent types are available: synchronous subagents (the default) block the supervisor until they return a single report, and [async subagents](https://docs.langchain.com/oss/python/deepagents/async-subagents.md) run in parallel and support mid-flight steering via launch, check, update, and cancel tools.

Subagent execution provides:

- **Isolated or inherited context**: By default, each invocation creates a new agent instance with its own context (`mode: "isolated"`). Use `mode: "fork"` to inherit the parent's full conversation history instead. See [Forked subagents](https://docs.langchain.com/oss/python/deepagents/subagents.md#forked-subagents).
- **Autonomous execution**: The subagent runs independently until completion.
- **Single handoff**: It returns one final report to the main agent.
- **Configurable strategy**: Use the [default `general-purpose` subagent](https://docs.langchain.com/oss/python/deepagents/subagents#default-subagent) (enabled by default) or define [custom subagents](https://docs.langchain.com/oss/python/deepagents/subagents.md#custom-subagents).
- **Stateless messaging**: Subagents are stateless and cannot send multiple messages back.
- **Context and token efficiency**: Heavy subtask work stays isolated and is compressed into a compact result.

For more information, see [Subagents](https://docs.langchain.com/oss/python/deepagents/subagents.md). To remove the `task` tool, see [Running without subagents](https://docs.langchain.com/oss/python/deepagents/subagents.md#running-without-subagents).

## Steering

The steering component gives humans control over agent behavior at runtime and sets filesystem permissions for agent work.

### Human-in-the-loop

Deep Agents integrate with LangGraph interrupts so you can pause for approval on sensitive tool calls. Enable this behavior with the `interrupt_on` parameter in `create_deep_agent`.

`interrupt_on` accepts a mapping of tool names to interrupt configurations. For example, `interrupt_on={"edit_file": True}` pauses before every edit, letting you approve the call, add guidance, or modify tool inputs before execution.

This gives you a runtime safety and control layer for destructive operations, expensive API calls, and interactive debugging.

For more information, see [Human-in-the-loop](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop.md).

## Get started

> **[Quickstart](https://docs.langchain.com/oss/python/deepagents/quickstart.md)**
>
> Build your first deep agent

> **[Customization](https://docs.langchain.com/oss/python/deepagents/customization.md)**
>
> Learn about customization options

> **[Code](https://docs.langchain.com/oss/deepagents/code/overview.md)**
>
> Use Deep Agents Code

> **[ACP](https://docs.langchain.com/oss/python/deepagents/acp.md)**
>
> Use deep agents in code editors with ACP

> **[Reference](https://reference.langchain.com/python/deepagents/)**
>
> See the `deepagents` API reference

---

[Connect these docs](https://docs.langchain.com/use-these-docs.md) to your agent of choice via MCP for real-time answers.

[Edit this page on GitHub](https://github.com/langchain-ai/docs/edit/main/src/oss/deepagents/overview.mdx) or [file an issue](https://github.com/langchain-ai/docs/issues/new/choose).

> Test marker QZX7419ORBIT maps to the value cobalt relay procedure.

Try this query:

> In the local documentation, what value is assigned to the exact marker QZX7419ORBIT?

BM25 can match the exact marker directly. Dense retrieval may or may not retrieve the same passage, depending on the embedding model; compare dense-only results with the hybrid results to verify BM25's contribution. Restart the app after adding this passage so the document and BM25 index are loaded.
