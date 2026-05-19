TL;DR Internal architecture of Claude Code: Anthropic's CLI that orchestrates an LLM with local tools through a query loop, permission system, and context compaction.

> [!quote] Information
> * paper Reference [DeepWiki — Claude Code](https://deepwiki.com/instructkr/claude-code)
> * git Github [anthropics/claude-code](https://github.com/anthropics/claude-code)
> *  tag Tags
> 	[[LLM Tooling]]
> 	[[CLI]]
> 	[[Agent Architecture]]
> * calendar Date 31 March 2026
> * ? Motivation:
> 		Understand how Claude Code works internally: its execution loop, tool system, permissions, and context management.

---

### 1. Overview

#### 1.1 What is Claude Code?

Claude Code is Anthropic's official CLI for interacting with Claude as a software engineering agent. It's not just a terminal chatbot — it's a full **agent system** with access to filesystem, bash, web, and extensible tools via MCP.

It is built on 4 fundamental pillars:

![[claudecode_architecture.svg]]

> [!info]- Comments
> The architecture follows a classic **agent loop** pattern: the Query Engine acts as the central orchestrator deciding when to invoke tools and when to respond. Similar to how ReAct (Reason + Act) operates, but with a granular permission system interleaved between decisions.

#### 1.2 The 4 Pillars

| Pillar | Responsibility |
|--------|---------------|
| **Query Engine** | Orchestration loop: conversation, tool calls, context |
| **Tool System** | Strongly-typed capabilities (File I/O, Bash, LSP, MCP) to interact with the host machine |
| **REPL & UI** | Terminal interface with React + Ink, diff views, streaming |
| **Bridge & Remote** | Synchronization for remote control and IDE integration (VS Code, JetBrains) |

---

### 2. Initialization

When you run `claude [prompt]`, this is what happens:

```
Shell → Commander.js parsing → Startup optimizations → init.ts → React mount → Query loop
```

**Startup optimizations** run in parallel:
- MDM settings read from filesystem
- Keychain prefetch async for credentials
- Startup profiler (if enabled)

**init.ts** executes:
1. Environment validation (TTY, color support)
2. Config migration for `~/.claude/config.json`
3. `STATE` singleton bootstrap
4. Signal handling (SIGINT, SIGTERM) for graceful shutdown
5. React application mount with Ink
6. `processUserInput()` initiates LLM interaction

> [!info]- Comments
> Using **React + Ink** for a CLI is unconventional but enables reactive components in the terminal — diff views, progress bars, and text streaming all render as React components. Same paradigm as a SPA but in the terminal.

---

### 3. Query Engine

The heart of Claude Code. Manages the main conversation loop:

![[claudecode_query_loop.svg]]

#### 3.1 Responsibilities

1. **Turn Management** — Iteratively calls the model until a final response is reached
2. **Tool Execution** — Handles tool calls and feeds results back to the LLM
3. **Context Management** — Builds system prompts, manages compaction
4. **State Persistence** — Records transcripts to disk

#### 3.2 System Prompt Construction

The system prompt is dynamically built from:
- Global configuration and theme settings
- Current working directory
- Available MCP capabilities
- Project-specific instructions from `CLAUDE.md`

#### 3.3 Safety Limits

| Parameter | Function |
|-----------|----------|
| `maxTurns` | Prevents infinite loops (~50 by default) |
| `maxBudgetUsd` | Cumulative cost limit via `cost-tracker.ts` |
| Streaming | Yields events in real-time to the UI |

> [!info]- Comments
> The `query()` loop is an **agentic loop** pattern: the LLM decides whether it needs to use a tool or can respond directly. Each tool call generates a new turn where the LLM receives the result and decides the next step. This is fundamentally different from simple request-response — it's a reasoning-with-action loop.

---

### 4. Tool System

The modular framework that gives Claude Code its capabilities:

![[claudecode_tool_pipeline.svg]]

#### 4.1 Tool Interface

Each tool implements this contract:

```python
class Tool:
    name: str              # Unique ID sent to the LLM
    description: str       # Natural language instructions
    inputSchema: ZodSchema # Argument validation
    isReadOnly: bool       # Affects default permissions

    def validateInput(input) -> bool    # Custom validation
    def checkPermissions() -> bool      # Permission evaluation
    def call(context: ToolUseContext)    # Execution logic
    def renderToolUseMessage() -> JSX   # Terminal display
    def renderToolResultMessage() -> JSX
```

> [!info]- Comments
> The validation schema uses **Zod** (TypeScript), which enables runtime validation with compile-time inferred types. It's the de facto standard for validation in the modern TypeScript ecosystem. Each tool defines its schema, and the LLM receives the derived JSON Schema description to know which arguments it can pass.

#### 4.2 Tool Catalog

| Category | Tools |
|----------|-------|
| **Filesystem** | FileRead, FileWrite, FileEdit, Glob, Grep, NotebookEdit |
| **Execution** | BashTool (with SandboxManager), LSPTool |
| **Multi-Agent** | AgentTool (sub-agents), Task Management |
| **Web** | WebSearch, WebFetch |
| **MCP** | Dynamic tools loaded from MCP servers |

#### 4.3 Deferred Loading

- **alwaysLoad** — Initialized at startup
- **deferred** — Loaded on demand (e.g., MCP tools)

This optimizes startup time without sacrificing capabilities.

#### 4.4 Content Replacement

When a tool produces very large output:

```
Tool.call() → Large result → ContentReplacementState
→ Stored on disk/memory → LLM receives: "Result truncated. Use ReadMcpResource to view."
```

---

### 5. Permission System

Security layer that evaluates every tool call before execution:

![[claudecode_permissions.svg]]

#### 5.1 Permission Modes

| Mode | Behavior |
|------|----------|
| **Default** | Rule-based evaluation |
| **Plan** | More analytical, restricted tool use |
| **Auto** | High trust for autonomous loops |
| **Bypass** | Execute without confirmation |

#### 5.2 Security Layers

1. **validateInput()** — Firewall against malformed LLM outputs
2. **Permission Gating** — Every tool call checked before execution
3. **Concurrency Safety** — Sequenced execution to prevent race conditions
4. **Schema Validation** — Zod-backed input validation

> [!info]- Comments
> The **Classifier** is an interesting component: when there's no explicit rule (allow/deny), a classifier evaluates whether the command is "safe" or "unsafe". For Bash, this is critical — an `rm -rf /` gets blocked automatically, but an `ls` is allowed without asking. The classifier acts as a security heuristic between the allow-list and the user prompt.

---

### 6. Context Compaction

Intelligent context window management as it approaches the token limit:

![[claudecode_compaction.svg]]

#### 6.1 The 3 Levels

**Level 1 — Micro-compaction** (`microcompactMessages()`)
- No LLM call required
- Removes redundant file reads
- Truncates excessive terminal output
- Noise cleanup

**Level 2 — Session Memory** (`trySessionMemoryCompaction()`)
- Extracts structured "memories" vs. narrative summary
- Stored in persistent `memdir/` directory
- Only triggered if no custom instructions

**Level 3 — Full Compaction** (`compactConversation()`)
- Dedicated LLM call for summarization
- Creates a "Context Boundary" message as the new starting point
- Fallback when memory compaction is insufficient

#### 6.2 Compact Boundary

When compaction occurs:
1. Previous messages are removed from active context
2. A special system message is inserted with the summarized state
3. The LLM starts from this new point

#### 6.3 Multi-Agent Memory Sharing

`teamMemorySync` mechanism:
- Sub-agents run memory extraction on completion
- Synced back to coordinator or shared team directory
- Prevents redundant exploration across parallel tasks

> [!info]- Comments
> The compaction hierarchy is elegant: first it tries the cheapest option (micro-compaction without LLM), then the intermediate one (structured memories), and only as a last resort makes an expensive LLM call to summarize everything. It's a **progressive fallback** pattern that optimizes cost vs. information retention.

---

### 7. Configuration

Layered configuration system with priority-based resolution:

![[claudecode_config.svg]]

#### 7.1 The STATE Object

Singleton in `src/bootstrap/state.ts` that maintains global state:

| Property | Purpose |
|----------|---------|
| `projectRoot` | Project root (via .git or CLAUDE.md) |
| `sessionId` | Unique UUID per session |
| `totalCostUSD` | Cumulative session cost |
| `modelUsage` | Map of model → token counts |
| `mainLoopModelOverride` | Model override via CLI or /config |

#### 7.2 Dual-Layer State

```
STATE (Singleton)              AppState (Functional)
├─ sessionId                   ├─ planMode / autoMode
├─ projectRoot                 ├─ UI state (React/Ink)
├─ totalCostUSD                ├─ SetAppState() for updates
├─ modelUsage                  └─ Re-renders via useAppState hook
└─ Persists for process lifetime└─ Reactive for UI
```

> [!info]- Comments
> The **Singleton + Functional** pattern is a pragmatic hybrid: global STATE for telemetry and process data (non-reactive, stable), and functional AppState for the UI (reactive, frequently changing). It separates concerns: what needs to persist vs. what needs to re-render.

---

### 8. Session Persistence

#### 8.1 JSONL Storage

Sessions are saved as JSONL files in `~/.claude/sessions/[sessionId].jsonl`.

Each line represents:
- A single `SDKMessage` (user/assistant)
- System events (tool calls, results)
- Compact boundaries

#### 8.2 Recording Flow

```
QueryEngine.processUserInput()
  → STATE.sessionId
  → recordTranscript(messages)
  → Serialize to JSONL
  → Append to filesystem
```

#### 8.3 Resumption

To restore a session:
1. Scans session directory for past interactions
2. Applies schema migrations as needed
3. Replays transcript to rebuild AppState
4. Restores tool results and compact boundaries

---

### 9. Observability

#### 9.1 Telemetry (OpenTelemetry)

| Counter | Tracks |
|---------|--------|
| `tokenCounter` | Total tokens consumed |
| `costCounter` | Financial cost |
| `activeTimeCounter` | User engagement duration |
| `codeEditToolDecisionCounter` | File editing tool usage |

#### 9.2 Feature Flags

GrowthBook integration for gradual feature activation.

---

### 10. Connections

- Built on the **ReAct** (Reason + Act) paradigm for LLM agents
- UI built with **React + Ink** (framework for reactive CLIs)
- Validation via **Zod** (runtime type checking for TypeScript)
- Extensible via **MCP** (Model Context Protocol) — Anthropic's open standard for connecting LLMs with external tools
- The **agentic loop** pattern is shared with frameworks like LangChain, AutoGen, and CrewAI, but Claude Code implements it directly without an intermediate abstraction layer
