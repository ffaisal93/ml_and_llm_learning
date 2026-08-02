# Claude Code Deep Dive (Configuration & Workflows)

> Covers exam domain **Claude Code Configuration & Workflows (20%)**. The recurring theme is a **hierarchy**: memory (`CLAUDE.md`), settings, and MCP servers all resolve from broad → specific, with the more specific level winning. Nail the precedence rules and half this domain is yours.

---

## 1. What Claude Code is (and what it is on the exam)

Claude Code is Anthropic's **agentic coding tool** that runs in your terminal (and IDEs). It can read and edit files, run commands, search the web, and call MCP tools — driving the full **agent loop** on your codebase. The Agent SDK is *the same engine exposed as a library* (see `AGENT_SDK_DEEP_DIVE.md`).

For the exam, Claude Code shows up mostly as a **configuration and workflow** system: how you steer it with `CLAUDE.md`, `settings.json`, slash commands, plan mode, subagents, skills, and hooks — and how those layer.

---

## 2. `CLAUDE.md` — the memory hierarchy

`CLAUDE.md` is a plain-Markdown file that Claude Code **automatically loads into context** to learn your project's conventions, commands, and guardrails ("always run `make test`", "we use pnpm not npm", "never touch `/legacy`"). It is *persistent instructions*, loaded every session.

It resolves at **multiple levels**, combined broad → specific:

| Level | Location | Scope | Committed? |
|---|---|---|---|
| **Enterprise / system** | system-managed policy path | whole organization | managed by admins |
| **Project (team)** | `CLAUDE.md` at repo root | everyone on the repo | yes (checked in) |
| **Project subdirectory** | `CLAUDE.md` in a subfolder | that part of the tree | yes |
| **User (personal)** | `~/.claude/CLAUDE.md` | you, across all projects | no |
| **Local / personal project** | `CLAUDE.local.md` (git-ignored) | you, this project only | no |

**How they combine:** they **stack**. Enterprise + user + project + the nearest subdirectory `CLAUDE.md` are all in play; the **more specific / closer-to-the-file** guidance takes priority when they conflict, and enterprise policy is the outer guardrail. When Claude works in `src/api/`, a `CLAUDE.md` in `src/api/` refines the root one.

> **Exam framing.** "Where do you put instructions that must apply to every engineer on the repo?" → **project `CLAUDE.md`, committed.** "Personal preferences across all your repos?" → **`~/.claude/CLAUDE.md`.** "A tweak only you want, only in this repo, not shared?" → **`CLAUDE.local.md`.**

---

## 3. `settings.json` — behavior and permissions

`settings.json` configures Claude Code's **behavior and permissions** (as opposed to `CLAUDE.md`, which is *instructional context*). It also layers:

- **User settings** — `~/.claude/settings.json` (your defaults everywhere).
- **Project settings** — `.claude/settings.json` (shared, committed).
- **Local project settings** — `.claude/settings.local.json` (personal, git-ignored).
- **Enterprise managed settings** — org policy that can *enforce* limits users cannot override.

Precedence follows the same spirit: **enterprise managed > local > project > user** for enforcement, with managed policy as the hard ceiling. Settings control things like **tool permissions** (which tools auto-run vs. require approval), allowed/denied commands, and MCP wiring.

**Key distinction the exam draws:** `CLAUDE.md` = *what Claude should know/do* (instructions/context). `settings.json` = *what Claude is allowed to do and how it behaves* (permissions/config). Don't put secrets or permission rules in `CLAUDE.md`; don't put prose coding conventions in `settings.json`.

---

## 4. Permissions & the tool-approval model

Claude Code gates powerful actions. Tools fall into "auto-allowed" vs "needs approval," configurable via settings (allow/deny lists) and, in the SDK, a permission callback. The **least-privilege** principle is the exam's north star: grant the narrowest set of tools/commands needed, require approval for destructive or irreversible actions, and let enterprise policy enforce non-negotiable limits.

**Hooks** (see also the Agent SDK file) let you run custom code at lifecycle points — e.g., a `PreToolUse` hook that **blocks** a dangerous command, or a hook that runs a linter after every edit. Hooks are how you enforce guarantees deterministically instead of hoping the model behaves.

---

## 5. Slash commands — reusable prompt workflows

**Slash commands** are saved prompt templates invoked with `/name`. They live as Markdown files in `.claude/commands/` (project, shared) or `~/.claude/commands/` (personal). A command file can take arguments and expand into a full instruction.

This connects back to MCP: **MCP prompts surface in Claude Code as slash commands** — the user-controlled primitive becomes a `/command`. So "reusable workflow the user launches" = prompt = slash command.

> **Exam framing.** "How do you give the team a one-word way to run your standard PR-review workflow?" → a **custom slash command** committed under `.claude/commands/` (or an MCP prompt).

---

## 6. Plan mode — think before touching

**Plan mode** makes Claude Code **research and propose a plan without making any changes**, so you can approve the approach before it edits files or runs commands. It is a **read-only, safety-first** stance: explore the codebase, outline steps, wait for go-ahead.

Use it for: large or risky changes, unfamiliar codebases, anything where an unwanted edit is costly. The exam likes plan mode as the answer to "how do you let Claude tackle a big refactor while keeping a human approval gate before mutations happen?"

---

## 7. Skills, subagents, and sessions (the newer surface area)

- **Skills** — packaged, reusable capabilities/instructions (a folder with a `SKILL.md` and optional scripts) that load automatically when relevant. They extend what Claude Code can do without bloating every prompt. Loaded from the project's `.claude/` and from `~/.claude/`.
- **Subagents** — specialized agents Claude Code can spawn for focused subtasks, each with its **own context window** and tool set. Delegating a self-contained subtask (e.g., "review this file for security issues") to a subagent keeps the main context clean and enables parallelism. (Orchestration patterns: `AGENTIC_PATTERNS_DEEP_DIVE.md`.)
- **Sessions** — conversations persist and can be **resumed or forked**; long sessions use **context summarization/compaction** to stay within the window. "Continue where I left off" = resuming a session with summarized prior context.
- **Plugins** — bundle skills, subagents, hooks, commands, and MCP servers into an installable package.

---

## 8. Configuration hierarchy — the unifying picture

Everything in Claude Code follows the same shape. Internalize this table; it answers a large share of the domain:

| Concern | File / mechanism | Broadest → most specific |
|---|---|---|
| Instructions / conventions | `CLAUDE.md` | enterprise → user → project → subdirectory |
| Behavior / permissions | `settings.json` | enterprise (managed) → local → project → user |
| MCP servers | `.mcp.json` / config | local → project → user |
| Workflows | slash commands | project `.claude/commands/` / user `~/.claude/commands/` |

**The mental model:** *the closer a setting is to the specific work being done (and the higher the enterprise policy), the more it wins.* Enterprise policy is the outer guardrail nobody overrides; within that, specific beats general.

---

## 9. Worked example: onboarding Claude Code to a team repo

An architect sets up a monorepo so every engineer's Claude Code behaves consistently:

1. **Root `CLAUDE.md`** (committed): build/test commands, code style, "never edit `generated/`," how to run the app.
2. **`services/payments/CLAUDE.md`**: extra rules for the sensitive payments service (must run compliance checks, PCI notes).
3. **`.claude/settings.json`** (committed): deny auto-running destructive shell commands; require approval for `git push`.
4. **`.mcp.json`** (committed, **project** scope): the team's issue-tracker and database MCP servers, so everyone gets them on clone.
5. **`.claude/commands/review.md`**: a `/review` slash command encoding the PR-review checklist.
6. Engineers keep personal tweaks in **`~/.claude/CLAUDE.md`** and **`CLAUDE.local.md`**, which never affect teammates.

Each choice maps to the right level of the hierarchy — that mapping *is* the exam skill.

---

## 10. Rapid-fire self-check

1. `CLAUDE.md` vs `settings.json` — what does each govern? *(Instructions/context vs. behavior/permissions.)*
2. Team-wide coding conventions on a repo: which file, committed or not? *(Root `CLAUDE.md`, committed.)*
3. A personal rule for just you in just this repo, not shared? *(`CLAUDE.local.md`, git-ignored.)*
4. What does plan mode guarantee? *(Research/propose only; no changes until you approve.)*
5. MCP prompts appear in Claude Code as…? *(Slash commands.)*
6. MCP server config precedence? *(local > project > user.)*
7. Why give a subtask to a subagent? *(Isolated context + tool set; keeps main context clean; enables parallelism.)*

---

## 11. Further reading

- Claude Code overview — `https://code.claude.com/docs/en/overview`
- Memory / `CLAUDE.md` — `https://code.claude.com/docs/en/memory`
- Settings — `https://code.claude.com/docs/en/settings`
- Slash commands — `https://code.claude.com/docs/en/slash-commands`
- Subagents — `https://code.claude.com/docs/en/sub-agents`
- Hooks — `https://code.claude.com/docs/en/hooks`
- MCP — `https://code.claude.com/docs/en/mcp`
