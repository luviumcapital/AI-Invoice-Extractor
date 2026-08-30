# Project routing configuration

This project ships a large local `.claude/` configuration: 287 skills
(`.claude/skills/`), 68 agents (`.claude/agents/`), and 23 language/framework
rule sets (`.claude/rules/`). Loading every one of those files into context
at session startup would exhaust the token budget, so this project uses a
dynamic, index-first loading strategy instead.

## Startup rule

On session startup, load **only** `.claude/SKILLS_INDEX.md`. Do **not**
bulk-scan `.claude/skills/`, `.claude/agents/`, or `.claude/rules/` — the
index is a generated summary (1-sentence description + trigger keywords +
file path) of every skill, agent, and rule file, organized by category.

## Dynamic loading rule

When a user request matches a skill, agent, or rule-set keyword in
`SKILLS_INDEX.md`, dynamically read only that targeted file from
`.claude/skills/<skill_name>/SKILL.md` (or the relevant
`.claude/agents/<agent_name>.md` / `.claude/rules/<lang>/<file>.md`) into
active context. Do not read sibling skills/agents/rules "just in case" —
load exactly the matched file(s) and nothing more.

If a request could plausibly match more than one entry, prefer the
narrowest, most specific match in `SKILLS_INDEX.md` rather than loading
several candidates speculatively.

If nothing in `SKILLS_INDEX.md` matches a request, proceed without loading
any skill/agent/rule file rather than scanning the directories to look for
one.

## Token efficiency rule

When working inside this repo, prefer targeted file tools over reading
whole files into context:

- Use `Grep`/`ripgrep` to locate a symbol, string, or section instead of
  reading an entire file to find it.
- Use ranged/offset reads (e.g. `sed -n`, or a file-read tool's line-range
  option) to pull only the relevant section of a large file, rather than
  reading it in full.
- Reserve full-file reads for files you are about to edit in full, or that
  are already small.

This applies to `.claude/` assets and to the rest of the codebase alike.

## Regenerating the index

`SKILLS_INDEX.md` is generated from the contents of `.claude/skills/`,
`.claude/agents/`, and `.claude/rules/`. If those directories change
(skills/agents/rules added, removed, or renamed), regenerate the index
rather than hand-editing it, so it stays an accurate map of what is
actually installed.
