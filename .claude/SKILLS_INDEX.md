# .claude Skills / Agents / Rules Index

Auto-generated summary of everything under `.claude/skills/`, `.claude/agents/`, and `.claude/rules/`. This file is the **only** thing loaded into context by default (see `.claude/CLAUDE.md`). When a request matches an entry below, read that entry's target file on demand rather than scanning the full `.claude/` tree.

Totals: **287 skills**, **68 agents**, **121 rule files** across **22 rule sets**.

## Skills

### Agent & AI Engineering (75)

| Skill | Description | Trigger keywords | File |
|---|---|---|---|
| `agent-eval` | Head-to-head comparison of coding agents (Claude Code, Aider, Codex, etc.) on custom tasks with pass rate, cost, time, and consistency metrics. | eval, head, comparison, coding, claude, code, aider, codex | `.claude/skills/agent-eval/SKILL.md` |
| `agent-harness-construction` | Design and optimize AI agent action spaces, tool definitions, and observation formatting for higher completion rates. | harness, construction, design, optimize, action, spaces, tool, definitions | `.claude/skills/agent-harness-construction/SKILL.md` |
| `agent-introspection-debugging` | Structured self-debugging workflow for AI agent failures using capture, diagnosis, contained recovery, and introspection reports. | introspection, debugging, structured, self, workflow, failures, capture, diagnosis | `.claude/skills/agent-introspection-debugging/SKILL.md` |
| `agent-payment-x402` | Add x402 payment execution to AI agents with per-task budgets, spending controls, and non-custodial wallets. | payment, x402, add, execution, per, task, budgets, spending | `.claude/skills/agent-payment-x402/SKILL.md` |
| `agent-self-evaluation` | Use after completing any non-trivial task. | self, evaluation, after, completing, non, trivial, task., rates | `.claude/skills/agent-self-evaluation/SKILL.md` |
| `agent-sort` | Build an evidence-backed ECC install plan for a specific repo by sorting skills, commands, rules, hooks, and extras into DAILY vs LIBRARY buckets using parallel repo-aware review passes. | sort, build, evidence, backed, ecc, install, plan, specific | `.claude/skills/agent-sort/SKILL.md` |
| `agentic-engineering` | Operate as an agentic engineer using eval-first execution, decomposition, and cost-aware model routing. | agentic, engineering, operate, engineer, eval, first, execution, decomposition | `.claude/skills/agentic-engineering/SKILL.md` |
| `agentic-os` | Build persistent multi-agent operating systems on Claude Code. | agentic, build, persistent, multi, operating, systems, claude, code. | `.claude/skills/agentic-os/SKILL.md` |
| `ai-first-engineering` | Engineering operating model for teams where AI agents generate a large share of implementation output. | first, engineering, operating, model, teams, where, generate, large | `.claude/skills/ai-first-engineering/SKILL.md` |
| `article-writing` | Write articles, guides, blog posts, tutorials, newsletter issues, and other long-form content in a distinctive voice derived from supplied examples or brand guidance. | article, writing, write, articles, guides, blog, posts, tutorials | `.claude/skills/article-writing/SKILL.md` |
| `automation-audit-ops` | Evidence-first automation inventory and overlap audit workflow for ECC. | automation, audit, ops, evidence, first, inventory, overlap, workflow | `.claude/skills/automation-audit-ops/SKILL.md` |
| `autonomous-agent-harness` | Transform Claude Code into a fully autonomous agent system with persistent memory, scheduled operations, computer use, and task queuing. | autonomous, harness, transform, claude, code, fully, system, persistent | `.claude/skills/autonomous-agent-harness/SKILL.md` |
| `autonomous-loops` | Patterns and architectures for autonomous Claude Code loops — from simple sequential pipelines to RFC-driven multi-agent DAG systems. | autonomous, loops, patterns, architectures, claude, code, simple, sequential | `.claude/skills/autonomous-loops/SKILL.md` |
| `blueprint` | >- Turn a one-line objective into a step-by-step construction plan for multi-session, multi-agent engineering projects. | blueprint, turn, one, line, objective, step, construction, plan | `.claude/skills/blueprint/SKILL.md` |
| `claude-devfleet` | Orchestrate multi-agent coding tasks via Claude DevFleet — plan projects, dispatch parallel agents in isolated worktrees, monitor progress, and read structured reports. | claude, devfleet, orchestrate, multi, coding, tasks, via, plan | `.claude/skills/claude-devfleet/SKILL.md` |
| `codehealth-mcp` | Real-time structural Code Health via CodeScene MCP — review before edits, verify score deltas after changes, gate commits and PRs. | codehealth, mcp, real, time, structural, code, health, via | `.claude/skills/codehealth-mcp/SKILL.md` |
| `config-gc` | Garbage collection for your Claude Code configuration. | config, garbage, collection, claude, code, configuration., periodically, scans | `.claude/skills/config-gc/SKILL.md` |
| `context-budget` | Audits Claude Code context window consumption across agents, skills, MCP servers, and rules. | context, budget, audits, claude, code, window, consumption, across | `.claude/skills/context-budget/SKILL.md` |
| `continuous-agent-loop` | Patterns for continuous autonomous agent loops with quality gates, evals, and recovery controls. | continuous, loop, patterns, autonomous, loops, quality, gates, evals | `.claude/skills/continuous-agent-loop/SKILL.md` |
| `continuous-learning-v2` | Instinct-based learning system that observes sessions via hooks, creates atomic instincts with confidence scoring, and evolves them into skills/commands/agents. | continuous, learning, instinct, based, system, observes, sessions, via | `.claude/skills/continuous-learning-v2/SKILL.md` |
| `cost-aware-llm-pipeline` | Cost optimization patterns for LLM API usage — model routing by task complexity, budget tracking, retry logic, and prompt caching. | cost, aware, llm, pipeline, optimization, patterns, api, usage | `.claude/skills/cost-aware-llm-pipeline/SKILL.md` |
| `data-scraper-agent` | Build a fully automated AI-powered data collection agent for any public source — job boards, prices, news, GitHub, sports, anything. | data, scraper, build, fully, automated, powered, collection, public | `.claude/skills/data-scraper-agent/SKILL.md` |
| `deep-research` | Multi-source deep research using firecrawl and exa MCPs. | deep, research, multi, source, firecrawl, exa, mcps., searches | `.claude/skills/deep-research/SKILL.md` |
| `dmux-workflows` | Multi-agent orchestration using dmux (tmux pane manager for AI agents). | dmux, workflows, multi, orchestration, tmux, pane, manager, patterns | `.claude/skills/dmux-workflows/SKILL.md` |
| `documentation-lookup` | Use up-to-date library and framework docs via Context7 MCP instead of training data. | documentation, lookup, date, library, framework, docs, via, context7 | `.claude/skills/documentation-lookup/SKILL.md` |
| `dynamic-workflow-mode` | Design task-local harnesses, eval gates, and reusable skill extraction for Claude dynamic workflow mode and other adaptive agent harnesses. | dynamic, workflow, mode, design, task, local, harnesses, eval | `.claude/skills/dynamic-workflow-mode/SKILL.md` |
| `ecc-guide` | Guide users through ECC's current agents, skills, commands, hooks, rules, install profiles, and project onboarding by reading the live repository surface before answering. | ecc, guide, users, through, current, commands, hooks, rules | `.claude/skills/ecc-guide/SKILL.md` |
| `ecc-recipes` | Map a described workflow to the right ECC command-GROUP with run-order and stop condition, and browse all command-group recipe families. | ecc, recipes, map, described, workflow, right, command, group | `.claude/skills/ecc-recipes/SKILL.md` |
| `energy-procurement` | > Codified expertise for electricity and gas procurement, tariff optimization, demand charge management, renewable PPA evaluation, and multi-facility energy cost management. | energy, procurement, codified, expertise, electricity, gas, tariff, optimization | `.claude/skills/energy-procurement/SKILL.md` |
| `eval-harness` | Formal evaluation framework for Claude Code sessions implementing eval-driven development (EDD) principles. | eval, harness, formal, evaluation, framework, claude, code, sessions | `.claude/skills/eval-harness/SKILL.md` |
| `exa-search` | Neural search via Exa MCP for web, code, and company research. | exa, search, neural, via, mcp, web, code, company | `.claude/skills/exa-search/SKILL.md` |
| `fal-ai-media` | Unified media generation via fal.ai MCP — image, video, and audio. | fal, media, unified, generation, via, fal.ai, mcp, image | `.claude/skills/fal-ai-media/SKILL.md` |
| `flox-environments` | Create reproducible, cross-platform (macOS/Linux) development environments with Flox, a declarative Nix-based environment manager. | flox, environments, create, reproducible, cross, platform, macos, linux | `.claude/skills/flox-environments/SKILL.md` |
| `foundation-models-on-device` | Apple FoundationModels framework for on-device LLM — text generation, guided generation with @Generable, tool calling, and snapshot streaming in iOS 26+. | foundation, models, device, apple, foundationmodels, framework, llm, text | `.claude/skills/foundation-models-on-device/SKILL.md` |
| `gan-style-harness` | GAN-inspired Generator-Evaluator agent harness for building high-quality applications autonomously. | gan, style, harness, inspired, generator, evaluator, building, high | `.claude/skills/gan-style-harness/SKILL.md` |
| `gateguard` | Fact-forcing gate that blocks Edit/Write/Bash (including MultiEdit) and demands concrete investigation (importers, data schemas, user instruction) before allowing the action. | gateguard, fact, forcing, gate, blocks, edit, write, bash | `.claude/skills/gateguard/SKILL.md` |
| `inherit-legacy-style` | Legacy-project style inheritance skill. | inherit, legacy, style, project, inheritance, skill., user, types | `.claude/skills/inherit-legacy-style/SKILL.md` |
| `iterative-retrieval` | Pattern for progressively refining context retrieval to solve the subagent context problem. | iterative, retrieval, pattern, progressively, refining, context, solve, subagent | `.claude/skills/iterative-retrieval/SKILL.md` |
| `ito-inference` | Inspect the availability of model serving on a completed Itô compute booking and, when the canonical backend becomes available, hand off an explicitly confirmed serving manifest. | ito, inference, inspect, availability, model, serving, completed, compute | `.claude/skills/ito-inference/SKILL.md` |
| `jira-integration` | Use this skill when retrieving Jira tickets, analyzing requirements, updating ticket status, adding comments, or transitioning issues. | jira, integration, retrieving, tickets, analyzing, requirements, updating, ticket | `.claude/skills/jira-integration/SKILL.md` |
| `knowledge-ops` | Knowledge base management, ingestion, sync, and retrieval across multiple storage layers (local files, MCP memory, vector stores, Git repos). | knowledge, ops, base, management, ingestion, sync, retrieval, across | `.claude/skills/knowledge-ops/SKILL.md` |
| `laravel-plugin-discovery` | Discover and evaluate Laravel packages via LaraPlugins.io MCP. | laravel, plugin, discovery, discover, evaluate, packages, via, laraplugins.io | `.claude/skills/laravel-plugin-discovery/SKILL.md` |
| `latency-critical-systems` | Use for latency-sensitive systems such as realtime dashboards, market data, streaming agents, execution gateways, queues, caches, or HFT-like infrastructure where freshness and p95 latency matter. | latency, critical, systems, sensitive, realtime, dashboards, market, data | `.claude/skills/latency-critical-systems/SKILL.md` |
| `lead-intelligence` | AI-native lead intelligence and outreach pipeline. | lead, intelligence, native, outreach, pipeline., replaces, apollo, clay | `.claude/skills/lead-intelligence/SKILL.md` |
| `living-docs-governance` | Keep a long-lived project's documentation from rotting by assigning existing project docs clear constitution, map, status, and history roles, then wiring the active agent harness to those canonical sources. | living, docs, governance, keep, long, lived, project, documentation | `.claude/skills/living-docs-governance/SKILL.md` |
| `loop-design-check` | Design a goal-oriented agent loop, and review it for the ways loops go wrong — spinning and burning tokens, Goodhart-gaming the verifier, or running a wrong answer to completion. | loop, design, check, goal, oriented, review, ways, loops | `.claude/skills/loop-design-check/SKILL.md` |
| `marketing-campaign` | End-to-end marketing campaign planning and execution. | marketing, campaign, end, planning, execution., covers, audience, research | `.claude/skills/marketing-campaign/SKILL.md` |
| `ml-adoption-playbook` | End-to-end methodology for AI agents and software engineers to add machine learning algorithms to existing non-ML codebases. | adoption, playbook, end, methodology, software, engineers, add, machine | `.claude/skills/ml-adoption-playbook/SKILL.md` |
| `mle-workflow` | Production machine-learning engineering workflow for data contracts, reproducible training, model evaluation, deployment, monitoring, and rollback. | mle, workflow, production, machine, learning, engineering, data, contracts | `.claude/skills/mle-workflow/SKILL.md` |
| `motion-advanced` | Advanced motion patterns for React / Next.js — drag & drop, gestures, text animations, SVG path drawing, custom hooks, imperative sequences (useAnimate), loaders, and the full API decision tree. | motion, advanced, patterns, react, next.js, drag, drop, gestures | `.claude/skills/motion-advanced/SKILL.md` |
| `nodejs-keccak256` | Prevent Ethereum hashing bugs in JavaScript and TypeScript. | nodejs, keccak256, prevent, ethereum, hashing, bugs, javascript, typescript. | `.claude/skills/nodejs-keccak256/SKILL.md` |
| `openclaw-persona-forge` | 为 OpenClaw AI Agent 锻造完整的龙虾灵魂方案。根据用户偏好或随机抽卡， 输出身份定位、灵魂描述(SOUL.md)、角色化底线规则、名字和头像生图提示词。 如当前环境提供已审核的生图 skill，可自动生成统一风格头像图片。 当用户需要创建、设计或定制 OpenClaw 龙虾灵魂时使用。 不适用于：微调已有 SOUL.md、非 OpenClaw 平台的角色设计、纯工具型无性格 Agent。... | openclaw, persona, forge, soul.md, npc, lobster, soul, character | `.claude/skills/openclaw-persona-forge/SKILL.md` |
| `parallel-execution-optimizer` | Use when the user wants a task done much faster through parallel work, concurrent agents, batched tool calls, isolated worktrees, or many independent verification lanes without losing correctness. | parallel, execution, optimizer, user, wants, task, done, much | `.claude/skills/parallel-execution-optimizer/SKILL.md` |
| `plan-orchestrate` | Read a plan document, decompose it into steps, design a per-step agent chain from the ECC catalogue, and emit ready-to-paste /orchestrate custom prompts. | plan, orchestrate, read, document, decompose, steps, design, per | `.claude/skills/plan-orchestrate/SKILL.md` |
| `prediction-market-oracle-research` | Research prediction markets as data sources or oracle signals for products, agents, dashboards, and corporate decision intelligence. | prediction, market, oracle, research, markets, data, sources, signals | `.claude/skills/prediction-market-oracle-research/SKILL.md` |
| `prompt-optimizer` | >- Analyze raw prompts, identify intent and gaps, match ECC components (skills/commands/agents/hooks), and output a ready-to-paste optimized prompt. | prompt, optimizer, analyze, raw, prompts, identify, intent, gaps | `.claude/skills/prompt-optimizer/SKILL.md` |
| `pubmed-database` | Direct PubMed and NCBI E-utilities search workflows for biomedical literature, MeSH queries, PMID lookup, citation retrieval, and API-backed literature monitoring. | pubmed, database, direct, ncbi, utilities, search, workflows, biomedical | `.claude/skills/scientific-db-pubmed-database/SKILL.md` |
| `ralphinho-rfc-pipeline` | RFC-driven multi-agent DAG execution pattern with quality gates, merge queues, and work unit orchestration. | ralphinho, rfc, pipeline, driven, multi, dag, execution, pattern | `.claude/skills/ralphinho-rfc-pipeline/SKILL.md` |
| `react-native-patterns` | React Native and Expo app patterns — Expo Router navigation, state separation (server/client/route/form), TanStack Query data fetching with Zod, performant lists, NativeWind/StyleSheet styling, native APIs, and secure... | react, native, patterns, expo, app, router, navigation, state | `.claude/skills/react-native-patterns/SKILL.md` |
| `react-performance` | React and Next.js performance optimization patterns adapted from Vercel Engineering's React Best Practices (https://github.com/vercel-labs/agent-skills). | react, performance, next.js, optimization, patterns, adapted, vercel, engineering | `.claude/skills/react-performance/SKILL.md` |
| `recsys-pipeline-architect` | Design composable recommendation, ranking, and feed pipelines using the six-stage Source→Hydrator→Filter→Scorer→Selector→SideEffect framework popularized by xAI's open-sourced For You algorithm. | recsys, pipeline, architect, design, composable, recommendation, ranking, feed | `.claude/skills/recsys-pipeline-architect/SKILL.md` |
| `regex-vs-llm-structured-text` | Decision framework for choosing between regex and LLM when parsing structured text — start with regex, add LLM only for low-confidence edge cases. | regex, llm, structured, text, decision, framework, choosing, between | `.claude/skills/regex-vs-llm-structured-text/SKILL.md` |
| `safety-guard` | Use this skill to prevent destructive operations when working on production systems or running agents autonomously. | safety, guard, prevent, destructive, operations, working, production, systems | `.claude/skills/safety-guard/SKILL.md` |
| `santa-method` | Multi-agent adversarial verification with convergence loop. | santa, method, multi, adversarial, verification, convergence, loop., two | `.claude/skills/santa-method/SKILL.md` |
| `scholar-evaluation` | Structured scholarly-work evaluation for papers, proposals, literature reviews, methods sections, evidence quality, citation support, and research-writing feedback. | scholar, evaluation, structured, scholarly, work, papers, proposals, literature | `.claude/skills/scientific-thinking-scholar-evaluation/SKILL.md` |
| `search-first` | Research-before-coding workflow. | search, first, research, before, coding, workflow., existing, tools | `.claude/skills/search-first/SKILL.md` |
| `skill-stocktake` | Use when auditing Claude skills and commands for quality. | stocktake, auditing, claude, commands, quality., supports, quick, scan | `.claude/skills/skill-stocktake/SKILL.md` |
| `social-publisher` | Agent-driven scheduling and publishing of social media posts across 13 platforms via SocialClaw. | social, publisher, driven, scheduling, publishing, media, posts, across | `.claude/skills/social-publisher/SKILL.md` |
| `swift-actor-persistence` | Thread-safe data persistence in Swift using actors — in-memory cache with file-backed storage, eliminating data races by design. | swift, actor, persistence, thread, safe, data, actors, memory | `.claude/skills/swift-actor-persistence/SKILL.md` |
| `taste` | A creative-direction (taste) layer for music videos and short-form edits in the angelcore / cloud-trance / hyperpop visual family. | taste, creative, direction, layer, music, videos, short, form | `.claude/skills/taste/SKILL.md` |
| `team-agent-orchestration` | Run team-based orchestration for agent squads using work items, ownership, agent Kanban, merge gates, and control pane handoffs. | team, orchestration, run, based, squads, work, items, ownership | `.claude/skills/team-agent-orchestration/SKILL.md` |
| `team-builder` | Interactive agent picker for composing and dispatching parallel teams. | team, builder, interactive, picker, composing, dispatching, parallel, teams. | `.claude/skills/team-builder/SKILL.md` |
| `unified-memory` | Share durable, inspectable context and handoffs between Claude, Codex, Hermes, Cursor, OpenCode, and other agents through the local ECC Memory Vault. | unified, memory, share, durable, inspectable, context, handoffs, between | `.claude/skills/unified-memory/SKILL.md` |
| `video-editing` | AI-assisted video editing workflows for cutting, structuring, and augmenting real footage. | video, editing, assisted, workflows, cutting, structuring, augmenting, real | `.claude/skills/video-editing/SKILL.md` |
| `workspace-surface-audit` | Audit the active repo, MCP servers, plugins, connectors, env surfaces, and harness setup, then recommend the highest-value ECC-native skills, hooks, agents, and operator workflows. | workspace, surface, audit, active, repo, mcp, servers, plugins | `.claude/skills/workspace-surface-audit/SKILL.md` |

### Architecture & Code Quality (6)

| Skill | Description | Trigger keywords | File |
|---|---|---|---|
| `code-tour` | Create CodeTour `.tour` files — persona-targeted, step-by-step walkthroughs with real file and line anchors. | code, tour, create, codetour, files, persona, targeted, step | `.claude/skills/code-tour/SKILL.md` |
| `continuous-learning` | [DEPRECATED - use continuous-learning-v2] Legacy v1 stop-hook skill extractor. | continuous, learning, deprecated, legacy, stop, hook, extractor., strict | `.claude/skills/continuous-learning/SKILL.md` |
| `growth-log` | Use after a complex task, failure, or when reviewing what was learned. | growth, log, after, complex, task, failure, reviewing, what | `.claude/skills/growth-log/SKILL.md` |
| `netmiko-ssh-automation` | Safe Python Netmiko patterns for read-only collection, bounded batch SSH, TextFSM parsing, guarded config changes, timeouts, and network automation error handling. | netmiko, ssh, automation, safe, python, patterns, read, only | `.claude/skills/netmiko-ssh-automation/SKILL.md` |
| `network-bgp-diagnostics` | Diagnostics-only BGP troubleshooting patterns for neighbor state, route exchange, prefix policy, AS path inspection, and safe evidence collection. | network, bgp, diagnostics, only, troubleshooting, patterns, neighbor, state | `.claude/skills/network-bgp-diagnostics/SKILL.md` |
| `repo-scan` | Bootstrap pointer that installs the external repo-scan skill from a pinned, reviewable commit. | repo, scan, bootstrap, pointer, installs, external, pinned, reviewable | `.claude/skills/repo-scan/SKILL.md` |

### Backend & APIs (10)

| Skill | Description | Trigger keywords | File |
|---|---|---|---|
| `clickhouse-io` | ClickHouse database patterns, query optimization, analytics, and data engineering best practices for high-performance analytical workloads. | clickhouse, database, patterns, query, optimization, analytics, data, engineering | `.claude/skills/clickhouse-io/SKILL.md` |
| `content-hash-cache-pattern` | Cache expensive file processing results using SHA-256 content hashes — path-independent, auto-invalidating, with service layer separation. | content, hash, cache, pattern, expensive, file, processing, results | `.claude/skills/content-hash-cache-pattern/SKILL.md` |
| `contract-first` | Use when multiple consumers and providers must evolve an API or event schema without field drift, integration surprises, or one side silently redefining the interface. | contract, first, multiple, consumers, providers, must, evolve, api | `.claude/skills/contract-first/SKILL.md` |
| `database-migrations` | Database migration best practices for schema changes, data migrations, rollbacks, and zero-downtime deployments across PostgreSQL, MySQL, and common ORMs (Prisma, Drizzle, Kysely, Django, TypeORM, golang-migrate). | database, migrations, migration, best, practices, schema, changes, data | `.claude/skills/database-migrations/SKILL.md` |
| `homelab-vlan-segmentation` | Segmenting home networks into VLANs for IoT, guest, trusted, and server traffic using UniFi, pfSense/OPNsense, and MikroTik — including switch trunk config, firewall rules, and wireless SSID mapping. | homelab, vlan, segmentation, segmenting, home, networks, vlans, iot | `.claude/skills/homelab-vlan-segmentation/SKILL.md` |
| `homelab-wireguard-vpn` | WireGuard VPN server setup, peer configuration, key generation, split tunneling vs full tunnel routing, and remote access to a home network from mobile and laptop clients. | homelab, wireguard, vpn, server, setup, peer, configuration, key | `.claude/skills/homelab-wireguard-vpn/SKILL.md` |
| `ios-icon-gen` | Generate iOS app icons as PNG imagesets for Xcode asset catalogs from SF Symbols (5000+ Apple-native) or Iconify API (275k+ open source icons from 200+ collections). | ios, icon, gen, generate, app, icons, png, imagesets | `.claude/skills/ios-icon-gen/SKILL.md` |
| `ito-training` | Inspect the availability of ML training on a completed Itô compute booking and, when the canonical backend becomes available, hand off an explicitly confirmed training manifest. | ito, training, inspect, availability, completed, compute, booking, canonical | `.claude/skills/ito-training/SKILL.md` |
| `nutrient-document-processing` | Process, convert, OCR, extract, redact, sign, and fill documents using the Nutrient DWS API. | nutrient, document, processing, process, convert, ocr, extract, redact | `.claude/skills/nutrient-document-processing/SKILL.md` |
| `uspto-database` | USPTO patent and trademark data workflow for official record lookup, PatentSearch queries, TSDR checks, assignment data, and reproducible IP research logs. | uspto, database, patent, trademark, data, workflow, official, record | `.claude/skills/scientific-db-uspto-database/SKILL.md` |

### Data & Analytics (5)

| Skill | Description | Trigger keywords | File |
|---|---|---|---|
| `data-throughput-accelerator` | Use when large data ingestion, backfill, export, ETL, warehouse loading, manifest catch-up, or table synchronization needs to become much faster while preserving data correctness. | data, throughput, accelerator, large, ingestion, backfill, export, etl | `.claude/skills/data-throughput-accelerator/SKILL.md` |
| `plan-canvas` | Open plans and HTML artifacts in a local browser canvas where the human annotates elements, chats, and approves or requests changes without leaving the page. | plan, canvas, open, plans, html, artifacts, local, browser | `.claude/skills/plan-canvas/SKILL.md` |
| `production-audit` | Local-evidence production readiness audit for shipped apps, pre-launch reviews, post-merge checks, and "what breaks in prod?" questions without sending repo data to an external audit service. | production, audit, local, evidence, readiness, shipped, apps, pre | `.claude/skills/production-audit/SKILL.md` |
| `quality-nonconformance` | > Codified expertise for quality control, non-conformance investigation, root cause analysis, corrective action, and supplier quality management in regulated manufacturing. | quality, nonconformance, codified, expertise, control, non, conformance, investigation | `.claude/skills/quality-nonconformance/SKILL.md` |
| `tasteforge-video` | Use for file-driven multimodal image, video, and 3D-asset discovery; taste interviews; distill or apply workflows; style-pack validation; editable EDL/FCPXML export; provenance audits; and offline planning that must... | tasteforge, video, file, driven, multimodal, image, asset, discovery | `.claude/skills/tasteforge-video/SKILL.md` |

### DevOps & Infrastructure (27)

| Skill | Description | Trigger keywords | File |
|---|---|---|---|
| `architecture-decision-records` | Capture architectural decisions made during Claude Code sessions as structured ADRs. | architecture, decision, records, capture, architectural, decisions, made, during | `.claude/skills/architecture-decision-records/SKILL.md` |
| `benchmark-methodology` | >- Use after competitive-platform-analysis has produced a tiered competitor set. | benchmark, methodology, after, competitive, platform, analysis, has, produced | `.claude/skills/benchmark-methodology/SKILL.md` |
| `competitive-platform-analysis` | >- Use when scoping a competitive landscape — identifying, categorising, and score-filtering a competitor set before any benchmarking begins. | competitive, platform, analysis, scoping, landscape, identifying, categorising, score | `.claude/skills/competitive-platform-analysis/SKILL.md` |
| `competitive-report-structure` | >- Use after benchmark-methodology has produced scored competitor profile cards. | competitive, report, structure, after, benchmark, methodology, has, produced | `.claude/skills/competitive-report-structure/SKILL.md` |
| `connections-optimizer` | Reorganize the user's X and LinkedIn network with review-first pruning, add/follow recommendations, and channel-specific warm outreach drafted in the user's real voice. | connections, optimizer, reorganize, user, linkedin, network, review, first | `.claude/skills/connections-optimizer/SKILL.md` |
| `content-engine` | Create platform-native content systems for X, LinkedIn, TikTok, YouTube, newsletters, and repurposed multi-platform campaigns. | content, engine, create, platform, native, systems, linkedin, tiktok | `.claude/skills/content-engine/SKILL.md` |
| `council` | Convene a four-voice council for ambiguous decisions, tradeoffs, and go/no-go calls. | council, convene, four, voice, ambiguous, decisions, tradeoffs, calls. | `.claude/skills/council/SKILL.md` |
| `crosspost` | Multi-platform content distribution across X, LinkedIn, Threads, and Bluesky. | crosspost, multi, platform, content, distribution, across, linkedin, threads | `.claude/skills/crosspost/SKILL.md` |
| `delivery-gate` | Stop hook that blocks Claude from finishing until quality checks pass. | delivery, gate, stop, hook, blocks, claude, finishing, until | `.claude/skills/delivery-gate/SKILL.md` |
| `evm-token-decimals` | Prevent silent decimal mismatch bugs across EVM chains. | evm, token, decimals, prevent, silent, decimal, mismatch, bugs | `.claude/skills/evm-token-decimals/SKILL.md` |
| `finance-billing-ops` | Evidence-first revenue, pricing, refunds, team-billing, and billing-model truth workflow for ECC. | finance, billing, ops, evidence, first, revenue, pricing, refunds | `.claude/skills/finance-billing-ops/SKILL.md` |
| `git-workflow` | Git workflow patterns including branching strategies, commit conventions, merge vs rebase, conflict resolution, and collaborative development best practices for teams of all sizes. | git, workflow, patterns, branching, strategies, commit, conventions, merge | `.claude/skills/git-workflow/SKILL.md` |
| `graphify` | Use for any question about a codebase, its architecture, file relationships, or project content — especially when graphify-out/ exists, where the question should be treated as a graphify query first. | graphify, question, about, codebase, its, architecture, file, relationships | `.claude/skills/graphify/SKILL.md` |
| `homelab-pihole-dns` | Pi-hole installation, blocklist management, DNS-over-HTTPS setup, DHCP integration, local DNS records, and troubleshooting broken DNS resolution on a home network. | homelab, pihole, dns, hole, installation, blocklist, management, over | `.claude/skills/homelab-pihole-dns/SKILL.md` |
| `investor-materials` | Create and update pitch decks, one-pagers, investor memos, accelerator applications, financial models, and fundraising materials. | investor, materials, create, update, pitch, decks, one, pagers | `.claude/skills/investor-materials/SKILL.md` |
| `investor-outreach` | Draft cold emails, warm intro blurbs, follow-ups, update emails, and investor communications for fundraising. | investor, outreach, draft, cold, emails, warm, intro, blurbs | `.claude/skills/investor-outreach/SKILL.md` |
| `literature-review` | Systematic literature-review workflow for academic, biomedical, technical, and scientific topics, including search planning, source screening, synthesis, citation checks, and evidence logging. | literature, review, systematic, workflow, academic, biomedical, technical, scientific | `.claude/skills/scientific-thinking-literature-review/SKILL.md` |
| `logistics-exception-management` | > Codified expertise for handling freight exceptions, shipment delays, damages, losses, and carrier disputes. | logistics, exception, management, codified, expertise, handling, freight, exceptions | `.claude/skills/logistics-exception-management/SKILL.md` |
| `market-research` | Conduct market research, competitive analysis, investor due diligence, and industry intelligence with source attribution and decision-oriented summaries. | market, research, conduct, competitive, analysis, investor, due, diligence | `.claude/skills/market-research/SKILL.md` |
| `product-capability` | Translate PRD intent, roadmap asks, or product discussions into an implementation-ready capability plan that exposes constraints, invariants, interfaces, and unresolved decisions before multi-service work starts. | product, capability, translate, prd, intent, roadmap, asks, discussions | `.claude/skills/product-capability/SKILL.md` |
| `production-scheduling` | > Codified expertise for production scheduling, job sequencing, line balancing, changeover optimization, and bottleneck resolution in discrete and batch manufacturing. | production, scheduling, codified, expertise, job, sequencing, line, balancing | `.claude/skills/production-scheduling/SKILL.md` |
| `project-flow-ops` | Operate execution flow across GitHub and Linear by triaging issues and pull requests, linking active work, and keeping GitHub public-facing while Linear remains the internal execution layer. | project, flow, ops, operate, execution, across, github, linear | `.claude/skills/project-flow-ops/SKILL.md` |
| `recursive-decision-ledger` | Use when the user asks for repeated rollouts, marked decision processes, high-dimensional search, stochastic optimization, local-optima exploration, ensemble comparison, or recursive reasoning with a visible evidence... | recursive, decision, ledger, user, asks, repeated, rollouts, marked | `.claude/skills/recursive-decision-ledger/SKILL.md` |
| `rules-distill` | Scan skills to extract cross-cutting principles and distill them into rules — append, revise, or create new rule files. | rules, distill, scan, extract, cross, cutting, principles, them | `.claude/skills/rules-distill/SKILL.md` |
| `social-graph-ranker` | Weighted social-graph ranking for warm intro discovery, bridge scoring, and network gap analysis across X and LinkedIn. | social, graph, ranker, weighted, ranking, warm, intro, discovery | `.claude/skills/social-graph-ranker/SKILL.md` |
| `terminal-ops` | Evidence-first repo execution workflow for ECC. | terminal, ops, evidence, first, repo, execution, workflow, ecc. | `.claude/skills/terminal-ops/SKILL.md` |
| `uncloud` | Use when managing an Uncloud cluster — deploying services, configuring Caddy ingress, adding static proxy routes for non-cluster devices, publishing ports, scaling, inspecting logs, or managing machines and volumes with... | uncloud, managing, cluster, deploying, services, configuring, caddy, ingress | `.claude/skills/uncloud/SKILL.md` |

### Documentation & Writing (3)

| Skill | Description | Trigger keywords | File |
|---|---|---|---|
| `cost-tracking` | Track and report Claude Code token usage, spending, and budgets from the local ECC cost-tracker metrics log. | cost, tracking, track, report, claude, code, token, usage | `.claude/skills/cost-tracking/SKILL.md` |
| `google-workspace-ops` | Operate across Google Drive, Docs, Sheets, and Slides as one workflow surface for plans, trackers, decks, and shared documents. | google, workspace, ops, operate, across, drive, docs, sheets | `.claude/skills/google-workspace-ops/SKILL.md` |
| `visa-doc-translate` | Translate visa application documents (images) to English and create a bilingual PDF with original and translation. | visa, doc, translate, application, documents, images, english, create | `.claude/skills/visa-doc-translate/SKILL.md` |

### Frontend & UI (63)

| Skill | Description | Trigger keywords | File |
|---|---|---|---|
| `accessibility` | Design, implement, and audit inclusive digital products using WCAG 2.2 Level AA. | accessibility, design, implement, audit, inclusive, digital, products, wcag | `.claude/skills/accessibility/SKILL.md` |
| `api-connector-builder` | Build a new API connector or provider by matching the target repo's existing integration pattern exactly. | api, connector, builder, build, new, provider, matching, target | `.claude/skills/api-connector-builder/SKILL.md` |
| `api-design` | REST API design patterns including resource naming, status codes, pagination, filtering, error responses, versioning, and rate limiting for production APIs. | api, design, rest, patterns, resource, naming, status, codes | `.claude/skills/api-design/SKILL.md` |
| `backend-patterns` | Backend architecture patterns, API design, database optimization, and server-side best practices for Node.js, Express, and Next.js API routes. | backend, patterns, architecture, api, design, database, optimization, server | `.claude/skills/backend-patterns/SKILL.md` |
| `blender-motion-state-inspection` | Use this skill when inspecting Blender characters, rigs, poses, animation retargeting, ground contact, facing direction, or model-vs-motion alignment where screenshots alone are not enough. | blender, motion, state, inspection, inspecting, characters, rigs, poses | `.claude/skills/blender-motion-state-inspection/SKILL.md` |
| `brand-voice` | Build a source-derived writing style profile from real posts, essays, launch notes, docs, or site copy, then reuse that profile across content, outreach, and social workflows. | brand, voice, build, source, derived, writing, style, profile | `.claude/skills/brand-voice/SKILL.md` |
| `click-path-audit` | Trace every user-facing button/touchpoint through its full state change sequence to find bugs where functions individually work but cancel each other out, produce wrong final state, or leave the UI in an inconsistent... | click, path, audit, trace, every, user, facing, button | `.claude/skills/click-path-audit/SKILL.md` |
| `codebase-onboarding` | Analyze an unfamiliar codebase and generate a structured onboarding guide with architecture map, key entry points, conventions, and a starter CLAUDE.md. | codebase, onboarding, analyze, unfamiliar, generate, structured, guide, architecture | `.claude/skills/codebase-onboarding/SKILL.md` |
| `coding-standards` | Baseline cross-project coding conventions for naming, readability, immutability, and code-quality review. | coding, standards, baseline, cross, project, conventions, naming, readability | `.claude/skills/coding-standards/SKILL.md` |
| `compose-multiplatform-patterns` | Compose Multiplatform and Jetpack Compose patterns for KMP projects — state management, navigation, theming, performance, and platform-specific UI. | compose, multiplatform, patterns, jetpack, kmp, projects, state, management | `.claude/skills/compose-multiplatform-patterns/SKILL.md` |
| `configure-ecc` | Guide ECC installation, update, or reconfiguration from inside Claude Code, Codex, or Kimi while respecting each harness's real plugin, scope, and hook capabilities. | configure, ecc, guide, installation, update, reconfiguration, inside, claude | `.claude/skills/configure-ecc/SKILL.md` |
| `council-multi-model` | Add one optional external Codex critique after the existing council has produced a decision draft. | council, multi, model, add, one, optional, external, codex | `.claude/skills/council-multi-model/SKILL.md` |
| `cpp-coding-standards` | C++ coding standards based on the C++ Core Guidelines (isocpp.github.io). | cpp, coding, standards, c++, based, core, guidelines, isocpp.github.io | `.claude/skills/cpp-coding-standards/SKILL.md` |
| `dashboard-builder` | Build monitoring dashboards that answer real operator questions for Grafana, SigNoz, and similar platforms. | dashboard, builder, build, monitoring, dashboards, answer, real, operator | `.claude/skills/dashboard-builder/SKILL.md` |
| `deployment-patterns` | Deployment workflows, CI/CD pipeline patterns, Docker containerization, health checks, rollback strategies, and production readiness checklists for web applications. | deployment, patterns, workflows, pipeline, docker, containerization, health, checks | `.claude/skills/deployment-patterns/SKILL.md` |
| `design-system` | Use this skill to generate or audit design systems, check visual consistency, and review PRs that touch styling. | design, system, generate, audit, systems, check, visual, consistency | `.claude/skills/design-system/SKILL.md` |
| `django-patterns` | Django architecture patterns, REST API design with DRF, ORM best practices, caching, signals, middleware, and production-grade Django apps. | django, patterns, architecture, rest, api, design, drf, orm | `.claude/skills/django-patterns/SKILL.md` |
| `dotnet-patterns` | Idiomatic C# and .NET patterns, conventions, dependency injection, async/await, and best practices for building robust, maintainable .NET applications. | dotnet, patterns, idiomatic, net, conventions, dependency, injection, async | `.claude/skills/dotnet-patterns/SKILL.md` |
| `error-handling` | Patterns for robust error handling across TypeScript, Python, and Go. | error, handling, patterns, robust, across, typescript, python, go. | `.claude/skills/error-handling/SKILL.md` |
| `frontend-a11y` | > Accessibility patterns for React and Next.js — semantic HTML, ARIA attributes, form labeling, keyboard navigation, focus management, and screen reader support. | frontend, a11y, accessibility, patterns, react, next.js, semantic, html | `.claude/skills/frontend-a11y/SKILL.md` |
| `frontend-design-direction` | Set an ECC-specific frontend design direction for production UI work. | frontend, design, direction, set, ecc, specific, production, work. | `.claude/skills/frontend-design-direction/SKILL.md` |
| `frontend-patterns` | Frontend development patterns for React, Next.js, state management, performance optimization, and UI best practices. | frontend, patterns, development, react, next.js, state, management, performance | `.claude/skills/frontend-patterns/SKILL.md` |
| `frontend-slides` | Create stunning, animation-rich HTML presentations from scratch or by converting PowerPoint files. | frontend, slides, create, stunning, animation, rich, html, presentations | `.claude/skills/frontend-slides/SKILL.md` |
| `gget` | gget CLI and Python workflow for quick genomic database queries, sequence lookup, BLAST-style searches, enrichment checks, and reproducible bioinformatics evidence logs. | gget, cli, python, workflow, quick, genomic, database, queries | `.claude/skills/scientific-pkg-gget/SKILL.md` |
| `golang-patterns` | Idiomatic Go patterns, best practices, and conventions for building robust, efficient, and maintainable Go applications. | golang, patterns, idiomatic, best, practices, conventions, building, robust | `.claude/skills/golang-patterns/SKILL.md` |
| `healthcare-cdss-patterns` | Clinical Decision Support System (CDSS) development patterns. | healthcare, cdss, patterns, clinical, decision, support, system, development | `.claude/skills/healthcare-cdss-patterns/SKILL.md` |
| `healthcare-emr-patterns` | EMR/EHR development patterns for healthcare applications. | healthcare, emr, patterns, ehr, development, applications., clinical, safety | `.claude/skills/healthcare-emr-patterns/SKILL.md` |
| `hookify-rules` | This skill should be used when the user asks to create a hookify rule, write a hook rule, configure hookify, add a hookify rule, or needs guidance on hookify rule syntax and patterns. | hookify, rules, user, asks, create, rule, write, hook | `.claude/skills/hookify-rules/SKILL.md` |
| `ito-baskets` | Read-only Itô basket and prediction-market data skill. | ito, baskets, read, only, basket, prediction, market, data | `.claude/skills/ito-baskets/SKILL.md` |
| `java-coding-standards` | Java coding standards for Spring Boot and Quarkus services: naming, immutability, Optional usage, streams, exceptions, generics, CDI, reactive patterns, and project layout. | java, coding, standards, spring, boot, quarkus, services, naming | `.claude/skills/java-coding-standards/SKILL.md` |
| `jpa-patterns` | JPA/Hibernate patterns for entity design, relationships, query optimization, transactions, auditing, indexing, pagination, and pooling in Spring Boot. | jpa, patterns, hibernate, entity, design, relationships, query, optimization | `.claude/skills/jpa-patterns/SKILL.md` |
| `kotlin-patterns` | Idiomatic Kotlin patterns, best practices, and conventions for building robust, efficient, and maintainable Kotlin applications with coroutines, null safety, and DSL builders. | kotlin, patterns, idiomatic, best, practices, conventions, building, robust | `.claude/skills/kotlin-patterns/SKILL.md` |
| `laravel-patterns` | Laravel architecture patterns, routing/controllers, Eloquent ORM, service layers, queues, events, caching, and API resources for production apps. | laravel, patterns, architecture, routing, controllers, eloquent, orm, service | `.claude/skills/laravel-patterns/SKILL.md` |
| `liquid-glass-design` | iOS 26 Liquid Glass design system — dynamic glass material with blur, reflection, and interactive morphing for SwiftUI, UIKit, and WidgetKit. | liquid, glass, design, ios, system, dynamic, material, blur | `.claude/skills/liquid-glass-design/SKILL.md` |
| `make-interfaces-feel-better` | Apply concrete design-engineering details that make interfaces feel polished. | make, interfaces, feel, better, apply, concrete, design, engineering | `.claude/skills/make-interfaces-feel-better/SKILL.md` |
| `manim-video` | Build reusable Manim explainers for technical concepts, graphs, system diagrams, and product walkthroughs, then hand off to the wider ECC video stack if needed. | manim, video, build, reusable, explainers, technical, concepts, graphs | `.claude/skills/manim-video/SKILL.md` |
| `motion-foundations` | Motion tokens, spring presets, performance rules, device adaptation, accessibility enforcement, and SSR safety for React / Next.js using motion/react. | motion, foundations, tokens, spring, presets, performance, rules, device | `.claude/skills/motion-foundations/SKILL.md` |
| `motion-patterns` | Production-ready animation patterns for React / Next.js — button, modal, toast, stagger, page transitions, exit animations, scroll, and layout — built on motion-foundations tokens and springs. | motion, patterns, production, ready, animation, react, next.js, button | `.claude/skills/motion-patterns/SKILL.md` |
| `motion-ui` | Production-ready UI motion system for React/Next.js. | motion, production, ready, system, react, next.js., implementing, animations | `.claude/skills/motion-ui/SKILL.md` |
| `mysql-patterns` | MySQL and MariaDB schema, query, indexing, transaction, replication, and connection-pool patterns for production backends. | mysql, patterns, mariadb, schema, query, indexing, transaction, replication | `.claude/skills/mysql-patterns/SKILL.md` |
| `nanoclaw-repl` | Operate and extend NanoClaw v2, ECC's zero-dependency session-aware REPL built on claude -p. | nanoclaw, repl, operate, extend, ecc, zero, dependency, session | `.claude/skills/nanoclaw-repl/SKILL.md` |
| `nestjs-patterns` | NestJS architecture patterns for modules, controllers, providers, DTO validation, guards, interceptors, config, and production-grade TypeScript backends. | nestjs, patterns, architecture, modules, controllers, providers, dto, validation | `.claude/skills/nestjs-patterns/SKILL.md` |
| `nextjs-turbopack` | Next.js 16+ and Turbopack — incremental bundling, FS caching, dev speed, and when to use Turbopack vs webpack. | nextjs, turbopack, next.js, incremental, bundling, caching, dev, speed | `.claude/skills/nextjs-turbopack/SKILL.md` |
| `nuxt4-patterns` | Nuxt 4 app patterns for hydration safety, performance, route rules, lazy loading, and SSR-safe data fetching with useFetch and useAsyncData. | nuxt4, patterns, nuxt, app, hydration, safety, performance, route | `.claude/skills/nuxt4-patterns/SKILL.md` |
| `perl-patterns` | Modern Perl 5.36+ idioms, best practices, and conventions for building robust, maintainable Perl applications. | perl, patterns, modern, idioms, best, practices, conventions, building | `.claude/skills/perl-patterns/SKILL.md` |
| `prisma-patterns` | Prisma ORM patterns for TypeScript backends — schema design, query optimization, transactions, pagination, and critical traps like updateMany returning count not records, $transaction timeouts, migrate dev resetting the... | prisma, patterns, orm, typescript, backends, schema, design, query | `.claude/skills/prisma-patterns/SKILL.md` |
| `python-patterns` | Pythonic idioms, PEP 8 standards, type hints, and best practices for building robust, efficient, and maintainable Python applications. | python, patterns, pythonic, idioms, pep, standards, type, hints | `.claude/skills/python-patterns/SKILL.md` |
| `pytorch-patterns` | PyTorch deep learning patterns and best practices for building robust, efficient, and reproducible training pipelines, model architectures, and data loading. | pytorch, patterns, deep, learning, best, practices, building, robust | `.claude/skills/pytorch-patterns/SKILL.md` |
| `quarkus-patterns` | Quarkus 3.x LTS architecture patterns with Camel for messaging, RESTful API design, CDI services, data access with Panache, and async processing. | quarkus, patterns, lts, architecture, camel, messaging, restful, api | `.claude/skills/quarkus-patterns/SKILL.md` |
| `react-patterns` | React 18/19 patterns including hooks discipline, server/client component boundaries, Suspense + error boundaries, form actions, data fetching, state management decision trees, and accessibility-first composition. | react, patterns, hooks, discipline, server, client, component, boundaries | `.claude/skills/react-patterns/SKILL.md` |
| `redis-patterns` | Redis data structure patterns, caching strategies, distributed locks, rate limiting, pub/sub, and connection management for production applications. | redis, patterns, data, structure, caching, strategies, distributed, locks | `.claude/skills/redis-patterns/SKILL.md` |
| `remotion-video-creation` | Best practices for Remotion - Video creation in React. | remotion, video, creation, best, practices, react., domain, specific | `.claude/skills/remotion-video-creation/SKILL.md` |
| `research-ops` | Evidence-first current-state research workflow for ECC. | research, ops, evidence, first, current, state, workflow, ecc. | `.claude/skills/research-ops/SKILL.md` |
| `rust-patterns` | Idiomatic Rust patterns, ownership, error handling, traits, concurrency, and best practices for building safe, performant applications. | rust, patterns, idiomatic, ownership, error, handling, traits, concurrency | `.claude/skills/rust-patterns/SKILL.md` |
| `seo` | Audit, plan, and implement SEO improvements across technical SEO, on-page optimization, structured data, Core Web Vitals, and content strategy. | seo, audit, plan, implement, improvements, across, technical, page | `.claude/skills/seo/SKILL.md` |
| `skill-scout` | Search existing local, marketplace, GitHub, and web skill sources before creating a new skill. | scout, search, existing, local, marketplace, github, web, sources | `.claude/skills/skill-scout/SKILL.md` |
| `springboot-patterns` | Spring Boot architecture patterns, REST API design, layered services, data access, caching, async processing, and logging. | springboot, patterns, spring, boot, architecture, rest, api, design | `.claude/skills/springboot-patterns/SKILL.md` |
| `swiftui-patterns` | SwiftUI architecture patterns, state management with @Observable, view composition, navigation, performance optimization, and modern iOS/macOS UI best practices. | swiftui, patterns, architecture, state, management, observable, view, composition | `.claude/skills/swiftui-patterns/SKILL.md` |
| `ui-demo` | Record polished UI demo videos using Playwright. | demo, record, polished, videos, playwright., user, asks, create | `.claude/skills/ui-demo/SKILL.md` |
| `ui-to-vue` | Use when the user has UI screenshots or design exports that need batch conversion into Vue 3 components, especially with Vant, Element Plus, or Ant Design Vue. | vue, user, has, screenshots, design, exports, need, batch | `.claude/skills/ui-to-vue/SKILL.md` |
| `videodb` | See, Understand, Act on video and audio. | videodb, see, understand, act, video, audio., ingest, local | `.claude/skills/videodb/SKILL.md` |
| `vite-patterns` | Vite build tool patterns including config, plugins, HMR, env variables, proxy setup, SSR, library mode, dependency pre-bundling, and build optimization. | vite, patterns, build, tool, config, plugins, hmr, env | `.claude/skills/vite-patterns/SKILL.md` |
| `vue-patterns` | Vue.js 3 Composition API patterns, component architecture, reactivity best practices, Pinia state management, Vue Router navigation, and Nuxt SSR patterns. | vue, patterns, vue.js, composition, api, component, architecture, reactivity | `.claude/skills/vue-patterns/SKILL.md` |

### General & Other (6)

| Skill | Description | Trigger keywords | File |
|---|---|---|---|
| `brand-discovery` | >- Use when a brand needs to discover or articulate its identity through structured multi-session interviews. | brand, discovery, needs, discover, articulate, its, identity, through | `.claude/skills/brand-discovery/SKILL.md` |
| `ck` | Persistent per-project memory for Claude Code. | persistent, per, project, memory, claude, code., auto, loads | `.claude/skills/ck/SKILL.md` |
| `homelab-network-readiness` | Readiness checklist for homelab VLAN segmentation, local DNS filtering, and WireGuard-style remote access before changing router, firewall, DHCP, or VPN configuration. | homelab, network, readiness, checklist, vlan, segmentation, local, dns | `.claude/skills/homelab-network-readiness/SKILL.md` |
| `plankton-code-quality` | Write-time code quality enforcement using Plankton — auto-formatting, linting, and Claude-powered fixes on every file edit via hooks. | plankton, code, quality, write, time, enforcement, auto, formatting | `.claude/skills/plankton-code-quality/SKILL.md` |
| `strategic-compact` | Suggests manual context compaction at logical intervals to preserve context through task phases rather than arbitrary auto-compaction. | strategic, compact, suggests, manual, context, compaction, logical, intervals | `.claude/skills/strategic-compact/SKILL.md` |
| `verification-loop` | A comprehensive verification system for Claude Code sessions. | verification, loop, comprehensive, system, claude, code, sessions., verifying | `.claude/skills/verification-loop/SKILL.md` |

### Mobile (5)

| Skill | Description | Trigger keywords | File |
|---|---|---|---|
| `android-clean-architecture` | Clean Architecture patterns for Android and Kotlin Multiplatform projects — module structure, dependency rules, UseCases, Repositories, and data layer patterns. | android, clean, architecture, patterns, kotlin, multiplatform, projects, module | `.claude/skills/android-clean-architecture/SKILL.md` |
| `cisco-ios-patterns` | Cisco IOS and IOS-XE review patterns for show commands, config hierarchy, wildcard masks, ACL placement, interface hygiene, and safe change-window verification. | cisco, ios, patterns, review, show, commands, config, hierarchy | `.claude/skills/cisco-ios-patterns/SKILL.md` |
| `dart-flutter-patterns` | Production-ready Dart and Flutter patterns covering null safety, immutable state, async composition, widget architecture, popular state management frameworks (BLoC, Riverpod, Provider), GoRouter navigation, Dio... | dart, flutter, patterns, production, ready, covering, null, safety | `.claude/skills/dart-flutter-patterns/SKILL.md` |
| `kotlin-exposed-patterns` | JetBrains Exposed ORM patterns including DSL queries, DAO pattern, transactions, HikariCP connection pooling, Flyway migrations, and repository pattern. | kotlin, exposed, patterns, jetbrains, orm, dsl, queries, dao | `.claude/skills/kotlin-exposed-patterns/SKILL.md` |
| `swift-concurrency-6-2` | Swift 6.2 Approachable Concurrency — single-threaded by default, @concurrent for explicit background offloading, isolated conformances for main actor types. | swift, concurrency, approachable, single, threaded, default, concurrent, explicit | `.claude/skills/swift-concurrency-6-2/SKILL.md` |

### Product & Process (10)

| Skill | Description | Trigger keywords | File |
|---|---|---|---|
| `customer-billing-ops` | Operate customer billing workflows such as subscriptions, refunds, churn triage, billing-portal recovery, and plan analysis using connected billing tools like Stripe. | customer, billing, ops, operate, workflows, subscriptions, refunds, churn | `.claude/skills/customer-billing-ops/SKILL.md` |
| `ecc-tools-cost-audit` | Evidence-first ECC Tools burn and billing audit workflow. | ecc, tools, cost, audit, evidence, first, burn, billing | `.claude/skills/ecc-tools-cost-audit/SKILL.md` |
| `email-ops` | Evidence-first mailbox triage, drafting, send verification, and sent-mail-safe follow-up workflow for ECC. | email, ops, evidence, first, mailbox, triage, drafting, send | `.claude/skills/email-ops/SKILL.md` |
| `hermes-imports` | Convert local Hermes operator workflows into sanitized ECC skills and release-pack artifacts. | hermes, imports, convert, local, operator, workflows, sanitized, ecc | `.claude/skills/hermes-imports/SKILL.md` |
| `homelab-network-setup` | Practical home and homelab network planning for gateways, switches, access points, IP ranges, DHCP reservations, DNS, cabling, and common beginner mistakes. | homelab, network, setup, practical, home, planning, gateways, switches | `.claude/skills/homelab-network-setup/SKILL.md` |
| `inventory-demand-planning` | > Codified expertise for demand forecasting, safety stock optimization, replenishment planning, and promotional lift estimation at multi-location retailers. | inventory, demand, planning, codified, expertise, forecasting, safety, stock | `.claude/skills/inventory-demand-planning/SKILL.md` |
| `messages-ops` | Evidence-first live messaging workflow for ECC. | messages, ops, evidence, first, live, messaging, workflow, ecc. | `.claude/skills/messages-ops/SKILL.md` |
| `network-interface-health` | Diagnose interface errors, drops, CRCs, duplex mismatches, flapping, speed negotiation issues, and counter trends on routers, switches, and Linux hosts. | network, interface, health, diagnose, errors, drops, crcs, duplex | `.claude/skills/network-interface-health/SKILL.md` |
| `terminal-opener` | Open an executable and its argument array in a visible terminal window through a reusable, shell-free launch plan with dry-run, JSON, capability detection, detached fallback, and standalone recovery modes. | terminal, opener, open, executable, its, argument, array, visible | `.claude/skills/terminal-opener/SKILL.md` |
| `unified-notifications-ops` | Operate notifications as one ECC-native workflow across GitHub, Linear, desktop alerts, hooks, and connected communication surfaces. | unified, notifications, ops, operate, one, ecc, native, workflow | `.claude/skills/unified-notifications-ops/SKILL.md` |

### Security & Compliance (39)

| Skill | Description | Trigger keywords | File |
|---|---|---|---|
| `carrier-relationship-management` | > Codified expertise for managing carrier portfolios, negotiating freight rates, tracking carrier performance, allocating freight, and maintaining strategic carrier relationships. | carrier, relationship, management, codified, expertise, managing, portfolios, negotiating | `.claude/skills/carrier-relationship-management/SKILL.md` |
| `customs-trade-compliance` | > Codified expertise for customs documentation, tariff classification, duty optimization, restricted party screening, and regulatory compliance across multiple jurisdictions. | customs, trade, compliance, codified, expertise, documentation, tariff, classification | `.claude/skills/customs-trade-compliance/SKILL.md` |
| `defi-amm-security` | Security checklist for Solidity AMM contracts, liquidity pools, and swap flows. | defi, amm, security, checklist, solidity, contracts, liquidity, pools | `.claude/skills/defi-amm-security/SKILL.md` |
| `django-security` | Django security best practices, authentication, authorization, CSRF protection, SQL injection prevention, XSS prevention, and secure deployment configurations. | django, security, best, practices, authentication, authorization, csrf, protection | `.claude/skills/django-security/SKILL.md` |
| `django-verification` | Verification loop for Django projects: migrations, linting, tests with coverage, security scans, and deployment readiness checks before release or PR. | django, verification, loop, projects, migrations, linting, tests, coverage | `.claude/skills/django-verification/SKILL.md` |
| `docker-patterns` | Docker and Docker Compose patterns for local development, hardened CLI installer harnesses, container security, networking, volumes, and multi-service orchestration. | docker, patterns, compose, local, development, hardened, cli, installer | `.claude/skills/docker-patterns/SKILL.md` |
| `enterprise-agent-ops` | Operate long-lived agent workloads with observability, security boundaries, and lifecycle management. | enterprise, ops, operate, long, lived, workloads, observability, security | `.claude/skills/enterprise-agent-ops/SKILL.md` |
| `fastapi-patterns` | FastAPI best practices covering project structure, Pydantic v2 schemas, dependency injection, async handlers, authentication, authorization, transactional service layers, and testing with httpx and pytest. | fastapi, patterns, best, practices, covering, project, structure, pydantic | `.claude/skills/fastapi-patterns/SKILL.md` |
| `flutter-dart-code-review` | Library-agnostic Flutter/Dart code review checklist covering widget best practices, state management patterns (BLoC, Riverpod, Provider, GetX, MobX, Signals), Dart idioms, performance, accessibility, security, and clean... | flutter, dart, code, review, library, agnostic, checklist, covering | `.claude/skills/flutter-dart-code-review/SKILL.md` |
| `github-ops` | GitHub repository operations, automation, and management. | github, ops, repository, operations, automation, management., issue, triage | `.claude/skills/github-ops/SKILL.md` |
| `healthcare-eval-harness` | Patient safety evaluation harness for healthcare application deployments. | healthcare, eval, harness, patient, safety, evaluation, application, deployments. | `.claude/skills/healthcare-eval-harness/SKILL.md` |
| `healthcare-phi-compliance` | Protected Health Information (PHI) and Personally Identifiable Information (PII) compliance patterns for healthcare applications. | healthcare, phi, compliance, protected, health, information, personally, identifiable | `.claude/skills/healthcare-phi-compliance/SKILL.md` |
| `hipaa-compliance` | HIPAA-specific entrypoint for healthcare privacy and security work. | hipaa, compliance, specific, entrypoint, healthcare, privacy, security, work. | `.claude/skills/hipaa-compliance/SKILL.md` |
| `intent-driven-development` | Turn ambiguous or high-impact product and engineering changes into scoped, verifiable acceptance criteria before or alongside implementation. | intent, driven, development, turn, ambiguous, high, impact, product | `.claude/skills/intent-driven-development/SKILL.md` |
| `ito-compute` | Query live GPU inventory, submit an authenticated Itô fixed-rate RFQ, inspect RFQ or procurement status, revoke device credentials, and run explicitly gated node qualification through the separately installed canonical... | ito, compute, query, live, gpu, inventory, submit, authenticated | `.claude/skills/ito-compute/SKILL.md` |
| `kotlin-ktor-patterns` | Ktor server patterns including routing DSL, plugins, authentication, Koin DI, kotlinx.serialization, WebSockets, and testApplication testing. | kotlin, ktor, patterns, server, routing, dsl, plugins, authentication | `.claude/skills/kotlin-ktor-patterns/SKILL.md` |
| `kubernetes-patterns` | Kubernetes workload patterns, resource management, RBAC, probes, autoscaling, ConfigMap/Secret handling, and kubectl debugging for production-grade deployments. | kubernetes, patterns, workload, resource, management, rbac, probes, autoscaling | `.claude/skills/kubernetes-patterns/SKILL.md` |
| `laravel-security` | Laravel security best practices — authentication, authorization, Eloquent safety, CSRF, XSS prevention, API security, and secure deployment configurations. | laravel, security, best, practices, authentication, authorization, eloquent, safety | `.claude/skills/laravel-security/SKILL.md` |
| `laravel-tdd` | Laravel testing strategies with PHPUnit, Pest, model factories, HTTP tests, Sanctum authentication testing, mocking, and coverage. | laravel, tdd, testing, strategies, phpunit, pest, model, factories | `.claude/skills/laravel-tdd/SKILL.md` |
| `laravel-verification` | Verification loop for Laravel projects: env checks, linting, static analysis, tests with coverage, security scans, and deployment readiness. | laravel, verification, loop, projects, env, checks, linting, static | `.claude/skills/laravel-verification/SKILL.md` |
| `llm-trading-agent-security` | Security patterns for autonomous trading agents with wallet or transaction authority. | llm, trading, security, patterns, autonomous, wallet, transaction, authority. | `.claude/skills/llm-trading-agent-security/SKILL.md` |
| `mailtrap-email-integration` | Guides agents through integrating transactional email sending via Mailtrap's Email API, including sandbox testing, domain verification, and API authentication. | mailtrap, email, integration, guides, through, integrating, transactional, sending | `.claude/skills/mailtrap-email-integration/SKILL.md` |
| `nasiko-control-plane` | Use the experimental Nasiko CLI lifecycle bridge for pinned installation, read-only status, and qualified uninstall with explicit consent and telemetry and secrets boundaries. | nasiko, control, plane, experimental, cli, lifecycle, bridge, pinned | `.claude/skills/nasiko-control-plane/SKILL.md` |
| `network-config-validation` | Pre-deployment checks for router and switch configuration, including dangerous commands, duplicate addresses, subnet overlaps, stale references, management-plane risk, and IOS-style security hygiene. | network, config, validation, pre, deployment, checks, router, switch | `.claude/skills/network-config-validation/SKILL.md` |
| `opensource-pipeline` | Open-source pipeline: fork, sanitize, and package private projects for safe public release. | opensource, pipeline, open, source, fork, sanitize, package, private | `.claude/skills/opensource-pipeline/SKILL.md` |
| `perl-security` | Comprehensive Perl security covering taint mode, input validation, safe process execution, DBI parameterized queries, web security (XSS/SQLi/CSRF), and perlcritic security policies. | perl, security, comprehensive, covering, taint, mode, input, validation | `.claude/skills/perl-security/SKILL.md` |
| `postgres-patterns` | PostgreSQL database patterns for query optimization, schema design, indexing, and security. | postgres, patterns, postgresql, database, query, optimization, schema, design | `.claude/skills/postgres-patterns/SKILL.md` |
| `prediction-market-risk-review` | Review prediction-market, basket, oracle, and trading-agent workflows for compliance, safety, data-quality, privacy, and execution risk. | prediction, market, risk, review, basket, oracle, trading, workflows | `.claude/skills/prediction-market-risk-review/SKILL.md` |
| `quarkus-security` | Quarkus Security best practices for authentication, authorization, JWT/OIDC, RBAC, input validation, CSRF, secrets management, and dependency security. | quarkus, security, best, practices, authentication, authorization, jwt, oidc | `.claude/skills/quarkus-security/SKILL.md` |
| `quarkus-verification` | Verification loop for Quarkus projects: build, static analysis, tests with coverage, security scans, native compilation, and diff review before release or PR. | quarkus, verification, loop, projects, build, static, analysis, tests | `.claude/skills/quarkus-verification/SKILL.md` |
| `returns-reverse-logistics` | > Codified expertise for returns authorization, receipt and inspection, disposition decisions, refund processing, fraud detection, and warranty claims management. | returns, reverse, logistics, codified, expertise, authorization, receipt, inspection | `.claude/skills/returns-reverse-logistics/SKILL.md` |
| `security-bounty-hunter` | Hunt for exploitable, bounty-worthy security issues in repositories. | security, bounty, hunter, hunt, exploitable, worthy, issues, repositories. | `.claude/skills/security-bounty-hunter/SKILL.md` |
| `security-review` | Use this skill when adding authentication, handling user input, working with secrets, creating API endpoints, or implementing payment/sensitive features. | security, review, adding, authentication, handling, user, input, working | `.claude/skills/security-review/SKILL.md` |
| `security-scan` | Scan your Claude Code configuration (.claude/ directory) for security vulnerabilities, misconfigurations, and injection risks using AgentShield. | security, scan, claude, code, configuration, directory, vulnerabilities, misconfigurations | `.claude/skills/security-scan/SKILL.md` |
| `skill-comply` | Visualize whether skills, rules, and agent definitions are actually followed — auto-generates scenarios at 3 prompt strictness levels, runs agents, classifies behavioral sequences, and reports compliance rates with full... | comply, visualize, whether, rules, definitions, actually, followed, auto | `.claude/skills/skill-comply/SKILL.md` |
| `springboot-security` | Spring Security best practices for authn/authz, validation, CSRF, secrets, headers, rate limiting, and dependency security in Java Spring Boot services. | springboot, security, spring, best, practices, authn, authz, validation | `.claude/skills/springboot-security/SKILL.md` |
| `springboot-verification` | Verification loop for Spring Boot projects: build, static analysis, tests with coverage, security scans, and diff review before release or PR. | springboot, verification, loop, spring, boot, projects, build, static | `.claude/skills/springboot-verification/SKILL.md` |
| `token-budget-advisor` | >- Offers the user an informed choice about how much response depth to consume before answering. | token, budget, advisor, offers, user, informed, choice, about | `.claude/skills/token-budget-advisor/SKILL.md` |
| `x-api` | X/Twitter API integration for posting tweets, threads, reading timelines, search, and analytics. | api, twitter, integration, posting, tweets, threads, reading, timelines | `.claude/skills/x-api/SKILL.md` |

### Testing & QA (38)

| Skill | Description | Trigger keywords | File |
|---|---|---|---|
| `agent-architecture-audit` | Full-stack diagnostic for agent and LLM applications. | architecture, audit, full, stack, diagnostic, llm, applications., audits | `.claude/skills/agent-architecture-audit/SKILL.md` |
| `ai-regression-testing` | Regression testing strategies for AI-assisted development. | regression, testing, strategies, assisted, development., sandbox, mode, api | `.claude/skills/ai-regression-testing/SKILL.md` |
| `angular-developer` | Generates Angular code and provides architectural guidance. | angular, developer, generates, code, provides, architectural, guidance., trigger | `.claude/skills/angular-developer/SKILL.md` |
| `benchmark` | Use this skill to measure performance baselines, detect regressions before/after PRs, and compare stack alternatives. | benchmark, measure, performance, baselines, detect, regressions, before, after | `.claude/skills/benchmark/SKILL.md` |
| `benchmark-optimization-loop` | Use when the user asks to make something faster, try many variants, run recursive optimization, benchmark latency/throughput/cost, or choose the best implementation by repeated measured tests. | benchmark, optimization, loop, user, asks, make, something, faster | `.claude/skills/benchmark-optimization-loop/SKILL.md` |
| `browser-qa` | Use this skill to automate visual testing and UI interaction verification using browser automation after deploying features. | browser, automate, visual, testing, interaction, verification, automation, after | `.claude/skills/browser-qa/SKILL.md` |
| `bun-runtime` | Bun as runtime, package manager, bundler, and test runner. | bun, runtime, package, manager, bundler, test, runner., choose | `.claude/skills/bun-runtime/SKILL.md` |
| `canary-watch` | Use this skill to monitor and verify a deployed URL after releases — checks HTTP endpoints, SSE streams, static assets, console errors, and performance regressions after deploys, merges, or dependency upgrades. | canary, watch, monitor, verify, deployed, url, after, releases | `.claude/skills/canary-watch/SKILL.md` |
| `cpp-testing` | Use only when writing/updating/fixing C++ tests, configuring GoogleTest/CTest, diagnosing failing or flaky tests, or adding coverage/sanitizers. | cpp, testing, only, writing, updating, fixing, c++, tests | `.claude/skills/cpp-testing/SKILL.md` |
| `csharp-testing` | C# and .NET testing patterns with xUnit, FluentAssertions, mocking, integration tests, and test organization best practices. | csharp, testing, net, patterns, xunit, fluentassertions, mocking, integration | `.claude/skills/csharp-testing/SKILL.md` |
| `dev-team` | Simulate a collaborative dev team session where multiple role-based personas (PM, Architect, Developer, QA) respond to the same problem together in one session. | dev, team, simulate, collaborative, session, where, multiple, role | `.claude/skills/dev-team/SKILL.md` |
| `django-celery` | Django + Celery async task patterns — configuration, task design, beat scheduling, retries, canvas workflows, monitoring, and testing. | django, celery, async, task, patterns, configuration, design, beat | `.claude/skills/django-celery/SKILL.md` |
| `django-tdd` | Django testing strategies with pytest-django, TDD methodology, factory_boy, mocking, coverage, and testing Django REST Framework APIs. | django, tdd, testing, strategies, pytest, methodology, factory, boy | `.claude/skills/django-tdd/SKILL.md` |
| `e2e-testing` | Playwright E2E testing patterns, Page Object Model, configuration, CI/CD integration, artifact management, and flaky test strategies. | e2e, testing, playwright, patterns, page, object, model, configuration | `.claude/skills/e2e-testing/SKILL.md` |
| `fsharp-testing` | F# testing patterns with xUnit, FsUnit, Unquote, FsCheck property-based testing, integration tests, and test organization best practices. | fsharp, testing, patterns, xunit, fsunit, unquote, fscheck, property | `.claude/skills/fsharp-testing/SKILL.md` |
| `generating-python-installer` | Commercial-grade Python installer expert for Windows: Nuitka extreme compilation, dist slimming, DLL footprint analysis, and Inno Setup packaging to ship the smallest, fastest installers. | generating, python, installer, commercial, grade, expert, windows, nuitka | `.claude/skills/generating-python-installer/SKILL.md` |
| `golang-testing` | Go testing patterns including table-driven tests, subtests, benchmarks, fuzzing, and test coverage. | golang, testing, patterns, table, driven, tests, subtests, benchmarks | `.claude/skills/golang-testing/SKILL.md` |
| `hexagonal-architecture` | Design, implement, and refactor Ports & Adapters systems with clear domain boundaries, dependency inversion, and testable use-case orchestration across TypeScript, Java, Kotlin, and Go services. | hexagonal, architecture, design, implement, refactor, ports, adapters, systems | `.claude/skills/hexagonal-architecture/SKILL.md` |
| `kotlin-coroutines-flows` | Kotlin Coroutines and Flow patterns for Android and KMP — structured concurrency, Flow operators, StateFlow, error handling, and testing. | kotlin, coroutines, flows, flow, patterns, android, kmp, structured | `.claude/skills/kotlin-coroutines-flows/SKILL.md` |
| `kotlin-testing` | Kotlin testing patterns with Kotest, MockK, coroutine testing, property-based testing, and Kover coverage. | kotlin, testing, patterns, kotest, mockk, coroutine, property, based | `.claude/skills/kotlin-testing/SKILL.md` |
| `mcp-server-patterns` | Build MCP servers with Node/TypeScript SDK — tools, resources, prompts, Zod validation, stdio vs Streamable HTTP. | mcp, server, patterns, build, servers, node, typescript, sdk | `.claude/skills/mcp-server-patterns/SKILL.md` |
| `orch-add-feature` | Orchestrate building a brand-new feature end to end — research, plan, TDD implementation, review, and gated commit — by delegating each phase to the matching ECC agent. | orch, add, feature, orchestrate, building, brand, new, end | `.claude/skills/orch-add-feature/SKILL.md` |
| `orch-build-mvp` | Orchestrate bootstrapping a working MVP from a design or spec document — ingest the doc, plan thin vertical slices, scaffold the first end-to-end slice, then TDD-implement, review, and gated commit. | orch, build, mvp, orchestrate, bootstrapping, working, design, spec | `.claude/skills/orch-build-mvp/SKILL.md` |
| `orch-change-feature` | Orchestrate altering an existing, working feature to new desired behavior — update its tests to the new spec, change the implementation to match, review, and gated commit. | orch, change, feature, orchestrate, altering, existing, working, new | `.claude/skills/orch-change-feature/SKILL.md` |
| `orch-fix-defect` | Orchestrate fixing a bug — reproduce it as a failing regression test, fix to green, review, and gated commit — by delegating each phase to the matching ECC agent. | orch, fix, defect, orchestrate, fixing, bug, reproduce, failing | `.claude/skills/orch-fix-defect/SKILL.md` |
| `orch-pipeline` | Shared orchestration engine for the orch-* skill family. | orch, pipeline, shared, orchestration, engine, family., defines, gated | `.claude/skills/orch-pipeline/SKILL.md` |
| `orch-refine-code` | Orchestrate a behavior-preserving refactor — confirm tests are green, restructure without changing behavior, keep tests green, review, and gated commit. | orch, refine, code, orchestrate, behavior, preserving, refactor, confirm | `.claude/skills/orch-refine-code/SKILL.md` |
| `perl-testing` | Perl testing patterns using Test2::V0, Test::More, prove runner, mocking, coverage with Devel::Cover, and TDD methodology. | perl, testing, patterns, test2, test, more, prove, runner | `.claude/skills/perl-testing/SKILL.md` |
| `product-lens` | Use this skill to validate the "why" before building, run product diagnostics, and pressure-test product direction before the request becomes an implementation contract. | product, lens, validate, why, before, building, run, diagnostics | `.claude/skills/product-lens/SKILL.md` |
| `python-testing` | Python testing strategies using pytest, TDD methodology, fixtures, mocking, parametrization, and coverage requirements. | python, testing, strategies, pytest, tdd, methodology, fixtures, mocking | `.claude/skills/python-testing/SKILL.md` |
| `quarkus-tdd` | Test-driven development for Quarkus 3.x LTS using JUnit 5, Mockito, REST Assured, Camel testing, and JaCoCo. | quarkus, tdd, test, driven, development, lts, junit, mockito | `.claude/skills/quarkus-tdd/SKILL.md` |
| `react-testing` | React component testing with React Testing Library, Vitest/Jest, MSW for network mocking, accessibility assertions with axe, and the decision boundary between component tests and Playwright/Cypress end-to-end runs. | react, testing, component, library, vitest, jest, msw, network | `.claude/skills/react-testing/SKILL.md` |
| `rust-testing` | Rust testing patterns including unit tests, integration tests, async testing, property-based testing, mocking, and coverage. | rust, testing, patterns, unit, tests, integration, async, property | `.claude/skills/rust-testing/SKILL.md` |
| `springboot-tdd` | Test-driven development for Spring Boot using JUnit 5, Mockito, MockMvc, Testcontainers, and JaCoCo. | springboot, tdd, test, driven, development, spring, boot, junit | `.claude/skills/springboot-tdd/SKILL.md` |
| `swift-protocol-di-testing` | Protocol-based dependency injection for testable Swift code — mock file system, network, and external APIs using focused protocols and Swift Testing. | swift, protocol, testing, based, dependency, injection, testable, code | `.claude/skills/swift-protocol-di-testing/SKILL.md` |
| `tdd-workflow` | Use this skill when writing new features, fixing bugs, or refactoring code. | tdd, workflow, writing, new, features, fixing, bugs, refactoring | `.claude/skills/tdd-workflow/SKILL.md` |
| `tinystruct-patterns` | Expert guidance for developing with the tinystruct Java framework. | tinystruct, patterns, expert, guidance, developing, java, framework., working | `.claude/skills/tinystruct-patterns/SKILL.md` |
| `windows-desktop-e2e` | E2E testing for Windows native desktop apps (WPF, WinForms, Win32/MFC, Qt) using pywinauto and Windows UI Automation. | windows, desktop, e2e, testing, native, apps, wpf, winforms | `.claude/skills/windows-desktop-e2e/SKILL.md` |

## Agents

### Agent & AI Engineering (9)

| Agent | Description | Trigger keywords | File |
|---|---|---|---|
| `agent-evaluator` | Evaluates agent output against 5-axis quality rubric (accuracy, completeness, clarity, actionability, conciseness). | evaluator, evaluates, output, against, axis, quality, rubric, accuracy | `.claude/agents/agent-evaluator.md` |
| `conversation-analyzer` | Use this agent when analyzing conversation transcripts to find behaviors worth preventing with hooks. | conversation, analyzer, analyzing, transcripts, find, behaviors, worth, preventing | `.claude/agents/conversation-analyzer.md` |
| `docs-lookup` | When the user asks how to use a library, framework, or API or needs up-to-date code examples, use Context7 MCP to fetch current documentation and return answers with examples. | docs, lookup, user, asks, how, library, framework, api | `.claude/agents/docs-lookup.md` |
| `gan-generator` | GAN Harness — Generator agent. | gan, generator, harness, agent., implements, features, according, spec | `.claude/agents/gan-generator.md` |
| `gan-planner` | GAN Harness — Planner agent. | gan, planner, harness, agent., expands, one, line, prompt | `.claude/agents/gan-planner.md` |
| `harness-optimizer` | Improve local agent-harness configuration reliability and cost using eval-driven grading (pass@k/pass^k) derived from the eval-harness skill. | harness, optimizer, improve, local, configuration, reliability, cost, eval | `.claude/agents/harness-optimizer.md` |
| `loop-operator` | Operate autonomous agent loops, monitor progress, and intervene safely when loops stall. | loop, operator, operate, autonomous, loops, monitor, progress, intervene | `.claude/agents/loop-operator.md` |
| `marketing-agent` | Marketing strategist and copywriter for campaign planning, audience research, positioning, copy creation, and content review. | marketing, strategist, copywriter, campaign, planning, audience, research, positioning | `.claude/agents/marketing-agent.md` |
| `mle-reviewer` | Production machine-learning engineering reviewer for data contracts, feature pipelines, training reproducibility, offline/online evaluation, model serving, monitoring, and rollback. | mle, reviewer, production, machine, learning, engineering, data, contracts | `.claude/agents/mle-reviewer.md` |

### Architecture & Code Quality (2)

| Agent | Description | Trigger keywords | File |
|---|---|---|---|
| `code-simplifier` | Simplifies and refines code for clarity, consistency, and maintainability while preserving behavior. | code, simplifier, simplifies, refines, clarity, consistency, maintainability, while | `.claude/agents/code-simplifier.md` |
| `silent-failure-hunter` | Review code for silent failures, swallowed errors, bad fallbacks, and missing error propagation. | silent, failure, hunter, review, code, failures, swallowed, errors | `.claude/agents/silent-failure-hunter.md` |

### Data & Analytics (1)

| Agent | Description | Trigger keywords | File |
|---|---|---|---|
| `opensource-packager` | Generate complete open-source packaging for a sanitized project. | opensource, packager, generate, complete, open, source, packaging, sanitized | `.claude/agents/opensource-packager.md` |

### DevOps & Infrastructure (8)

| Agent | Description | Trigger keywords | File |
|---|---|---|---|
| `code-explorer` | Deeply analyzes existing codebase features by tracing execution paths, mapping architecture layers, and documenting dependencies to inform new development. | code, explorer, deeply, analyzes, existing, codebase, features, tracing | `.claude/agents/code-explorer.md` |
| `cpp-reviewer` | Expert C++ code reviewer specializing in memory safety, modern C++ idioms, concurrency, and performance. | cpp, reviewer, expert, c++, code, specializing, memory, safety | `.claude/agents/cpp-reviewer.md` |
| `fsharp-reviewer` | Expert F# code reviewer specializing in functional idioms, type safety, pattern matching, computation expressions, and performance. | fsharp, reviewer, expert, code, specializing, functional, idioms, type | `.claude/agents/fsharp-reviewer.md` |
| `go-reviewer` | Expert Go code reviewer specializing in idiomatic Go, concurrency patterns, error handling, and performance. | reviewer, expert, code, specializing, idiomatic, concurrency, patterns, error | `.claude/agents/go-reviewer.md` |
| `performance-optimizer` | Performance analysis and optimization specialist. | performance, optimizer, analysis, optimization, specialist., proactively, identifying, bottlenecks | `.claude/agents/performance-optimizer.md` |
| `planner` | Expert planning specialist for complex features and refactoring. | planner, expert, planning, specialist, complex, features, refactoring., proactively | `.claude/agents/planner.md` |
| `refactor-cleaner` | Dead code cleanup and consolidation specialist. | refactor, cleaner, dead, code, cleanup, consolidation, specialist., proactively | `.claude/agents/refactor-cleaner.md` |
| `rust-reviewer` | Expert Rust code reviewer specializing in ownership, lifetimes, error handling, unsafe usage, and idiomatic patterns. | rust, reviewer, expert, code, specializing, ownership, lifetimes, error | `.claude/agents/rust-reviewer.md` |

### Frontend & UI (21)

| Agent | Description | Trigger keywords | File |
|---|---|---|---|
| `architect` | Software architecture specialist for system design, scalability, and technical decision-making. | architect, software, architecture, specialist, system, design, scalability, technical | `.claude/agents/architect.md` |
| `build-error-resolver` | Build and TypeScript error resolution specialist. | build, error, resolver, typescript, resolution, specialist., proactively, fails | `.claude/agents/build-error-resolver.md` |
| `chief-of-staff` | Personal communication chief of staff that triages email, Slack, LINE, and Messenger. | chief, staff, personal, communication, triages, email, slack, line | `.claude/agents/chief-of-staff.md` |
| `code-architect` | Designs feature architectures by analyzing existing codebase patterns and conventions, then providing implementation blueprints with concrete files, interfaces, data flow, and build order. | code, architect, designs, feature, architectures, analyzing, existing, codebase | `.claude/agents/code-architect.md` |
| `cpp-build-resolver` | C++ build, CMake, and compilation error resolution specialist. | cpp, build, resolver, c++, cmake, compilation, error, resolution | `.claude/agents/cpp-build-resolver.md` |
| `dart-build-resolver` | Dart/Flutter build, analysis, and dependency error resolution specialist. | dart, build, resolver, flutter, analysis, dependency, error, resolution | `.claude/agents/dart-build-resolver.md` |
| `django-build-resolver` | Django/Python build, migration, and dependency error resolution specialist. | django, build, resolver, python, migration, dependency, error, resolution | `.claude/agents/django-build-resolver.md` |
| `doc-updater` | Documentation and codemap specialist. | doc, updater, documentation, codemap, specialist., proactively, updating, codemaps | `.claude/agents/doc-updater.md` |
| `flutter-reviewer` | Flutter and Dart code reviewer. | flutter, reviewer, dart, code, reviewer., reviews, widget, best | `.claude/agents/flutter-reviewer.md` |
| `go-build-resolver` | Go build, vet, and compilation error resolution specialist. | build, resolver, vet, compilation, error, resolution, specialist., fixes | `.claude/agents/go-build-resolver.md` |
| `homelab-architect` | Designs home and small-lab network plans from hardware inventory, goals, and operator experience level, with safe staged changes and rollback guidance. | homelab, architect, designs, home, small, lab, network, plans | `.claude/agents/homelab-architect.md` |
| `java-build-resolver` | Java/Maven/Gradle build, compilation, and dependency error resolution specialist. | java, build, resolver, maven, gradle, compilation, dependency, error | `.claude/agents/java-build-resolver.md` |
| `kotlin-build-resolver` | Kotlin/Gradle build, compilation, and dependency error resolution specialist. | kotlin, build, resolver, gradle, compilation, dependency, error, resolution | `.claude/agents/kotlin-build-resolver.md` |
| `network-architect` | Designs enterprise or multi-site network architecture from requirements, using existing network skills for focused routing, validation, automation, and troubleshooting detail. | network, architect, designs, enterprise, multi, site, architecture, requirements | `.claude/agents/network-architect.md` |
| `pytorch-build-resolver` | PyTorch runtime, CUDA, and training error resolution specialist. | pytorch, build, resolver, runtime, cuda, training, error, resolution | `.claude/agents/pytorch-build-resolver.md` |
| `react-build-resolver` | Diagnose and fix React build failures across Vite, webpack, Next.js, CRA, Parcel, esbuild, and Bun. | react, build, resolver, diagnose, fix, failures, across, vite | `.claude/agents/react-build-resolver.md` |
| `rust-build-resolver` | Rust build, compilation, and dependency error resolution specialist. | rust, build, resolver, compilation, dependency, error, resolution, specialist. | `.claude/agents/rust-build-resolver.md` |
| `seo-specialist` | SEO specialist for technical SEO audits, on-page optimization, structured data, Core Web Vitals, and content/keyword mapping. | seo, specialist, technical, audits, page, optimization, structured, data | `.claude/agents/seo-specialist.md` |
| `swift-build-resolver` | Swift/Xcode build, compilation, and dependency error resolution specialist. | swift, build, resolver, xcode, compilation, dependency, error, resolution | `.claude/agents/swift-build-resolver.md` |
| `swift-reviewer` | Expert Swift code reviewer specializing in protocol-oriented design, value semantics, ARC memory management, Swift Concurrency, and idiomatic patterns. | swift, reviewer, expert, code, specializing, protocol, oriented, design | `.claude/agents/swift-reviewer.md` |
| `type-design-analyzer` | Analyze type design for encapsulation, invariant expression, usefulness, and enforcement. | type, design, analyzer, analyze, encapsulation, invariant, expression, usefulness | `.claude/agents/type-design-analyzer.md` |

### General & Other (1)

| Agent | Description | Trigger keywords | File |
|---|---|---|---|
| `comment-analyzer` | Analyze code comments for accuracy, completeness, maintainability, and comment rot risk. | comment, analyzer, analyze, code, comments, accuracy, completeness, maintainability | `.claude/agents/comment-analyzer.md` |

### Mobile (1)

| Agent | Description | Trigger keywords | File |
|---|---|---|---|
| `kotlin-reviewer` | Kotlin and Android/KMP code reviewer. | kotlin, reviewer, android, kmp, code, reviewer., reviews, idiomatic | `.claude/agents/kotlin-reviewer.md` |

### Product & Process (1)

| Agent | Description | Trigger keywords | File |
|---|---|---|---|
| `network-troubleshooter` | Diagnoses network connectivity, routing, DNS, interface, and policy symptoms with a read-only OSI-layer workflow and evidence-backed root cause summary. | network, troubleshooter, diagnoses, connectivity, routing, dns, interface, policy | `.claude/agents/network-troubleshooter.md` |

### Security & Compliance (18)

| Agent | Description | Trigger keywords | File |
|---|---|---|---|
| `a11y-architect` | Accessibility Architect specializing in WCAG 2.2 compliance for Web and Native platforms. | a11y, architect, accessibility, specializing, wcag, compliance, web, native | `.claude/agents/a11y-architect.md` |
| `code-reviewer` | Expert code review specialist. | code, reviewer, expert, review, specialist., proactively, reviews, quality | `.claude/agents/code-reviewer.md` |
| `csharp-reviewer` | Expert C# code reviewer specializing in .NET conventions, async patterns, security, nullable reference types, and performance. | csharp, reviewer, expert, code, specializing, net, conventions, async | `.claude/agents/csharp-reviewer.md` |
| `database-reviewer` | PostgreSQL database specialist for query optimization, schema design, security, and performance. | database, reviewer, postgresql, specialist, query, optimization, schema, design | `.claude/agents/database-reviewer.md` |
| `django-reviewer` | Expert Django code reviewer specializing in ORM correctness, DRF patterns, migration safety, security misconfigurations, and production-grade Django practices. | django, reviewer, expert, code, specializing, orm, correctness, drf | `.claude/agents/django-reviewer.md` |
| `fastapi-reviewer` | Reviews FastAPI applications for async correctness, dependency injection, Pydantic schemas, security, OpenAPI quality, testing, and production readiness. | fastapi, reviewer, reviews, applications, async, correctness, dependency, injection | `.claude/agents/fastapi-reviewer.md` |
| `harmonyos-app-resolver` | HarmonyOS application development expert specializing in ArkTS and ArkUI. | harmonyos, app, resolver, application, development, expert, specializing, arkts | `.claude/agents/harmonyos-app-resolver.md` |
| `healthcare-reviewer` | Reviews healthcare application code for clinical safety, CDSS accuracy, PHI compliance, and medical data integrity. | healthcare, reviewer, reviews, application, code, clinical, safety, cdss | `.claude/agents/healthcare-reviewer.md` |
| `java-reviewer` | Expert Java code reviewer for Spring Boot and Quarkus projects. | java, reviewer, expert, code, spring, boot, quarkus, projects. | `.claude/agents/java-reviewer.md` |
| `network-config-reviewer` | Reviews router and switch configurations for security, correctness, stale references, risky change-window commands, and missing operational guardrails. | network, config, reviewer, reviews, router, switch, configurations, security | `.claude/agents/network-config-reviewer.md` |
| `opensource-forker` | Fork any project for open-sourcing. | opensource, forker, fork, project, open, sourcing., copies, files | `.claude/agents/opensource-forker.md` |
| `opensource-sanitizer` | Verify an open-source fork is fully sanitized before release. | opensource, sanitizer, verify, open, source, fork, fully, sanitized | `.claude/agents/opensource-sanitizer.md` |
| `php-reviewer` | Expert PHP code reviewer specializing in PSR-12 compliance, PHP type system, Eloquent ORM patterns, security, and performance. | php, reviewer, expert, code, specializing, psr, compliance, type | `.claude/agents/php-reviewer.md` |
| `python-reviewer` | Expert Python code reviewer specializing in PEP 8 compliance, Pythonic idioms, type hints, security, and performance. | python, reviewer, expert, code, specializing, pep, compliance, pythonic | `.claude/agents/python-reviewer.md` |
| `react-reviewer` | Expert React/JSX code reviewer specializing in hook correctness, render performance, server/client component boundaries, accessibility, and React-specific security. | react, reviewer, expert, jsx, code, specializing, hook, correctness | `.claude/agents/react-reviewer.md` |
| `security-reviewer` | Security vulnerability detection and remediation specialist. | security, reviewer, vulnerability, detection, remediation, specialist., proactively, after | `.claude/agents/security-reviewer.md` |
| `typescript-reviewer` | Expert TypeScript/JavaScript code reviewer specializing in type safety, async correctness, Node/web security, and idiomatic patterns. | typescript, reviewer, expert, javascript, code, specializing, type, safety | `.claude/agents/typescript-reviewer.md` |
| `vue-reviewer` | Expert Vue.js code reviewer specializing in Composition API correctness, reactivity pitfalls, component architecture, template security, and Vue-specific performance. | vue, reviewer, expert, vue.js, code, specializing, composition, api | `.claude/agents/vue-reviewer.md` |

### Testing & QA (6)

| Agent | Description | Trigger keywords | File |
|---|---|---|---|
| `e2e-runner` | End-to-end testing specialist using Vercel Agent Browser (preferred) with Playwright fallback. | e2e, runner, end, testing, specialist, vercel, browser, preferred | `.claude/agents/e2e-runner.md` |
| `gan-evaluator` | GAN Harness — Evaluator agent. | gan, evaluator, harness, agent., tests, live, running, application | `.claude/agents/gan-evaluator.md` |
| `pr-test-analyzer` | Review pull request test coverage quality and completeness, with emphasis on behavioral coverage and real bug prevention. | test, analyzer, review, pull, request, coverage, quality, completeness | `.claude/agents/pr-test-analyzer.md` |
| `rag-pipeline-reviewer` | Reviews RAG (Retrieval-Augmented Generation) pipelines for retrieval quality, chunking strategy, embedding choices, and evaluation coverage. | rag, pipeline, reviewer, reviews, retrieval, augmented, generation, pipelines | `.claude/agents/rag-pipeline-reviewer.md` |
| `spec-miner` | Extracts behavioral specs from existing codebases for OpenSpec. | spec, miner, extracts, behavioral, specs, existing, codebases, openspec. | `.claude/agents/spec-miner.md` |
| `tdd-guide` | Test-Driven Development specialist enforcing write-tests-first methodology. | tdd, guide, test, driven, development, specialist, enforcing, write | `.claude/agents/tdd-guide.md` |

## Rules

Rules are plain markdown guides grouped by language/framework directory (`common/` applies to every project; the rest are language-specific).

### angular (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | Angular Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with Angular specific content. | `.claude/rules/angular/coding-style.md` |
| `hooks` | Angular Hooks | > This file extends [common/hooks.md](../common/hooks.md) with Angular specific content. | `.claude/rules/angular/hooks.md` |
| `patterns` | Angular Patterns | > This file extends [common/patterns.md](../common/patterns.md) with Angular specific content. | `.claude/rules/angular/patterns.md` |
| `security` | Angular Security | > This file extends [common/security.md](../common/security.md) with Angular specific content. | `.claude/rules/angular/security.md` |
| `testing` | Angular Testing | > This file extends [common/testing.md](../common/testing.md) with Angular specific content. | `.claude/rules/angular/testing.md` |

### arkts (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | HarmonyOS / ArkTS Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with HarmonyOS and ArkTS-specific content. | `.claude/rules/arkts/coding-style.md` |
| `hooks` | HarmonyOS / ArkTS Hooks | > This file extends [common/hooks.md](../common/hooks.md) with HarmonyOS-specific build and validation hooks. | `.claude/rules/arkts/hooks.md` |
| `patterns` | HarmonyOS / ArkTS Patterns | > This file extends [common/patterns.md](../common/patterns.md) with HarmonyOS and ArkTS-specific patterns. | `.claude/rules/arkts/patterns.md` |
| `security` | HarmonyOS / ArkTS Security | > This file extends [common/security.md](../common/security.md) with HarmonyOS-specific security practices. | `.claude/rules/arkts/security.md` |
| `testing` | HarmonyOS / ArkTS Testing | > This file extends [common/testing.md](../common/testing.md) with HarmonyOS-specific testing practices. | `.claude/rules/arkts/testing.md` |

### common (10 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `agents` | Agent Orchestration | Located in `~/.claude/agents/`: | `.claude/rules/common/agents.md` |
| `code-review` | Code Review Standards | Code review ensures quality, security, and maintainability before code is merged. This rule defines when and how to conduct code reviews. | `.claude/rules/common/code-review.md` |
| `coding-style` | Coding Style | ALWAYS create new objects, NEVER mutate existing ones: | `.claude/rules/common/coding-style.md` |
| `development-workflow` | Development Workflow | > This file extends [common/git-workflow.md](./git-workflow.md) with the full feature development process that happens before git operations. | `.claude/rules/common/development-workflow.md` |
| `git-workflow` | Git Workflow | <type>: <description> | `.claude/rules/common/git-workflow.md` |
| `hooks` | Hooks System | - **PreToolUse**: Before tool execution (validation, parameter modification) | `.claude/rules/common/hooks.md` |
| `patterns` | Common Patterns | When implementing new functionality: | `.claude/rules/common/patterns.md` |
| `performance` | Performance Optimization | **Haiku** (90% of Sonnet capability, 3x cost savings): | `.claude/rules/common/performance.md` |
| `security` | Security Guidelines | - [ ] No hardcoded secrets (API keys, passwords, tokens) | `.claude/rules/common/security.md` |
| `testing` | Testing Requirements | Test Types (ALL required): | `.claude/rules/common/testing.md` |

### cpp (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | C++ Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with C++ specific content. | `.claude/rules/cpp/coding-style.md` |
| `hooks` | C++ Hooks | > This file extends [common/hooks.md](../common/hooks.md) with C++ specific content. | `.claude/rules/cpp/hooks.md` |
| `patterns` | C++ Patterns | > This file extends [common/patterns.md](../common/patterns.md) with C++ specific content. | `.claude/rules/cpp/patterns.md` |
| `security` | C++ Security | > This file extends [common/security.md](../common/security.md) with C++ specific content. | `.claude/rules/cpp/security.md` |
| `testing` | C++ Testing | > This file extends [common/testing.md](../common/testing.md) with C++ specific content. | `.claude/rules/cpp/testing.md` |

### csharp (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | C# Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with C#-specific content. | `.claude/rules/csharp/coding-style.md` |
| `hooks` | C# Hooks | > This file extends [common/hooks.md](../common/hooks.md) with C#-specific content. | `.claude/rules/csharp/hooks.md` |
| `patterns` | C# Patterns | > This file extends [common/patterns.md](../common/patterns.md) with C#-specific content. | `.claude/rules/csharp/patterns.md` |
| `security` | C# Security | > This file extends [common/security.md](../common/security.md) with C#-specific content. | `.claude/rules/csharp/security.md` |
| `testing` | C# Testing | > This file extends [common/testing.md](../common/testing.md) with C#-specific content. | `.claude/rules/csharp/testing.md` |

### dart (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | Dart/Flutter Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with Dart and Flutter-specific content. | `.claude/rules/dart/coding-style.md` |
| `hooks` | Dart/Flutter Hooks | > This file extends [common/hooks.md](../common/hooks.md) with Dart and Flutter-specific content. | `.claude/rules/dart/hooks.md` |
| `patterns` | Dart/Flutter Patterns | > This file extends [common/patterns.md](../common/patterns.md) with Dart, Flutter, and common ecosystem-specific content. | `.claude/rules/dart/patterns.md` |
| `security` | Dart/Flutter Security | > This file extends [common/security.md](../common/security.md) with Dart, Flutter, and mobile-specific content. | `.claude/rules/dart/security.md` |
| `testing` | Dart/Flutter Testing | > This file extends [common/testing.md](../common/testing.md) with Dart and Flutter-specific content. | `.claude/rules/dart/testing.md` |

### fsharp (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | F# Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with F#-specific content. | `.claude/rules/fsharp/coding-style.md` |
| `hooks` | F# Hooks | > This file extends [common/hooks.md](../common/hooks.md) with F#-specific content. | `.claude/rules/fsharp/hooks.md` |
| `patterns` | F# Patterns | > This file extends [common/patterns.md](../common/patterns.md) with F#-specific content. | `.claude/rules/fsharp/patterns.md` |
| `security` | F# Security | > This file extends [common/security.md](../common/security.md) with F#-specific content. | `.claude/rules/fsharp/security.md` |
| `testing` | F# Testing | > This file extends [common/testing.md](../common/testing.md) with F#-specific content. | `.claude/rules/fsharp/testing.md` |

### golang (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | Go Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with Go specific content. | `.claude/rules/golang/coding-style.md` |
| `hooks` | Go Hooks | > This file extends [common/hooks.md](../common/hooks.md) with Go specific content. | `.claude/rules/golang/hooks.md` |
| `patterns` | Go Patterns | > This file extends [common/patterns.md](../common/patterns.md) with Go specific content. | `.claude/rules/golang/patterns.md` |
| `security` | Go Security | > This file extends [common/security.md](../common/security.md) with Go specific content. | `.claude/rules/golang/security.md` |
| `testing` | Go Testing | > This file extends [common/testing.md](../common/testing.md) with Go specific content. | `.claude/rules/golang/testing.md` |

### java (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | Java Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with Java-specific content. | `.claude/rules/java/coding-style.md` |
| `hooks` | Java Hooks | > This file extends [common/hooks.md](../common/hooks.md) with Java-specific content. | `.claude/rules/java/hooks.md` |
| `patterns` | Java Patterns | > This file extends [common/patterns.md](../common/patterns.md) with Java-specific content. | `.claude/rules/java/patterns.md` |
| `security` | Java Security | > This file extends [common/security.md](../common/security.md) with Java-specific content. | `.claude/rules/java/security.md` |
| `testing` | Java Testing | > This file extends [common/testing.md](../common/testing.md) with Java-specific content. | `.claude/rules/java/testing.md` |

### kotlin (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | Kotlin Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with Kotlin-specific content. | `.claude/rules/kotlin/coding-style.md` |
| `hooks` | Kotlin Hooks | > This file extends [common/hooks.md](../common/hooks.md) with Kotlin-specific content. | `.claude/rules/kotlin/hooks.md` |
| `patterns` | Kotlin Patterns | > This file extends [common/patterns.md](../common/patterns.md) with Kotlin and Android/KMP-specific content. | `.claude/rules/kotlin/patterns.md` |
| `security` | Kotlin Security | > This file extends [common/security.md](../common/security.md) with Kotlin and Android/KMP-specific content. | `.claude/rules/kotlin/security.md` |
| `testing` | Kotlin Testing | > This file extends [common/testing.md](../common/testing.md) with Kotlin and Android/KMP-specific content. | `.claude/rules/kotlin/testing.md` |

### nuxt (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | Nuxt Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with Nuxt specific content. | `.claude/rules/nuxt/coding-style.md` |
| `hooks` | Nuxt Hooks | > This file extends [common/hooks.md](../common/hooks.md) with Nuxt specific content. | `.claude/rules/nuxt/hooks.md` |
| `patterns` | Nuxt Patterns | > This file extends [common/patterns.md](../common/patterns.md) with Nuxt specific content. | `.claude/rules/nuxt/patterns.md` |
| `security` | Nuxt Security | > This file extends [common/security.md](../common/security.md) with Nuxt specific content. | `.claude/rules/nuxt/security.md` |
| `testing` | Nuxt Testing | > This file extends [common/testing.md](../common/testing.md) with Nuxt specific content. | `.claude/rules/nuxt/testing.md` |

### perl (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | Perl Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with Perl-specific content. | `.claude/rules/perl/coding-style.md` |
| `hooks` | Perl Hooks | > This file extends [common/hooks.md](../common/hooks.md) with Perl-specific content. | `.claude/rules/perl/hooks.md` |
| `patterns` | Perl Patterns | > This file extends [common/patterns.md](../common/patterns.md) with Perl-specific content. | `.claude/rules/perl/patterns.md` |
| `security` | Perl Security | > This file extends [common/security.md](../common/security.md) with Perl-specific content. | `.claude/rules/perl/security.md` |
| `testing` | Perl Testing | > This file extends [common/testing.md](../common/testing.md) with Perl-specific content. | `.claude/rules/perl/testing.md` |

### php (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | PHP Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with PHP specific content. | `.claude/rules/php/coding-style.md` |
| `hooks` | PHP Hooks | > This file extends [common/hooks.md](../common/hooks.md) with PHP specific content. | `.claude/rules/php/hooks.md` |
| `patterns` | PHP Patterns | > This file extends [common/patterns.md](../common/patterns.md) with PHP specific content. | `.claude/rules/php/patterns.md` |
| `security` | PHP Security | > This file extends [common/security.md](../common/security.md) with PHP specific content. | `.claude/rules/php/security.md` |
| `testing` | PHP Testing | > This file extends [common/testing.md](../common/testing.md) with PHP specific content. | `.claude/rules/php/testing.md` |

### python (6 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | Python Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with Python specific content. | `.claude/rules/python/coding-style.md` |
| `fastapi` | FastAPI Rules | Use these rules for FastAPI projects alongside the general Python rules. | `.claude/rules/python/fastapi.md` |
| `hooks` | Python Hooks | > This file extends [common/hooks.md](../common/hooks.md) with Python specific content. | `.claude/rules/python/hooks.md` |
| `patterns` | Python Patterns | > This file extends [common/patterns.md](../common/patterns.md) with Python specific content. | `.claude/rules/python/patterns.md` |
| `security` | Python Security | > This file extends [common/security.md](../common/security.md) with Python specific content. | `.claude/rules/python/security.md` |
| `testing` | Python Testing | > This file extends [common/testing.md](../common/testing.md) with Python specific content. | `.claude/rules/python/testing.md` |

### react (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | React Coding Style | > This file extends [typescript/coding-style.md](../typescript/coding-style.md) and [common/coding-style.md](../common/coding-style.md) with React specific content. | `.claude/rules/react/coding-style.md` |
| `hooks` | React Hooks | > This file covers **React hooks** (`useState`, `useEffect`, `useMemo`, `useCallback`, custom hooks) — NOT the Claude Code `hooks/` runtime system. Naming matches the per-language convention `rules/<lang>/hooks.md` used across this repo. | `.claude/rules/react/hooks.md` |
| `patterns` | React Patterns | > This file extends [typescript/patterns.md](../typescript/patterns.md) and [common/patterns.md](../common/patterns.md) with React specific content. For hook-specific rules see [hooks.md](./hooks.md). | `.claude/rules/react/patterns.md` |
| `security` | React Security | > This file extends [typescript/security.md](../typescript/security.md) and [common/security.md](../common/security.md) with React specific content. | `.claude/rules/react/security.md` |
| `testing` | React Testing | > This file extends [typescript/testing.md](../typescript/testing.md) and [common/testing.md](../common/testing.md) with React specific content. | `.claude/rules/react/testing.md` |

### react-native (8 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `accessibility` | React Native / Expo Accessibility | > Extends the ECC quality bar to accessibility (a11y). Treat a11y as a release requirement, not an afterthought. | `.claude/rules/react-native/accessibility.md` |
| `coding-style` | React Native / Expo Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with React Native / Expo specific content. | `.claude/rules/react-native/coding-style.md` |
| `hooks` | React Native / Expo Hooks | > This file extends [common/hooks.md](../common/hooks.md) with React Native / Expo-specific automation guidance. | `.claude/rules/react-native/hooks.md` |
| `patterns` | React Native / Expo Patterns | > This file extends [common/patterns.md](../common/patterns.md) with React Native / Expo specific patterns. | `.claude/rules/react-native/patterns.md` |
| `performance` | React Native / Expo Performance | > This file extends [common/performance.md](../common/performance.md) with React Native / Expo specific content. | `.claude/rules/react-native/performance.md` |
| `production-readiness` | React Native / Expo Production Readiness | > Extends the ECC philosophy to ship-grade concerns that style/pattern rules cannot encode by themselves. | `.claude/rules/react-native/production-readiness.md` |
| `security` | React Native / Expo Security | > This file extends [common/security.md](../common/security.md) with React Native / Expo specific content. | `.claude/rules/react-native/security.md` |
| `testing` | React Native / Expo Testing | > This file extends [common/testing.md](../common/testing.md) with React Native / Expo specific content. | `.claude/rules/react-native/testing.md` |

### ruby (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | Ruby Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with Ruby and Rails specific content. | `.claude/rules/ruby/coding-style.md` |
| `hooks` | Ruby Hooks | > This file extends [common/hooks.md](../common/hooks.md) with Ruby and Rails specific content. | `.claude/rules/ruby/hooks.md` |
| `patterns` | Ruby Patterns | > This file extends [common/patterns.md](../common/patterns.md) with Ruby and Rails specific content. | `.claude/rules/ruby/patterns.md` |
| `security` | Ruby Security | > This file extends [common/security.md](../common/security.md) with Ruby and Rails specific content. | `.claude/rules/ruby/security.md` |
| `testing` | Ruby Testing | > This file extends [common/testing.md](../common/testing.md) with Ruby and Rails specific content. | `.claude/rules/ruby/testing.md` |

### rust (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | Rust Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with Rust-specific content. | `.claude/rules/rust/coding-style.md` |
| `hooks` | Rust Hooks | > This file extends [common/hooks.md](../common/hooks.md) with Rust-specific content. | `.claude/rules/rust/hooks.md` |
| `patterns` | Rust Patterns | > This file extends [common/patterns.md](../common/patterns.md) with Rust-specific content. | `.claude/rules/rust/patterns.md` |
| `security` | Rust Security | > This file extends [common/security.md](../common/security.md) with Rust-specific content. | `.claude/rules/rust/security.md` |
| `testing` | Rust Testing | > This file extends [common/testing.md](../common/testing.md) with Rust-specific content. | `.claude/rules/rust/testing.md` |

### swift (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | Swift Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with Swift specific content. | `.claude/rules/swift/coding-style.md` |
| `hooks` | Swift Hooks | > This file extends [common/hooks.md](../common/hooks.md) with Swift specific content. | `.claude/rules/swift/hooks.md` |
| `patterns` | Swift Patterns | > This file extends [common/patterns.md](../common/patterns.md) with Swift specific content. | `.claude/rules/swift/patterns.md` |
| `security` | Swift Security | > This file extends [common/security.md](../common/security.md) with Swift specific content. | `.claude/rules/swift/security.md` |
| `testing` | Swift Testing | > This file extends [common/testing.md](../common/testing.md) with Swift specific content. | `.claude/rules/swift/testing.md` |

### typescript (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | TypeScript/JavaScript Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with TypeScript/JavaScript specific content. | `.claude/rules/typescript/coding-style.md` |
| `hooks` | TypeScript/JavaScript Hooks | > This file extends [common/hooks.md](../common/hooks.md) with TypeScript/JavaScript specific content. | `.claude/rules/typescript/hooks.md` |
| `patterns` | TypeScript/JavaScript Patterns | > This file extends [common/patterns.md](../common/patterns.md) with TypeScript/JavaScript specific content. | `.claude/rules/typescript/patterns.md` |
| `security` | TypeScript/JavaScript Security | > This file extends [common/security.md](../common/security.md) with TypeScript/JavaScript specific content. | `.claude/rules/typescript/security.md` |
| `testing` | TypeScript/JavaScript Testing | > This file extends [common/testing.md](../common/testing.md) with TypeScript/JavaScript specific content. | `.claude/rules/typescript/testing.md` |

### vue (5 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | Vue Coding Style | > This file extends [common/coding-style.md](../common/coding-style.md) with Vue specific content. | `.claude/rules/vue/coding-style.md` |
| `hooks` | Vue Hooks | > This file extends [common/hooks.md](../common/hooks.md) with Vue specific content. | `.claude/rules/vue/hooks.md` |
| `patterns` | Vue Patterns | > This file extends [common/patterns.md](../common/patterns.md) with Vue specific content. | `.claude/rules/vue/patterns.md` |
| `security` | Vue Security | > This file extends [common/security.md](../common/security.md) with Vue specific content. | `.claude/rules/vue/security.md` |
| `testing` | Vue Testing | > This file extends [common/testing.md](../common/testing.md) with Vue specific content. | `.claude/rules/vue/testing.md` |

### web (7 files)

| Rule file | Topic | Summary | File |
|---|---|---|---|
| `coding-style` | Web Coding Style | Organize by feature or surface area, not by file type: | `.claude/rules/web/coding-style.md` |
| `design-quality` | Web Design Quality Standards | Do not ship generic template-looking UI. Frontend output should look intentional, opinionated, and specific to the product. | `.claude/rules/web/design-quality.md` |
| `hooks` | Web Hooks | Prefer project-local tooling. Do not wire hooks to remote one-off package execution. | `.claude/rules/web/hooks.md` |
| `patterns` | Web Patterns | Use compound components when related UI shares state and interaction semantics: | `.claude/rules/web/patterns.md` |
| `performance` | Web Performance Rules | \| Page Type \| JS Budget (gzipped) \| CSS Budget \| | `.claude/rules/web/performance.md` |
| `security` | Web Security Rules | Always configure a production CSP. | `.claude/rules/web/security.md` |
| `testing` | Web Testing Rules | - Screenshot key breakpoints: 320, 768, 1024, 1440 | `.claude/rules/web/testing.md` |

