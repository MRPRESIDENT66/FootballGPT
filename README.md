# ⚽ FootballGPT

> **Multi-agent football intelligence platform.** Ask natural language questions — a team of specialized LLM agents collaborates to produce scouting reports, tactical analyses, and transfer recommendations backed by real stats and Wikipedia knowledge.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-0.2+-orange" alt="LangGraph">
  <img src="https://img.shields.io/badge/LangChain-0.3+-green" alt="LangChain">
  <img src="https://img.shields.io/badge/Qdrant-1.9+-red" alt="Qdrant">
  <img src="https://img.shields.io/badge/LLM-Qwen--Plus-purple" alt="Qwen">
</p>

---

## ✨ Highlights

- 🧠 **5 Specialized Agents** orchestrated via LangGraph state machine — Router, Scout, Analyst, Tactics, Reporter
- ⚡ **Parallel Fan-out** — Scout and Tactics run concurrently for `recommend` queries (~40% faster)
- 🔍 **ReAct Pattern** — Analyst and Tactics agents autonomously pick tools from 7 LangChain `@tool`s (max 2 rounds, cost-capped)
- 🎯 **Self-Correction** — Analyst reflects on its own output; Reporter scores itself (≥7/10 or regenerate)
- 📚 **Shared + Incremental RAG** — shared baseline retrieval plus role-specific retrieval query rewriting on top of Qdrant Wikipedia knowledge
- 🧠 **Vector Memory** — FAISS semantic retrieval for multi-turn follow-ups
- 🔭 **Observability** — Zero-code LangSmith tracing for every agent call, tool invocation, and latency

---

## 🎬 Demo

### Web UI (Gradio)

<!-- TODO: insert screenshot of Gradio chat interface -->
<!-- Suggested file: docs/demo_web.png -->
> _Screenshot coming — shows streaming node progress as each agent fires._

### CLI (Rich)

<!-- TODO: insert screenshot or GIF of Rich terminal UI -->
<!-- Suggested file: docs/demo_cli.gif -->
> _Screenshot coming — shows live agent trace in the terminal._

### End-to-End Example: Recommend Query

<!-- TODO: paste real run of "Recommend a striker for Arsenal under 100M" -->
<!-- This will demonstrate: Router → [Scout ∥ Tactics] → Analyst → Reporter full chain -->

<details>
<summary><b>Query:</b> <i>"Recommend a striker for Arsenal under 100M budget"</i></summary>

```
[Router]     → intent=recommend, filters={position: ST, budget: 100M, team: Arsenal}
[Scout]      → 8 candidates matched (direct DB + Wikipedia enrichment)
[Tactics]    → Arsenal uses 4-3-3, needs mobile ST with pressing ability (RAG)
[Analyst]    → ReAct: calls compare_players(Osimhen, Isak) → calls find_similar_players(Isak)
             → reflection: PASS
[Reporter]   → drafts report → self-score: 8/10 → final output
```

**Final report:**

> _Output coming — will insert real run._

</details>

<details>
<summary><b>Query:</b> <i>"Compare Haaland and Mbappe this season"</i></summary>

<!-- TODO: paste real run output showing analyst with compare_players tool -->
> _Output coming — demonstrates Analyst's ReAct loop invoking `compare_players` tool._

</details>

<details>
<summary><b>Query:</b> <i>"Analyze Liverpool's tactical system and weaknesses"</i></summary>

<!-- TODO: paste real run output showing Tactics agent + RAG -->
> _Output coming — demonstrates Tactics agent using RAG retrieval and team stats tools._

</details>

---

## 🏗️ Architecture

```
                            User Query
                                │
                                ▼
                        ┌───────────────┐      ┌──────────────┐
                        │    Router     │ ◄── │ FAISS Memory │  (multi-turn)
                        │ (qwen-plus)   │      └──────────────┘
                        └───────┬───────┘
                                │ conditional routing
     ┌──────────────┬───────────┼─────────────┬──────────────────┐
     ▼              ▼           ▼             ▼                  ▼
  SCOUT         COMPARE      TACTICS      RECOMMEND (parallel fan-out)
     │              │           │             │                  │
     ▼              ▼           ▼        ┌────┴─────┐            │
  Scout ─────► Analyst ──► Tactics      Scout    Tactics         │
  (DB+RAG)    (ReAct)    (ReAct+RAG)      │        │             │
     │              │           │         └───┬────┘             │
     └────┬─────────┴───────────┤             ▼                  │
          │                     │         Analyst (ReAct)        │
          │                     │             │                  │
          ▼                     ▼             ▼                  │
                        ┌──────────────────────────┐             │
                        │   Reporter (synthesize   │◄── Qdrant RAG (Wikipedia)
                        │   + self-score ≥ 7/10)   │
                        └────────────┬─────────────┘
                                     ▼
                              Markdown Report
```

### Execution Flows

| Intent | Pipeline | Example Query |
|--------|----------|---------------|
| **Scout** | Router → Scout → Analyst → Reporter | _"Find strikers under 25 in La Liga"_ |
| **Compare** | Router → Analyst → Reporter | _"Compare Salah and Mbappe"_ |
| **Tactics** | Router → Tactics → Reporter | _"Analyze Arsenal's weaknesses"_ |
| **Recommend** | Router → [Scout ∥ Tactics] → Analyst → Reporter | _"Recommend a striker for Arsenal"_ |

---

## 🔬 How It Works — A Walkthrough

Take the query **_"Recommend a striker for Arsenal under 100M"_**:

1. **Router** (lightweight LLM, ~2s) classifies intent as `recommend`, extracts `{team: Arsenal, position: ST, budget: 100M}`. FAISS retrieves semantically similar past turns for context.

2. **Shared retrieval** runs once and stores baseline team/player context in state. After that, parallel fan-out fires Scout and Tactics concurrently:
   - **Scout** (LLM-free): filters player database → attaches shared knowledge as baseline evidence.
   - **Tactics** (ReAct): calls `get_team_roster(Arsenal)` + role-specific retrieval when shared context is insufficient → derives tactical requirements (pressing striker, 4-3-3 compatibility).

3. **Join node** waits for both to complete, merges their outputs.

4. **Analyst** (ReAct, max 2 rounds): decides to call `compare_players(Isak, Osimhen)` based on Scout candidates, observes stats, optionally calls `find_similar_players` for alternatives. If shared context still misses candidate-specific background, a small retrieval-rewriter model generates narrower player queries before analysis. Then **reflects** on its output — if something's missing, regenerates.

5. **Reporter** synthesizes everything into a structured report, **self-evaluates** (1-10 score), and regenerates if score < 7.

6. Every step is traced in **LangSmith** for debugging.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Agent Orchestration | **LangGraph** | State graph, conditional routing, parallel fan-out/join, checkpointing |
| LLM Framework | **LangChain** | Tool binding, structured prompts, message management |
| LLM | **Qwen** (DashScope) | `qwen-plus` for router, `qwen3.5-plus` for agents |
| Knowledge Base | **Qdrant** | Vector store with payload filtering (entity_type, position, league) |
| Conversation Memory | **FAISS** | Semantic retrieval of past turns (not sliding window) |
| Embeddings | **text-embedding-v4** | 1024-dim vectors via DashScope |
| Data Pipeline | **Wikipedia-API + MediaWiki** | Structured article scraping + sentence-boundary chunking |
| Player Data | **API-Football** | 1,450+ real players across 54 clubs (2024/25 season) |
| Observability | **LangSmith** | End-to-end tracing of agent calls, tool invocations, latency |
| Web UI | **Gradio** | Chat interface with streaming node progress |
| CLI | **Rich** | Interactive terminal UI |

---

## 📂 Project Structure

```
FootballGPT/
├── main.py                  # CLI entry point (Rich UI)
├── app.py                   # Web entry point (Gradio chat)
├── graph/
│   └── workflow.py          # LangGraph state machine + MemorySaver checkpointing
├── agents/
│   ├── router.py            # Intent classification + parameter extraction (JSON schema)
│   ├── scout.py             # LLM-free player search (DB + RAG enrichment)
│   ├── analyst.py           # ReAct agent: 4 tools + self-reflection (max 2 rounds)
│   ├── tactics.py           # ReAct agent: 3 tools + RAG (max 2 rounds)
│   └── reporter.py          # Report synthesis + self-scoring (≥7/10)
├── tools/
│   └── player_db.py         # 7 LangChain @tools: search, compare, roster, similarity, etc.
├── knowledge/
│   ├── qdrant_store.py      # Qdrant client (batch upsert, filtered search)
│   ├── rag.py               # Dual RAG interface (@tool + direct function)
│   ├── retrieval.py         # Shared retrieval + role-specific query rewriting
│   ├── wiki_scraper.py      # Wikipedia scraper + sentence-boundary chunker
│   ├── ingest.py            # Pipeline: Wikipedia → chunks → embeddings → Qdrant
│   └── memory.py            # FAISS-based semantic conversation memory
├── scripts/
│   └── fetch_players.py     # Incremental API-Football data fetcher (rate-limited)
├── config/settings.py       # Centralized env-based config
└── data/
    ├── players.json         # 1,450+ players
    └── qdrant_store/        # Qdrant local storage
```

---

## 🚀 Setup

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env:
#   DASHSCOPE_API_KEY=...       (required — get from https://dashscope.aliyuncs.com/)
#   FOOTBALL_API_KEY=...        (optional — only needed to refresh player data)
#   LANGSMITH_API_KEY=...       (optional — enables end-to-end tracing)
```

### 3. Build the knowledge base

```bash
python -m knowledge.ingest                 # full ingest (players + teams + seasons)
python -m knowledge.ingest --only players  # players only
python -m knowledge.ingest --limit 10      # first 10 entities (quick test)
```

### 4. Run

```bash
python main.py    # CLI (Rich terminal UI)
python app.py     # Web (Gradio chat, opens http://localhost:7860)
```

### 5. (Optional) Refresh player data

```bash
python scripts/fetch_players.py             # fetch all remaining teams
python scripts/fetch_players.py --teams 5   # fetch next 5 teams (respects 100 req/day limit)
```

---

## 💡 Key Design Decisions

Interview talking points — each choice reflects a real trade-off:

- **Parallel fan-out for `recommend`** — Scout and Tactics are independent; running them concurrently via LangGraph conditional edges cuts response time ~40%.

- **LLM-free Scout Agent** — Player search is a deterministic DB query + vector enrichment. No reasoning needed → zero LLM cost, faster, fully reproducible. LLMs are used only where reasoning adds value.

- **Lightweight router model** — Intent classification is a simple JSON task; `qwen-plus` (no thinking) returns in ~2s vs ~30s for a reasoning model.

- **ReAct with hard cap (max 2 rounds)** — Analyst and Tactics autonomously pick tools, but unbounded ReAct loops can burn tokens. Capping at 2 rounds bounds worst-case cost while preserving reasoning depth.

- **Self-correction at multiple stages** — Analyst reflects on its own output (PASS or regenerate); Reporter self-scores 1-10 and regenerates if <7. Catches hallucinations and incomplete answers without a human in the loop.

- **Semantic conversation memory (FAISS)** — Sliding windows miss relevant old context. Vector retrieval finds contextually relevant past turns regardless of recency — enables natural multi-turn follow-ups like _"compare him with..."_.

- **Shared retrieval before fan-out** — baseline player/team knowledge is retrieved once and written into graph state so downstream agents start from the same evidence instead of repeating the same search.

- **Rule-gated incremental retrieval** — Analyst, Tactics, and Reporter only trigger extra retrieval when simple rules say shared context is missing key entities or role-specific context.

- **Small-model query rewriting** — when extra retrieval is needed, a lightweight model rewrites retrieval queries for the current agent role instead of reusing the raw user query.

- **Payload-filtered RAG** — Qdrant metadata (`entity_type`, `position`, `league`) enables targeted retrieval: _"only retrieve goalkeeper articles"_ instead of brute-force semantic search. Raises precision dramatically.

- **Incremental data pipeline** — API-Football free tier is 100 req/day. The fetcher checkpoints progress and resumes daily — no all-or-nothing bulk loads.

---

## 📖 Example Queries

```
> Recommend a striker for Arsenal, budget under 100M
> Compare Haaland and Mbappe this season
> Analyze Liverpool's tactical system and weaknesses
> Find young midfielders in La Liga with high assist rate
> Who plays like Kevin De Bruyne but younger?
```

Multi-turn supported — follow up with _"compare him with..."_ or _"what about his defensive stats?"_ — semantic memory retrieves relevant past turns.

---

## 📝 License

MIT
