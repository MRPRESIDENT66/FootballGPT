# ⚽ FootballGPT

AI-powered football intelligence platform using multi-agent collaboration. Ask natural language questions about players, tactics, and transfers — multiple specialized AI agents work together to generate comprehensive analysis reports.

## Architecture

```
User Query
    │
    ▼
┌──────────┐     ┌───────────┐
│  Router   │ ◄──│   FAISS   │  ← Semantic conversation memory (multi-turn)
└────┬─────┘     │  Memory   │
     │           └───────────┘
     │ conditional routing                    ┌────────────┐
     ├── scout ──────► Scout Agent ──► Analyst Agent ──┐  │
     ├── compare ────► Analyst Agent ──────────────────┤  │
     ├── tactics ────► Tactics Agent ──────────────────┤  │  Qdrant
     └── recommend ──► ┌─Scout Agent──┐                │  │  Knowledge
                       │              ├► Analyst Agent ─┤◄─┤  Base
                       └─Tactics Agent┘  (parallel)    │  │  (Wikipedia)
                                                       │  │
                                                       ▼  │
                                                 ┌──────────┐
                                                 │ Reporter  │◄─┘
                                                 └────┬─────┘
                                                      ▼
                                              Markdown Report
```

**Four execution flows:**

| Intent | Agent Pipeline | Example |
|--------|---------------|---------|
| Scout | Router → Scout → Analyst → Reporter | "Find strikers under 25 in La Liga" |
| Compare | Router → Analyst → Reporter | "Compare Salah and Mbappe" |
| Tactics | Router → Tactics(RAG + team data) → Reporter | "Analyze Arsenal's weaknesses" |
| Recommend | Router → [Scout ∥ Tactics] → Analyst → Reporter | "Recommend a striker for Arsenal" |

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Agent Orchestration | **LangGraph** | State graph, conditional routing, parallel fan-out/join |
| LLM Framework | **LangChain** | Tool binding, structured prompts, message management |
| LLM | **Qwen** (via DashScope) | Intent classification, analysis, report generation |
| Knowledge Base | **Qdrant** (local) | Vector store for Wikipedia knowledge (players, teams, seasons) |
| Conversation Memory | **FAISS** | Semantic retrieval of relevant past turns |
| Data Pipeline | **Wikipedia-API** | Automated scraping + chunking of football articles |
| Player Data | **API-Football** | 1,450+ real players, 54 clubs, 2024/25 season stats |
| Observability | **LangSmith** | End-to-end tracing of agent calls, tool invocations, latency |
| CLI | **Rich** | Interactive terminal UI with streaming progress |
| Web UI | **Gradio** | Chat interface with streaming node progress |

## Project Structure

```
FootballGPT/
├── main.py                         # CLI entry point (Rich UI)
├── app.py                          # Web entry point (Gradio chat UI)
├── graph/
│   └── workflow.py                 # LangGraph state machine + MemorySaver checkpointing
├── agents/
│   ├── router.py                   # Intent classification + parameter extraction
│   ├── scout.py                    # Player search (LLM-free, direct DB lookup + RAG enrichment)
│   ├── analyst.py                  # ReAct agent: compare, similarity, top scorers (max 2 rounds)
│   ├── tactics.py                  # ReAct agent: RAG + team stats + roster (max 2 rounds)
│   └── reporter.py                 # Final report synthesis
├── tools/
│   └── player_db.py                # 7 LangChain @tools: search, compare, roster, similarity, top scorers, team stats
├── knowledge/
│   ├── qdrant_store.py             # Qdrant vector store (CRUD, filtered search)
│   ├── rag.py                      # RAG interface (tool + direct injection)
│   ├── wiki_scraper.py             # Wikipedia scraper + chunker (player/team/season)
│   ├── ingest.py                   # Data pipeline: Wikipedia → chunks → Qdrant
│   └── memory.py                   # FAISS-based semantic conversation memory
├── data/
│   ├── players.json                # 1,450+ real players (from API-Football)
│   └── qdrant_store/               # Qdrant local storage (auto-generated)
├── scripts/
│   ├── fetch_players.py            # Incremental player data fetcher
│   ├── eval_router.py              # Router intent classification eval (20 test cases)
│   └── eval_rag.py                 # RAG retrieval quality eval (Recall@K, MRR)
├── config/
│   └── settings.py                 # Centralized configuration
└── requirements.txt
```

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in:
- `DASHSCOPE_API_KEY` — **Required**. Get from [DashScope](https://dashscope.aliyuncs.com/)
- `FOOTBALL_API_KEY` — Optional. Only needed to refresh player data via `scripts/fetch_players.py`
- `LANGSMITH_API_KEY` — Optional. Enables end-to-end tracing in [LangSmith](https://smith.langchain.com)

### 3. Build the knowledge base

Scrape Wikipedia articles and load into Qdrant:

```bash
python -m knowledge.ingest                    # full ingest (players + teams + seasons)
python -m knowledge.ingest --only players     # players only
python -m knowledge.ingest --only teams       # teams only
python -m knowledge.ingest --only seasons     # seasons only
python -m knowledge.ingest --limit 10         # first 10 entities (for quick testing)
```

### 4. Run

**CLI** (Rich terminal UI):
```bash
python main.py
```

**Web UI** (Gradio chat interface):
```bash
python app.py
```

### 5. (Optional) Refresh player data

```bash
python scripts/fetch_players.py            # fetch all remaining teams
python scripts/fetch_players.py --teams 5  # fetch next 5 teams only
```

Supports incremental fetching — run across multiple days to stay within the free API limit (100 requests/day).

## Evaluation

```bash
python scripts/eval_router.py    # Router intent classification accuracy (20 test cases, 4 intents)
python scripts/eval_rag.py       # RAG retrieval quality (Recall@K, MRR over 15 queries)
```

## Example Queries

```
Query > Recommend a striker for Arsenal, budget under 100M
Query > Compare Haaland and Mbappe this season
Query > Analyze Liverpool's tactical system
Query > Find young midfielders in La Liga with high assist rate
```

Multi-turn supported — follow up with "compare him with..." or "what about his stats?" (semantic memory retrieves relevant past turns).

## Key Design Decisions

- **Parallel fan-out**: Scout and Tactics run concurrently for `recommend` queries via LangGraph conditional edges, cutting response time ~40%
- **Router uses lightweight model** (qwen-plus): Simple JSON classification doesn't need a thinking model — 2s vs 30s
- **Thinking mode disabled** for worker agents: `enable_thinking: False` keeps responses fast and focused
- **Scout is LLM-free**: Direct database lookup + RAG enrichment, no LLM reasoning overhead
- **ReAct pattern in Analyst & Tactics**: LLM autonomously decides which tools to call (from 4 and 3 tools respectively), observes results, and re-reasons — max 2 rounds to prevent runaway API costs
- **Semantic conversation memory**: FAISS vector retrieval instead of fixed sliding window — finds contextually relevant history regardless of recency
- **Dual RAG interfaces**: `@tool` for agents that decide when to search (Tactics), plain function for direct context injection (Scout, Analyst, Reporter)
- **Wikipedia data pipeline**: Automated scraping → section extraction → sentence-boundary chunking → Qdrant upsert with structured metadata (entity type, position, league) for filtered retrieval
- **Incremental data fetching**: Build up the player database across multiple days within free API limits
