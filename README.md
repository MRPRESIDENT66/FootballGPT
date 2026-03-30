# ⚽ FootballGPT

AI-powered football intelligence platform using multi-agent collaboration. Ask natural language questions about players, tactics, and transfers — multiple specialized AI agents work together to generate comprehensive analysis reports.

## Architecture

```
User Query
    │
    ▼
┌─────────┐
│  Router  │  ← Intent classification + parameter extraction (qwen-plus)
└────┬────┘
     │
     ├── scout ──────► Scout Agent     → Search player database
     ├── compare ────► Analyst Agent   → Data comparison
     ├── tactics ────► Tactics Agent   → RAG-based tactical analysis
     └── recommend ──► [Scout ∥ Tactics] → Analyst → Reporter
                              │
                              ▼
                      ┌──────────┐
                      │ Reporter │  ← Final report generation
                      └──────────┘
```

**Four execution flows:**

| Intent | Agent Pipeline | Example |
|--------|---------------|---------|
| Scout | Router → Scout → Analyst → Reporter | "Find strikers under 25 in La Liga" |
| Compare | Router → Analyst → Reporter | "Compare Salah and Mbappe" |
| Tactics | Router → Tactics(RAG) → Reporter | "Analyze Arsenal's weaknesses" |
| Recommend | Router → [Scout ∥ Tactics] → Analyst → Reporter | "Recommend a striker for Arsenal" |

## Tech Stack

- **LangGraph** — Multi-agent orchestration with state graph, parallel execution
- **LangChain** — LLM integration, tool binding, prompt management
- **Qwen (via DashScope)** — LLM for reasoning and report generation
- **FAISS** — Vector store for RAG-based tactical knowledge retrieval
- **API-Football** — Real player data (800+ players, 30 clubs, 2024/25 season)

## Project Structure

```
FootballGPT/
├── main.py                     # CLI entry point with Rich UI
├── graph/
│   └── workflow.py             # LangGraph state machine (core orchestration)
├── agents/
│   ├── router.py               # Intent classification + parameter extraction
│   ├── scout.py                # Player search (direct DB lookup, no LLM)
│   ├── analyst.py              # Data analysis with tool calling
│   ├── tactics.py              # Tactical analysis with RAG
│   └── reporter.py             # Report synthesis
├── tools/
│   └── player_db.py            # LangChain tools: search, compare, roster
├── knowledge/
│   ├── team_profiles.json      # Tactical knowledge base (8 top clubs)
│   └── rag.py                  # FAISS + DashScope embeddings
├── data/
│   └── players.json            # 800+ real players (from API-Football)
├── scripts/
│   └── fetch_players.py        # Incremental data fetcher
└── config/
    └── settings.py             # Configuration
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
- `DASHSCOPE_API_KEY` — Required. Get from [DashScope](https://dashscope.aliyuncs.com/)
- `FOOTBALL_API_KEY` — Optional. Only needed to refresh player data. Get from [API-Football](https://dashboard.api-football.com/)

### 3. Run

```bash
python main.py
```

### 4. (Optional) Refresh player data

```bash
python scripts/fetch_players.py            # fetch all remaining teams
python scripts/fetch_players.py --teams 5  # fetch next 5 teams only
```

The script supports incremental fetching — run it across multiple days to stay within the free API limit (100 requests/day).

## Example Queries

```
Query > Recommend a striker for Arsenal, budget under 100M
Query > Compare Haaland and Mbappe this season
Query > Analyze Liverpool's tactical system
Query > Find young midfielders in La Liga with high assist rate
```

## Key Design Decisions

- **Parallel execution**: Scout and Tactics agents run concurrently for `recommend` queries, cutting response time by ~40%
- **Router uses lightweight model** (qwen-plus): Simple classification doesn't need a thinking model — 2s vs 30s
- **Thinking mode disabled** for worker agents: `enable_thinking: False` for faster responses
- **Scout is LLM-free**: Direct database lookup instead of LLM-driven tool calling — instant results
- **Tool call limits**: Analyst max 2 rounds, Tactics max 1 round — prevents unnecessary API calls
- **Incremental data fetching**: Build up the player database across multiple days within free API limits
