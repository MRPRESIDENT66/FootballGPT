"""Reporter Agent — synthesizes all findings into a final report, enriched with knowledge base."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import settings
from knowledge.rag import retrieve_knowledge

REPORTER_PROMPT = """You are a football report writer. Synthesize the analysis below into a clear, detailed report.

Format:
1. **Summary** — 3-5 sentences overview
2. **Top Picks** — for each candidate:
   - Name (Age, Club, Nationality)
   - Key stats and strengths
   - Wikipedia background (career highlights, honours, playing style)
   - Risks or concerns
3. **Verdict** — final recommendation with reasoning

Guidelines:
- Use bullet points, avoid wide tables
- Reference Wikipedia background to add depth (career trajectory, honours, style descriptions)
- Be specific with stats and facts, not vague
- Reply in the same language as the user's query
- No filler or disclaimers"""


def run_reporter(query: str, scout_data: str, analysis: str, tactical_context: str) -> str:
    """Generate final report from all gathered information."""
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
        temperature=0.5,
        **settings.NO_THINKING,
    )

    parts = [f"Original user query: {query}"]
    if scout_data:
        parts.append(f"## Scout Data\n{scout_data}")
    if analysis:
        parts.append(f"## Analysis\n{analysis}")
    if tactical_context:
        parts.append(f"## Tactical Context\n{tactical_context}")

    # RAG: add team/season background for richer reports
    knowledge = retrieve_knowledge(query, limit=5)
    if knowledge:
        parts.append(f"## Wikipedia Background\n{knowledge}")

    messages = [
        SystemMessage(content=REPORTER_PROMPT),
        HumanMessage(content="\n\n".join(parts)),
    ]

    return llm.invoke(messages).content
