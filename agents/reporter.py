"""Reporter Agent — synthesizes all findings into a final report."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import settings

REPORTER_PROMPT = """You are a football report writer. Synthesize the analysis below into a clear final report.

Format:
1. Executive summary (2-3 sentences)
2. Key findings with specific stats
3. Recommendation / conclusion

Guidelines:
- Use markdown formatting
- Include data tables where appropriate
- For scouting: rank players, include key stats, note risks
- For comparison: clear verdict with evidence
- For tactics: formation diagram, specific suggestions
- Always output in English
- Keep it focused — no fluff"""


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

    messages = [
        SystemMessage(content=REPORTER_PROMPT),
        HumanMessage(content="\n\n".join(parts)),
    ]

    return llm.invoke(messages).content
