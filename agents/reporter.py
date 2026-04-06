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

    report = llm.invoke(messages).content

    # Self-correction: LLM evaluates its own report, regenerates if quality is low
    eval_prompt = f"""Rate the following report on a scale of 1-10 based on:
- Does it directly answer the user's query: "{query}"?
- Does it include specific stats and facts (not vague)?
- Is it well-structured and complete?

Report:
{report}

Respond with ONLY a JSON object: {{"score": <int>, "feedback": "<what's missing or wrong>"}}"""

    eval_response = llm.invoke([HumanMessage(content=eval_prompt)]).content

    try:
        import json
        eval_result = json.loads(eval_response.strip().strip("`").strip("json").strip())
        score = eval_result.get("score", 10)
        feedback = eval_result.get("feedback", "")

        if score < 7 and feedback:
            # Regenerate with self-correction feedback
            messages.append(HumanMessage(
                content=f"Your report scored {score}/10. Issues: {feedback}\n"
                        f"Please regenerate the report addressing these issues."
            ))
            report = llm.invoke(messages).content
    except (json.JSONDecodeError, KeyError):
        pass  # If eval parsing fails, keep original report

    return report
