"""FootballGPT — AI Football Analysis Platform with Multi-Agent Collaboration."""

import time
import uuid

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from graph.workflow import build_workflow

console = Console()


def display_banner():
    banner = """
⚽ **FootballGPT** — AI Football Intelligence Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multi-Agent System: Router → Scout → Analyst → Tactics → Reporter

Example queries:
  • Find a right winger for Liverpool, under 25, fast
  • Compare Salah and Mbappe this season
  • Analyze Barcelona's tactical system and weaknesses
  • Recommend a striker for Arsenal, budget under 100M

Multi-turn supported — follow up with "compare him with..." or "what about his stats?"

Type quit to exit
"""
    console.print(Panel(banner, border_style="green"))


INTENT_LABELS = {
    "scout": "🔍 Player Scouting",
    "compare": "📊 Player Comparison",
    "tactics": "📋 Tactical Analysis",
    "recommend": "🎯 Player Recommendation",
}

NODE_LABELS = {
    "router": "🧭 Router — Classifying intent...",
    "shared_retrieval": "📚 Retrieval — Gathering shared context...",
    "scout": "🔍 Scout — Searching players...",
    "analyst": "📊 Analyst — Analyzing data...",
    "tactics": "📋 Tactics — Evaluating fit...",
    "reporter": "📝 Reporter — Generating report...",
    "join_recommend": "⏳ Merging parallel results...",
}


def run():
    display_banner()
    app = build_workflow()

    # Each session gets a unique thread_id for conversation memory
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": "FootballGPT-CLI",           # LangSmith trace name
        "metadata": {"interface": "cli", "session_id": thread_id},
    }
    turn_count = 0

    while True:
        console.print()
        query = console.input("[bold green]Query > [/]").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye! ⚽[/]")
            break

        turn_count += 1
        # LangGraph 所用，每个agent工作完都只是填充这个state
        # Only set query — other fields reset per turn, chat_history accumulates via MemorySaver
        initial_state = {
            "query": query,
            "intent": "",
            "parameters": {},
            "scout_data": "",
            "analysis": "",
            "tactical_context": "",
            "report": "",
            "shared_knowledge": "",
            "analyst_knowledge": "",
            "tactics_knowledge": "",
            "reporter_knowledge": "",
        }

        try:
            start = time.time()
            result = None

            console.print(f"  [dim]Turn {turn_count}[/]")

            for event in app.stream(initial_state, config=config):
                for node_name, node_output in event.items():
                    label = NODE_LABELS.get(node_name, node_name)
                    elapsed = time.time() - start
                    console.print(f"  [dim]{elapsed:.1f}s[/] [bold cyan]{label}[/] [green]✓[/]")

                    if node_name == "reporter":
                        result = node_output

            total = time.time() - start
            console.print(f"  [dim]Total: {total:.1f}s[/]")
            console.print()

            if result and result.get("report"):
                console.print()
                console.print(Markdown(result["report"]))
            else:
                console.print("[yellow]No report generated.[/]")

        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    run()
