"""FootballGPT — AI Football Analysis Platform with Multi-Agent Collaboration."""

import time

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
    "scout": "🔍 Scout — Searching players...",
    "analyst": "📊 Analyst — Analyzing data...",
    "tactics": "📋 Tactics — Evaluating fit...",
    "reporter": "📝 Reporter — Generating report...",
    "join_recommend": "⏳ Merging parallel results...",
}


def run():
    display_banner()
    app = build_workflow()

    while True:
        console.print()
        query = console.input("[bold green]Query > [/]").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye! ⚽[/]")
            break

        initial_state = {
            "query": query,
            "intent": "",
            "parameters": {},
            "scout_data": "",
            "analysis": "",
            "tactical_context": "",
            "report": "",
        }

        try:
            start = time.time()
            result = None

            for event in app.stream(initial_state):
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
                console.print(Panel(Markdown(result["report"]), title="📝 Analysis Report", border_style="blue"))
            else:
                console.print("[yellow]No report generated.[/]")

        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    run()
