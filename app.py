"""FootballGPT — Gradio Web Interface."""

import uuid
import time

import gradio as gr

from graph.workflow import build_workflow

# Build workflow once at startup
_app = build_workflow()

# Node progress labels
NODE_LABELS = {
    "router": "🧭 Router — Classifying intent...",
    "scout": "🔍 Scout — Searching players...",
    "analyst": "📊 Analyst — Analyzing data...",
    "tactics": "📋 Tactics — Evaluating fit...",
    "reporter": "📝 Reporter — Generating report...",
    "join_recommend": "⏳ Merging parallel results...",
}


def chat(message: str, history: list, session_state: dict):
    """Process a user message and yield streaming progress + final report."""
    if not session_state.get("thread_id"):
        session_state["thread_id"] = str(uuid.uuid4())
        session_state["turn"] = 0

    session_state["turn"] += 1
    thread_id = session_state["thread_id"]
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": "FootballGPT-Web",           # LangSmith trace name
        "metadata": {"interface": "gradio", "session_id": thread_id},
    }

    initial_state = {
        "query": message,
        "intent": "",
        "parameters": {},
        "scout_data": "",
        "analysis": "",
        "tactical_context": "",
        "report": "",
    }

    # Stream node progress
    progress_lines = []
    result = None
    start = time.time()

    try:
        for event in _app.stream(initial_state, config=config):
            for node_name, node_output in event.items():
                elapsed = time.time() - start
                label = NODE_LABELS.get(node_name, node_name)
                progress_lines.append(f"`{elapsed:.1f}s` {label} ✓")

                if node_name == "reporter":
                    result = node_output

                # Show progress as it happens
                yield "\n\n".join(progress_lines)

        total = time.time() - start
        progress_lines.append(f"\n---\n*Total: {total:.1f}s*\n")

        if result and result.get("report"):
            final = "\n\n".join(progress_lines) + "\n\n" + result["report"]
        else:
            final = "\n\n".join(progress_lines) + "\n\n*No report generated.*"

        yield final

    except Exception as e:
        yield f"**Error:** {e}"


# Build Gradio UI
with gr.Blocks(
    title="FootballGPT",
    theme=gr.themes.Soft(),
    css="""
    .contain { max-width: 900px; margin: auto; }
    footer { display: none !important; }
    """,
) as demo:
    gr.Markdown(
        """
        # ⚽ FootballGPT
        **AI Football Intelligence Platform** — Multi-Agent System with RAG Knowledge Base

        Example queries:
        - Find a right winger for Liverpool, under 25, fast
        - Compare Salah and Mbappé
        - Analyze Barcelona's tactical system
        - Recommend a striker for Arsenal
        """
    )

    session_state = gr.State(value={})

    chatbot = gr.Chatbot(
        height=400,
        show_label=False,
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask about players, tactics, transfers...",
            show_label=False,
            scale=9,
            container=False,
        )
        send_btn = gr.Button("Send", scale=1, variant="primary")

    # Example queries as quick buttons
    gr.Examples(
        examples=[
            "Find me a left-footed winger who plays like Salah",
            "Compare Haaland and Mbappé",
            "Recommend a young striker for Liverpool",
            "Analyze Arsenal's playing philosophy",
        ],
        inputs=msg,
    )

    def user_submit(message, history, state):
        """Add user message to history and clear input."""
        history = history + [{"role": "user", "content": message}]
        return "", history, state

    def bot_respond(history, state):
        """Stream bot response."""
        user_message = history[-1]["content"]
        history.append({"role": "assistant", "content": ""})

        for partial in chat(user_message, history, state):
            history[-1]["content"] = partial
            yield history, state

    # Wire up events
    msg.submit(
        user_submit, [msg, chatbot, session_state], [msg, chatbot, session_state]
    ).then(
        bot_respond, [chatbot, session_state], [chatbot, session_state]
    )

    send_btn.click(
        user_submit, [msg, chatbot, session_state], [msg, chatbot, session_state]
    ).then(
        bot_respond, [chatbot, session_state], [chatbot, session_state]
    )


if __name__ == "__main__":
    demo.launch()
