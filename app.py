"""Gradio frontend for Survival AI."""

import os

import gradio as gr

from agent import chat as agent_chat


WELCOME_MESSAGE = {
    "role": "assistant",
    "content": (
        "Welcome to **Survival AI**. Ask a question about survival planning, "
        "risk awareness, equipment, water, or first aid."
    ),
}

ERROR_RESPONSES = {
    "Enter your Groq API key to start chatting.": (
        "⚠️ **API key required.** Enter your Groq API key above, then try again."
    ),
    "Groq rejected the API key. Please check it and try again.": (
        "⚠️ **API key rejected.** Check the key and try again."
    ),
    "Groq's request limit was reached. Please try again later.": (
        "⚠️ **API rate limit reached.** Please wait a moment and try again."
    ),
    "The Groq request failed. Check your connection and model access, then try again.": (
        "⚠️ **API request failed.** Check your connection and Groq model access, then try again."
    ),
}

APP_CSS = """
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    padding: clamp(12px, 3vw, 28px) !important;
}
#api-key-panel {
    margin: 16px 0 20px;
    padding: 14px;
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
}
#survival-chat .message {
    line-height: 1.6;
    padding: 12px 16px;
}
#tools-panel {
    margin-top: 12px;
    padding: 10px 14px;
    border-radius: 10px;
    background: var(--background-fill-secondary);
}
@media (max-width: 640px) {
    #api-key-panel { padding: 10px; }
    #survival-chat .message { padding: 10px 12px; }
}
"""


def _display_history(history: list[dict]) -> list[dict]:
    return [WELCOME_MESSAGE.copy(), *history]


def _format_tools(tool_names: list[str]) -> str:
    if not tool_names:
        return "**Tools used for the latest response:** None"
    names = ", ".join(f"`{name.replace('`', '')}`" for name in tool_names)
    return f"**Tools used for the latest response:** {names}"


def respond(message: str, history: list[dict], api_key: str):
    """Call the backend and update only this session's conversation state."""
    message = message.strip() if isinstance(message, str) else ""
    if not message:
        return "", _display_history(history), history, _format_tools([])

    if not isinstance(api_key, str) or not api_key.strip():
        error_chat = [
            *_display_history(history),
            {"role": "user", "content": message},
            {"role": "assistant", "content": ERROR_RESPONSES["Enter your Groq API key to start chatting."]},
        ]
        return message, error_chat, history, _format_tools([])

    try:
        response, tools_used = agent_chat(message, list(history), api_key or "")
        if not isinstance(response, str) or not isinstance(tools_used, list):
            raise ValueError("Unexpected response from agent.chat")
        tools_used = [str(tool_name) for tool_name in tools_used]
    except Exception:
        response = (
            "⚠️ **API request failed.** The assistant could not process the request. "
            "Please try again."
        )
        tools_used = []

    presented_response = ERROR_RESPONSES.get(response, response)
    if response in ERROR_RESPONSES:
        error_chat = [
            *_display_history(history),
            {"role": "user", "content": message},
            {"role": "assistant", "content": presented_response},
        ]
        return message, error_chat, history, _format_tools(tools_used)

    updated_history = [
        *history,
        {"role": "user", "content": message},
        {"role": "assistant", "content": presented_response},
    ]
    return "", _display_history(updated_history), updated_history, _format_tools(tools_used)


def clear_chat():
    """Clear conversation data for the current Gradio session."""
    return _display_history([]), [], "", _format_tools([])


with gr.Blocks(title="Survival AI", analytics_enabled=False) as demo:
    gr.Markdown(
        "# Survival AI\n"
        "A conversational assistant for practical survival guidance, planning, "
        "risk awareness, and first-aid information."
    )

    with gr.Group(elem_id="api-key-panel"):
        gr.Markdown("### Groq API key\nRequired to send messages. It is used only for this session.")
        api_key = gr.Textbox(
            label="API key",
            type="password",
            placeholder="Enter your Groq API key",
        )

    chatbot = gr.Chatbot(
        value=_display_history([]),
        label="Conversation",
        height=460,
        layout="bubble",
        elem_id="survival-chat",
    )
    session_history = gr.State([])

    with gr.Row():
        message = gr.Textbox(
            label="Message",
            placeholder="Ask a survival question...",
            lines=2,
            scale=5,
        )
        send_button = gr.Button("Send", variant="primary", scale=1)

    clear_button = gr.Button("Clear Chat")
    tools_display = gr.Markdown(_format_tools([]), elem_id="tools-panel")

    submit_outputs = [message, chatbot, session_history, tools_display]
    send_button.click(
        respond,
        inputs=[message, session_history, api_key],
        outputs=submit_outputs,
    )
    message.submit(
        respond,
        inputs=[message, session_history, api_key],
        outputs=submit_outputs,
    )
    clear_button.click(
        clear_chat,
        outputs=[chatbot, session_history, message, tools_display],
    )


def _server_port() -> int:
    try:
        return int(os.environ.get("PORT", "7860"))
    except ValueError:
        return 7860


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=_server_port(),
        css=APP_CSS,
    )
