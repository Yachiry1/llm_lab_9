"""Gradio frontend for Survival AI."""

import os

import gradio as gr

try:
    from agent import chat as agent_chat
except ImportError:
    agent_chat = None


def _temporary_chat(message: str, history: list[dict], api_key: str) -> tuple[str, list[str]]:
    """Temporary fallback; remove after agent.chat is implemented."""
    if not api_key.strip():
        return "Please enter your Groq API key before sending a message.", []
    return (
        "The conversational backend is not available yet. "
        "This temporary response confirms that the Gradio interface is working.",
        [],
    )


def _format_tools(tool_names: list[str]) -> str:
    if not tool_names:
        return "**Tools used for the latest response:** None"
    return "**Tools used for the latest response:** " + ", ".join(tool_names)


def respond(message: str, history: list[dict], api_key: str):
    """Call the backend and update only this session's conversation state."""
    message = message.strip() if isinstance(message, str) else ""
    if not message:
        return "", history, history, _format_tools([])

    backend = agent_chat or _temporary_chat
    try:
        response, tools_used = backend(message, list(history), api_key or "")
        if not isinstance(response, str) or not isinstance(tools_used, list):
            raise ValueError("Unexpected response from agent.chat")
        tools_used = [str(tool_name) for tool_name in tools_used]
    except Exception:
        response = "Sorry, the assistant could not process that message. Please try again."
        tools_used = []

    updated_history = [
        *history,
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return "", updated_history, updated_history, _format_tools(tools_used)


def clear_chat():
    """Clear conversation data for the current Gradio session."""
    return [], [], "", _format_tools([])


with gr.Blocks(title="Survival AI") as demo:
    gr.Markdown(
        "# Survival AI\n"
        "A conversational assistant for practical survival guidance, planning, "
        "risk awareness, and first-aid information."
    )

    api_key = gr.Textbox(
        label="Groq API key",
        type="password",
        placeholder="Enter your API key for this session",
    )
    chatbot = gr.Chatbot(label="Conversation", height=420)
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
    tools_display = gr.Markdown(_format_tools([]))

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
    demo.queue().launch(server_name="0.0.0.0", server_port=_server_port())
