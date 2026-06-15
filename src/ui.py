import gradio as gr
import torch
from src.config import MODEL_CONFIGS, SYSTEM_PROMPT, SYSTEM_PROMPTS, CLAUDE_CSS
from src.engine import execute_chat, HAS_SPACES

def get_hardware_status():
    """Returns a user-friendly string indicating the current runtime hardware."""
    if HAS_SPACES:
        return "🟢 Hugging Face Zero-GPU (A100 Dynamic Allocation)"
    elif torch.cuda.is_available():
        return f"🟢 Local GPU: {torch.cuda.get_device_name(0)}"
    else:
        return "⚪ Standard CPU Mode (Free Tier)"

def update_model_dropdown(mode):
    """Updates the model choice list when the backend mode is toggled."""
    models = [m["name"] for m in MODEL_CONFIGS[mode]]
    default_model = next(m["name"] for m in MODEL_CONFIGS[mode] if m["default"])
    return gr.Dropdown(choices=models, value=default_model, label="Active Model")

def add_user_message(message, history):
    """Adds the user message to the chat container and clears the input box."""
    if not message or not message.strip():
        return "", history
    if history is None:
        history = []
    return "", history + [[message, "⏳ Initializing inference engine..."]]

def execute_chat_ui(
    history,
    mode,
    model_name,
    system_prompt_preset,
    max_new_tokens,
    temperature,
    top_p,
    enable_search,
    hf_token
):
    """
    UI Wrapper that processes the active chatbot history state,
    runs the backend generator, and streams response updates.
    """
    if history is None or len(history) == 0:
        return
        
    # Extract latest user message and the history preceding it
    user_message = history[-1][0]
    past_history = history[:-1]
    
    # Run chat execution generator
    chat_generator = execute_chat(
        message=user_message,
        history=past_history,
        mode=mode,
        model_name=model_name,
        system_prompt_preset=system_prompt_preset,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_search=enable_search,
        hf_token=hf_token
    )
    
    for updated_history, artifacts in chat_generator:
        yield updated_history, artifacts

def load_selected_artifact(selected_title, artifacts):
    """Retrieves and formats code/render outputs for a selected artifact."""
    if not selected_title or not artifacts:
        return "", "", "plaintext"
        
    for art in artifacts:
        if art["title"] == selected_title:
            content = art["content"]
            if art["type"] == "html":
                # Render HTML inside a secure data URI iframe to isolate it from Gradio styles
                import urllib.parse
                escaped_content = urllib.parse.quote(content)
                iframe_render = f'<iframe src="data:text/html;charset=utf-8,{escaped_content}" style="width: 100%; height: 500px; border: none; border-radius: 8px; background-color: white;"></iframe>'
                return content, iframe_render, art["language"]
            elif art["type"] == "svg":
                # Render SVG directly
                svg_render = f'<div style="background-color: white; padding: 20px; border-radius: 8px; text-align: center; display: flex; justify-content: center; align-items: center;">{content}</div>'
                return content, svg_render, "xml"
            else:
                # Code content (no render preview available)
                no_render_placeholder = '<div style="padding: 40px; text-align: center; color: #9ca3af;">No visual render preview available for this code type. Use the "Source Code" tab to view.</div>'
                return content, no_render_placeholder, art["language"]
                
    return "", "", "plaintext"

def update_artifacts_ui(artifacts):
    """Refreshes the state and visibility of components inside the Artifacts Panel."""
    if not artifacts:
        return (
            gr.Dropdown(choices=[], value=None, visible=False),
            gr.Markdown(visible=True),
            gr.Code(value="", visible=False),
            gr.HTML(value="", visible=False)
        )
        
    choices = [art["title"] for art in artifacts]
    default_val = choices[-1]
    
    code_content, render_html, lang = load_selected_artifact(default_val, artifacts)
    
    return (
        gr.Dropdown(choices=choices, value=default_val, visible=True),
        gr.Markdown(visible=False),
        gr.Code(value=code_content, language=lang, visible=True),
        gr.HTML(value=render_html, visible=True)
    )


def build_interface():
    """Constructs the Gradio user interface using custom styles and themes."""
    # Custom light/dark theme initialization
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Roboto Mono"), "ui-monospace", "SFMono-Regular", "monospace"]
    ).set(
        body_background_fill="#0b0f19",
        body_background_fill_dark="#0b0f19",
        block_background_fill="rgba(17, 24, 39, 0.5)",
        block_background_fill_dark="rgba(17, 24, 39, 0.5)",
        border_color_primary="rgba(255, 255, 255, 0.08)",
        border_color_primary_dark="rgba(255, 255, 255, 0.08)"
    )

    with gr.Blocks(theme=theme, css=CLAUDE_CSS, title="Saffan Chat") as demo:
        # State to store the raw message during submission sequence
        
        with gr.Row():
            with gr.Column(scale=12):
                gr.HTML(
                    """
                    <div style="text-align: center; margin-bottom: 24px; margin-top: 10px;">
                        <h1 style="font-size: 2.8em; margin-bottom: 5px; background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            SAFFAN CHAT
                        </h1>
                        <p style="font-size: 1.1em; color: #9ca3af; max-width: 600px; margin: 0 auto;">
                            A premium Claude-style chatbot environment designed for Hugging Face free tier.
                            Equipped with real-time web search, page scraping, and cognitive system reasoning.
                        </p>
                    </div>
                    """
                )

        with gr.Row():
            # Side Control Panel (Sidebar)
            with gr.Column(scale=3, elem_classes=["sidebar-panel"]):
                gr.Markdown("### ⚙️ System Settings")
                
                hardware_text = get_hardware_status()
                gr.HTML(
                    f"""
                    <div style="font-size: 0.85em; padding: 8px 12px; background-color: rgba(255,255,255,0.03); border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 15px;">
                        <span style="color: #9ca3af;">Host Hardware:</span><br/>
                        <strong style="color: #38bdf8;">{hardware_text}</strong>
                    </div>
                    """
                )
                
                # Mode selection
                mode_dropdown = gr.Dropdown(
                    choices=list(MODEL_CONFIGS.keys()),
                    value="Local CPU (Lightweight)",
                    label="Inference Backend Mode",
                    interactive=True
                )
                
                # Model selection (changes dynamically based on mode)
                model_choices = [m["name"] for m in MODEL_CONFIGS["Local CPU (Lightweight)"]]
                default_model = next(m["name"] for m in MODEL_CONFIGS["Local CPU (Lightweight)"] if m["default"])
                
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=default_model,
                    label="Active Model",
                    interactive=True
                )
                
                # Web Search Toggle
                enable_search = gr.Checkbox(
                    label="🔍 Enable Web Search (DuckDuckGo)",
                    value=False,
                    interactive=True
                )
                
                # Token field (hidden input for HF Serverless inference token)
                hf_token = gr.Textbox(
                    label="Hugging Face API Token (optional)",
                    placeholder="hf_...",
                    type="password",
                    info="Required for gated Serverless models (e.g. Llama 3.3). Get one at hf.co/settings/tokens"
                )
                
                # Advanced Settings Accordion
                with gr.Accordion("🛠️ Advanced Parameters", open=False):
                    system_preset_dropdown = gr.Dropdown(
                        choices=list(SYSTEM_PROMPTS.keys()),
                        value="Saffan Chat (Default)",
                        label="Select AI Persona / Skill Mode",
                        interactive=True
                    )
                    
                    system_prompt = gr.Textbox(
                        label="System Instruction Prompt",
                        value=SYSTEM_PROMPT,
                        lines=8,
                        max_lines=15
                    )
                    
                    max_tokens = gr.Slider(
                        minimum=64,
                        maximum=4096,
                        value=1024,
                        step=64,
                        label="Max New Tokens"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.2,
                        value=0.7,
                        step=0.1,
                        label="Temperature (0.0 = deterministic)"
                    )
                    
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-P Sampling"
                    )
                
                # System actions
                clear_btn = gr.Button("🗑️ Clear Chat History", variant="secondary", elem_classes=["secondary-btn"])
            
            # Main Chat & Artifacts Area
            with gr.Column(scale=9):
                # Conversation-scoped state to hold parsed artifacts
                artifacts_state = gr.State(value=[])
                
                with gr.Row():
                    # Left Column: Chat Viewport
                    with gr.Column(scale=6):
                        chatbot = gr.Chatbot(
                            label="Chat Window",
                            elem_classes=["chatbot-container"],
                            show_label=False,
                            avatar_images=(None, "https://huggingface.co/front/assets/huggingface_logo-noborder.svg"),
                            height=580,
                            bubble_full_width=False,
                            type="tuples"
                        )
                        
                        with gr.Row():
                            input_box = gr.Textbox(
                                placeholder="Ask Saffan anything... (e.g., 'Draft a clean Python function using asyncio to scrape web data.')",
                                show_label=False,
                                scale=10
                            )
                            submit_btn = gr.Button("Send", variant="primary", scale=1, elem_classes=["action-btn"])
                            
                        # Prompts suggestions
                        gr.Markdown("💡 **Quick Prompts**")
                        with gr.Row():
                            suggestion_1 = gr.Button("Draft a clean Python function using asyncio to scrape web data.", variant="secondary", elem_classes=["secondary-btn"])
                            suggestion_2 = gr.Button("Search the web for the latest advancements in LLM reasoning models.", variant="secondary", elem_classes=["secondary-btn"])
                            suggestion_3 = gr.Button("Explain quantum computing superposition using a simple real-life analogy.", variant="secondary", elem_classes=["secondary-btn"])
                            
                    # Right Column: Claude-Style Artifacts Panel
                    with gr.Column(scale=5, elem_classes=["sidebar-panel"]):
                        gr.HTML(
                            """
                            <div style="border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 10px; margin-bottom: 15px;">
                                <h3 style="margin: 0; font-size: 1.25em; color: #60a5fa; display: flex; align-items: center; gap: 8px;">
                                    🎨 Claude-Style Artifacts
                                </h3>
                                <p style="margin: 3px 0 0 0; font-size: 0.8em; color: #9ca3af;">
                                    Interactive HTML/SVG rendering and source code viewer.
                                </p>
                            </div>
                            """
                        )
                        
                        # Artifact Selector Dropdown
                        artifact_selector = gr.Dropdown(
                            label="Select Artifact",
                            choices=[],
                            value=None,
                            visible=False,
                            interactive=True
                        )
                        
                        # No Artifacts Placeholder Description
                        artifact_placeholder = gr.Markdown(
                            "**No active artifacts.**\n\nWhen Saffan generates complete webpages, SVG graphics, or scripts, they will appear here side-by-side automatically.",
                            visible=True
                        )
                        
                        # Tabs for Preview Render and Code Source
                        with gr.Tabs() as artifact_tabs:
                            with gr.Tab("Preview"):
                                artifact_render = gr.HTML(
                                    value="",
                                    visible=False
                                )
                            with gr.Tab("Source Code"):
                                artifact_code = gr.Code(
                                    value="",
                                    language="plaintext",
                                    interactive=False,
                                    wrap_lines=True,
                                    visible=False
                                )

        # Define UI event linkages
        
        # 1. Mode dropdown change updates the Model selection dropdown options
        mode_dropdown.change(
            fn=update_model_dropdown,
            inputs=[mode_dropdown],
            outputs=[model_dropdown]
        )
        
        # Preset dropdown change updates the system prompt textbox content
        system_preset_dropdown.change(
            fn=lambda preset: SYSTEM_PROMPTS.get(preset, SYSTEM_PROMPT),
            inputs=[system_preset_dropdown],
            outputs=[system_prompt]
        )
        
        # 2. Main submit event chain (for Enter key submit)
        submit_event = input_box.submit(
            fn=add_user_message,
            inputs=[input_box, chatbot],
            outputs=[input_box, chatbot],
            queue=False
        ).then(
            fn=execute_chat_ui,
            inputs=[
                chatbot,
                mode_dropdown,
                model_dropdown,
                system_prompt,
                max_tokens,
                temperature,
                top_p,
                enable_search,
                hf_token
            ],
            outputs=[chatbot, artifacts_state]
        )
        
        # 3. Submit button click event chain
        click_event = submit_btn.click(
            fn=add_user_message,
            inputs=[input_box, chatbot],
            outputs=[input_box, chatbot],
            queue=False
        ).then(
            fn=execute_chat_ui,
            inputs=[
                chatbot,
                mode_dropdown,
                model_dropdown,
                system_prompt,
                max_tokens,
                temperature,
                top_p,
                enable_search,
                hf_token
            ],
            outputs=[chatbot, artifacts_state]
        )
        
        # 4. State change triggers UI update for the Artifacts panel
        artifacts_state.change(
            fn=update_artifacts_ui,
            inputs=[artifacts_state],
            outputs=[artifact_selector, artifact_placeholder, artifact_code, artifact_render]
        )
        
        # 5. Dropdown change updates the content panel
        def handle_selector_change(selected_title, artifacts):
            if not selected_title or not artifacts:
                return gr.update(value="", visible=False), gr.update(value="", visible=False)
            code_content, render_html, lang = load_selected_artifact(selected_title, artifacts)
            return (
                gr.Code(value=code_content, language=lang, visible=True),
                gr.HTML(value=render_html, visible=True)
            )

        artifact_selector.change(
            fn=handle_selector_change,
            inputs=[artifact_selector, artifacts_state],
            outputs=[artifact_code, artifact_render]
        )
        
        # 6. Clear chat history button event (also clears artifacts state)
        clear_btn.click(fn=lambda: ([], []), outputs=[chatbot, artifacts_state], queue=False)
        
        # 7. Suggestion prompt buttons click events
        def load_suggestion(text):
            search_enabled = "Search the web" in text or "latest advancements" in text
            return text, search_enabled

        suggestion_1.click(
            fn=lambda: load_suggestion("Draft a clean Python function using asyncio to scrape web data."),
            outputs=[input_box, enable_search]
        )
        suggestion_2.click(
            fn=lambda: load_suggestion("Search the web for the latest advancements in LLM reasoning models."),
            outputs=[input_box, enable_search]
        )
        suggestion_3.click(
            fn=lambda: load_suggestion("Explain quantum computing superposition using a simple real-life analogy."),
            outputs=[input_box, enable_search]
        )

    return demo
