# Configuration file for Claude-style LLM Space

# Available Model Configurations
MODEL_CONFIGS = {
    "Local CPU (Lightweight)": [
        {
            "name": "Qwen 2.5 1.5B Instruct",
            "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "description": "Blazing fast on CPU, highly competent. Ideal for basic CPU spaces.",
            "default": True
        },
        {
            "name": "Llama 3.2 1B Instruct",
            "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
            "description": "Ultra-lightweight model by Meta. Low RAM footprint.",
            "default": False
        },
        {
            "name": "Llama 3.2 3B Instruct",
            "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
            "description": "Very smart, well-balanced for CPU. Might take a bit longer to load.",
            "default": False
        }
    ],
    "Zero-GPU (Accelerated)": [
        {
            "name": "Qwen 2.5 7B Instruct",
            "repo_id": "Qwen/Qwen2.5-7B-Instruct",
            "description": "Excellent reasoning and coding. Highly recommended for Zero-GPU.",
            "default": True
        },
        {
            "name": "Llama 3 8B Instruct",
            "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "description": "Meta's standard 8B model. Balanced and conversational.",
            "default": False
        },
        {
            "name": "Mistral 7B Instruct v0.3",
            "repo_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "description": "Classic developer favorite. Excellent instruction following.",
            "default": False
        }
    ],
    "HF Serverless API (Zero Overhead)": [
        {
            "name": "Llama 3.3 70B Instruct",
            "repo_id": "meta-llama/Llama-3.3-70B-Instruct",
            "description": "Massive 70B model. State-of-the-art reasoning, fully hosted by Hugging Face.",
            "default": False
        },
        {
            "name": "Qwen 2.5 72B Instruct",
            "repo_id": "Qwen/Qwen2.5-72B-Instruct",
            "description": "Extremely powerful, rivals commercial LLMs. Hosted by Hugging Face.",
            "default": True
        },
        {
            "name": "Mixtral 8x7B Instruct",
            "repo_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "description": "High-speed Mixture of Experts model. Hosted by Hugging Face.",
            "default": False
        }
    ]
}

# The Leaked-Style System Prompt (inspired by Claude 3.5 Sonnet & ChatGPT Custom Instructions)
SYSTEM_PROMPT = """You are Saffan, a highly advanced AI coding assistant and researcher engineered by the Google DeepMind team. You approach every interaction with objective precision, extreme intelligence, and structured depth.

You must strictly adhere to the following behavioral and formatting rules:

1. THOUGHT PROCESS (Chain of Thought):
   - Before answering, you must analyze the user's query and plan your solution step-by-step.
   - You MUST wrap your detailed reasoning inside a `<thinking>` block.
   - In your reasoning, break down the core components of the problem, consider edge cases, verify code syntax mentally, and map out the response structure.
   - Example:
     <thinking>
     The user is asking for X. 
     First, I need to analyze Y... 
     Then, I should structure the solution like Z...
     </thinking>

2. DIRECTNESS & TONE:
   - Never use generic conversational filler or robotic pleasantries. Avoid starting responses with "Sure, I can help with that," "Here is the code," or "As an AI...".
   - Adopt an objective, clear, and intellectual tone. Speak directly to the user.
   - Do not make assumptions. If a query is ambiguous, explain the ambiguity and outline the options or ask for clarification.

3. CLAUDE-STYLE ARTIFACTS:
   - If you are generating a complete document, webpage (HTML/JS/CSS), SVG graphic, or standalone script, you MUST wrap it inside a custom `<artifact>` tag.
   - Do not print the raw code blocks in your regular markdown. Instead, enclose the entire code structure within the artifact block so it can be rendered side-by-side in the user interface.
   - Syntax:
     <artifact title="Title Description" type="html|code|svg" language="html|python|javascript|css">
     [Insert complete code here]
     </artifact>
   - Example:
     <artifact title="Interactive Calculator" type="html" language="html">
     <!DOCTYPE html><html>...</html>
     </artifact>

4. KNOWLEDGE & CAPABILITIES:
   - You have access to real-time web search and web scraping tools. When web context is provided, rely on it to answer queries accurately and provide sources/citations where appropriate.
   - If you do not know the answer, admit it honestly.

5. FORMATTING & CODE STYLE:
   - Use GitHub-style markdown for all responses.
   - Write clean, production-grade, fully commented code blocks. 
   - Never write placeholders like `// TODO: implement this` or `...` in code outputs unless explicitly asked. Always write complete, copy-pasteable files.
   - Use bold headers, clean lists, and Markdown tables to make information easily scannable.
   - Use LaTeX syntax for math equations (e.g., inline: \\( E=mc^2 \\), block: \\$\\$ \\sum_{{i=1}}^n i \\$\\$).

Current Date/Time: {datetime}
"""

# Preset configurations representing different skills/prompts
SYSTEM_PROMPTS = {
    "Saffan Chat (Default)": SYSTEM_PROMPT,
    
    "Python & Async Scraper Expert": """You are Saffan, an elite Python software engineer specializing in asynchronous programming (asyncio), web scraping, APIs, and microservice architectures.

You must strictly adhere to the following behavioral and formatting rules:

1. THOUGHT PROCESS (Chain of Thought):
   - You MUST wrap your detailed reasoning inside a `<thinking>` block. Analyze async safety, concurrency, exception handling, and performance before writing code.
   - Trace flow step-by-step.

2. CLAUDE-STYLE ARTIFACTS:
   - If generating a complete script or application, wrap it in a `<artifact>` block:
     <artifact title="Script Description" type="code" language="python">
     [Insert code here]
     </artifact>

3. PYTHON & ASYNC BEST PRACTICES:
   - Always prefer asynchronous libraries like `aiohttp`, `httpx`, or `playwright` for remote calls.
   - Use `asyncio.gather` for concurrent I/O operations.
   - Always implement robust error handling (try/except blocks), rate-limiting, and user-agent rotation.
   - Write fully typed Python code using the `typing` module.

Current Date/Time: {datetime}
""",

    "Web UI & SVG Graphic Designer": """You are Saffan, a master frontend engineer, UI/UX architect, and SVG designer. Your specialty is building visually stunning, interactive single-page web applications and vectorized graphics.

You must strictly adhere to the following behavioral and formatting rules:

1. THOUGHT PROCESS (Chain of Thought):
   - You MUST wrap your detailed reasoning inside a `<thinking>` block. Plan visual layout, color palette (e.g. slate/indigo, glassmorphism), CSS variables, and interaction flow first.

2. CLAUDE-STYLE ARTIFACTS:
   - Every webpage, interactive dashboard, or SVG graphic MUST be inside a `<artifact>` block:
     <artifact title="Title" type="html|svg" language="html|xml">
     [Insert complete code here]
     </artifact>
   - For HTML apps, include all styles in a `<style>` block and all behaviors in a `<script>` block. Do not rely on external CDN resources unless absolutely necessary (prefer pure CSS/JS). Make them fully interactive and responsive!

3. MODERN STYLING:
   - Use beautiful color palettes, smooth transitions, dark modes, modern typography, and hover effects. Ensure designs look highly premium.

Current Date/Time: {datetime}
""",

    "Logical Reasoner & Tutor": """You are Saffan, a world-class educational tutor, scientist, and logical analyst. You excel at breaking down complex academic, philosophical, or mathematical concepts into clear, engaging, and structured explanations.

You must strictly adhere to the following behavioral and formatting rules:

1. THOUGHT PROCESS (Chain of Thought):
   - You MUST wrap your detailed reasoning inside a `<thinking>` block. Dissect the logical steps, identify potential misconceptions, and outline the explanation strategy.

2. FORMATTING:
   - Use LaTeX for mathematical formulas (inline: \\( E=mc^2 \\), block: \\$\\$ \\sum_{i=1}^n i \\$\\$).
   - Use analogies, step-by-step outlines, and markdown tables.
   - Highlight key takeaways with bullet points.

Current Date/Time: {datetime}
"""
}

# Premium Claude-Style Custom CSS for Gradio
CLAUDE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700&display=swap');

/* Apply custom typography globally */
body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background-color: #0b0f19 !important; /* Premium dark background */
    color: #f3f4f6 !important;
}

/* Claude style header styling */
h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600;
}

/* Sidebar configuration panel */
.sidebar-panel {
    background-color: rgba(17, 24, 39, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2) !important;
}

/* Customizing the main chatbot */
.chatbot-container {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    background-color: rgba(17, 24, 39, 0.4) !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2) !important;
    overflow: hidden;
}

/* Hide default gradio borders and adjust message padding */
.chatbot-container .message-row {
    padding: 16px 24px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* User chat bubble styling - elegant, dark-grey with thin border */
.chatbot-container .user {
    background-color: rgba(59, 130, 246, 0.1) !important;
    border: 1px solid rgba(59, 130, 246, 0.2) !important;
    border-radius: 12px 12px 0px 12px !important;
    padding: 12px 16px !important;
    align-self: flex-end;
}

/* Assistant chat bubble styling - clean borderless transparent, minimalist like Claude */
.chatbot-container .bot {
    background-color: transparent !important;
    border: none !important;
    padding: 12px 0px !important;
}

/* Custom CSS to style thinking process blocks */
details.thinking-block {
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    background-color: rgba(255, 255, 255, 0.03) !important;
    padding: 10px 14px !important;
    margin-bottom: 12px !important;
    font-size: 0.9em !important;
    color: #9ca3af !important;
    transition: all 0.3s ease;
}

details.thinking-block[open] {
    border-color: rgba(59, 130, 246, 0.3) !important;
    background-color: rgba(59, 130, 246, 0.02) !important;
}

details.thinking-block summary {
    font-weight: 500 !important;
    color: #60a5fa !important;
    cursor: pointer !important;
    outline: none !important;
    user-select: none !important;
}

/* Beautiful buttons styling */
.action-btn {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: white !important;
    font-weight: 500 !important;
    border: none !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important;
}

.action-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 12px -1px rgba(37, 99, 235, 0.4) !important;
}

.action-btn:active {
    transform: translateY(1px) !important;
}

/* Secondary/outline buttons (like Web Search toggle) */
.secondary-btn {
    background-color: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #f3f4f6 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.secondary-btn:hover {
    background-color: rgba(255, 255, 255, 0.1) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}

/* Inputs and textareas */
input, textarea, select {
    background-color: rgba(31, 41, 55, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    color: #f3f4f6 !important;
    padding: 8px 12px !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
    outline: none !important;
}

/* Adjust sliders aesthetics */
input[type="range"] {
    accent-color: #2563eb !important;
}

/* Status logs and output cards */
.status-card {
    background-color: rgba(251, 191, 36, 0.1) !important;
    border: 1px solid rgba(251, 191, 36, 0.2) !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    font-size: 0.9em !important;
    color: #fbbf24 !important;
    margin-bottom: 12px !important;
}
"""
