---
title: Claude-Style Modular LLM Space
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
python_version: "3.10"
app_file: app.py

pinned: false
license: mit
short_description: Claude-style chatbot with search and memory.


---

# Claude-Style Modular LLM Chatbot

A production-grade, premium LLM chatbot interface optimized for Hugging Face Spaces (free tier). 

It features a minimalist **Claude-style interface** (built with custom CSS and Gradio), an advanced reasoning system prompt, **real-time web search (via DuckDuckGo)**, **automatic web page scraping**, and **conversational memory**.

## 🚀 Key Features

* **Claude-Style Minimalist UI**: Sleek, elegant light/dark mode inspired by Anthropic's Claude, complete with high-quality typography, clean card structures, and responsive layouts.
* **Tri-Mode Inference Engine**:
  1. **Local CPU (Quantized PyTorch)**: Run lightweight models (e.g., `Qwen/Qwen2.5-1.5B-Instruct`) locally on the free CPU tier (fits within 16GB RAM).
  2. **Zero-GPU (Free GPU Sharing)**: Dynamic GPU acceleration utilizing Hugging Face's shared Zero-GPU pool (for 7B/8B models like `Qwen/Qwen2.5-7B-Instruct`).
  3. **HF Serverless Inference API**: Instant, zero-overhead connectivity to massive models (like `Qwen/Qwen2.5-72B-Instruct` or `Meta-Llama-3.3-70B-Instruct`) using your Hugging Face API token.
* **Real-time Web Search & Scraper**: Toggleable web-search capability that searches DuckDuckGo, scrapes page contents, and injects context directly into the prompt.
* **Conversational Memory**: Maintains session chat history dynamically.
* **State-of-the-Art System Prompt**: Tailored prompts that guide the model to provide detailed, step-by-step thinking, well-formatted markdown, and objective answers.

## 🛠️ Hugging Face Space Deployment

Click the button below to start your deployment:

[![Deploy to Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-to-spaces-lg.svg)](https://huggingface.co/new-space?sdk=gradio)

### How to Deploy in 3 Simple Steps:
1.  **Create the Space**: Click the button above, type **`llm`** as the Space Name, and click **Create Space** at the bottom of the Hugging Face page.
2.  **Add your HF_TOKEN Secret**: Go to your GitHub repository -> **Settings** -> **Secrets and variables** -> **Actions** -> Add a secret named `HF_TOKEN` containing your Hugging Face write-access token (from [hf.co/settings/tokens](https://huggingface.co/settings/tokens)).
3.  **Run GitHub Sync Action**: Go to the **Actions** tab in your GitHub repository, select **Sync to Hugging Face Spaces**, and click **Run workflow**.

---

### Method 1: Manual Git Push
If you prefer to deploy manually via terminal:
1. Create a new blank Space on [Hugging Face](https://huggingface.co/new-space) (SDK: Gradio).
2. Push these files directly to your Space's repository:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/YOUR_SPACE_NAME
   git push -u hf main --force
   ```

### Method 2: GitHub Action Auto-Sync (Fully Automated)
Your repository is pre-configured with a GitHub Action that automatically syncs code to Hugging Face on every commit:

1. Create your Space on Hugging Face (SDK: Gradio).
2. Go to your GitHub repository -> **Settings** -> **Secrets and variables** -> **Actions**.
3. Create a **New repository secret**:
   - **Name**: `HF_TOKEN`
   - **Value**: Your Hugging Face Access Token (Write access).
4. Create a file in your repo at `.github/workflows/hf_sync.yml` using the template below. Every time you run `git push origin main`, GitHub will build and update your Hugging Face Space automatically!

### 🔒 Access Control (Private Deployment)
To restrict access so **only you** can use the chatbot:
1. In your Hugging Face Space, navigate to **Settings** -> **Variables and secrets**.
2. Create a new **Secret** with:
   - **Name**: `APP_PASSWORD`
   - **Value**: Your chosen private passcode (e.g., `my_private_passcode123`).
3. Once saved, Hugging Face will prompt a secure login page when accessing the Space. You will log in using `admin` as the username and your passcode as the password.

### 💻 Manual Git Deployment
To push manually:
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Choose **Gradio** as the SDK.

3. Push these files to your Space's repository:
   ```bash
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git add .
   git commit -m "Deploying Claude-style chatbot"
   git push -u origin main --force
   ```


## ⚙️ Project Structure

```
├── app.py                      # Space entrypoint
├── requirements.txt            # Python dependencies
├── README.md                   # Space configuration & docs
└── src/
    ├── __init__.py
    ├── config.py               # Theme styling, system prompts, & model settings
    ├── tools.py                # Web Search & Web Scraping engine
    ├── engine.py               # Modular multi-backend inference runner
    └── ui.py                   # Custom Gradio interface components
```
