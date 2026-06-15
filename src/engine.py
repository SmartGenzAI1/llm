import os
import gc
import time
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from huggingface_hub import InferenceClient
from src.config import SYSTEM_PROMPT, MODEL_CONFIGS
from src.tools import web_search, scrape_url, format_search_results_for_prompt

# Conditional Zero-GPU Spaces import
try:
    import spaces
    HAS_SPACES = True
    gpu_decorator = spaces.GPU
except ImportError:
    HAS_SPACES = False
    # Dummy decorator if not on HF Zero-GPU
    def gpu_decorator(f):
        return f

# Global Model Cache variables
_current_model = None
_current_tokenizer = None
_current_repo_id = None

def unload_model():
    """Unloads the currently cached model and tokenizer to free RAM/GPU memory."""
    global _current_model, _current_tokenizer, _current_repo_id
    if _current_model is not None:
        print(f"Unloading model: {_current_repo_id} to free memory...")
        del _current_model
        del _current_tokenizer
        _current_model = None
        _current_tokenizer = None
        _current_repo_id = None
        # Force garbage collection and CUDA cache clearing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1)

def get_local_model(repo_id: str, hf_token: str = None):
    """
    Retrieves the local tokenizer and model, loading them from Hugging Face
    cache if not already loaded in the memory cache.
    """
    global _current_model, _current_tokenizer, _current_repo_id
    
    if _current_repo_id == repo_id and _current_model is not None:
        return _current_model, _current_tokenizer

    # Unload previous model to avoid out-of-memory errors
    unload_model()

    print(f"Loading model: {repo_id}...")
    token = hf_token or os.environ.get("HF_TOKEN")
    
    # Determine the device mapping (GPU if available, else CPU)
    if torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.float16
    else:
        device_map = "cpu"
        # On CPU, float32 is most stable, bfloat16 can be used if CPU supports it
        torch_dtype = torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            token=token
        )
    except Exception as e:
        error_msg = str(e)
        if "gated repo" in error_msg.lower() or "401" in error_msg or "unauthorized" in error_msg.lower() or "gatedrepoerror" in error_msg.lower():
            raise ValueError(
                f"❌ Hugging Face Access Error: The model you selected (`{repo_id}`) is a Gated Model.\n\n"
                f"To access this model, please:\n"
                f"1. Accept the licensing agreement on the model page: [huggingface.co/{repo_id}](https://huggingface.co/{repo_id})\n"
                f"2. Provide your Hugging Face API Read Token in the *Advanced Parameters* input in the sidebar, or set it as a Space Secret named `HF_TOKEN` in your Hugging Face Space settings."
            )
        else:
            raise ValueError(f"❌ Failed to load model `{repo_id}`: {e}")

    _current_model = model
    _current_tokenizer = tokenizer
    _current_repo_id = repo_id
    
    print(f"Successfully loaded {repo_id} into memory.")
    return model, tokenizer

def sample_next_token(logits, temperature: float, top_p: float):
    """Samples the next token from logits using temperature and top-p (nucleus) filtering."""
    # Apply temperature
    if temperature > 0.0 and temperature != 1.0:
        logits = logits / temperature
        
    # Apply top-p (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least the first token (prevent empty sets)
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits[sorted_indices_to_remove] = -float("Inf")
        logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(-1))
        
    # Sample or Argmax
    if temperature > 0.0:
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
    return next_token

def generate_local_inference_cpu(prompt_text: str, repo_id: str, max_new_tokens: int, temperature: float, top_p: float, hf_token: str = None):
    """
    Executes local text generation using a pure-python KV-cache loop.
    This avoids background threads entirely, making it 100% compatible
    with both standard CPU spaces and Hugging Face Zero-GPU environments.
    """
    model, tokenizer = get_local_model(repo_id, hf_token)
    device = next(model.parameters()).device
    
    # Tokenize prompt
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    
    # Check stopping criteria (all special tokens like <|endoftext|>, <|im_end|>, etc.)
    stop_tokens = tokenizer.all_special_ids
    
    past_key_values = None
    generated_ids = []
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        # First step (process the whole prompt)
        outputs = model(input_ids, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        
        next_token = sample_next_token(next_token_logits, temperature, top_p)
        next_token_id = next_token.item()
        
        if next_token_id in stop_tokens:
            return
            
        generated_ids.append(next_token_id)
        yield tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Loop steps (generate one token at a time passing KV cache)
        for _ in range(max_new_tokens - 1):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            next_token = sample_next_token(next_token_logits, temperature, top_p)
            next_token_id = next_token.item()
            
            if next_token_id in stop_tokens:
                break
                
            generated_ids.append(next_token_id)
            yield tokenizer.decode(generated_ids, skip_special_tokens=True)

# GPU-accelerated version wrapped with Hugging Face Zero-GPU decorator
generate_local_inference_gpu = gpu_decorator(generate_local_inference_cpu)


def run_serverless_api_inference(messages: list, repo_id: str, max_new_tokens: int, temperature: float, top_p: float, hf_token: str = None):
    """
    Runs text generation via HF Serverless Inference API client.
    Streams tokens in real time.
    """
    # Retrieve token from environment variables if not provided explicitly
    token = hf_token or os.environ.get("HF_TOKEN")
    
    # Initialize Client
    client = InferenceClient(model=repo_id, token=token)
    
    generated_text = ""
    try:
        response_stream = client.chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True
        )
        
        for chunk in response_stream:
            content = chunk.choices[0].delta.content
            if content:
                generated_text += content
                yield generated_text
    except Exception as e:
        error_msg = f"Serverless API Error: {str(e)}\n\n"
        if not token:
            error_msg += "💡 Tip: Many models require a valid Hugging Face Token for serverless inference. Please enter your HF Token in the sidebar panel."
        yield error_msg

def build_prompt_with_history(messages: list, system_prompt: str, tokenizer=None) -> str:
    """
    Formats the conversation history using standard chat templates.
    """
    formatted_messages = [{"role": "system", "content": system_prompt}] + messages
    
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
            
    # Fallback to general formatting if template is unavailable
    prompt_str = ""
    for msg in formatted_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt_str += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            prompt_str += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            prompt_str += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    prompt_str += "<|im_start|>assistant\n"
    return prompt_str

def format_thinking_tags(text: str) -> str:
    """
    Replaces model <thinking></thinking> tags with clean, modern HTML Details panels
    for premium rendering in the Gradio chat viewport.
    """
    if "<thinking>" in text:
        parts = text.split("<thinking>", 1)
        before_thinking = parts[0]
        rest = parts[1]
        
        if "</thinking>" in rest:
            thinking_parts = rest.split("</thinking>", 1)
            thinking_content = thinking_parts[0]
            after_thinking = thinking_parts[1]
            return f"{before_thinking}<details class='thinking-block'><summary>Thought Process</summary>\n\n{thinking_content.strip()}\n\n</details>\n\n{after_thinking}"
        else:
            # Thinking block is still generating, render it open
            return f"{before_thinking}<details open class='thinking-block'><summary>Thinking Process...</summary>\n\n{rest.strip()}\n\n</details>"
    
def extract_artifacts(text: str) -> list:
    """
    Extracts all <artifact> blocks (including in-progress ones)
    from the streaming text.
    """
    artifacts = []
    # Match tags: <artifact title="x" type="y" language="z">content</artifact> or open tags at the end of text
    pattern = r'<artifact\s+title="([^"]*)"\s+type="([^"]*)"\s+language="([^"]*)"\s*>(.*?)(?:</artifact>|$)'
    matches = re.finditer(pattern, text, re.DOTALL)
    for m in matches:
        title = m.group(1) or "Untitled"
        type_ = m.group(2) or "code"
        lang = m.group(3) or "plaintext"
        content = m.group(4)
        artifacts.append({
            "title": title,
            "type": type_,
            "language": lang,
            "content": content.strip()
        })
    return artifacts

def clean_chatbot_response(text: str) -> str:
    """
    Replaces <artifact> blocks in the response with a clean visual badge
    so the raw code doesn't clutter the main chat viewport.
    """
    pattern = r'<artifact\s+title="([^"]*)"\s+type="([^"]*)"\s+language="([^"]*)"\s*>.*?(?:</artifact>|$)'
    
    def replace_with_badge(match):
        title = match.group(1) or "Untitled Artifact"
        type_ = match.group(2) or "code"
        return f"\n\n> ⚙️ **Artifact Generated:** *{title}* ({type_}) — *Rendered in the right-hand panel* ↗️\n\n"
        
    return re.sub(pattern, replace_with_badge, text, flags=re.DOTALL)

def execute_chat(

    message: str,
    history: list,
    mode: str,
    model_name: str,
    system_prompt_preset: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_search: bool,
    hf_token: str
):
    """
    Orchestrates the chat request, performs search if toggled, builds the history,
    and runs inference on the selected backend mode (Local CPU, Zero-GPU, or API).
    """
    # 1. Look up the repo_id from configs
    repo_id = None
    for item in MODEL_CONFIGS.get(mode, []):
        if item["name"] == model_name:
            repo_id = item["repo_id"]
            break
            
    if not repo_id:
        yield history + [[message, "Configuration Error: Selected model details not found."]], ""
        return

    # 2. Handle web search if enabled
    search_context = ""
    status_update = ""
    
    if enable_search:
        status_update = f"🔍 Searching web for: '{message}'...\n"
        yield history + [[message, status_update]], ""
        
        results = web_search(message, max_results=3)
        if results:
            status_update += f"📄 Scraped {len(results)} relevant web sources. Integrating context...\n"
            yield history + [[message, status_update]], ""
            
            # Scrape details from the top result to enrich context
            top_url = results[0]["url"]
            scraped_content = scrape_url(top_url, max_chars=3000)
            
            # Format combined search results
            search_context = format_search_results_for_prompt(message, results)
            search_context += f"\nDetailed body scraped from source [1] ({top_url}):\n{scraped_content}\n---\n"
        else:
            status_update += "❌ Web search returned no results. Proceeding with model knowledge...\n"
            yield history + [[message, status_update]], ""
            time.sleep(1)

    # 3. Compile history into standard Gradio message formats
    chat_messages = []
    for user_msg, bot_msg in history:
        # If the bot response has status logs from web search, strip them so LLM doesn't read them as its own words
        clean_bot_msg = bot_msg
        if "🔍 Searching web" in bot_msg:
            # Split and get the text after the final status separator if it exists
            parts = bot_msg.split("---\n")
            if len(parts) > 1:
                clean_bot_msg = parts[-1]
            else:
                # Fallback if structure is different
                clean_bot_msg = bot_msg.split("\n")[-1]
        
        chat_messages.append({"role": "user", "content": user_msg})
        chat_messages.append({"role": "assistant", "content": clean_bot_msg})

    # Prepare active prompt contents
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    compiled_system_prompt = system_prompt_preset.replace("{datetime}", current_time)
    
    # Prepend search context to user query if found
    if search_context:
        user_query_content = f"{search_context}User Query: {message}"
    else:
        user_query_content = message
        
    chat_messages.append({"role": "user", "content": user_query_content})

    # 4. Invoke inference backend
    if mode == "HF Serverless API (Zero Overhead)":
        # Stream response from API
        api_stream = run_serverless_api_inference(
            messages=chat_messages,
            repo_id=repo_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            hf_token=hf_token
        )
        
        for partial_text in api_stream:
            formatted_text = format_thinking_tags(partial_text)
            artifacts = extract_artifacts(formatted_text)
            clean_text = clean_chatbot_response(formatted_text)
            full_response = status_update + clean_text if status_update else clean_text
            yield history + [[message, full_response]], artifacts

            
    else:
        # Local CPU or Zero-GPU mode
        # Load local tokenizer (temporarily to build prompt or load model)
        # Note: loading tokenizer is fast and lightweight
        token = hf_token or os.environ.get("HF_TOKEN")
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)
        except Exception:
            tokenizer = None
            
        prompt_text = build_prompt_with_history(chat_messages, compiled_system_prompt, tokenizer)
        
        # Free up variables
        del tokenizer
        
        # Conditionally invoke GPU or CPU engine based on selected UI mode
        if mode == "Zero-GPU (Accelerated)":
            local_stream = generate_local_inference_gpu(
                prompt_text=prompt_text,
                repo_id=repo_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                hf_token=token
            )
        else:
            local_stream = generate_local_inference_cpu(
                prompt_text=prompt_text,
                repo_id=repo_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                hf_token=token
            )

        try:
            for partial_text in local_stream:
                formatted_text = format_thinking_tags(partial_text)
                artifacts = extract_artifacts(formatted_text)
                clean_text = clean_chatbot_response(formatted_text)
                full_response = status_update + clean_text if status_update else clean_text
                yield history + [[message, full_response]], artifacts
        except Exception as e:
            error_message = f"\n\n### ⚠️ Inference Failure\n{e}"
            yield history + [[message, status_update + error_message if status_update else error_message]], []

