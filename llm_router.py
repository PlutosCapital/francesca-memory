import os, requests
from datetime import datetime, timezone
from typing import Optional

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
DEEPSEEK_KEY   = os.getenv("DEEPSEEK_API_KEY")
XAI_KEY        = os.getenv("XAI_API_KEY")
OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "francesca-local").strip()

def is_deepseek_offpeak():
    now = datetime.now(timezone.utc)
    t = now.hour + now.minute / 60
    return 16.5 <= t <= 24.5

def _post(url, headers, payload, timeout=120):
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def minimax_agent(messages, max_tokens=4096, temperature=0.3, tools=None):
    payload = {"model": "minimax/minimax-m2.5", "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if tools: payload["tools"] = tools
    return _post("https://openrouter.ai/api/v1/chat/completions",
                 {"Authorization": f"Bearer {OPENROUTER_KEY}", "HTTP-Referer": "https://atlas.local", "X-Title": "Francesca-Atlas"},
                 payload)

def deepseek_heavy(prompt, system=None, model="deepseek-chat", max_tokens=4000, temperature=0.2):
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    data = _post("https://api.deepseek.com/chat/completions",
                 {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
                 {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature})
    return data["choices"][0]["message"]["content"]

def xai_fallback(prompt, system=None, model="grok-4.20-reasoning", max_tokens=4000):
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    data = _post("https://api.x.ai/v1/chat/completions",
                 {"Authorization": f"Bearer {XAI_KEY}", "Content-Type": "application/json"},
                 {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2})
    return data["choices"][0]["message"]["content"]

def ollama_local(prompt, model=None, format=None, keep_alive="5m"):
    payload = {"model": model or OLLAMA_MODEL, "prompt": prompt, "stream": False, "keep_alive": keep_alive}
    if format: payload["format"] = format
    data = _post(f"{OLLAMA_HOST}/api/generate", {"Content-Type": "application/json"}, payload, timeout=300)
    return data["response"]

def synthesize(prompt, system=None, max_tokens=4000, prefer="deepseek"):
    try:
        if prefer == "deepseek": return deepseek_heavy(prompt, system, max_tokens=max_tokens)
        if prefer == "xai": return xai_fallback(prompt, system, max_tokens=max_tokens)
        if prefer == "ollama": return ollama_local(prompt)
    except Exception as e:
        if prefer == "deepseek":
            print(f"[WARN] DeepSeek failed ({e}), trying xAI...")
            return xai_fallback(prompt, system, max_tokens=max_tokens)
        raise
