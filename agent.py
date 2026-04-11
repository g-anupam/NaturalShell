# ============================================================
#  agent.py — ReAct loop, zero LangChain, Python 3.14 safe
# ============================================================

from __future__ import annotations
import json, sys, re, os
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent))
import config
import os_context
import history_rag


# ── Tool handlers ─────────────────────────────────────────────

def tool_inspect_directory(args: dict) -> str:
    return os_context.get_context()

def tool_search_recent_history(args: dict) -> str:
    return history_rag.get_recent(int(args.get("n", 5)))

def tool_search_semantic_history(args: dict) -> str:
    return history_rag.search_semantic(args.get("query", ""))

def tool_search_hybrid_history(args: dict) -> str:
    return history_rag.search_hybrid(args.get("query", ""), days=args.get("days"))

TOOLS = {
    "inspect_directory": {
        "fn":   tool_inspect_directory,
        "desc": "Inspect current working directory. Returns real filenames, sizes, git status, disk space. Args: { reason: str }",
    },
    "search_recent_history": {
        "fn":   tool_search_recent_history,
        "desc": "Get the N most recent commands sorted by timestamp. Use for: 'last N commands', 'what did i just run'. Args: { n: int }",
    },
    "search_semantic_history": {
        "fn":   tool_search_semantic_history,
        "desc": "Semantic similarity search over full zsh history. Use for: 'that ffmpeg command', 'how did i install X'. Args: { query: str }",
    },
    "search_hybrid_history": {
        "fn":   tool_search_hybrid_history,
        "desc": "Semantic search scoped to a time window. Use for: 'pip command from last week'. Args: { query: str, days: int }",
    },
}

TOOL_LIST = "\n".join(f"- {n}: {i['desc']}" for n, i in TOOLS.items())


# ── System prompt ─────────────────────────────────────────────

SYSTEM = f"""You are NaturalShell, an expert at converting natural language into precise shell commands for macOS/Linux.

You reason step by step using this loop:

Thought: <your reasoning>
Action: <tool_name>
Action Input: <valid JSON dict>
Observation: <filled in by system>
... repeat as needed ...
Thought: I have everything I need.
Final Answer:
<use ONE of the two formats below>

FORMAT A — when a shell command is needed:
COMMAND
<exact shell command>
END
EXPLANATION
<1-2 sentences>
END
DANGER
<yes or no>
END

FORMAT B — when the answer is purely informational (history recall, explanations, questions):
ANSWER
<your full plain text answer>
END

Available tools:
{TOOL_LIST}

RULES:
- Always start with a Thought
- Call inspect_directory for any file/folder/git/disk task
- Pick the right history tool:
    * "last N commands" / "what did i just run"  -> search_recent_history
    * "that command i used to..."                 -> search_semantic_history
    * "command from last week"                    -> search_hybrid_history
- When history tools return commands, use them DIRECTLY — never fall back to history | grep
- List ALL returned history commands, never skip any
- If asked for last N commands, show all N
- Never guess filenames — only use names confirmed by inspect_directory
- Output Final Answer EXACTLY ONCE. Do not repeat the format multiple times.
- After writing Final Answer, STOP. Do not add more Thoughts after it.
"""


# ── LLM callers ───────────────────────────────────────────────

def _call_groq(messages: list) -> str:
    try:
        from groq import Groq
    except ImportError:
        sys.exit("[error] Run: pip install groq")
    client = Groq(api_key=config.GROQ_API_KEY)
    resp   = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        stop=["Observation:"],
    )
    return resp.choices[0].message.content or ""


def _call_gemini(messages: list) -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        sys.exit("[error] Run: pip install google-generativeai")
    genai.configure(api_key=config.GEMINI_API_KEY)
    model   = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        system_instruction=messages[0]["content"],
    )
    history = []
    for m in messages[1:-1]:
        history.append({"role": "user" if m["role"] == "user" else "model",
                        "parts": [m["content"]]})
    chat = model.start_chat(history=history)
    resp = chat.send_message(
        messages[-1]["content"],
        generation_config={"temperature": 0, "max_output_tokens": 1024,
                           "stop_sequences": ["Observation:"]},
    )
    return resp.text or ""


def _call_ollama(messages: list) -> str:
    import urllib.request
    payload = json.dumps({
        "model":    config.OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": 0, "stop": ["Observation:"]},
    }).encode()
    req = urllib.request.Request(
        f"{config.OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["message"]["content"] or ""


def _call_llm(provider: str, messages: list) -> str:
    p = provider.lower()
    if p == "groq":   return _call_groq(messages)
    if p == "gemini": return _call_gemini(messages)
    if p == "ollama": return _call_ollama(messages)
    sys.exit(f"[error] Unknown provider: {provider}")


# ── Parsers ───────────────────────────────────────────────────

def _parse_action(text: str):
    action_m = re.search(r"Action:\s*(\w+)", text)
    input_m  = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
    if not action_m:
        return None, None
    tool_name = action_m.group(1).strip()
    args = {}
    if input_m:
        try:
            args = json.loads(input_m.group(1))
        except json.JSONDecodeError:
            raw = input_m.group(1)
            for k, v in re.findall(r'"(\w+)"\s*:\s*"([^"]*)"', raw):
                args[k] = v
            for k, v in re.findall(r'"(\w+)"\s*:\s*(\d+)', raw):
                args[k] = int(v)
    return tool_name, args


def _parse_final(text: str) -> Dict[str, Any]:
    def block(tag: str) -> str:
        m = re.search(rf"{tag}\n(.*?)\nEND", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    # Informational answer
    answer = block("ANSWER")
    if answer:
        return {"command": "", "explanation": answer,
                "is_dangerous": False, "success": True, "informational": True}

    command     = block("COMMAND")
    explanation = block("EXPLANATION")
    danger_raw  = block("DANGER").lower()

    if not command:
        m = re.search(r"```(?:bash|sh|zsh)?\n?(.*?)```", text, re.DOTALL)
        if m:
            command = m.group(1).strip()

    is_dangerous = danger_raw == "yes" or any(
        p in command for p in config.DANGEROUS_PATTERNS
    )
    return {
        "command":      command,
        "explanation":  explanation or "Shell command generated.",
        "is_dangerous": is_dangerous,
        "success":      bool(command),
        "informational": False,
    }


def _has_final_answer(text: str) -> bool:
    """Return True if text contains a complete parseable final answer."""
    r = _parse_final(text)
    return r["success"] or r.get("informational", False)


# ── Verbose printer ───────────────────────────────────────────

def _vprint(text: str):
    for line in text.strip().splitlines():
        l = line.strip()
        if not l:
            continue
        if l.startswith("Thought:"):
            print(f"  \033[33m{l}\033[0m")       # yellow
        elif l.startswith("Action Input:"):
            print(f"  \033[36m{l}\033[0m")        # cyan
        elif l.startswith("Action:"):
            print(f"  \033[36m{l}\033[0m")        # cyan
        elif l.startswith("Final Answer:"):
            print(f"  \033[32m{l}\033[0m")        # green
        elif l.startswith("Observation:"):
            print(f"  \033[35m{l[:200]}\033[0m")  # magenta, truncated
        else:
            print(f"  \033[2m{l}\033[0m")         # dim


# ── Main ReAct loop ───────────────────────────────────────────

def run(query: str, provider: str = None, verbose: bool = False) -> Dict[str, Any]:
    p = (provider or config.ACTIVE_LLM).lower()

    if p == "groq"   and not config.GROQ_API_KEY:   sys.exit("[error] Set GROQ_API_KEY in .env")
    if p == "gemini" and not config.GEMINI_API_KEY: sys.exit("[error] Set GEMINI_API_KEY in .env")

    messages   = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"Query: {query}"},
    ]
    scratchpad = ""
    steps      = []

    for iteration in range(8):
        call_messages = messages.copy()
        if scratchpad:
            call_messages.append({"role": "assistant", "content": scratchpad})

        try:
            llm_output = _call_llm(p, call_messages)
        except Exception as e:
            return {"command": "", "explanation": "", "is_dangerous": False,
                    "success": False, "steps": steps, "error": str(e)}

        if verbose:
            _vprint(llm_output)

        scratchpad += llm_output

        # ── Stop as soon as we have a parseable final answer ───
        # Check both after "Final Answer:" and in the raw output
        # (LLM sometimes skips the prefix)
        search_in = scratchpad.split("Final Answer:")[-1] if "Final Answer:" in scratchpad else ""
        if search_in and _has_final_answer(search_in):
            parsed = _parse_final(search_in)
            parsed["steps"] = steps
            parsed["error"] = None
            return parsed

        # Also check raw llm_output for an answer block (no Final Answer prefix)
        if _has_final_answer(llm_output) and "Action:" not in llm_output:
            parsed = _parse_final(llm_output)
            parsed["steps"] = steps
            parsed["error"] = None
            return parsed

        # ── Tool call ──────────────────────────────────────────
        tool_name, args = _parse_action(llm_output)

        if tool_name and tool_name in TOOLS:
            try:
                observation = TOOLS[tool_name]["fn"](args or {})
            except Exception as e:
                observation = f"[tool error] {e}"

            steps.append({"tool": tool_name, "input": str(args)})
            obs_line = f"\nObservation: {observation}\n"

            if verbose:
                preview = str(observation)[:250].replace("\n", " ")
                print(f"  \033[35mObservation: {preview}{'...' if len(str(observation)) > 250 else ''}\033[0m")

            scratchpad += obs_line

        elif tool_name:
            scratchpad += f"\nObservation: [error] Unknown tool '{tool_name}'. Use one of: {list(TOOLS.keys())}\n"

        else:
            # No tool call, no answer yet — add Final Answer trigger once
            if "Final Answer:" not in scratchpad:
                scratchpad += "\nThought: I have enough information.\nFinal Answer:\n"
            else:
                # Already nudged — parse whatever we have
                break

    # Last resort parse
    final_text = scratchpad.split("Final Answer:")[-1] if "Final Answer:" in scratchpad else scratchpad
    parsed = _parse_final(final_text)
    if parsed["success"] or parsed.get("informational"):
        parsed["steps"] = steps
        parsed["error"] = None
        return parsed

    return {"command": "", "explanation": "", "is_dangerous": False,
            "success": False, "steps": steps,
            "error": "Could not produce a final answer."}
