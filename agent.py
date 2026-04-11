# ============================================================
#  agent.py
#  Proper ReAct loop — explicit Thought/Action/Observation steps.
#  Text-based loop, not relying on native tool-call APIs.
#  Zero LangChain. Direct Groq/Gemini/Ollama API calls.
# ============================================================

from __future__ import annotations
import json, sys, re, os
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent))
import config
import os_context
import history_rag


# ── Tools ─────────────────────────────────────────────────────
# Each tool is a plain function. The LLM calls them by name
# via text output, not via native API tool-call schemas.

def tool_inspect_directory(args: dict) -> str:
    return os_context.get_context()


def tool_search_recent_history(args: dict) -> str:
    """Returns the N most recent commands by actual timestamp."""
    n = int(args.get("n", 5))
    return history_rag.get_recent(n)


def tool_search_semantic_history(args: dict) -> str:
    """Semantic similarity search over full history."""
    query = args.get("query", "")
    return history_rag.search_semantic(query)


def tool_search_hybrid_history(args: dict) -> str:
    """Semantic search scoped to a time window (days back)."""
    query = args.get("query", "")
    days  = args.get("days", None)
    return history_rag.search_hybrid(query, days=days)


TOOLS = {
    "inspect_directory": {
        "fn": tool_inspect_directory,
        "desc": "Inspect the current working directory. Returns real filenames, sizes, git status, venv, disk space, project type. Args: { reason: str }",
    },
    "search_recent_history": {
        "fn": tool_search_recent_history,
        "desc": "Get the N most recent commands from zsh history, sorted by timestamp. Use for: 'last N commands', 'what did i just run', 'most recent command'. Args: { n: int }",
    },
    "search_semantic_history": {
        "fn": tool_search_semantic_history,
        "desc": "Semantic similarity search over full zsh history. Use for: 'that ffmpeg command i used', 'how did i install X', 'the docker command for pruning'. Args: { query: str }",
    },
    "search_hybrid_history": {
        "fn": tool_search_hybrid_history,
        "desc": "Semantic search scoped to a time window. Use for: 'pip command from last week', 'git command i ran yesterday'. Args: { query: str, days: int }",
    },
}

TOOL_LIST = "\n".join(
    f"- {name}: {info['desc']}" for name, info in TOOLS.items()
)


# ── ReAct prompt ──────────────────────────────────────────────

SYSTEM = f"""You are NaturalShell, an expert at converting natural language into precise shell commands for macOS/Linux.

You reason step by step using this EXACT loop format:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <valid JSON dict of args>
Observation: <tool result will be filled in here>
... repeat Thought/Action/Observation as needed ...
Thought: I now have everything I need.
Final Answer:

If the query needs a shell command, use this format:
COMMAND
<the exact shell command>
END
EXPLANATION
<1-2 sentences>
END
DANGER
<yes or no>
END

If the query is informational (e.g. "what were my last commands", "what did i run"), use this format instead:
ANSWER
<plain text answer to the user's question>
END

Available tools:
{TOOL_LIST}

RULES:
- Always start with a Thought
- Always call inspect_directory for any file/folder/git/disk task
- For history queries, pick the RIGHT tool:
    * "last N commands" / "what did i just run"  → search_recent_history
    * "that command I used to..."                 → search_semantic_history
    * "pip command from last week"                → search_hybrid_history
- NEVER guess filenames — only use names confirmed by inspect_directory
- NEVER use history | grep — the history tools already do the search
- If asked for "last N commands", your answer must show all N commands
- For purely informational queries (what did i run, show my history), set COMMAND to "echo \'<the answer>\'" so the output is meaningful when run, and set DANGER to no
- When history tools return commands, LIST ALL of them in your answer exactly as returned — never summarize or pick just one
- Only output one Thought/Action pair at a time and wait for the Observation
"""


# ── LLM clients ───────────────────────────────────────────────

def _call_groq(messages: list) -> str:
    from groq import Groq
    client   = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        stop=["Observation:"],   # stop when it expects a tool result
    )
    return response.choices[0].message.content or ""


def _call_gemini(messages: list) -> str:
    import google.generativeai as genai
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        system_instruction=messages[0]["content"],
    )
    # Build history for Gemini
    history = []
    for m in messages[1:-1]:
        role = "user" if m["role"] == "user" else "model"
        history.append({"role": role, "parts": [m["content"]]})

    chat     = model.start_chat(history=history)
    response = chat.send_message(
        messages[-1]["content"],
        generation_config={"temperature": 0, "max_output_tokens": 1024,
                           "stop_sequences": ["Observation:"]},
    )
    return response.text or ""


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
        data = json.loads(r.read())
    return data["message"]["content"] or ""


def _call_llm(provider: str, messages: list) -> str:
    p = provider.lower()
    if p == "groq":
        return _call_groq(messages)
    elif p == "gemini":
        return _call_gemini(messages)
    elif p == "ollama":
        return _call_ollama(messages)
    else:
        sys.exit(f"[error] Unknown provider: {provider}")


# ── ReAct parser ──────────────────────────────────────────────

def _parse_action(text: str):
    """
    Extract Action and Action Input from LLM output.
    Returns (tool_name, args_dict) or (None, None) if not found.
    """
    action_m = re.search(r"Action:\s*(\w+)", text)
    input_m  = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)

    if not action_m:
        return None, None

    tool_name = action_m.group(1).strip()
    args      = {}
    if input_m:
        try:
            args = json.loads(input_m.group(1))
        except json.JSONDecodeError:
            # Try to extract key values manually if JSON is malformed
            raw = input_m.group(1)
            for k, v in re.findall(r'"(\w+)"\s*:\s*"([^"]*)"', raw):
                args[k] = v
            for k, v in re.findall(r'"(\w+)"\s*:\s*(\d+)', raw):
                args[k] = int(v)

    return tool_name, args


def _parse_final(text: str) -> Dict[str, Any]:
    """Extract COMMAND block (shell task) or ANSWER block (informational query)."""
    def block(tag: str) -> str:
        m = re.search(rf"{tag}\n(.*?)\nEND", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    # Informational answer — no shell command needed
    answer = block("ANSWER")
    if answer:
        return {
            "command":      "",
            "explanation":  answer,
            "is_dangerous": False,
            "is_answer":    True,
            "success":      True,
        }

    command     = block("COMMAND")
    explanation = block("EXPLANATION")
    danger_raw  = block("DANGER").lower()

    # Fallback to ```bash block
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
        "is_answer":    False,
        "success":      bool(command),
    }


# ── Main ReAct loop ───────────────────────────────────────────

def run(query: str, provider: str = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Runs the full ReAct loop:
      1. LLM emits Thought + Action
      2. We stop at "Observation:" and run the tool
      3. Append observation, let LLM continue
      4. Repeat until LLM emits "Final Answer:"
    """
    p = (provider or config.ACTIVE_LLM).lower()

    if p == "groq" and not config.GROQ_API_KEY:
        sys.exit("[error] Set GROQ_API_KEY in .env")
    if p == "gemini" and not config.GEMINI_API_KEY:
        sys.exit("[error] Set GEMINI_API_KEY in .env")

    try:
        from groq import Groq
    except ImportError:
        if p == "groq":
            sys.exit("[error] Run: pip install groq")

    messages = [
        {"role": "system",  "content": SYSTEM},
        {"role": "user",    "content": f"Query: {query}"},
    ]

    steps        = []
    scratchpad   = ""   # accumulates the full Thought/Action/Observation trace

    for iteration in range(8):
        # Ask LLM to continue the ReAct trace
        # We append the scratchpad as an assistant turn so the LLM
        # continues from where it left off
        call_messages = messages.copy()
        if scratchpad:
            call_messages.append({"role": "assistant", "content": scratchpad})

        try:
            llm_output = _call_llm(p, call_messages)
        except Exception as e:
            return {
                "command": "", "explanation": "", "is_dangerous": False,
                "success": False, "steps": steps, "error": str(e),
            }

        scratchpad += llm_output

        # ── Print reasoning trace if verbose ──────────────────
        if verbose:
            print()
            for line in llm_output.strip().splitlines():
                if line.startswith("Thought:"):
                    print(f"  [33m{line}[0m")        # yellow
                elif line.startswith("Action:"):
                    print(f"  [36m{line}[0m")        # cyan
                elif line.startswith("Action Input:"):
                    print(f"  [36m{line}[0m")        # cyan
                elif line.startswith("Final Answer:"):
                    print(f"  [32m{line}[0m")        # green
                elif line.strip():
                    print(f"  [2m{line}[0m")         # dim

        # ── Did the LLM produce a Final Answer? ────────────────
        if "Final Answer:" in scratchpad:
            final_text = scratchpad.split("Final Answer:")[-1]
            parsed     = _parse_final(final_text)
            parsed["steps"] = steps
            parsed["error"] = None
            return parsed

        # ── Did the LLM call a tool? ───────────────────────────
        tool_name, args = _parse_action(llm_output)

        if tool_name and tool_name in TOOLS:
            tool_fn = TOOLS[tool_name]["fn"]
            try:
                observation = tool_fn(args or {})
            except Exception as e:
                observation = f"[tool error] {e}"

            steps.append({"tool": tool_name, "input": str(args)})

            if verbose:
                preview = str(observation)[:300].replace("\n", " ")
                print(f"  [35mObservation: {preview}{'...' if len(str(observation)) > 300 else ''}[0m")

            # Append observation and let the loop continue
            scratchpad += f"\nObservation: {observation}\n"

        elif tool_name:
            # Unknown tool — tell the LLM
            scratchpad += f"\nObservation: [error] Unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}\n"

        else:
            # LLM produced neither a tool call nor a Final Answer
            # Nudge it to continue
            scratchpad += "\nThought: Let me provide the final answer now.\n"

    return {
        "command": "", "explanation": "", "is_dangerous": False,
        "success": False, "steps": steps,
        "error": "Max iterations reached without a final answer.",
    }
