# ============================================================
#  agent.py
#  Zero LangChain. Direct Groq/Gemini/Ollama API calls.
#  ReAct tool-calling loop from scratch.
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

def _run_inspect_directory(args: dict) -> str:
    return os_context.get_context()


def _run_search_zsh_history(args: dict) -> str:
    query = args.get("query", "")
    hits  = history_rag.search_history(query)
    if not hits:
        return "No relevant past commands found in history."

    lines = [
        "These are REAL commands from the user's actual zsh history.",
        "Use the most relevant one directly in your answer — do NOT suggest generic shell commands.\n",
    ]
    for i, h in enumerate(hits, 1):
        lines.append(f"  {i}. command: `{h['command']}`")
        lines.append(f"     relevance: {h['relevance']}%  |  run on: {h['when']}")
    return "\n".join(lines)


TOOL_HANDLERS = {
    "inspect_directory":  _run_inspect_directory,
    "search_zsh_history": _run_search_zsh_history,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "inspect_directory",
            "description": (
                "Inspect the current working directory. "
                "Returns real filenames, sizes, git status, project type, disk space. "
                "Call this for ANY query involving files, directories, or git."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Short note on what you are looking for."}
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_zsh_history",
            "description": (
                "Semantic search over the user's full zsh command history. "
                "Returns REAL commands the user has previously run. "
                "Use this when the user references something they ran before. "
                "Always use the actual returned commands in your answer — never suggest generic alternatives."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language description of the command to find."}
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM = """You are NaturalShell, an expert at converting natural language into precise shell commands for macOS/Linux.

You have two tools:
- inspect_directory  ->  see the REAL files, folders, git status in the current directory
- search_zsh_history ->  returns REAL commands from the user's actual zsh history

CRITICAL RULES:
- When search_zsh_history returns results, you MUST use one of those actual commands in your answer.
  Never fall back to generic alternatives like `history | grep` — the tool already did the search.
- When inspect_directory returns results, use the actual filenames and paths shown.
- Always call inspect_directory for file/folder/git/disk tasks.
- Always call search_zsh_history if the user says "i ran", "last time", "previously", "what was that command".

Respond in EXACTLY this format after using tools — no extra text:

COMMAND
<the exact shell command>
END

EXPLANATION
<1-2 sentences explaining what it does>
END

DANGER
<yes or no>
END
"""


# ── Groq ──────────────────────────────────────────────────────

def _groq_loop(query: str) -> tuple[str, list]:
    try:
        from groq import Groq
    except ImportError:
        sys.exit("[error] Run: pip install groq")

    client   = Groq(api_key=config.GROQ_API_KEY)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": query},
    ]
    steps = []

    for _ in range(6):
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message

        messages.append({
            "role":    "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (msg.tool_calls or [])
            ] or None,
        })

        if not msg.tool_calls:
            return msg.content or "", steps

        for tc in msg.tool_calls:
            name   = tc.function.name
            args   = json.loads(tc.function.arguments)
            output = TOOL_HANDLERS.get(name, lambda _: "[unknown tool]")(args)
            steps.append({"tool": name, "input": str(args), "output_preview": str(output)[:120]})
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      str(output),
            })

    return "", steps


# ── Gemini ────────────────────────────────────────────────────

def _gemini_loop(query: str) -> tuple[str, list]:
    try:
        import google.generativeai as genai
    except ImportError:
        sys.exit("[error] Run: pip install google-generativeai")

    genai.configure(api_key=config.GEMINI_API_KEY)

    from google.generativeai.types import FunctionDeclaration, Tool as GeminiTool

    tool_decls = [
        FunctionDeclaration(
            name="inspect_directory",
            description="Inspect the current working directory. Returns real filenames, sizes, git status, project type, disk space.",
            parameters={
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        ),
        FunctionDeclaration(
            name="search_zsh_history",
            description="Semantic search over the user's full zsh history. Returns REAL past commands. Use the returned commands directly.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
    ]

    model = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        system_instruction=SYSTEM,
        tools=[GeminiTool(function_declarations=tool_decls)],
    )

    chat  = model.start_chat()
    steps = []
    msg   = query

    for _ in range(6):
        response = chat.send_message(msg)
        part     = response.candidates[0].content.parts[0]

        if hasattr(part, "function_call") and part.function_call.name:
            fc     = part.function_call
            name   = fc.name
            args   = dict(fc.args)
            output = TOOL_HANDLERS.get(name, lambda _: "[unknown tool]")(args)
            steps.append({"tool": name, "input": str(args), "output_preview": str(output)[:120]})

            from google.generativeai import protos
            msg = protos.Content(parts=[protos.Part(
                function_response=protos.FunctionResponse(
                    name=name,
                    response={"output": output},
                )
            )])
        else:
            text = "".join(
                p.text for p in response.candidates[0].content.parts
                if hasattr(p, "text")
            )
            return text, steps

    return "", steps


# ── Ollama ────────────────────────────────────────────────────

def _ollama_loop(query: str) -> tuple[str, list]:
    import urllib.request

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": query},
    ]
    steps = []

    for _ in range(6):
        payload = json.dumps({
            "model":    config.OLLAMA_MODEL,
            "messages": messages,
            "tools":    TOOLS_SCHEMA,
            "stream":   False,
        }).encode()

        req = urllib.request.Request(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())

        msg        = data["message"]
        tool_calls = msg.get("tool_calls", [])
        messages.append(msg)

        if not tool_calls:
            return msg.get("content", ""), steps

        for tc in tool_calls:
            fn     = tc["function"]
            name   = fn["name"]
            args   = fn.get("arguments", {})
            output = TOOL_HANDLERS.get(name, lambda _: "[unknown tool]")(args)
            steps.append({"tool": name, "input": str(args), "output_preview": str(output)[:120]})
            messages.append({"role": "tool", "content": str(output)})

    return "", steps


# ── Parser ────────────────────────────────────────────────────

def _parse(raw: str) -> Dict[str, Any]:
    def block(tag: str) -> str:
        m = re.search(rf"{tag}\n(.*?)\nEND", raw, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    command     = block("COMMAND")
    explanation = block("EXPLANATION")
    danger_raw  = block("DANGER").lower()

    if not command:
        m = re.search(r"```(?:bash|sh|zsh)?\n?(.*?)```", raw, re.DOTALL)
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
    }


# ── Entry point ───────────────────────────────────────────────

def run(query: str, provider: str = None) -> Dict[str, Any]:
    p = (provider or config.ACTIVE_LLM).lower()

    try:
        if p == "groq":
            if config.GROQ_API_KEY == "your-groq-api-key-here":
                sys.exit("[error] Set GROQ_API_KEY in config.py")
            raw, steps = _groq_loop(query)
        elif p == "gemini":
            if config.GEMINI_API_KEY == "your-gemini-api-key-here":
                sys.exit("[error] Set GEMINI_API_KEY in config.py")
            raw, steps = _gemini_loop(query)
        elif p == "ollama":
            raw, steps = _ollama_loop(query)
        else:
            sys.exit(f"[error] Unknown provider '{p}'. Choose: groq | gemini | ollama")

    except Exception as e:
        return {
            "command": "", "explanation": "", "is_dangerous": False,
            "success": False, "steps": [], "error": str(e),
        }

    if not raw:
        return {
            "command": "", "explanation": "", "is_dangerous": False,
            "success": False, "steps": steps, "error": "No final answer produced.",
        }

    parsed = _parse(raw)
    parsed["steps"] = steps
    parsed["error"] = None
    return parsed
