#!/usr/bin/env python3
# ============================================================
#  agent_cli.py
#  alias agent="python3 ~/projects/natural/agent_cli.py"
# ============================================================

import sys, os, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _divider():
    print("─" * 60)


def _print_steps(steps: list):
    icons = {
        "inspect_directory":      "📁",
        "search_recent_history":  "🕐",
        "search_semantic_history":"🗂 ",
        "search_hybrid_history":  "🗂 ",
    }
    for s in steps:
        icon = icons.get(s["tool"], "⚙️ ")
        inp  = str(s["input"])[:80]
        print(f"  {icon} {s['tool']}  ←  {inp}")


def _ask(prompt_text: str) -> str:
    try:
        return input(prompt_text).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return ""


def _copy(text: str) -> bool:
    try:
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
            return True
        for tool in [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]:
            if subprocess.run(["which", tool[0]], capture_output=True).returncode == 0:
                subprocess.run(tool, input=text.encode(), check=True)
                return True
    except Exception:
        pass
    return False


def _execute(cmd: str):
    import time
    print()
    subprocess.run(cmd, shell=True, text=True, env=os.environ.copy())
    log = Path.home() / ".naturalshell" / "executed.log"
    log.parent.mkdir(exist_ok=True)
    with open(log, "a") as f:
        f.write(f": {int(time.time())}:0;{cmd}\n")


def main():
    args = sys.argv[1:]

    if "--reindex" in args:
        from history_rag import index_history, history_count
        print("Indexing zsh history...", end=" ", flush=True)
        added = index_history(force=True)
        print(f"done. {added} entries. Total: {history_count()}")
        return

    provider = None
    if "--llm" in args:
        idx = args.index("--llm")
        if idx + 1 < len(args):
            provider = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
        else:
            print("[error] --llm requires a value: groq | gemini | ollama")
            sys.exit(1)

    query = " ".join(args).strip()
    if not query:
        print("Usage:  agent \"your query\"")
        print("        agent --llm gemini \"your query\"")
        print("        agent --reindex")
        return

    from history_rag import index_history, history_count
    if history_count() == 0:
        print("First run — indexing zsh history...", end=" ", flush=True)
        added = index_history()
        print(f"done. {added} commands indexed.")

    import config
    active = provider or config.ACTIVE_LLM
    print(f"\n[{active}] thinking...\n")

    from agent import run
    result = run(query, provider=provider)

    if result.get("error"):
        print(f"[error] {result['error']}")
        sys.exit(1)

    if not result["success"]:
        print("[error] Could not generate a command. Try rephrasing.")
        sys.exit(1)

    if result.get("steps"):
        _print_steps(result["steps"])
        print()

    _divider()
    print(result["command"])
    _divider()
    print(f"  {result['explanation']}")

    if result["is_dangerous"]:
        print("\n  ⚠  WARNING: this command may be irreversible")

    print()
    choice = _ask("[r]un  [c]opy  [s]kip  › ")

    if choice in ("r", "run", ""):
        _execute(result["command"])
    elif choice in ("c", "copy"):
        if _copy(result["command"]):
            print("copied.")
        else:
            print("[warn] clipboard not available")


if __name__ == "__main__":
    main()
