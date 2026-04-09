# NaturalShell 🐚

Convert natural language into shell commands, powered by your own zsh history and real filesystem context.

```
agent "find all python files modified this week"
agent "what was that ffmpeg command I used to compress a video"
agent "create a new file called delete.html"
```

## How it works

Two things make it smarter than a generic AI assistant:

1. **Zsh History RAG** — indexes your `~/.zsh_history` into a local vector database. Ask about commands you've run before and it retrieves the actual command, not a guess.

2. **OS Context Injection** — before generating any command, it reads your current directory (real filenames, git status, disk space, project type) so commands are specific to what actually exists on your machine.

## Requirements

- Python 3.10+
- A free API key from [Groq](https://console.groq.com) or [Gemini](https://aistudio.google.com)
- zsh (for history indexing)

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/naturalshell.git
cd naturalshell
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first run will download the embedding model (~80MB, one time only).

### 4. Set up your API keys

Create `.env` and fill in your keys:

```
ACTIVE_LLM=groq
GROQ_API_KEY=your-groq-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here
```

Get free keys here:
- **Groq** (recommended, fast): https://console.groq.com/keys
- **Gemini**: https://aistudio.google.com/app/apikey

### 5. Add the alias

Add this line to your `~/.zshrc`:

```bash
alias agent="python3 /path/to/naturalshell/agent_cli.py"
```

Replace `/path/to/naturalshell` with your actual clone path, for example:
```bash
alias agent="python3 ~/projects/naturalshell/agent_cli.py"
```

Then reload your shell:

```bash
source ~/.zshrc
```

## Usage

```bash
# Ask anything in natural language
agent "find all files larger than 100mb and show their sizes"
agent "what git commands did I run to undo changes recently"
agent "create a virtualenv and install requirements.txt"

# Switch LLM provider on the fly
agent --llm gemini "show me disk usage sorted by size"

# Re-index your zsh history (run after a long session)
agent --reindex
```

After each command you get three options:

```
[r]un  [c]opy  [s]kip  ›
```

- `r` — runs the command directly in your shell
- `c` — copies it to your clipboard
- `s` — skips (just wanted to see it)

## Switching LLMs

If you run out of free credits on one provider, switch in `.env`:

```
ACTIVE_LLM=gemini
```

Or on a per-query basis with `--llm`:

```bash
agent --llm gemini "your query"
agent --llm ollama "your query"   # fully local, no API key needed
```

For Ollama, install it from https://ollama.ai and pull a model:
```bash
ollama pull llama3.2
```

## Project structure

```
naturalshell/
├── .env.example      ← copy to .env and add your keys
├── config.py         ← loads settings from .env
├── agent_cli.py      ← CLI entry point (the thing you alias)
├── agent.py          ← ReAct loop, calls Groq/Gemini/Ollama directly
├── history_rag.py    ← zsh history parser + ChromaDB vector search
├── os_context.py     ← real-time directory and git context
└── requirements.txt
```

## Troubleshooting

**`[error] .env file not found`**
Run `cp .env.example .env` and fill in your API key.

**`agent: command not found`**
Make sure you added the alias to `~/.zshrc` and ran `source ~/.zshrc`.

**History search returns generic commands instead of your actual ones**
Run `agent --reindex` to rebuild the history index from scratch.

**Slow on first run**
The embedding model downloads once on first use (~80MB). Subsequent runs are fast.
