# ============================================================
#  os_context.py
#  Collects real filesystem + git + env context before the LLM call.
#  The richer this context, the more accurate the generated command.
# ============================================================

from __future__ import annotations
import os, subprocess, platform, shutil, stat
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def get_context() -> str:
    """
    Returns a plain-text block describing the current environment.
    This is injected verbatim into the LLM prompt.
    """
    cwd  = Path.cwd()
    home = Path.home()

    lines = []

    # ── Basic ──────────────────────────────────────────────
    lines += [
        f"## Environment",
        f"- Platform : {platform.system()} {platform.machine()}",
        f"- Shell    : {os.environ.get('SHELL', 'zsh')}",
        f"- User     : {os.environ.get('USER', 'user')}",
        f"- CWD      : {cwd}",
        f"- Home     : {home}",
    ]

    # ── Disk ───────────────────────────────────────────────
    try:
        u = shutil.disk_usage(cwd)
        lines.append(
            f"- Disk     : {_human(u.free)} free / {_human(u.total)} total"
        )
    except Exception:
        pass

    # ── Git ────────────────────────────────────────────────
    git = _git_context(cwd)
    if git:
        lines.append("")
        lines.append("## Git")
        lines += git

    # ── Directory listing ──────────────────────────────────
    lines.append("")
    lines.append("## Current Directory Contents")
    try:
        all_entries = sorted(cwd.iterdir(), key=lambda p: (p.is_file(), p.name))
        dirs  = [e for e in all_entries if e.is_dir()]
        files = [e for e in all_entries if e.is_file()]

        if dirs:
            lines.append("### Subdirectories")
            for d in dirs[:30]:
                try:
                    n = sum(1 for _ in d.iterdir())
                except PermissionError:
                    n = "?"
                lines.append(f"  {d.name}/  ({n} items)")

        if files:
            lines.append("### Files")
            for f in files[:50]:
                try:
                    s    = f.stat()
                    size = _human(s.st_size)
                    mtime = datetime.fromtimestamp(s.st_mtime).strftime("%Y-%m-%d")
                    xbit = " [executable]" if s.st_mode & stat.S_IXUSR else ""
                    lines.append(f"  {f.name}  {size}  modified {mtime}{xbit}")
                except Exception:
                    lines.append(f"  {f.name}")

        if len(all_entries) > 50:
            lines.append(f"  ... and {len(all_entries) - 50} more entries")

    except PermissionError:
        lines.append("  (permission denied)")

    # ── Python env ─────────────────────────────────────────
    venv = _venv(cwd)
    if venv:
        lines.append("")
        lines.append(f"## Python Environment")
        lines.append(venv)

    # ── Notable project files ──────────────────────────────
    notable = _notable(cwd)
    if notable:
        lines.append("")
        lines.append(f"## Detected Project Type")
        lines.append("  " + ", ".join(notable))

    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────

def _human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _run(cmd: list, cwd: Path, timeout: int = 3) -> str:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _git_context(cwd: Path) -> list[str] | None:
    if not _run(["git", "rev-parse", "--is-inside-work-tree"], cwd):
        return None

    branch      = _run(["git", "branch", "--show-current"], cwd) or "HEAD detached"
    last_commit = _run(["git", "log", "-1", "--pretty=format:%h %s (%cr)"], cwd)

    status_raw  = _run(["git", "status", "--porcelain"], cwd)
    staged, modified, untracked = [], [], []
    for line in status_raw.splitlines():
        if not line:
            continue
        idx, wt, fname = line[0], line[1], line[3:]
        if idx not in (" ", "?"):
            staged.append(fname)
        if wt == "M":
            modified.append(fname)
        if idx == "?" and wt == "?":
            untracked.append(fname)

    out = [f"- Branch      : {branch}"]
    if last_commit:
        out.append(f"- Last commit : {last_commit}")
    if staged:
        out.append(f"- Staged      : {', '.join(staged[:10])}")
    if modified:
        out.append(f"- Modified    : {', '.join(modified[:10])}")
    if untracked:
        out.append(f"- Untracked   : {len(untracked)} file(s)")
    if not (staged or modified or untracked):
        out.append("- Status      : clean")

    return out


def _venv(cwd: Path) -> str | None:
    for name in [".venv", "venv", "env"]:
        p = cwd / name
        if p.is_dir() and (p / "bin" / "python").exists():
            return f"  virtualenv at ./{name}/"
    conda = os.environ.get("CONDA_DEFAULT_ENV")
    if conda and conda != "base":
        return f"  conda env: {conda}"
    if (cwd / "pyproject.toml").exists():
        return "  pyproject.toml detected (poetry/hatch)"
    return None


def _notable(cwd: Path) -> list[str]:
    markers = {
        "package.json":        "Node.js",
        "requirements.txt":    "Python",
        "pyproject.toml":      "Python (poetry/hatch)",
        "Dockerfile":          "Docker",
        "docker-compose.yml":  "Docker Compose",
        "docker-compose.yaml": "Docker Compose",
        "Makefile":            "Makefile",
        "Cargo.toml":          "Rust",
        "go.mod":              "Go",
        "pom.xml":             "Java/Maven",
    }
    return [label for fname, label in markers.items() if (cwd / fname).exists()]
