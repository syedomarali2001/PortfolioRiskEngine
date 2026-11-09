"""Subprocess adapter to the *actual* V8.4 command-line contract.

Only the existing engine computes finance metrics. We deliberately use a subprocess:
the CLI already handles data, cache and FX; this adapter only exchanges CSV/JSON.
The V8.4 exposure command emits its JSON on stdout and has NO --output flag.
Other V8.4 analysis commands have a --output flag.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .core import normalized_symbol, to_csv


class BackendError(RuntimeError):
    """A user-facing failure when the engine cannot produce verified results."""


def run_report(command: str, rows: list[dict], base: str, args: list[str] | None = None,
               *, root: Path | None = None, timeout: int = 300) -> dict[str, Any]:
    if command not in {"insights", "exposure", "dividends"}:
        raise BackendError(f"Unsupported portfolio report command: {command}")
    return _run(command, rows, base, args or [], root=root, timeout=timeout)


def run_fundamentals(symbol: str, args: list[str] | None = None,
                     *, root: Path | None = None, timeout: int = 120) -> dict[str, Any]:
    return _run("fundamentals", None, None, args or [], root=root, timeout=timeout,
                symbol=normalized_symbol(symbol))


def _run(command: str, rows: list[dict] | None, base: str | None, args: list[str],
         *, root: Path | None, timeout: int, symbol: str | None = None) -> dict[str, Any]:
    project_root = Path(root or os.environ.get("PORTFOLIO_RISK_PROJECT_ROOT") or Path.cwd()).resolve()
    if not project_root.is_dir():
        raise BackendError("The configured engine directory does not exist")

    # A fixed command allow-list and argv sequence. Never use a shell.
    # UI-controlled args are also assembled from fixed flags, not raw user CLI strings.
    with tempfile.TemporaryDirectory(prefix="portfolio_v9_") as td:
        inp, out = Path(td) / "portfolio.csv", Path(td) / "report.json"
        if rows is not None:
            inp.write_text(to_csv(rows), encoding="utf-8")
        target = symbol if command == "fundamentals" else str(inp)
        argv = [sys.executable, "-m", "portfolio_risk", command, target]
        if base is not None:
            argv += ["--base", base]
        argv += list(args)
        if command == "exposure":
            # V8.4 exposure has --json but NOT --output.
            argv += ["--json"]
        else:
            argv += ["--output", str(out)]
        env = os.environ.copy()
        src = str(project_root / "src")
        env["PYTHONPATH"] = src + os.pathsep + str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            completed = subprocess.run(argv, cwd=str(project_root), env=env, shell=False,
                                       capture_output=True, text=True, timeout=timeout,
                                       encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired as exc:
            raise BackendError(
                f"The {command} request timed out. Try fewer enrichment lookups or a shorter window."
            ) from exc
        except OSError as exc:
            raise BackendError(f"Could not launch V8.4: {exc}") from exc
        if completed.returncode:
            message = (completed.stderr or completed.stdout or "Unknown backend error").strip()
            raise BackendError(f"V8.4 {command} failed: {message[-2000:]}")
        if command == "exposure":
            content = completed.stdout
        else:
            if not out.is_file():
                raise BackendError(f"V8.4 {command} returned no JSON report file")
            content = out.read_text(encoding="utf-8-sig")
        try:
            result = json.loads(content.lstrip("\ufeff"))
        except ValueError as exc:
            raise BackendError(f"V8.4 {command} returned invalid JSON. {completed.stderr[-500:]}") from exc
        if not isinstance(result, dict):
            raise BackendError("V8.4 JSON report must be an object")
        return result
