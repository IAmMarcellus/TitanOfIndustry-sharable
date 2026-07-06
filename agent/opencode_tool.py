"""OpenSage -> OpenCode delegation tool (the core TitanOfIndustry integration).

The OpenSage orchestrator (running on local Qwen) calls ``opencode_run`` to hand a
concrete coding subtask to OpenCode, which performs the actual file edits via the
*same* local Qwen model served by vLLM. The high-volume "write the code" tokens stay
on the executor; the orchestrator only plans and verifies.

Process boundary by design: we shell out to the ``opencode`` CLI rather than import it
(it is a Node app). See ../CLAUDE.md.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

# Default model ref is "<provider>/<model>" from opencode/opencode.json.
_DEFAULT_MODEL = os.environ.get("OPENCODE_MODEL", "local-vllm/qwen-codex")
# Absolute path to our provider config, injected so OpenCode resolves the local-vllm
# provider regardless of the working directory (no global install step required).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_OPENCODE_CONFIG = _REPO_ROOT / "opencode" / "opencode.json"
_DEFAULT_TIMEOUT_S = int(os.environ.get("OPENCODE_TIMEOUT_S", "900"))

# Bound concurrent opencode subprocesses (host RAM / process count). The LiteLLM proxy's global cap
# bounds GPU requests; this bounds the Node executor processes themselves. `async with` guarantees the
# permit is released on every exit path (return, timeout, error).
try:
    _MAX_PARALLEL = max(1, int(os.environ.get("OPENCODE_MAX_PARALLEL", "2")))
except (TypeError, ValueError):
    _MAX_PARALLEL = 2
_SEM = asyncio.Semaphore(_MAX_PARALLEL)


async def opencode_run(task: str, cwd: str = ".", model: str = "") -> dict[str, Any]:
    """Delegate a complete coding unit to OpenCode (an autonomous agent on local Qwen via vLLM).

    Use this whenever the task requires creating or editing files. OpenCode runs headless in
    ``cwd`` and can edit files, run shell commands, run the tests/build, and fix failures on its
    own — so give it a COMPLETE, SELF-VERIFYING unit of work (implement -> run tests/build -> fix
    until green) in one call rather than one edit at a time. It returns its transcript.

    Args:
        task: A complete, self-verifying coding instruction — name the target file(s), the expected
            result, AND how to verify it, e.g. "add a function add(a, b) to scratch.py, add a pytest
            test for it, run pytest, and fix until it passes". Be explicit about file paths and the
            expected result.
        cwd: Working directory OpenCode operates in (its project root). Defaults to the
            current directory.
        model: Optional "<provider>/<model>" override; defaults to local-vllm/qwen-27b.

    Returns:
        A dict with ``success`` (bool), ``returncode`` (int), ``stdout``, ``stderr``, and
        the resolved ``cwd``/``model``. On failure ``success`` is False and ``stderr``
        explains why (missing CLI, timeout, bad cwd, or a non-zero OpenCode exit).
    """
    resolved_model = model or _DEFAULT_MODEL
    work_dir = Path(cwd).expanduser().resolve()
    if not work_dir.is_dir():
        return _result(False, -1, "", f"cwd is not a directory: {work_dir}", work_dir, resolved_model)

    env = dict(os.environ)
    if _OPENCODE_CONFIG.is_file():
        env.setdefault("OPENCODE_CONFIG", str(_OPENCODE_CONFIG))

    # `--dir` is authoritative: OpenCode resolves its project root from the config's git
    # root and ignores the subprocess cwd, so the working dir must be passed explicitly.
    cmd = ["opencode", "run", "--dir", str(work_dir), "--model", resolved_model, task]
    # Hold a permit for the subprocess's whole lifetime so at most _MAX_PARALLEL run concurrently.
    async with _SEM:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(work_dir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return _result(
                False, -1, "",
                "`opencode` CLI not found on PATH. Install with: npm install -g opencode-ai",
                work_dir, resolved_model,
            )

        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=_DEFAULT_TIMEOUT_S)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return _result(False, -1, "", f"opencode run timed out after {_DEFAULT_TIMEOUT_S}s", work_dir, resolved_model)

        return _result(
            proc.returncode == 0,
            proc.returncode if proc.returncode is not None else -1,
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
            work_dir,
            resolved_model,
        )


def _result(success: bool, returncode: int, stdout: str, stderr: str, cwd: Path, model: str) -> dict[str, Any]:
    return {
        "success": success,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "cwd": str(cwd),
        "model": model,
    }
