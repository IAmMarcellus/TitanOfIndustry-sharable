"""TitanOfIndustry coding agent.

A local-Qwen OpenSage orchestrator that PLANS and VERIFIES, delegating the actual file edits to
OpenCode (also on local Qwen) via ``opencode_run``, and using a shared Neo4j-backed memory via
``remember``/``recall``. The OpenSage CLI loads ``mk_agent`` from this module:
``opensage web --agent <this dir>``.

Swap the orchestrator to a cloud model later by changing ``_MODEL``/``_API_KEY`` (see
../README.md and ../CLAUDE.md "Hard rules" first).
"""

from __future__ import annotations

import os

from google.adk.models.lite_llm import LiteLlm

from opensage.agents.opensage_agent import OpenSageAgent
from opensage.toolbox.finish_task.finish_task import finish_task
from opensage.toolbox.general.agent_tools import plan, think
from opensage.toolbox.general.dynamic_subagent import (
    call_subagent_as_tool,
    create_subagent,
    list_active_agents,
)

from .approved_mcp import build_approved_mcp_toolsets
from .memory import recall, remember
from .opencode_tool import opencode_run
from .paperclip_tool import (
    paperclip_checkout,
    paperclip_create_subtask,
    paperclip_get_issue,
    paperclip_list_assignments,
    paperclip_post_comment,
    paperclip_release,
    paperclip_update_issue,
)

def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set in .env")
    return value


# All-local by default: LiteLLM in OpenAI-compatible mode pointed at the local model endpoint.
_MODEL = os.environ.get("OPENSAGE_MODEL", "openai/qwen-codex").strip()
_BASE_URL = _required_env("OPENAI_BASE_URL")
_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# The planner runs on the EXPENSIVE cloud route and re-sends the whole transcript each turn, so the
# instruction pushes hard on minimizing paid round-trips. `{browser_block}` is filled with the
# Playwright paragraph only when that opt-in toolset is attached (see mk_agent) — otherwise the prompt
# never advertises an unavailable tool and the prefix stays smaller.
_INSTRUCTION_TMPL = """\
You are the TitanOfIndustry coding orchestrator. You PLAN, DELEGATE, and VERIFY; you do NOT edit files yourself.

Models a subagent can be pinned to (create_subagent `model_name`):
- "openai/qwen-codex" — the fast text/code model (the default; what you run on). Use it for all code,
  tests, and reasoning.
- "openai/qwen-vl" — the VISION model. Use ONLY when a task requires you to SEE an image, screenshot,
  diagram, PDF page, or rendered UI. It runs on a separate engine that is woken on demand and briefly
  reduces text throughput, so reach for it only when vision is genuinely needed.

Workflow — you are the EXPENSIVE cloud brain and every turn re-sends the whole transcript, so
minimize round-trips: plan once, delegate fat self-verifying units of work, and come back only when a
unit is done or genuinely blocked.
- Before planning, call `recall` ONCE to check shared memory for relevant prior decisions/facts — but
  only if the task plausibly builds on prior work; SKIP it for small, self-contained tasks.
- Use `think`/`plan` to lay out the WHOLE change up front in as few calls as possible — call these
  tools (do NOT write reasoning or plans in plain text), and do not re-plan between every edit. One
  planning pass is enough for a small change; don't call both `think` and `plan` for it.
- `opencode_run` is your executor and is itself an autonomous agent on the local model (free): it can
  edit files, run shell commands, run the tests/build, read the output, and fix failures on its own.
  So delegate a COMPLETE, SELF-VERIFYING unit of work in ONE call — state the goal, the target
  file(s), the `cwd`, HOW to verify it (e.g. "run pytest" / "run the build"), and tell it to fix until
  green — then read the single result. Do NOT issue one `opencode_run` per tiny edit with a separate
  turn to verify, and NEVER run it then immediately re-run it to fix the same unit — tell it to fix
  until green in that one call and let OpenCode loop locally instead of bouncing back to you.
- Split into separate `opencode_run` calls only what is LOGICALLY INDEPENDENT (the system bounds how
  many run at once); each such call must still be self-verifying.
- For VISUAL work (anything that needs you to look at an image/screenshot/diagram): reuse an existing
  viewer via `list_active_agents`, else `create_subagent(agent_name="viewer",
  model_name="openai/qwen-vl", ...)`, then `call_subagent_as_tool`. BATCH all your visual questions
  into as few viewer calls as possible — each switch to the vision model is costly. Never use the
  viewer for non-visual work.
{browser_block}- Keep only a SMALL number of subagents active at once; the system caps concurrency, so extra calls
  just queue.
- After a change is verified, call `remember` to store the durable decision/fact (concise, tagged)
  so future sessions and agents benefit.
- When everything is done and verified, summarize what changed and call `finish_task` right away — do
  not spend an extra turn just to confirm.

Coordinate on the Paperclip control plane (you run as a Paperclip task; creds are injected per-run —
if any `paperclip_*` tool returns a "no credentials" error, skip these steps and just do the work):
- Before mutating an issue you own, `paperclip_checkout` it; `paperclip_release` if you stop without
  finishing. A 409 means another agent owns it — pick different work and never retry the 409.
- Leave durable progress with `paperclip_post_comment`, and keep the issue current with
  `paperclip_update_issue` (status/priority, optional `comment`): `blocked` (naming who must act)
  when stuck, `in_review` when handing off, `done` only when verified and complete.
- Read your work and context with `paperclip_list_assignments` / `paperclip_get_issue`.
- Delegate independent or follow-up work with `paperclip_create_subtask` (set `parent_id`/`goal_id`,
  assign to another agent when appropriate). Always do the actual code edits via `opencode_run`.
Keep every instruction you send to a subagent or to OpenCode concrete and unambiguous.
"""

# Appended into _INSTRUCTION_TMPL only when the Playwright toolset is attached (opt-in).
_BROWSER_BLOCK = """\
- For INTERACTIVE BROWSER work (open pages, click/type, inspect live DOM/network state, or verify a
  running web app): create or reuse a dedicated browser subagent with `tools_list=["playwright"]`,
  then call it with concrete browser actions. Keep static image/screenshot/PDF inspection on the
  qwen-vl viewer; use Playwright only when the page itself must be driven or inspected live.
"""


def mk_agent(opensage_session_id: str) -> OpenSageAgent:
    """Build the root agent. Called by the OpenSage CLI (`opensage web --agent`).

    Args:
        opensage_session_id: Session id supplied by the OpenSage runtime.
    """
    approved_mcp_toolsets = build_approved_mcp_toolsets()
    playwright_on = any(ts.name == "playwright" for ts in approved_mcp_toolsets)
    instruction = _INSTRUCTION_TMPL.format(browser_block=_BROWSER_BLOCK if playwright_on else "")

    # Pin a stable per-session prompt-cache key so a cloud planner's long shared prefix (system
    # instruction + tool schemas + early history) routes to the same OpenAI cache across this thread's
    # turns. ADK forwards LiteLlm(**kwargs) into the completion call, so this reaches the model without
    # touching vendor/.
    # MEASURED (2026-06-23): on the live `chatgpt-codex` route (ChatGPT *subscription* via litellm's
    # chatgpt provider) this is INERT — that backend caches only stored stateful sessions (store=true +
    # previous_response_id, what the Codex CLI does), and litellm hard-forces store=False, so even a
    # stable ~10k-token prefix returns cached_tokens=0 (probe across 4 back-to-back calls). Kept because
    # it is harmless and becomes effective verbatim if the planner is repointed at a token-billed OpenAI
    # key, which DOES do automatic stateless prefix caching. Skipped on the local qwen route.
    cache_kwargs = (
        {"prompt_cache_key": f"opensage:{opensage_session_id}"}
        if _MODEL and "qwen" not in _MODEL
        else {}
    )
    model = LiteLlm(model=_MODEL, api_key=_API_KEY, base_url=_BASE_URL, **cache_kwargs)
    return OpenSageAgent(
        name="titanofindustry_coder",
        model=model,
        description="Plans coding tasks, delegates edits to OpenCode (local Qwen), and uses shared Neo4j memory.",
        instruction=instruction,
        # opencode_run delegates edits; think/plan structure reasoning; remember/recall are the
        # shared memory; create_subagent/list_active_agents/call_subagent_as_tool give role-pinned
        # specialists (notably a qwen-vl "viewer" for vision); the paperclip_* tools coordinate on
        # the Paperclip control plane (comment/status/checkout/delegate, creds injected per-run via
        # session state); finish_task signals completion.
        tools=[
            opencode_run,
            think,
            plan,
            recall,
            remember,
            create_subagent,
            list_active_agents,
            call_subagent_as_tool,
            paperclip_list_assignments,
            paperclip_get_issue,
            paperclip_post_comment,
            paperclip_update_issue,
            paperclip_checkout,
            paperclip_release,
            paperclip_create_subtask,
            *approved_mcp_toolsets,
            finish_task,
        ],
    )
