# TitanOfIndustry-sharable

TitanOfIndustry-sharable is an public-safe snapshot of a
multi-agent operations stack for autonomous workflows. It wires together
Paperclip, OpenSage, OpenCode, LiteLLM, local Qwen/Ollama/vLLM models, and Neo4j
memory so agents can plan work, delegate coding tasks, coordinate through issues,
remember prior decisions, and optionally run through a voice interface.

It includes a custom local multi-agent stack of an orchestrator, an executor, shared
memory, and an optional voice sidecar:
```text
Paperclip UI/control plane
  -> OpenSage orchestrator
    -> OpenCode executor
      -> LiteLLM proxy
        -> local LLM backend, usually Ollama
  -> Neo4j shared memory
  -> optional Pipecat voice sidecar
```
This stack runs as single agent to assign tasks and interact with other agents.

This repository is the glue layer: configuration, launch scripts, OpenSage agent
code, memory tools, Paperclip integration, and local service wiring. It also
heavily customizes the forked repos and implements a mobile app to manage the platform.

## Sanitized Snapshot

Production prompts, company names, live deployment details, screenshots, local
paths, capital amounts, and other sensitive material have been removed or
redacted. The project still preserves the architecture and the code paths that
show how the system is assembled.

Important submodules:

| Path | Source |
| --- | --- |
| `vendor/opensage-adk/` | OpenSage ADK upstream submodule. |
| `vendor/paperclip/` | A true GitHub fork of `paperclipai/paperclip`, pinned to the sanitized `main` branch at `IAmMarcellus/paperclip`. |

If a clone was not made with `--recurse-submodules`, initialize the submodules:

```bash
git submodule update --init --recursive
```

## What Runs

The normal local stack starts these services. Local port assignments and
loopback URLs are intentionally omitted from this public snapshot; configure
them in `.env` from `.env.example`.

| Service | Purpose |
| --- | --- |
| Neo4j | Shared `remember`/`recall` memory store |
| LiteLLM proxy | One OpenAI-compatible endpoint for local models, routing, and concurrency caps |
| OpenSage | Planner/orchestrator and role-pinned subagents |
| Paperclip | Governance plane, agent control, runs, issues, approvals |
| memory-mcp | Optional MCP wrapper around the shared memory tools |
| voice sidecar | Optional self-hosted Pipecat voice pipeline |

The default OpenSage agent and the default self-hosted voice-agent configuration
are designed for a local LLM. They expect the LiteLLM proxy to expose model
aliases such as `qwen-codex`, `qwen-vl`, and `qwen-voice`. They will not work
out of the box on a machine that does not have a compatible local LLM backend
configured.

## Requirements

Install these first:

- `git`
- `uv`
- Docker with Docker Compose
- Node.js 20+
- `corepack`/`pnpm`
- `tmux`
- `opencode-ai`
- A local OpenAI-compatible LLM backend, usually Ollama

The checked-in proxy config assumes Ollama is reachable through
`OLLAMA_BASE_URL` and can serve the model aliases used by the stack. If your
local model names differ, update `proxy/litellm-proxy.yaml`, `.env`, and
`opencode/opencode.json` together.

The optional `chatgpt-codex` proxy route uses a ChatGPT/Codex cloud account and
is not required for the default sanitized configuration.

## Setup

From the repository root:

```bash
git submodule update --init --recursive
cp .env.example .env
```

Edit `.env` at minimum:

```bash
NEO4J_PASSWORD=<choose-a-local-password>
OPENAI_BASE_URL=<your-local-openai-compatible-endpoint>
OLLAMA_BASE_URL=<your-local-ollama-endpoint>
```

Install Python dependencies for OpenSage and the agent:

```bash
cd vendor/opensage-adk
uv python install
uv sync
cd ../..
```

Install JavaScript dependencies:

```bash
npm install -g opencode-ai
corepack enable
corepack prepare pnpm@9.15.4 --activate

cd vendor/paperclip
pnpm install
cd ../..
```

Make sure your local model backend is running before starting the stack. For the
default config, the proxy must be able to route:

- `qwen-codex` for OpenSage, OpenCode, memory dreaming, and research intern work
- `qwen-vl` for vision tasks
- `qwen-voice` for the Pipecat voice sidecar

## Run The Stack

Start everything in dependency order:

```bash
make stack
```

This starts a `tmux` session named `titanofindustry`, which is the service session
name retained by the launch scripts. Attach to logs with:

```bash
tmux attach -t titanofindustry
```

Access the Paperclip website after the stack is up:

```bash
set -a; . ./.env; set +a
echo "${PAPERCLIP_BASE_URL:-http://localhost:${PAPERCLIP_PORT}}"
```

Open the printed URL in a browser. The Paperclip service is the web
governance/control-plane UI; the exact host and port come from your private
`.env` file. If you expose Paperclip through a LAN hostname, Tailscale name, or
tunnel, set `PAPERCLIP_BASE_URL` to that URL and include the hostname in
`PAPERCLIP_ALLOWED_HOSTNAMES`.

`make stack` does not start the Expo mobile app. To run it against the same
Paperclip backend, start it in a second terminal after the stack is up:

```bash
set -a; . ./.env; set +a
export EXPO_PUBLIC_API_BASE_URL="${PAPERCLIP_BASE_URL:-http://localhost:${PAPERCLIP_PORT}}"

cd vendor/paperclip/mobile
pnpm start
```

Then open the app from the Expo/Metro prompt. For a physical phone, use a URL
the phone can reach, such as a LAN IP, Tailscale hostname, or tunnel URL,
instead of `localhost`; Android emulators usually need `http://10.0.2.2:<port>`.
If you need a native dev build, run `pnpm ios` or `pnpm android` from
`vendor/paperclip/mobile`. For an iOS build that reaches a plain `http://`
non-localhost hostname, set `EXPO_PUBLIC_ATS_INSECURE_DOMAIN` to that hostname
before running the app.

Stop the stack:

```bash
make stack-down
```

Run individual services when debugging:

```bash
make neo4j
make proxy
make opensage
make paperclip
```

Service URLs are determined by the local `.env` file and are not published in
this snapshot.

## Voice Sidecar

The optional voice path is a self-hosted Pipecat service:

```bash
make voice
```

It uses CPU Whisper for speech-to-text, CPU Kokoro for text-to-speech, and the
local `qwen-voice` model alias through the LiteLLM proxy for the spoken
assistant brain. Without a working local LLM route for `qwen-voice`, the voice
agent will not work.

Paperclip uses this sidecar when its voice provider is set to `pipecat`. The
default environment comments in `.env.example` document the relevant
`VOICE_*` settings.

## Verify

After `make stack`, basic checks are:

```bash
curl -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  "${OPENAI_BASE_URL}/models"

make opensage-sessions
```

In Paperclip, create or open an agent using the `opensage` adapter. The default
adapter config expects OpenSage to be reachable at the local endpoint configured
for your machine.

Give the agent a small task. Paperclip should stream the run from OpenSage, and
OpenSage should call OpenCode for edits through the local model route.

## Memory

Shared memory is backed by Neo4j. The OpenSage agent can `remember` and
`recall`; those facts persist across OpenSage sessions. Optional maintenance
targets:

```bash
make memory-graph
make embed-cpu
make memory-embed-backfill
make memory-dream
make memory-mcp
```

`memory-dream` and the research intern are designed to use the local
`qwen-codex` worker through the proxy, not a cloud planner.

## Research Intern

The research intern is an optional idle-time scout:

```bash
make research-intern ARGS='--dry-run'
```

It reads beats from `agent/intern_beats.toml`, uses the local model route, and
can write low-confidence candidate findings into memory. Keep
`INTERN_ENABLE_WEB=0` and `INTERN_ENABLE_DRAFTS=0` until you have inspected the
behavior locally.

## Main Paths

| Path | Purpose |
| --- | --- |
| `agent/` | OpenSage agent factory, OpenCode glue, memory tools, plugins, and intern. |
| `proxy/` | LiteLLM proxy config and hooks. |
| `opencode/opencode.json` | OpenCode provider config for the local proxy. |
| `scripts/start-stack.sh` | `make stack` orchestration. |
| `voice/` | Self-hosted Pipecat voice sidecar. |
| `deploy/` | Redacted service/unit examples for the sanitized snapshot. |
| `vendor/opensage-adk/` | OpenSage submodule. |
| `vendor/paperclip/` | Sanitized Paperclip fork submodule. |

## Notes For Reviewers

This snapshot is meant to demonstrate architecture and integration work, not to
be a turnkey SaaS install. It needs local credentials, local model availability,
and local service setup before the agents can run end to end.

The most important setup constraint is the local LLM backend. Paperclip can
start without one, but OpenSage agents, OpenCode execution, memory dreaming, the
research intern, and the default voice agent configuration all depend on a
working local model route through the LiteLLM proxy.
