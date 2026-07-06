# TitanOfIndustry-sharable

This is a sanitized, interview-safe snapshot of a local-first agent stack. It
contains the glue code and configuration that connect:

- Paperclip as the governance and agent-control plane.
- OpenSage as the planner/orchestrator.
- OpenCode as the local code executor.
- LiteLLM as the local model gateway.
- Neo4j as shared memory.
- An optional Pipecat sidecar for the Mergatroid voice agent.

See `README.md` for setup and run instructions. This file is intentionally
brief; live operator details, private ingress setup, local port assignments,
prompts, company-specific workflows, and deployment hostnames are redacted from
this public snapshot.

## Repository Layout

- `agent/` contains the custom OpenSage agent glue, memory tools, Paperclip
  control-plane tools, and optional research intern.
- `opencode/`, `proxy/`, `scripts/`, `docker-compose.neo4j.yml`, and
  `.env.example` contain local configuration and launch wiring.
- `voice/` contains the self-hosted Pipecat implementation for the Mergatroid
  voice agent.
- `vendor/opensage-adk/` is an upstream OpenSage ADK submodule.
- `vendor/paperclip/` is a GitHub fork submodule of Paperclip, pinned to the
  sanitized shareable branch.

## Operating Notes

- Keep OpenSage, Paperclip, OpenCode, model servers, and Neo4j separated by
  their normal process/API boundaries.
- Keep endpoint values, local ports, model endpoints, credentials, and private
  hostnames in `.env`; do not hardcode them in source files.
- The default OpenSage agent and Mergatroid voice configuration are designed
  for a local OpenAI-compatible LLM backend. They will not work out of the box
  without one.
- The `agent/` package should stay thin. Prefer existing component CLIs and
  APIs over reimplementing model serving, database behavior, or orchestration.
- Never commit secrets, local `.env` files, generated auth material, private
  screenshots, or live deployment details.

## Verification

Use the repository-level checks before sharing changes:

```bash
git submodule update --init --recursive
python3 -m compileall agent scripts voice
gitleaks detect --source . --redact --exit-code 1 --no-banner
trufflehog git file://$(pwd) --only-verified --no-update --fail --json
```

The submodules are external projects and may have their own test fixtures or
upstream history. Review them independently before changing their visibility.
