# TitanOfIndustry service launchers.
# Bring up the backends (neo4j + vLLM chat + vLLM embed), then the agents (opencode -> opensage ->
# paperclip). Each runs in the foreground unless noted; use one terminal per service.
.PHONY: help up stack stack-down neo4j neo4j-stop memory-graph memory-dream memory-embed-backfill memory-mcp memory-migrate embed-cpu research-intern vllm vllm-keepwarm vllm-text vllm-vision proxy model-stack vllm-embed opencode opensage opensage-sessions paperclip voice voice-check codex-auth-link upstream-digest
.DEFAULT_GOAL := help

REPO_ROOT := $(CURDIR)

# Load .env (gitignored) so endpoints / model ids / NEO4J_PASSWORD / VLLM_* reach the recipes.
ifneq (,$(wildcard ./.env))
include .env
export
endif

# Session resume (opt-in): `make opensage RESUME=last` (most recent) or `RESUME=<id>` (specific id).
RESUME ?=
_RESUME := $(if $(RESUME),$(if $(filter last latest 1,$(RESUME)),--resume,--resume-from $(RESUME)),)

help up:
	@echo "TitanOfIndustry services — start the backends first, then the agents:"
	@echo "  make neo4j       # Neo4j Community (shared memory)"
	@echo "  make memory-graph# (re)build topic clusters + importance (free GDS, on-demand)"
	@echo "  make memory-dream# consolidate: decay, dedup, reflect, contradict, forget (run AFTER memory-graph)"
	@echo "  make embed-cpu   # CPU bge-base embeddings for memory (no VRAM)"
	@echo "  make memory-embed-backfill# embed any memories still missing a vector (after starting embed-cpu)"
	@echo "  make memory-mcp  # serve remember/recall over MCP for non-OpenSage agents"
	@echo "  make memory-migrate# backfill company/agent scope on existing memories (one-off; ARGS='--dry-run')"
	@echo "  make research-intern# the intern: one idle-time research cycle (ARGS='--dry-run'); loop = systemd unit"
	@echo "  make vllm        # qwen-codex AWQ chat (CUDA graphs on)"
	@echo "  make vllm-keepwarm# keep the text engine resident in VRAM (WSL anti-paging; run in bg)"
	@echo "  --- dual-model (vision) path — use INSTEAD of 'make vllm' ---"
	@echo "  make model-stack # model-manager owns text+vision engines (C3b)"
	@echo "  make proxy       # LiteLLM gateway: routing + concurrency cap"
	@echo "  #   vllm-text / vllm-vision run BY the manager (or standalone to debug)"
	@echo "  make vllm-embed  # bge-base embeddings (GPU; for memory, optional)"
	@echo "  make opencode    # OpenCode headless server (optional)"
	@echo "  make opensage    # OpenSage web (the orchestrator)"
	@echo "  make paperclip   # Paperclip governance plane"
	@echo "  make voice-check # voice ingress diagnostics redacted in this snapshot"
	@echo "  make codex-auth-link # keep codex_local agents authed (auto-symlink shared auth; ARGS='--once')"
	@echo ""
	@echo "  make stack       # bring the WHOLE stack up (Ollama + memory-mcp; tmux + health gates)"
	@echo "  make stack-down  # tear the whole stack down"
	@echo "       skip memory-mcp: make stack ARGS='--no-memory-mcp'  ·  WSL vLLM: make stack ARGS='--with-vllm'"
	@echo "       pass through: make stack RESUME=last | make stack ARGS='--with-opencode --attach'"
	@echo ""
	@echo "Resume a prior OpenSage thread: make opensage RESUME=last (or RESUME=<id>);"
	@echo "list saved ids with: make opensage-sessions"
	@echo "Memory falls back to keyword search if vllm-embed is down (it's optional)."

stack:
	bash scripts/start-stack.sh $(if $(RESUME),--resume $(RESUME),) $(ARGS)

stack-down:
	bash scripts/start-stack.sh --down

memory-mcp:
	uv run --project vendor/opensage-adk --with mcp --with uvicorn --with starlette \
		python -m agent.memory_mcp

memory-migrate:
	uv run --project vendor/opensage-adk python -m agent.migrate_memory_scope $(ARGS)

neo4j:
	docker compose -f docker-compose.neo4j.yml up -d

neo4j-stop:
	docker compose -f docker-compose.neo4j.yml down

memory-graph:
	uv run --project vendor/opensage-adk python -m agent.memory_graph

# Dreaming: offline consolidation over the :Memory store. Run AFTER `make memory-graph` (it reads
# topic/importance/SIMILAR_TO). LLM stages use the LOCAL qwen-codex via the proxy — never the planner.
memory-dream:
	uv run --project vendor/opensage-adk python -m agent.memory_dream

# One-off / nightly: embed any :Memory still missing a vector (e.g. the first time embed-cpu is up).
memory-embed-backfill:
	uv run --project vendor/opensage-adk python -m agent.memory_dream --backfill-only

# CPU embedder for semantic memory (bge-base, 768-dim) — does not touch the 3090.
embed-cpu:
	bash scripts/serve-embed-cpu.sh

# The intern: one-shot research cycle (testing / operators). Picks one beat, runs it, prints JSON.
# `ARGS='--dry-run'` computes findings WITHOUT writing memory or filing drafts. The CONTINUOUS loop is
# the deploy/research-intern.service systemd unit (not a make target). Local qwen-codex — never the planner.
research-intern:
	uv run --project vendor/opensage-adk python -m agent.research_intern --once $(ARGS)

vllm:
	bash scripts/serve-vllm.sh

vllm-keepwarm:
	bash scripts/vllm-keepwarm.sh

vllm-text:
	bash scripts/serve-vllm-engine.sh text

vllm-vision:
	bash scripts/serve-vllm-engine.sh vision

# UsageLogger [usage] lines (proxy/hooks.py) tee to logs/proxy-<date>.log so cloud-planner burn +
# prompt-cache hit rate survive a restart (the tmux pane keeps only ~24 lines of scrollback).
proxy:
	cd $(REPO_ROOT) && mkdir -p $(REPO_ROOT)/logs && PYTHONPATH=$(REPO_ROOT) \
		OLLAMA_BASE_URL=$${OLLAMA_BASE_URL:?set OLLAMA_BASE_URL in .env} \
		OLLAMA_API_KEY=$${OLLAMA_API_KEY:-ollama} \
		CHATGPT_TOKEN_DIR=$${CHATGPT_TOKEN_DIR:-$$HOME/.config/litellm/chatgpt} \
		uv run --with 'litellm[proxy]' \
		litellm --config proxy/litellm-proxy.yaml --port $${PROXY_PORT:?set PROXY_PORT in .env} 2>&1 \
		| tee -a $(REPO_ROOT)/logs/proxy-$$(date +%F).log

model-stack:
	cd $(REPO_ROOT) && uv run --project vendor/opensage-adk --with fastapi --with uvicorn --with httpx \
		python scripts/model_manager.py

vllm-embed:
	bash scripts/serve-vllm-embed.sh

opencode:
	OPENCODE_CONFIG=$(REPO_ROOT)/opencode/opencode.json \
		opencode serve --port $${OPENCODE_PORT:?set OPENCODE_PORT in .env}

opensage:
	cd vendor/opensage-adk && \
		uv run opensage web --agent $(REPO_ROOT)/agent --port $${OPENSAGE_PORT:?set OPENSAGE_PORT in .env} $(_RESUME)

opensage-sessions:
	@ls -1t ~/.local/opensage/sessions 2>/dev/null || echo "(no saved sessions yet)"

paperclip:
	cd vendor/paperclip && pnpm dev

# Self-hosted voice sidecar (Pipecat): whisper-cpu -> qwen-voice (proxy) -> kokoro-cpu.
# Used when VOICE_PROVIDER=pipecat (paperclip .env); ElevenLabs remains the default provider.
voice:
	bash scripts/serve-voice.sh

# Voice ingress diagnostics were redacted from the public interview snapshot.
voice-check:
	bash scripts/voice-check.sh

# On-demand backstop for codex_local auth: symlink the shared codex auth.json into every agent's
# isolated CODEX_HOME (and clear any 401). The Paperclip fork now seeds this at agent-create time, so
# this is a repair tool — use ARGS='--once' to fix already-stuck agents or re-link after the shared
# auth moves. Bare loop form needs Paperclip reachable through PAPERCLIP_BASE_URL.
codex-auth-link:
	bash scripts/codex-auth-link.sh $(ARGS)

# Upstream lifeline for the vendored (folded-in) Paperclip: fetch the read-only paperclip-upstream
# remote and list new commits since the baseline we've synced to (vendor/paperclip/.upstream-base).
# See vendor/paperclip/MAINTAINING.md for how to cherry-pick a fix in and bump the baseline.
upstream-digest:
	@git remote get-url paperclip-upstream >/dev/null 2>&1 || \
		{ echo "no 'paperclip-upstream' remote; add it with:"; \
		  echo "  git remote add paperclip-upstream https://github.com/paperclipai/paperclip"; exit 1; }
	@git fetch --quiet paperclip-upstream
	@base=$$(cat vendor/paperclip/.upstream-base); \
		echo "baseline (.upstream-base): $$base"; \
		n=$$(git rev-list --count $$base..paperclip-upstream/master); \
		echo "new upstream commits on paperclip-upstream/master: $$n"; \
		echo "----------------------------------------------------------------"; \
		git log --oneline --no-merges $$base..paperclip-upstream/master
