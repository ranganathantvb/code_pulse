# Code Pulse

AI-ready Python service with MCP connectors (Git, SonarQube, Jira), pluggable agents, retrieval-augmented generation (RAG) with document upload, and simple memory management. Built with FastAPI and LangChain-friendly components and tuned for local LLMs (Ollama).

## Features
- MCP-style connectors: Git metadata fetch, SonarQube issues, Jira search/create (token-based)
- Agents: tool-executing agent loop with memory and RAG-aware context provider
- RAG: upload docs, vectorize with FAISS, query endpoint; CSV/PDF/MD/TXT supported via LangChain loaders
- Memory: conversation store backed by JSONL for durability
- Config: .env driven, Pydantic settings, structured logging
- API-first: FastAPI server exposing ingestion, RAG query, agent runner, and memory inspection

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env  # fill tokens/URLs
uvicorn code_pulse.app:app --reload
```

## Run the API locally
- Install deps and copy `.env.example` as above, then fill in Git/Sonar/Jira tokens and any optional RAG settings.
- Start the API: `uvicorn code_pulse.app:app --host 0.0.0.0 --port 8000 --reload`
- OpenAPI docs live at `http://localhost:8000/docs`; you can exercise endpoints from there.
- Run tests with `pytest -q` once dependencies are installed.

### Local LLM (Ollama deepseek-coder)
- Install Ollama and pull the model: `ollama pull deepseek-coder:6.7b`
- (optional) set `OLLAMA_BASE_URL` or `OLLAMA_HOST` if not on the default `http://localhost:11434`
- The agent responder only uses the local model (no OpenAI fallback is wired in).

### UI chatbot (optional)
- No bundled UI is included; the API is ready to be wired to a lightweight chat frontend (e.g., React/Vite or simple HTMX). You can start from the OpenAPI-powered `/docs` page to exercise endpoints, then point a UI at `/agents/run` for chat-style interactions.
- Typical flow in a UI: maintain a `memory_key` per conversation, send user text as the `task`, set `tools` (e.g., `["git", "sonar", "rag"]`), and render the returned `answer` plus tool outputs.

## API endpoints
- `GET /health` — liveness check, returns `{"status": "ok"}`.
- `POST /ingest` — multipart file upload with optional `namespace` query param; returns chunked documents stored for RAG.
- `POST /rag/query` — JSON `{"question": "...", "namespace": "..."}`; returns matching chunks (`content` + `metadata`).
- `POST /agents/run` — JSON `{"task": "...", "tools": ["git","sonar","rag"], "memory_key": "session-id", "namespace": "default", "tool_args": {...}}`; returns agent answer and tool outputs.
- `GET /memory/{memory_key}` — returns stored conversation messages; `limit` query param defaults to 50.
- `GET /sonar/rules` — proxies SonarCloud rules with filters (`query`, `languages`, `severities`, `types`, `page`, `page_size`).

### Seed Sonar rules into RAG
If you already have exported Sonar rule JSON (e.g., `.data/ingest/sonarr_rule_data.json`), you can index it for RAG answers:
```bash
source .venv/bin/activate
python -m code_pulse.rag.seed_sonar_rules  # accepts Enter for default path/namespace
```
This stores chunks under the `sonar` namespace by default. When invoking the agent with the `sonar` tool, pass your question (and namespace if you changed it) in `tool_args` so it can pull RAG matches.

## RAG how-to (setup, train, run, test)
- **Install + env**: `python -m venv .venv && source .venv/bin/activate && pip install -e .`; copy `.env.example` to `.env` and fill tokens/URLs (Sonar/Git/Jira as needed).
- **Train (ingest) Sonar rules**: place your rules export at `.data/ingest/sonarr_rule_data.json` (or another path), then run `python -m code_pulse.rag.seed_sonar_rules` and accept defaults (namespace `sonar`). Re-run this command whenever you refresh rules.
- **Run API**: `uvicorn code_pulse.app:app --reload`; Swagger at `http://localhost:8000/docs`.
- **Direct RAG query**: after seeding, call `POST /rag/query` with `{"question": "How to fix TLS 1.0 issues?", "namespace": "sonar"}`.
- **Agent with Sonar RAG**: `POST /agents/run`:
  ```bash
  curl -X POST http://localhost:8000/agents/run \
    -H "Content-Type: application/json" \
    -d '{
          "task": "Explain how to remediate weak TLS in my Sonar findings",
          "tools": ["sonar"],
          "memory_key": "sonar-chat",
          "tool_args": {
            "sonar": {
              "question": "TLS 1.0 finding remediation",
              "namespace": "sonar"
            }
          }
        }'
  ```
  The agent will pull RAG matches from the `sonar` namespace; if nothing relevant is found it responds politely that guidance is unavailable.
- **Run tests**: `pytest -q` (after installing deps). Add more RAG tests as needed.

### RAG workflow
1. `POST /ingest` with files: `curl -F "files=@docs/guide.pdf" -F "namespace=demo" http://localhost:8000/ingest`
2. `POST /rag/query` with JSON: `{"question": "How to deploy?", "namespace": "demo"}`

### Agent workflow
```bash
curl -X POST http://localhost:8000/agents/run \
  -H "Content-Type: application/json" \
  -d '{
        "task": "Summarize code health and open PRs",
        "tools": ["git", "sonar", "rag"],
        "memory_key": "demo-session",
        "namespace": "demo",
        "tool_args": {
          "git": {"owner": "octocat", "repo": "Hello-World"},
          "sonar": {"project_key": "org:project"},
          "rag": {"question": "deployment checklist"}
        }
      }'
```
The responder will use your local Ollama model to craft the answer and include any tool outputs in the response.

### Issue/report workflow with an existing repo
Use the agent to scan repo metadata and issue sources, then ask it for a report:
```bash
curl -X POST http://localhost:8000/agents/run \
  -H "Content-Type: application/json" \
  -d '{
        "task": "Scan the repo issues and summarize key risks",
        "tools": ["git", "sonar"],
        "memory_key": "issue-audit",
        "tool_args": {
          "git": {"owner": "your-org", "repo": "your-repo"},
          "sonar": {"project_key": "your-org:your-repo"}
        }
      }'
```
If you also want RAG context from uploaded docs, add the `rag` tool and `namespace`.

## Configuration
Environment variables (see `.env.example`):
- `GIT_BASE_URL`, `GIT_TOKEN`
- `SONAR_BASE_URL`, `SONAR_TOKEN`, `SONAR_PROJECT_KEY`, `SONAR_ORGANIZATION`
- `JIRA_BASE_URL`, `JIRA_USER_EMAIL`, `JIRA_API_TOKEN`
- `RAG_EMBEDDINGS_MODEL` (defaults to `sentence-transformers/all-MiniLM-L6-v2` via LangChain)
- `DATA_DIR` (defaults to `.data`)
- Webhook/agent relay: `GITHUB_WEBHOOK_SECRET` (required), `GIT_TOKEN` or `GITHUB_TOKEN` (to comment on PRs), `AGENT_API_URL` (defaults to `http://127.0.0.1:8000/agents/run`), `AGENT_API_KEY` (if you secure `/agents/run`), `AGENT_API_TIMEOUT` (seconds, defaults to 30).

## GitHub webhook integration (PR + issue comments)
- Webhook path: `/webhook-9d83cbbcf7f1`. Point GitHub to `http(s)://<host>:9000/webhook-9d83cbbcf7f1` with content type `application/json` and the same `GITHUB_WEBHOOK_SECRET`.
- Subscribe to `Pull request` and `Issue comment` events:
  - PR events (`opened`, `synchronize`, `reopened`) trigger the agent with `git` + `sonar` tools and post a summary comment back to the PR.
  - Issue/PR comments containing `/agent <task>` trigger the agent with `git` + `sonar` (PR-aware when applicable) and post the response as a new comment.
- Start the webhook relay (after exporting secrets/tokens):
  ```bash
  export GITHUB_WEBHOOK_SECRET="long-random-string"
  export GITHUB_TOKEN="ghp_..."            # or GIT_TOKEN
  export AGENT_API_URL="http://127.0.0.1:8000/agents/run"
  export AGENT_API_KEY="super-secret-agent-key"  # only if /agents/run is secured
  uvicorn code_pulse.webhook.webhook_server:app --host 0.0.0.0 --port 9000 --reload
  ```
- For public access during local dev, run `ngrok http 9000` (or similar) and use the forwarded URL plus the webhook path in GitHub settings.
- The webhook server validates the GitHub signature (`X-Hub-Signature-256`) using `GITHUB_WEBHOOK_SECRET` before calling your agent API and posting PR/issue comments.

## Tests
```bash
pytest -q
```
`tests/test_smoke.py` covers the API health check and memory round-trip. Add integration tests for RAG/agents once credentials and models are available.

## Notes
- Networked APIs are thin wrappers; enable/disable per deployment.
- RAG artifacts and memory live under `.data` by default.
- Extend agents or connectors by adding new tool classes in `code_pulse/agents/tools`.



<!-- # 
 export GITHUB_WEBHOOK_SECRET="e102965ec2f74fb5ca721fb0fe843004401b4d6df8254208b1f3c554117a5054"   # same string in GitHub webhook
 export GITHUB_TOKEN=" <github_token>"                            # PAT to post PR comments
 export AGENT_API_URL="http://127.0.0.1:8000/agents/run"  # your agent endpoint
 export AGENT_API_KEY="super-secret-agent-key"            # used to protect agents run
 export AGENT_API_TIMEOUT=30                              # seconds to wait on the agent call from webhook
 export CODEPULSE_WORKSPACE_PATH="Users/ranganathan/workspace/svc_catalog"
 export OLLAMA_HOST="http://localhost:11434"   # or OLLAMA_BASE_URL -->
#export GITHUB_TOKEN=" <github_token>"
export CODEPULSE_TARGET_BRANCH="fix_code_quality_issues"  # branch to create PR against
export CODEPULSE_POST_PR_SUMMARY=true                    # post agent summary back to the PR as a comment (default on)
export CODEPULSE_FORCE_PUSH=true  
#### How to start 
<!-- set -a          # auto-export all variables
source .env
set +a -->

## terminal 1
 <!-- export GITHUB_WEBHOOK_SECRET="e102965ec2f74fb5ca721fb0fe843004401b4d6df8254208b1f3c554117a5054"   # same as in GitHub webhook
 export GITHUB_TOKEN="your-github-pat"                # PAT with repo access
 export AGENT_API_KEY="6cf488ac11da5c73fdd4e170401accea13e77762d7659e65f6d4874c2478d4f8"  # optional
 uvicorn webhook_server:app --host 0.0.0.0 --port 9000 --reload -->

## terminal 2
# ngrok http 9000  

## terminal 3
# uvicorn code_pulse.app:app --reload    

## terminal 4
# test using the curl