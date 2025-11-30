# Code Pulse

AI-ready Python service with MCP connectors (Git, SonarQube, Jira), pluggable agents, retrieval-augmented generation (RAG) with document upload, and simple memory management. Built with FastAPI and LangChain-friendly components.

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

### Local LLM (Ollama deepseek-coder)
- Install Ollama and pull the model: `ollama pull deepseek-coder:6.7b`
- (optional) set `OLLAMA_BASE_URL` or `OLLAMA_HOST` if not on the default `http://localhost:11434`
- The agent responder will automatically use the local model; if unavailable and `OPENAI_API_KEY` is set, it falls back to OpenAI.

## API (high level)
- `POST /ingest` upload files (multipart) with optional `namespace`
- `POST /rag/query` ask questions with `question` and `namespace`
- `POST /agents/run` run an agent with `task`, `tools` (git|sonar|jira|rag), `memory_key`, `namespace`
- `GET /memory/{memory_key}` inspect stored messages

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
If `OPENAI_API_KEY` is set, responses are LLM-generated via `langchain-openai`; otherwise a deterministic summary is returned.

## Configuration
Environment variables (see `.env.example`):
- `GIT_BASE_URL`, `GIT_TOKEN`
- `SONAR_BASE_URL`, `SONAR_TOKEN`
- `JIRA_BASE_URL`, `JIRA_USER_EMAIL`, `JIRA_API_TOKEN`
- `RAG_EMBEDDINGS_MODEL` (defaults to `sentence-transformers/all-MiniLM-L6-v2` via LangChain)
- `DATA_DIR` (defaults to `.data`)

## Tests
```bash
pytest -q
```
`tests/test_smoke.py` covers the API health check and memory round-trip. Add integration tests for RAG/agents once credentials and models are available.

## Notes
- Networked APIs are thin wrappers; enable/disable per deployment.
- RAG artifacts and memory live under `.data` by default.
- Extend agents or connectors by adding new tool classes in `code_pulse/agents/tools`.
