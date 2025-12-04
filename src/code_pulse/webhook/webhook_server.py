import hmac
import hashlib
import os
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request, Header, HTTPException

app = FastAPI()

GITHUB_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")
GIT_TOKEN = os.getenv("GIT_TOKEN", "")
AGENT_API_URL = "http://127.0.0.1:8000/agents/run"  # You gave this correctly
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")       # If you want to secure the agent

WEBHOOK_PATH = "/webhook-9d83cbbcf7f1"  # random path to hide endpoint


# ------------------------------
# SECURITY: VERIFY WEBHOOK SIGNATURE
# ------------------------------
def verify_github_signature(raw_body: bytes, signature_header: str | None) -> None:
    if not GITHUB_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret is missing")

    if not signature_header:
        raise HTTPException(status_code=401, detail="No signature header")

    algo, _, signature = signature_header.partition("=")
    if algo != "sha256":
        raise HTTPException(status_code=401, detail="Invalid signature algorithm")

    mac = hmac.new(GITHUB_SECRET.encode(), msg=raw_body, digestmod=hashlib.sha256)
    expected = mac.hexdigest()

    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="Invalid signature")


# ------------------------------
# CALL YOUR AGENT ENDPOINT
# ------------------------------
async def call_agent() -> str:
    """Call your /agents/run API with the correct request format."""

    payload = {
        "task": "Scan repo issues and summarize risks",
        "tools": ["git", "sonar"],
        "memory_key": "local-chat",
        "tool_args": {
            "git": {"owner": "ranganathantvb", "repo": "svc_catalog"},
            "sonar": {"project_key": "ranganathantvb:svc_catalog"}
        }
    }

    headers = {"Content-Type": "application/json"}
    if AGENT_API_KEY:
        headers["X-Agent-Key"] = AGENT_API_KEY

    async with httpx.AsyncClient() as client:
        resp = await client.post(AGENT_API_URL, json=payload, headers=headers)
        resp.raise_for_status()

    try:
        data = resp.json()

        # Try different keys based on your API output
        if "output" in data:
            return data["output"]
        if "comment" in data:
            return data["comment"]

        return str(data)
    except Exception:
        return resp.text


# ------------------------------
# POST COMMENT BACK TO GITHUB PR
# ------------------------------
async def post_comment(owner: str, repo: str, pr_number: int, body: str):
    if not GIT_TOKEN:
        raise RuntimeError("GIT_TOKEN not set")

    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {GIT_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json={"body": body})
        resp.raise_for_status()


# ------------------------------
# MAIN GITHUB WEBHOOK ENDPOINT
# ------------------------------
@app.post(WEBHOOK_PATH)
async def github_webhook(
    request: Request,
    x_github_event: str = Header(None),
    x_hub_signature_256: str | None = Header(None),
):
    raw_body = await request.body()

    # 1️⃣ Validate GitHub signature
    verify_github_signature(raw_body, x_hub_signature_256)

    # 2️⃣ Read event
    payload = await request.json()

    if x_github_event == "pull_request":
        action = payload.get("action")
        pr = payload.get("pull_request", {})
        repo = payload.get("repository", {})

        if action in ["opened", "synchronize", "reopened"]:
            pr_number = pr.get("number")
            repo_full = repo.get("full_name")  # "ranganathantvb/svc_catalog"
            owner, name = repo_full.split("/")

            # 3️⃣ Call your agent (Sonar scan + git)
            agent_output = await call_agent()

            # 4️⃣ Post comment back to PR
            await post_comment(owner, name, pr_number, agent_output)

    return {"status": "ok"}
