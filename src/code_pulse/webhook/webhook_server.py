import hmac
import hashlib
import os
import logging
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Header, HTTPException

load_dotenv()

app = FastAPI()
logger = logging.getLogger(__name__)

GITHUB_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


AGENT_API_URL = os.getenv("AGENT_API_URL", "http://127.0.0.1:8000/agents/run")
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")       # If you want to secure the agent
AGENT_API_TIMEOUT = _get_float_env("AGENT_API_TIMEOUT", 30.0)

WEBHOOK_PATH = "/webhook-9d83cbbcf7f1"  # random path to hide endpoint


def _truncate(value: bytes | str | None, limit: int = 1000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            value = str(value)
    return value if len(value) <= limit else f"{value[:limit]}... [truncated]"


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
def _resolve_project_key(owner: str, repo: str) -> str:
    return os.getenv("SONAR_PROJECT_KEY") or f"{owner}:{repo}"


async def call_agent(
    task: str,
    tools: Optional[List[str]] = None,
    tool_args: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Call your /agents/run API with the correct request format."""

    payload = {
        "task": task,
        "tools": tools or ["sonar"],
        "memory_key": "local-chat",
        "tool_args": tool_args or {
            "sonar": {"project_key": "ranganathantvb:svc_catalog"}
        },
    }

    headers = {"Content-Type": "application/json"}
    if AGENT_API_KEY:
        headers["X-Agent-Key"] = AGENT_API_KEY

    logger.info(
        "Sending agent request",
        extra={"agent_url": AGENT_API_URL, "task": task, "tools": payload["tools"]}
    )

    try:
        async with httpx.AsyncClient(timeout=AGENT_API_TIMEOUT) as client:
            resp = await client.post(AGENT_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Agent API timed out") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Agent API request failed: {exc}") from exc

    try:
        data = resp.json()
        logger.info(
            "Agent response received",
            extra={
                "status_code": resp.status_code,
                "agent_url": AGENT_API_URL,
                "keys": list(data.keys()),
            },
        )

        # Try different keys based on your API output
        if "answer" in data:
            return data["answer"]
        if "output" in data:
            return data["output"]
        if "comment" in data:
            return data["comment"]

        return str(data)
    except Exception:
        logger.warning(
            "Agent response was not JSON, returning raw text",
            extra={"status_code": resp.status_code, "agent_url": AGENT_API_URL},
        )
        return resp.text


# ------------------------------
# POST COMMENT BACK TO GITHUB PR
# ------------------------------
async def post_comment(owner: str, repo: str, pr_number: int, body: str):
    token = os.getenv("GIT_TOKEN") or os.getenv("GITHUB_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="GIT_TOKEN or GITHUB_TOKEN not set")

    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json={"body": body})
        resp.raise_for_status()


def _parse_agent_command(body: str) -> str | None:
    """Return the command payload after '/agent', or None if not present."""
    if not body:
        return None
    stripped = body.strip()
    if not stripped.lower().startswith("/agent"):
        return None
    command_text = stripped[len("/agent"):].strip()
    return command_text or None


def _with_completion_footer(agent_response: str) -> str:
    """Append a short completion note to the agent output."""
    footer = (
        "\n\n---\n"
        "✅ Agent run completed. If this took a bit, thanks for waiting."
    )
    return f"{agent_response}{footer}"


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
    logger.info("Webhook raw body %s", _truncate(raw_body, 1500))

    # 1 Validate GitHub signature
    logger.info("Validating GitHub signature")
    verify_github_signature(raw_body, x_hub_signature_256)

    # 2️ Read event
    payload = await request.json()
    repo_full = payload.get("repository", {}).get("full_name")
    pr_number = payload.get("pull_request", {}).get("number")
    issue_number = payload.get("issue", {}).get("number")
    logger.info(
        "Webhook request event=%s action=%s repo=%s pr=%s issue=%s",
        x_github_event,
        payload.get("action"),
        repo_full,
        pr_number,
        issue_number,
    )

    if x_github_event == "pull_request":
        action = payload.get("action")
        pr = payload.get("pull_request", {})
        repo = payload.get("repository", {})

        if action in ["opened", "synchronize", "reopened"]:
            logger.info("Processing PR event", extra={"action": action, "repo": repo.get("full_name")})
            pr_number = pr.get("number")
            repo_full = repo.get("full_name")  # "ranganathantvb/svc_catalog"
            owner, name = repo_full.split("/")

            # 3️ Call your agent (Sonar scan + git)
            logger.info("Calling agent for PR", extra={"owner": owner, "repo": name, "pr": pr_number})
            agent_output = await call_agent(
                f"Summarize Sonar issues and key risks for {owner}/{name}",
                tools=["git", "sonar"],
                tool_args={
                    "git": {"owner": owner, "repo": name, "pull_number": pr_number},
                    "sonar": {
                        "project_key": _resolve_project_key(owner, name),
                        "pull_request": pr_number,
                    },
                },
            )

            # 4️ Post comment back to PR
            logger.info("Posting agent summary comment to PR", extra={"owner": owner, "repo": name, "pr": pr_number})
            await post_comment(owner, name, pr_number, _with_completion_footer(agent_output))
    elif x_github_event == "issue_comment":
        action = payload.get("action")
        comment = payload.get("comment", {})
        issue = payload.get("issue", {})
        repo = payload.get("repository", {})

        # Only respond when someone leaves a command like "/agent do something"
        command_text = _parse_agent_command(comment.get("body", ""))
        if action in {"created", "edited"} and command_text:
            logger.info("Processing issue comment agent command", extra={"action": action})
            issue_number = issue.get("number")
            repo_full = repo.get("full_name", "")
            if not issue_number or "/" not in repo_full:
                raise HTTPException(status_code=400, detail="Missing issue or repo info")
            owner, name = repo_full.split("/", 1)

            # Acknowledge immediately so the user knows work is in progress
            ack = (
                f"👋 Received agent command: `{command_text}`.\n"
                "Running now; results will follow in a separate comment."
            )
            await post_comment(owner, name, issue_number, ack)

            logger.info(
                "Issue comment agent trigger",
                extra={
                    "repo": repo_full,
                    "issue": issue_number,
                    "comment_id": comment.get("id"),
                    "action": action,
                    "task": command_text,
                },
            )

            pr_number = issue_number if issue.get("pull_request") else None
            git_args: Dict[str, Any] = {"owner": owner, "repo": name}
            sonar_args: Dict[str, Any] = {"project_key": _resolve_project_key(owner, name)}
            if pr_number:
                git_args["pull_number"] = pr_number
                sonar_args["pull_request"] = pr_number

            logger.info("Calling agent for issue comment", extra={"owner": owner, "repo": name, "issue": issue_number, "pr": pr_number})
            agent_output = await call_agent(
                command_text,
                tools=["git", "sonar"],
                tool_args={"git": git_args, "sonar": sonar_args},
            )
            logger.info("Posting agent response to issue/PR", extra={"owner": owner, "repo": name, "issue": issue_number})
            await post_comment(owner, name, issue_number, _with_completion_footer(agent_output))

    response_payload = {"status": "ok"}
    logger.info("Webhook response %s", response_payload)
    return response_payload
