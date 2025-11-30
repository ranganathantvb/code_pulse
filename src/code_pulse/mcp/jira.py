from typing import Any, Dict, Optional
import httpx

from code_pulse.config import get_settings
from code_pulse.mcp.base import MCPClient


class JiraClient(MCPClient):
    def __init__(self, base_url: str, api_token: Optional[str], user_email: Optional[str]):
        super().__init__(base_url, api_token)
        self.user_email = user_email

    async def __aenter__(self) -> "JiraClient":
        headers = {"User-Agent": "code-pulse/0.1.0"}
        auth = None
        if self.token and self.user_email:
            auth = (self.user_email, self.token)
        self._client = httpx.AsyncClient(
            base_url=self.base_url.rstrip("/") + "/rest/api/3",
            headers=headers,
            auth=auth,
            timeout=20.0,
        )
        return self

    async def search(self, jql: str, max_results: int = 25) -> Dict[str, Any]:
        return await self.get("/search", params={"jql": jql, "maxResults": max_results})

    async def create_issue(
        self, project_key: str, summary: str, issue_type: str = "Task", description: str = ""
    ) -> Dict[str, Any]:
        payload = {
            "fields": {
                "project": {"key": project_key},
                "summary": summary,
                "description": description,
                "issuetype": {"name": issue_type},
            }
        }
        return await self.post("/issue", json=payload)


def default_jira_client() -> JiraClient:
    settings = get_settings()
    return JiraClient(settings.jira_base_url, settings.jira_api_token, settings.jira_user_email)
