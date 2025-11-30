from typing import Any, Dict, List
import httpx

from code_pulse.logger import setup_logging
from code_pulse.mcp.base import MCPClient

logger = setup_logging(__name__)


class GitClient(MCPClient):
    async def __aenter__(self) -> "GitClient":
        # GitHub PATs expect the "token" scheme (Bearer is rejected for PATs).
        headers = {
            "User-Agent": "code-pulse/0.1.0",
            "Accept": "application/vnd.github+json",
        }
        if self.token:
            # Fine-grained tokens require Bearer; classic tokens accept both.
            headers["Authorization"] = f"Bearer {self.token}"
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=20.0)
        logger.info("Git client ready base_url=%s token_set=%s", self.base_url, bool(self.token))
        return self

    async def repo(self, owner: str, repo: str) -> Dict[str, Any]:
        logger.info("Fetching repo owner=%s repo=%s", owner, repo)
        return await self.get(f"/repos/{owner}/{repo}")

    async def pull_requests(self, owner: str, repo: str, state: str = "open") -> List[Dict[str, Any]]:
        return await self.get(f"/repos/{owner}/{repo}/pulls", params={"state": state})

    async def commits(self, owner: str, repo: str, branch: str = "main") -> List[Dict[str, Any]]:
        return await self.get(f"/repos/{owner}/{repo}/commits", params={"sha": branch})
