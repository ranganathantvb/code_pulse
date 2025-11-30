from typing import Any, Dict, List

from code_pulse.mcp.base import MCPClient


class GitClient(MCPClient):
    async def repo(self, owner: str, repo: str) -> Dict[str, Any]:
        return await self.get(f"/repos/{owner}/{repo}")

    async def pull_requests(self, owner: str, repo: str, state: str = "open") -> List[Dict[str, Any]]:
        return await self.get(f"/repos/{owner}/{repo}/pulls", params={"state": state})

    async def commits(self, owner: str, repo: str, branch: str = "main") -> List[Dict[str, Any]]:
        return await self.get(f"/repos/{owner}/{repo}/commits", params={"sha": branch})
