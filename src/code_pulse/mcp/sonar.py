from typing import Any, Dict, List
import httpx

from code_pulse.mcp.base import MCPClient


class SonarClient(MCPClient):
    def __init__(self, base_url: str, token: str | None, organization: str | None = None):
        super().__init__(base_url, token)
        self.organization = organization

    def _component_key(self, raw_key: str) -> str:
        # SonarCloud project keys often use underscores instead of colons.
        if "sonarcloud.io" in self.base_url and ":" in raw_key:
            return raw_key.replace(":", "_")
        return raw_key

    async def __aenter__(self) -> "SonarClient":
        headers = {
            "User-Agent": "code-pulse/0.1.0",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=20.0)
        return self

    async def project_issues(self, project_key: str, statuses: str = "OPEN") -> Dict[str, Any]:
        component = self._component_key(project_key)
        params = {"componentKeys": component, "statuses": statuses}
        if self.organization:
            params["organization"] = self.organization
        return await self.get("/issues/search", params=params)

    async def measures(self, project_key: str, metric_keys: str = "bugs,vulnerabilities,code_smells") -> List[Dict[str, Any]]:
        component = self._component_key(project_key)
        params = {"component": component, "metricKeys": metric_keys}
        if self.organization:
            params["organization"] = self.organization
        data = await self.get("/measures/component", params=params)
        return data.get("component", {}).get("measures", [])
