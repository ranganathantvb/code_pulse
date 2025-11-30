from typing import Any, Dict, List

from code_pulse.mcp.base import MCPClient


class SonarClient(MCPClient):
    async def project_issues(self, project_key: str, statuses: str = "OPEN") -> Dict[str, Any]:
        return await self.get(
            "/issues/search",
            params={"componentKeys": project_key, "statuses": statuses},
        )

    async def measures(self, project_key: str, metric_keys: str = "bugs,vulnerabilities,code_smells") -> List[Dict[str, Any]]:
        data = await self.get(
            "/measures/component",
            params={"component": project_key, "metricKeys": metric_keys},
        )
        return data.get("component", {}).get("measures", [])
