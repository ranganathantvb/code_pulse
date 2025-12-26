from typing import Any, Dict, List, Optional
import httpx

from code_pulse.mcp.base import MCPClient


class SonarClient(MCPClient):
    def __init__(self, base_url: str, token: str | None, organization: str | None = None):
        # SonarCloud can scope requests by organization; store so each call can include it.
        super().__init__(base_url, token)
        self.organization = organization

    def _component_key(self, raw_key: str) -> str:
        # SonarCloud project keys often use underscores instead of colons; normalize to match API expectations.
        if "sonarcloud.io" in self.base_url and ":" in raw_key:
            return raw_key.replace(":", "_")
        return raw_key

    async def __aenter__(self) -> "SonarClient":
        headers = {"User-Agent": "code-pulse/0.1.0"}
        auth = httpx.BasicAuth(self.token, "") if self.token else None
        # SonarQube/SonarCloud expect token via HTTP Basic with empty password instead of bearer tokens.
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, auth=auth, timeout=20.0)
        return self

    async def project_issues(self, project_key: str, statuses: str = "OPEN") -> Dict[str, Any]:
        """List issues on a project, defaulting to open issues only."""
        component = self._component_key(project_key)
        params = {"componentKeys": component, "statuses": statuses}
        if self.organization:
            params["organization"] = self.organization
        return await self.get(
            "/issues/search",
            params=params,
            purpose="Fetch project issues from Sonar",
        )

    async def pr_issues(
        self,
        project_key: str,
        pull_request: str,
        statuses: str = "OPEN",
        issue_types: str | None = None,
    ) -> Dict[str, Any]:
        """List issues on a pull request, optionally narrowed by type (BUG, VULNERABILITY, etc.)."""
        component = self._component_key(project_key)
        params: Dict[str, Any] = {
            "componentKeys": component,
            "pullRequest": pull_request,
            "statuses": statuses,
        }
        if issue_types:
            params["types"] = issue_types
        if self.organization:
            params["organization"] = self.organization
        return await self.get(
            "/issues/search",
            params=params,
            purpose="Fetch pull request issues from Sonar",
        )

    async def pr_hotspots(
        self,
        project_key: str,
        pull_request: str,
        status: str | None = None,
    ) -> Dict[str, Any]:
        """Fetch security hotspots tied to a pull request."""
        component = self._component_key(project_key)
        params: Dict[str, Any] = {
            "projectKey": component,
            "pullRequest": pull_request,
        }
        if status:
            params["status"] = status
        if self.organization:
            params["organization"] = self.organization
        return await self.get(
            "/hotspots/search",
            params=params,
            purpose="Fetch pull request security hotspots from Sonar",
        )

    async def pr_quality_gate(self, project_key: str, pull_request: str) -> Dict[str, Any]:
        """Return the quality gate result for a given pull request."""
        component = self._component_key(project_key)
        params: Dict[str, Any] = {"projectKey": component, "pullRequest": pull_request}
        if self.organization:
            params["organization"] = self.organization
        return await self.get(
            "/qualitygates/project_status",
            params=params,
            purpose="Fetch pull request quality gate status",
        )

    async def measures(self, project_key: str, metric_keys: str = "bugs,vulnerabilities,code_smells") -> List[Dict[str, Any]]:
        """Return project-level aggregate metrics (bugs, vulnerabilities, code smells by default)."""
        component = self._component_key(project_key)
        params = {"component": component, "metricKeys": metric_keys}
        if self.organization:
            params["organization"] = self.organization
        data = await self.get(
            "/measures/component",
            params=params,
            purpose="Fetch project measures from Sonar",
        )
        return data.get("component", {}).get("measures", [])

    async def projects(
        self,
        query: Optional[str] = None,
        page: int = 1,
        page_size: int = 100,
    ) -> Dict[str, Any]:
        """Search Sonar projects with optional text query and pagination."""
        params: Dict[str, Any] = {"p": page, "ps": page_size}
        if query:
            params["q"] = query
        if self.organization:
            params["organization"] = self.organization
        return await self.get(
            "/projects/search",
            params=params,
            purpose="Search projects in Sonar",
        )

    async def rules(
        self,
        query: Optional[str] = None,
        languages: Optional[str] = None,
        severities: Optional[str] = None,
        types: Optional[str] = None,
        page: int = 1,
        page_size: int = 100,
    ) -> Dict[str, Any]:
        """Fetch Sonar rules for downstream ingestion (e.g., RAG training)."""
        params: Dict[str, Any] = {"p": page, "ps": page_size}
        if query:
            params["q"] = query
        if languages:
            params["languages"] = languages
        if severities:
            params["severities"] = severities
        if types:
            params["types"] = types
        if self.organization:
            params["organization"] = self.organization

        try:
            data = await self.get(
                "/rules/search",
                params=params,
                purpose="Fetch Sonar rules",
            )
        except httpx.HTTPStatusError as exc:
            # Enrich the default HTTP error with Sonar's response payload to make failures actionable.
            detail = ""
            try:
                payload = exc.response.json()
                if isinstance(payload, dict) and "errors" in payload:
                    detail = "; ".join(
                        e.get("msg", "") for e in payload.get("errors", []) if isinstance(e, dict)
                    )
            except Exception:  # noqa: BLE001
                detail = exc.response.text

            if self.organization and exc.response.status_code == 404:
                raise ValueError(
                    f"Sonar organization '{self.organization}' not found or inaccessible: {detail}"
                ) from exc
            if self.organization and exc.response.status_code == 400 and "organization" in detail.lower():
                raise ValueError(
                    f"Sonar organization '{self.organization}' is required or invalid: {detail}"
                ) from exc
            raise

        return {
            "rules": data.get("rules", []),
            "paging": {
                "page": data.get("p", page),
                "page_size": data.get("ps", page_size),
                "total": data.get("total", len(data.get("rules", []))),
            },
        }
