from typing import Any, Dict, Optional

from code_pulse.mcp.base import MCPClient


class SonarMCPClient(MCPClient):
    async def __aenter__(self) -> "SonarMCPClient":
        # Inherit base headers; no auth required for internal FastAPI calls.
        await super().__aenter__()
        return self

    async def rules(
        self,
        query: Optional[str] = None,
        languages: Optional[str] = None,
        severities: Optional[str] = None,
        types: Optional[str] = None,
        page: int = 1,
        page_size: int = 100,
    ) -> Dict[str, Any]:
        """Call into the local MCP server to retrieve Sonar rule metadata."""
        params: Dict[str, Any] = {
            "page": page,
            "page_size": page_size,
        }
        if query:
            params["query"] = query
        if languages:
            params["languages"] = languages
        if severities:
            params["severities"] = severities
        if types:
            params["types"] = types

        return await self.get(
            "/sonar/rules",
            params=params,
            purpose="Fetch cached Sonar rules via MCP",
        )
