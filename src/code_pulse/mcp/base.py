import httpx
from typing import Any, Dict, Optional

from code_pulse.logger import setup_logging

logger = setup_logging(__name__)


class MCPClient:
    def __init__(self, base_url: str, token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "MCPClient":
        headers = {"User-Agent": "code-pulse/0.1.0"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=20.0)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client:
            await self._client.aclose()

    async def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        assert self._client, "Client not initialized"
        response = await self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    async def post(
        self, path: str, data: Optional[Dict[str, Any]] = None, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        assert self._client, "Client not initialized"
        response = await self._client.post(path, data=data, json=json)
        response.raise_for_status()
        return response.json()


def default_clients() -> Dict[str, MCPClient]:
    from code_pulse.mcp.clients import client_factory  # lazy to avoid circular import

    return client_factory()
