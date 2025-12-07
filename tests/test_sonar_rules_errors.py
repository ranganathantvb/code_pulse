import json
import httpx
import pytest

from code_pulse.mcp.sonar import SonarClient


@pytest.mark.asyncio
async def test_rules_org_not_found(monkeypatch):
    async def fake_get(self, path, params=None):  # noqa: ARG001
        request = httpx.Request("GET", "https://sonarcloud.io/api/rules/search")
        content = json.dumps({"errors": [{"msg": "Organization does not exist"}]}).encode()
        response = httpx.Response(404, request=request, content=content)
        raise httpx.HTTPStatusError("not found", request=request, response=response)

    monkeypatch.setattr(SonarClient, "get", fake_get)

    async with SonarClient("https://sonarcloud.io/api", "token", "bad-org") as client:
        with pytest.raises(ValueError) as err:
            await client.rules()

    assert "bad-org" in str(err.value)
    assert "Organization does not exist" in str(err.value)
