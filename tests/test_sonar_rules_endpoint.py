from fastapi.testclient import TestClient

from code_pulse.app import app
from code_pulse.mcp.sonar import SonarClient


def test_sonar_rules_endpoint(monkeypatch):
    async def fake_rules(
        self,
        query=None,
        languages=None,
        severities=None,
        types=None,
        page=1,
        page_size=100,
    ):
        return {
            "rules": [
                {"key": "python:S123", "name": "Example rule", "severity": "MAJOR"},
            ],
            "paging": {"page": page, "page_size": page_size, "total": 1},
        }

    monkeypatch.setattr(SonarClient, "rules", fake_rules)

    client = TestClient(app)
    res = client.get("/sonar/rules", params={"query": "python"})
    assert res.status_code == 200
    payload = res.json()
    assert payload["rules"][0]["key"] == "python:S123"
    assert payload["paging"]["total"] == 1
