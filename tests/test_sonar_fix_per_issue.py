import json
import subprocess
from types import SimpleNamespace

import pytest

from code_pulse.agents.tools import Tooling
from code_pulse.config import get_settings


class FakeRag:
    def lookup_by_metadata(self, namespace, metadata_filter, query_text):  # noqa: D401
        return [
            SimpleNamespace(
                page_content="Use explicit checks to avoid unsafe defaults.",
                metadata={"section": "remediation"},
            )
        ]


class FakeChain:
    def __init__(self, responses):
        self._responses = list(responses)

    async def ainvoke(self, *_args, **_kwargs):
        if not self._responses:
            raise RuntimeError("No more responses available")
        return self._responses.pop(0)


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


@pytest.mark.asyncio
async def test_fix_sonar_per_issue_commits(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    _run(["git", "config", "user.name", "Test User"], cwd=repo)

    file_a = repo / "file_a.py"
    file_b = repo / "file_b.py"
    file_a.write_text("a = 1\n", encoding="utf-8")
    file_b.write_text("b = 1\n", encoding="utf-8")
    _run(["git", "add", "-A"], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)

    responses = [
        json.dumps(
            {
                "rule": "S1111",
                "plan": "Update constant",
                "edits": [
                    {
                        "file": "file_a.py",
                        "before": "a = 1",
                        "after": "a = 2",
                        "rag_quote": "Use explicit checks to avoid unsafe defaults.",
                    }
                ],
                "commands": [],
                "notes": "",
            }
        ),
        json.dumps(
            {
                "rule": "S2222",
                "plan": "Update constant",
                "edits": [
                    {
                        "file": "file_b.py",
                        "before": "b = 1",
                        "after": "b = 2",
                        "rag_quote": "Use explicit checks to avoid unsafe defaults.",
                    }
                ],
                "commands": [],
                "notes": "",
            }
        ),
    ]

    def _fake_runnable(_fn):
        return FakeChain(responses)

    monkeypatch.setenv("CODEPULSE_WORKSPACE_PATH", str(repo))
    monkeypatch.setenv("CODEPULSE_FIX_MODEL", "mock")
    monkeypatch.setenv("CODEPULSE_FIX_MOCK_RESPONSE", responses[0])
    monkeypatch.setattr("langchain_core.runnables.RunnableLambda", _fake_runnable)

    tooling = Tooling.__new__(Tooling)
    tooling.settings = get_settings()
    tooling.rag = FakeRag()

    payload = {
        "pr_issues": {
            "issues": [
                {"rule": "S1111", "type": "BUG", "message": "Issue A", "component": "file_a.py", "textRange": {"startLine": 1}},
                {"rule": "S2222", "type": "BUG", "message": "Issue B", "component": "file_b.py", "textRange": {"startLine": 1}},
            ]
        },
        "pr_hotspots": {"hotspots": []},
    }

    result = await tooling.fix_sonar_locally(
        "fix sonar issues",
        payload,
        git_payload=None,
        workspace_path=str(repo),
        commit_mode="per_issue",
    )

    commits = result.output.get("commits") or []
    assert len(commits) == 2
