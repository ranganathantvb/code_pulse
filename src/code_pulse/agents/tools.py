from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from code_pulse.mcp.clients import client_factory
from code_pulse.mcp.git import GitClient
from code_pulse.mcp.sonar import SonarClient
from code_pulse.mcp.jira import JiraClient
from code_pulse.rag.service import RAGService


@dataclass
class ToolResult:
    name: str
    output: Any


class Tooling:
    def __init__(self):
        clients = client_factory()
        self.git: GitClient = clients["git"]  # type: ignore[assignment]
        self.sonar: SonarClient = clients["sonar"]  # type: ignore[assignment]
        self.jira: JiraClient = clients["jira"]  # type: ignore[assignment]
        self.rag = RAGService()

    async def use_git(self, owner: str, repo: str) -> ToolResult:
        async with self.git as client:
            repo_data = await client.repo(owner, repo)
            pulls = await client.pull_requests(owner, repo)
            return ToolResult(
                name="git",
                output={"repo": repo_data, "open_prs": pulls},
            )

    async def use_sonar(self, project_key: str) -> ToolResult:
        async with self.sonar as client:
            issues = await client.project_issues(project_key)
            measures = await client.measures(project_key)
            return ToolResult(name="sonar", output={"issues": issues, "measures": measures})

    async def use_jira(
        self, jql: str, create: Optional[Dict[str, str]] = None
    ) -> ToolResult:
        async with self.jira as client:
            search_results = await client.search(jql)
            created = None
            if create:
                created = await client.create_issue(
                    create["project_key"],
                    create["summary"],
                    create.get("issue_type", "Task"),
                    create.get("description", ""),
                )
            return ToolResult(name="jira", output={"search": search_results, "created": created})

    def use_rag(self, question: str, namespace: str = "default") -> ToolResult:
        docs = self.rag.query(question, namespace=namespace)
        excerpts = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]
        return ToolResult(name="rag", output={"matches": excerpts})
