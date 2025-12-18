import asyncio
import os
import re
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from code_pulse.agents.tools import Tooling, ToolResult
from code_pulse.logger import setup_logging
from code_pulse.memory import MemoryStore, Message

logger = setup_logging(__name__)


class Responder:
    def __init__(self):
        self.llm = None

        # Prefer local Ollama deepseek-coder if available; no OpenAI fallback.
        try:
            from langchain_ollama import ChatOllama  # type: ignore

            base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
            self.llm = ChatOllama(
                model="deepseek-coder:6.7b",
                temperature=0.2,
                base_url=base_url,
            )
        except Exception:
            self.llm = None

        # Intent helpers (formatting choice).
        self._qa_patterns = [
            r"\bhow many\b",
            r"\bvalidation status\b",
            r"\bfiles? (were )?changed\b",
            r"\bsecurity hotspots?\b",
            r"\bprojects? (are )?linked\b",
            r"\bsonarqube projects\b",
        ]

        self.prompt = ChatPromptTemplate.from_template(
            "You are a helpful engineering agent.\n"
            "Task: {task}\n"
            "Context:\n{context}\n"
            "When responding with summary, follow these formatting rules strictly:\n"
            "1. Use markdown format.\n"
            "2. Use \"AI-Generated Summary:\" as the main header.\n"
            "3. Write a concise summary in one paragraph under the header about total SonarQube issues found in the given PR. "
            "Do not list issues individually. Specify additional statistics for each issue category (type) and severity (numbers only, no details).\n"
            "4. When calculating totals, include both issues and security hotspots. For example, if there are 3 BUG, 2 VULNERABILITY "
            "and 2 SECURITY_HOTSPOTs total number is 7.\n"
            "5. Write a single, concise sentence that clearly states the main changes while fixing issues in the PR related to SonarQube, "
            "generalizing across all issues found and fixed. Add it only if the fixes were made.\n"
            "6. Include the sentence 'A link to the new branch with proposed fixes can be found here.' ONLY when the user requested fixes "
            "or the task includes applying fixes; omit it otherwise.\n"
            "Provide the requested summary only."
        )
        self.parser = StrOutputParser()

    def _truncate(self, value: str, limit: int = 500) -> str:
        if value is None:
            return ""
        return value if len(value) <= limit else f"{value[:limit]}... [truncated]"

    async def generate(self, task: str, context: str) -> str:
        # Q&A tasks -> bullet points.
        is_qa = any(re.search(p, task, re.IGNORECASE) for p in self._qa_patterns)
        if is_qa:
            task = task + "\n\nFormatting requirement: Answer each question as short bullet points. Do not output any code or scripts."

        logger.info(
            "Model input task=%s context=%s",
            self._truncate(task, 300),
            self._truncate(context, 800),
        )

        if self.llm:
            chain = self.prompt | self.llm | self.parser
            answer = await chain.ainvoke({"task": task, "context": context})
        else:
            answer = f"Task: {task}\nContext Summary:\n{context}"

        wants_fix = ("fix" in task.lower()) or ("apply the changes" in task.lower()) or ("apply changes" in task.lower())

        # If a local_fix tool result exists, append deterministic branch/commit line.
        if wants_fix and "Branch:" not in answer:
            m_branch = re.search(r"'branch':\s*'([^']+)'", context)
            m_commit = re.search(r"'commit':\s*'([0-9a-fA-F]+)'", context)
            if m_branch:
                branch = m_branch.group(1)
                commit = m_commit.group(1) if m_commit else ""
                extra = f"Branch: {branch}"
                if commit:
                    extra += f", Commit: {commit}"
                answer = f"{answer}\n{extra}"

        # Ensure branch link sentence is present only when user asked for fixes.
        branch_line = "A link to the new branch with proposed fixes can be found here."
        if wants_fix and branch_line not in answer:
            separator = "\n" if not answer.endswith("\n") else ""
            answer = f"{answer}{separator}{branch_line}"

        # Enforce truthful reporting for fix requests: no commit/applied -> justification mode.
        try:
            if wants_fix and ("local_fix" in context):
                no_commit = ("'commit': None" in context) or ('"commit": None' in context) or ("commit': None" in context)
                no_applied = ("'applied': []" in context) or ('\"applied\": []' in context) or ("applied=0" in context)
                if no_commit or no_applied:
                    justification_lines = [
                        "No automatic fixes were applied because patches could not be safely generated or applied. See tool output for exact git apply/model diff reasons."
                    ]
                    failed = re.findall(r"'file': '([^']+)'.*?'reason': '([^']+)'", context)
                    if failed:
                        justification_lines.append("Top failed files:")
                        for file_name, reason in failed[:3]:
                            justification_lines.append(f"- {file_name}: {reason}")
                    justification = "\n".join(justification_lines)
                    answer = answer.rstrip() + "\n\n" + justification
        except Exception:
            pass

        logger.info("Model response %s", self._truncate(answer, 800))

        return answer


class AgentRunner:
    def __init__(self, responder: Optional[Responder] = None):
        self.tooling = Tooling()
        self.memory = MemoryStore()
        self.responder = responder or Responder()

    def _parse_task_repo(self, task: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        pr_match = re.search(r"github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)", task)
        if pr_match:
            return pr_match.group(1), pr_match.group(2), pr_match.group(3)
        repo_match = re.search(r"(?:repo|repository)\s+([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", task, re.IGNORECASE)
        if repo_match:
            return repo_match.group(1), repo_match.group(2), None
        return None, None, None

    def _augment_tool_args(
        self,
        task: str,
        tools: List[str],
        tool_args: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        owner, repo, pr_number = self._parse_task_repo(task)
        if "git" in tools:
            git_args = dict(tool_args.get("git", {}))
            if owner and "owner" not in git_args:
                git_args["owner"] = owner
            if repo and "repo" not in git_args:
                git_args["repo"] = repo
            if pr_number and "pull_number" not in git_args:
                git_args["pull_number"] = pr_number
            if git_args:
                tool_args["git"] = git_args
        if "sonar" in tools:
            sonar_args = dict(tool_args.get("sonar", {}))
            if pr_number and "pull_request" not in sonar_args:
                sonar_args["pull_request"] = pr_number
            if "project_key" not in sonar_args:
                if self.tooling.settings.sonar_project_key:
                    sonar_args["project_key"] = self.tooling.settings.sonar_project_key
                elif owner and repo:
                    sonar_args["project_key"] = f"{owner}:{repo}"
            if owner and repo and "project_search" not in sonar_args:
                if "project" in task.lower():
                    sonar_args["project_search"] = repo
            if sonar_args:
                tool_args["sonar"] = sonar_args
        return tool_args

    async def _execute_tools(
        self, tools: List[str], tool_args: Dict[str, Dict[str, Any]], namespace: str
    ) -> List[ToolResult]:
        tasks = []
        for name in tools:
            if name == "git" and "git" in tool_args:
                args = tool_args["git"]
                tasks.append(
                    self.tooling.use_git(
                        args.get("owner"),
                        args.get("repo"),
                        pull_number=args.get("pull_number"),
                        pull_url=args.get("pull_url"),
                    )
                )
            elif name == "sonar":
                args = tool_args.get("sonar", {})
                project_key = args.get("project_key")
                question = args.get("question") or args.get("task") or ""
                sonar_namespace = args.get("namespace") or namespace or "sonar"
                tasks.append(
                    self.tooling.use_sonar(
                        project_key,
                        question=question or "",
                        namespace=sonar_namespace,
                        pull_request=args.get("pull_request"),
                        project_search=args.get("project_search"),
                        issue_types=args.get("issue_types"),
                        hotspot_status=args.get("hotspot_status"),
                    )
                )
            elif name == "jira" and "jira" in tool_args:
                args = tool_args["jira"]
                tasks.append(self.tooling.use_jira(args["jql"], args.get("create")))
            elif name == "rag":
                args = tool_args.get("rag", {})
                question = args.get("question") or args.get("task") or ""
                tasks.append(asyncio.to_thread(self.tooling.use_rag, question, namespace))
        if not tasks:
            return []
        results = await asyncio.gather(*tasks)
        return results

    async def run(
        self,
        task: str,
        tools: List[str],
        memory_key: str,
        namespace: str = "default",
        tool_args: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, object]:
        logger.info("TASK RECEIVED: %s", task)
        logger.info("TOOLS REQUESTED (initial): %s", tools)

        wants_fix = ("fix" in task.lower()) or ("apply the changes" in task.lower()) or ("apply changes" in task.lower())
        if wants_fix and "sonar" not in tools:
            logger.info("Auto-injecting sonar tool for fix request")
            tools = list(tools) + ["sonar"]

        tool_args = self._augment_tool_args(task, tools, tool_args or {})
        logger.info("TOOL ARGS (augmented): %s", {k: list(v.keys()) for k, v in tool_args.items()})

        self.memory.append(memory_key, Message(role="user", content=task))
        results = await self._execute_tools(tools, tool_args, namespace)

        logger.info("TOOLS EXECUTED: %s", [r.name for r in results])

        # Post-step: local autofix (non-breaking, runs only on fix/apply requests)
        if wants_fix:
            sonar_res = next((r for r in results if r.name == "sonar"), None)
            logger.info("Local fix requested. Sonar result found=%s", sonar_res is not None)
            if sonar_res and isinstance(sonar_res.output, dict):
                try:
                    local_fix = await self.tooling.fix_sonar_locally(task, sonar_res.output)
                    results.append(local_fix)
                    logger.info("Local fix result: %s", local_fix.output)

                    git_res = next((r for r in results if r.name == "git"), None)
                    if git_res and isinstance(git_res.output, dict):
                        pr_comment = await self.tooling.comment_pr_with_branch(task, git_res.output, local_fix.output)
                        results.append(pr_comment)
                        logger.info("PR comment result: %s", pr_comment.output)
                except Exception as exc:
                    logger.exception("Local fix failed")
                    results.append(ToolResult(name="local_fix", output={"error": str(exc)}))
            else:
                results.append(ToolResult(name="local_fix", output={"skipped": True, "reason": "Sonar tool did not run or returned no payload."}))

        context_parts = []
        for res in results:
            context_parts.append(f"{res.name}: {res.output}")
        context = "\n".join(context_parts) if context_parts else "No tools invoked."
        answer = await self.responder.generate(task, context)
        self.memory.append(memory_key, Message(role="assistant", content=answer))
        return {"answer": answer, "tools": [res.__dict__ for res in results], "memory_key": memory_key}
