import datetime
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

# Avoid tokenizer fork warnings/deadlocks when forking LLM workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}
from code_pulse.mcp.clients import client_factory
from code_pulse.mcp.git import GitClient
from code_pulse.mcp.sonar import SonarClient
from code_pulse.mcp.jira import JiraClient
from code_pulse.rag.service import RAGService
from code_pulse.config import get_settings, Settings
from code_pulse.logger import logger
from code_pulse import telemetry


@dataclass
class ToolResult:
    name: str
    output: Any


class Tooling:
    # Orchestrates access to Git/Sonar/Jira plus local RAG.
    def __init__(self):
        self.settings: Settings = get_settings()
        # Build concrete MCP clients once so tool calls can reuse connections/config.
        clients = client_factory(self.settings)
        self.git: GitClient = clients["git"]  # type: ignore[assignment]
        self.sonar: SonarClient = clients["sonar"]  # type: ignore[assignment]
        self.jira: JiraClient = clients["jira"]  # type: ignore[assignment]
        self.rag = RAGService()

    def _parse_pull_url(self, url: str) -> Optional[Dict[str, str]]:
        # Lightweight parser to avoid pulling in full URL libs.
        match = re.search(r"github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)", url)
        if not match:
            return None
        return {"owner": match.group(1), "repo": match.group(2), "number": match.group(3)}

    async def use_git(
        self,
        owner: Optional[str],
        repo: Optional[str],
        pull_number: Optional[int | str] = None,
        pull_url: Optional[str] = None,
    ) -> ToolResult:
        # Allow callers to provide a PR URL instead of individual pieces.
        # Logical switch: PR URL present -> decompose into owner/repo/number.
        if pull_url:
            parsed = self._parse_pull_url(pull_url)
            if parsed:
                owner = owner or parsed["owner"]
                repo = repo or parsed["repo"]
                if pull_number is None:
                    pull_number = parsed["number"]
        # Logical switch: require repository identifiers to continue.
        if not owner or not repo:
            raise ValueError("Git tool requires owner and repo")

        # Collect repo metadata plus PR details/status/checks/files in one pass.
        async with self.git as client:
            repo_data = await client.repo(owner, repo)
            pulls = await client.pull_requests(owner, repo)
            pr_data = pr_files = pr_status = pr_checks = None
            files_count = None
            changed_files = None
            check_summary = None
            validation = None
            if pull_number is not None:
                pr_number_int = int(pull_number)
                pr_data = await client.pull_request(owner, repo, pr_number_int)
                pr_files = await client.pull_request_files(owner, repo, pr_number_int)
                files_count = len(pr_files)
                changed_files = pr_data.get("changed_files")
                head_sha = pr_data.get("head", {}).get("sha")
                # Logical switch: only fetch status/checks when we know the head SHA.
                if head_sha:
                    pr_status = await client.commit_status(owner, repo, head_sha)
                    pr_checks = await client.check_runs(owner, repo, head_sha)
                    if pr_checks and isinstance(pr_checks, dict):
                        summary: Dict[str, int] = {}
                        for check in pr_checks.get("check_runs", []):
                            conclusion = check.get("conclusion") or "neutral"
                            summary[conclusion] = summary.get(conclusion, 0) + 1
                        check_summary = {
                            "total": pr_checks.get("total_count", 0),
                            "by_conclusion": summary,
                        }
                    validation = {
                        "combined_status": pr_status.get("state") if pr_status else None,
                        "check_summary": check_summary,
                        "mergeable_state": pr_data.get("mergeable_state"),
                    }
            return ToolResult(
                name="git",
                output={
                    "repo": repo_data,
                    "open_prs": pulls,
                    "pull_request": pr_data,
                    "pull_request_files": pr_files,
                    "pull_request_files_count": files_count,
                    "pull_request_changed_files": changed_files,
                    "pull_request_status": pr_status,
                    "pull_request_check_runs": pr_checks,
                    "pull_request_check_summary": check_summary,
                    "pull_request_validation": validation,
                },
            )

    async def use_sonar(
        self,
        project_key: Optional[str] = None,
        question: str = "",
        namespace: str = "sonar",
        pull_request: Optional[int | str] = None,
        project_search: Optional[str] = None,
        issue_types: Optional[str] = None,
        hotspot_status: Optional[str] = None,
    ) -> ToolResult:
        # Sonar tool blends RAG guidance with live SonarQube/Cloud data when available.
        # Always try RAG for Sonar guidance so we can respond even if live Sonar calls fail.
        rag_docs = self.rag.query(question or "sonar issue", namespace=namespace)
        rag_matches = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in rag_docs
        ]

        # Pull live Sonar data if configured.
        project_key = project_key or self.settings.sonar_project_key
        issues = measures = None
        pr_issues = pr_hotspots = pr_quality_gate = None
        pr_issues_count = pr_hotspots_count = None
        projects = None
        # Logical switch: only call Sonar APIs when we have a project target or search term.
        if project_key or project_search:
            async with self.sonar as client:
                if project_key:
                    # Baseline project data.
                    issues = await client.project_issues(project_key)
                    measures = await client.measures(project_key)
                    if pull_request is not None:
                        pr_number = str(pull_request)
                        try:
                            pr_issues = await client.pr_issues(
                                project_key, pr_number, issue_types=issue_types
                            )
                            pr_issues_count = pr_issues.get("total")
                        except Exception as exc:  # noqa: BLE001
                            pr_issues = {"message": f"PR issues unavailable: {exc}"}
                        try:
                            # Hotspots often require specific Sonar edition/permissions.
                            pr_hotspots = await client.pr_hotspots(
                                project_key, pr_number, status=hotspot_status
                            )
                            pr_hotspots_count = pr_hotspots.get("paging", {}).get("total")
                        except Exception:  # noqa: BLE001
                            pr_hotspots = {"message": "Hotspots endpoint not available."}
                        try:
                            # Quality gate gives the per-PR gate decision if configured.
                            pr_quality_gate = await client.pr_quality_gate(project_key, pr_number)
                        except Exception:  # noqa: BLE001
                            pr_quality_gate = {"message": "Quality gate endpoint not available."}
                if project_search:
                    # Optional search when only text query is provided.
                    projects = await client.projects(query=project_search)

        message = None
        if not rag_matches:
            message = "No matching Sonar guidance found in the knowledge base."

        return ToolResult(
            name="sonar",
            output={
                "rag_matches": rag_matches,
                "issues": issues,
                "pr_issues": pr_issues,
                "pr_issues_count": pr_issues_count,
                "pr_hotspots": pr_hotspots,
                "pr_hotspots_count": pr_hotspots_count,
                "pr_quality_gate": pr_quality_gate,
                "measures": measures,
                "projects": projects,
                "message": message,
            },
        )

    async def use_jira(
        self, jql: str, create: Optional[Dict[str, str]] = None
    ) -> ToolResult:
        # Fetch issues by JQL and optionally create a new one in a single call.
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
        # Thin wrapper around vector search so agents can retrieve snippets quickly.
        docs = self.rag.query(question, namespace=namespace)
        excerpts = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]
        message = None if excerpts else "No relevant documents found."
        return ToolResult(name="rag", output={"matches": excerpts, "message": message})

    # Heavyweight path: attempt to auto-apply Sonar fixes locally via LLM-generated edits.
    async def fix_sonar_locally(
        self,
        task: str,
        sonar_payload: Dict[str, Any],
        git_payload: Optional[Dict[str, Any]] = None,
        workspace_path: Optional[str] = None,
        commit_mode: str = "single",
    ) -> ToolResult:
        """
        Attempt local Sonar remediations with guarded application + commit.
        - Supports BUG/VULNERABILITY/CODE_SMELL and SECURITY_HOTSPOT.
        - Prefers minimal hotspot changes (validation/safe API/annotations) over refactors.
        """
        import subprocess
        from typing import List, Dict, Any, Optional, Tuple

        # Run a subprocess while capturing output for logging.
        def _run_cmd(cmd: List[str], cwd: str, check: bool = True) -> subprocess.CompletedProcess:
            logger.info("CMD: %s (cwd=%s)", " ".join(cmd), cwd)
            return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)

        # Validate the workspace is a git repo (needed for apply/commit/push).
        def _is_git_repo(path: str) -> bool:
            try:
                cp = _run_cmd(["git", "rev-parse", "--is-inside-work-tree"], cwd=path, check=True)
                return cp.stdout.strip().lower() == "true"
            except Exception:
                return False

        def _normalize_workspace_path(path: str) -> str:
            """
            Resolve common path issues (missing leading slash on macOS-style paths, ~ expansion).
            """
            expanded = os.path.expanduser(path)
            if expanded.startswith("Users/"):
                expanded = f"/{expanded}"
            return expanded

        def _next_incremental_branch(prefix: str, cwd: str) -> str:
            """Find the next available incremental branch name with the given prefix."""
            try:
                existing = _run_cmd(["git", "branch", "--list", f"{prefix}_*"], cwd=cwd, check=False).stdout
                numbers: List[int] = []
                for line in existing.splitlines():
                    m = re.search(rf"{re.escape(prefix)}_(\d+)", line.strip())
                    if m:
                        numbers.append(int(m.group(1)))
                next_num = (max(numbers) + 1) if numbers else 1
                return f"{prefix}_{next_num:02d}"
            except Exception:
                stamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
                return f"{prefix}_{stamp}"

        def _current_branch(cwd: str) -> Optional[str]:
            try:
                return _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd).stdout.strip()
            except Exception:
                return None

        def _resolve_branch(
            task_text: str,
            sonar_payload_in: Dict[str, Any],
            git_payload_in: Optional[Dict[str, Any]],
            cwd: str,
        ) -> tuple[str, str]:
            """
            Pick a target branch in the following priority:
            1) Provided in the Git payload (pull_request head ref) or Sonar payload (branch/git_branch keys)
            2) Environment override (CODEPULSE_TARGET_BRANCH or CODEPULSE_GIT_BRANCH)
            3) Current branch (git)
            4) Fallback to main
            """
            pr = (git_payload_in or {}).get("pull_request") or {}
            from_git = (pr.get("head") or {}).get("ref")
            from_sonar = sonar_payload_in.get("branch") or sonar_payload_in.get("git_branch")
            from_env = os.getenv("CODEPULSE_TARGET_BRANCH") or os.getenv("CODEPULSE_GIT_BRANCH")
            base_branch = from_git or from_sonar or from_env or _current_branch(cwd) or "main"
            target_branch = _next_incremental_branch("Fix_sonar_issues", cwd)
            return base_branch, target_branch

        # Guardrail: only proceed when caller asked for a fix.
        def _wants_fix(text: str) -> bool:
            t = text.lower()
            return ("fix" in t) or ("apply the changes" in t) or ("apply changes" in t)

        # Map Sonar component keys to workspace-relative file paths.
        def _normalize_component_to_path(component: str, project_key: Optional[str] = None) -> str:
            if ":" in component:
                return component.split(":", 1)[1]
            if project_key and component.startswith(project_key):
                return component[len(project_key):].lstrip("/\\")
            return component

        def _normalize_edit_path(raw_path: str, base_path: str) -> Optional[str]:
            if not raw_path:
                return None
            cleaned = raw_path.strip()
            if cleaned.startswith(("a/", "b/")):
                cleaned = cleaned[2:]
            if os.path.isabs(cleaned):
                try:
                    cleaned = os.path.relpath(cleaned, base_path)
                except ValueError:
                    return None
            if cleaned.startswith(".."):
                return None
            return cleaned

        def _contains_forbidden_markers(text: str) -> bool:
            if not text:
                return True
            lowered = text.lower()
            forbidden = ["```", "```diff", "--- a/", "+++ b/", "@@", "diff --git"]
            return any(token in lowered for token in forbidden)

        def _normalize_whitespace(text: str) -> str:
            return " ".join((text or "").split())

        def _replace_once_with_flexible_whitespace(
            content: str, before: str, after: str
        ) -> Optional[str]:
            normalized_before = _normalize_whitespace(before)
            if not normalized_before:
                return None
            # First try an exact match for safety.
            if before in content:
                return content.replace(before, after, 1)
            # Fallback: build a whitespace-flexible regex for the before text.
            pattern = re.sub(r"\s+", r"\\s+", re.escape(before))
            matches = list(re.finditer(pattern, content, flags=re.DOTALL))
            if len(matches) != 1:
                return None
            return re.sub(pattern, after, content, count=1, flags=re.DOTALL)

        def _replace_once_in_window(
            content: str,
            before: str,
            after: str,
            start_line: int,
            end_line: int,
        ) -> Optional[str]:
            if start_line < 1 or end_line < start_line:
                return None
            lines = content.splitlines(keepends=True)
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            window_text = "".join(lines[start_idx:end_idx])
            if before in window_text:
                window_text = window_text.replace(before, after, 1)
            else:
                pattern = re.sub(r"\s+", r"\\s+", re.escape(before))
                matches = list(re.finditer(pattern, window_text, flags=re.DOTALL))
                if len(matches) != 1:
                    return None
                window_text = re.sub(pattern, after, window_text, count=1, flags=re.DOTALL)
            lines[start_idx:end_idx] = [window_text]
            return "".join(lines)

        def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
            if _contains_forbidden_markers(raw):
                return None
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return None
            if not isinstance(payload, dict):
                return None
            if "edits" not in payload or not isinstance(payload.get("edits"), list):
                return None
            return payload

        def _order_rule_chunks(chunks: List[Any]) -> List[Any]:
            order = {"compliant": 0, "noncompliant": 1, "remediation": 2, "rationale": 3, "summary": 4}
            return sorted(chunks, key=lambda doc: order.get(doc.metadata.get("section"), 99))

        def _windowed_lines(content: str, start_line: int, radius: int = 120) -> Tuple[int, int, str]:
            lines = content.splitlines()
            start_idx = max(0, start_line - 1 - radius)
            end_idx = min(len(lines), start_line - 1 + radius)
            return start_idx + 1, end_idx, "\n".join(lines[start_idx:end_idx])

        run_id = str(uuid.uuid4())
        run_start = time.monotonic()
        repo_name = None
        if git_payload:
            repo_obj = git_payload.get("repo") or {}
            repo_name = repo_obj.get("full_name") or repo_obj.get("name")
        telemetry.set_sonar_run_id(run_id)

        def _duration_ms(start: float) -> int:
            return max(0, int((time.monotonic() - start) * 1000))

        def _emit_run(
            status: str,
            duration_ms: Optional[int] = None,
            reason: Optional[str] = None,
            branch_name: Optional[str] = None,
            sonar_found: Optional[bool] = None,
        ) -> None:
            payload: Dict[str, Any] = {
                "event": "sonar_fix_run",
                "run_id": run_id,
                "status": status,
                "branch": branch_name,
                "repo": repo_name,
                "sonar_found": sonar_found,
            }
            if duration_ms is not None:
                payload["duration_ms"] = duration_ms
            if reason:
                payload["reason"] = reason
            telemetry.emit_structured_log(payload)

        def _emit_item(
            rule_key: str,
            rule_type: str,
            action: str,
            reason: Optional[str] = None,
            duration_ms: Optional[int] = None,
            file_path: Optional[str] = None,
        ) -> None:
            payload: Dict[str, Any] = {
                "event": "sonar_fix_item",
                "run_id": run_id,
                "rule_key": rule_key,
                "rule_type": rule_type,
                "action": action,
                "file_path": file_path,
            }
            if duration_ms is not None:
                payload["duration_ms"] = duration_ms
            if reason:
                payload["reason"] = reason
            telemetry.emit_structured_log(payload)

        def _emit_stage(
            stage: str,
            status: str,
            duration_ms: Optional[int] = None,
            reason: Optional[str] = None,
        ) -> None:
            payload: Dict[str, Any] = {
                "event": "sonar_fix_stage",
                "run_id": run_id,
                "stage": stage,
                "status": status,
            }
            if duration_ms is not None:
                payload["duration_ms"] = duration_ms
            if reason:
                payload["reason"] = reason
            telemetry.emit_structured_log(payload)

        _emit_run(status="started")

        logger.info("ENTER fix_sonar_locally task=%s", task)

        if not _wants_fix(task):
            # Guardrail: only run heavy remediation flow when explicitly requested.
            logger.info("Local fix skipped: task did not request fixes")
            _emit_run(status="skipped", reason="Task does not request fixes.")
            return ToolResult(name="local_fix", output={"skipped": True, "reason": "Task does not request fixes."})

        workspace_path = workspace_path or os.getenv("CODEPULSE_WORKSPACE_PATH")
        if not workspace_path:
            logger.error("CODEPULSE_WORKSPACE_PATH not set")
            _emit_run(status="skipped", reason="CODEPULSE_WORKSPACE_PATH not set.")
            return ToolResult(name="local_fix", output={"skipped": True, "reason": "CODEPULSE_WORKSPACE_PATH not set."})

        workspace_path = _normalize_workspace_path(workspace_path)
        logger.info("Workspace=%s", workspace_path)

        if not os.path.isdir(workspace_path):
            logger.error("Workspace path does not exist: %s", workspace_path)
            _emit_run(status="skipped", reason=f"Workspace path does not exist: {workspace_path}")
            return ToolResult(name="local_fix", output={"skipped": True, "reason": f"Workspace path does not exist: {workspace_path}"})

        if not _is_git_repo(workspace_path):
            logger.error("Workspace is not a git repository: %s", workspace_path)
            _emit_run(status="skipped", reason=f"Workspace is not a git repository: {workspace_path}")
            return ToolResult(name="local_fix", output={"skipped": True, "reason": f"Workspace is not a git repository: {workspace_path}"})

        pr_issues = (sonar_payload or {}).get("pr_issues") or {}
        pr_hotspots = (sonar_payload or {}).get("pr_hotspots") or {}
        issues: List[Dict[str, Any]] = pr_issues.get("issues") or []
        hotspots: List[Dict[str, Any]] = pr_hotspots.get("hotspots") or []

        # Combine issues and hotspots into a single list of findings to remediate.
        findings: List[Dict[str, Any]] = []
        for iss in issues:
            findings.append({
                "kind": (iss.get("type") or "issue").upper(),
                "rule": iss.get("rule"),
                "message": iss.get("message"),
                "component": iss.get("component"),
                "textRange": iss.get("textRange") or {},
                "issue_key": iss.get("key") or iss.get("issue") or iss.get("id"),
            })
        for hs in hotspots:
            findings.append({
                "kind": "SECURITY_HOTSPOT",
                "rule": hs.get("rule") or hs.get("ruleKey"),
                "message": hs.get("message"),
                "component": hs.get("component") or hs.get("project"),
                "textRange": hs.get("textRange") or {},
                "issue_key": hs.get("key") or hs.get("hotspotKey") or hs.get("id"),
            })

        logger.info("Findings total=%d (issues=%d, hotspots=%d)", len(findings), len(issues), len(hotspots))

        if not findings:
            branch = None
            logger.info("Local fix skipped: Sonar payload contained no findings")
            report_dir = os.path.join(workspace_path, ".codepulse", "reports")
            os.makedirs(report_dir, exist_ok=True)
            report_stamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            report_path = os.path.join(report_dir, f"sonar_fix_report_{report_stamp}.json")
            report_payload = {
                "workspace": workspace_path,
                "branch": branch,
                "applied_count": 0,
                "skipped_count": 0,
                "applied": [],
                "skipped": [],
                "entries": [{"status": "skipped", "reason": "No Sonar findings available to fix."}],
            }
            with open(report_path, "w", encoding="utf-8") as fh:
                json.dump(report_payload, fh, indent=2)
            _emit_run(status="skipped", reason="No Sonar findings available to fix.", branch_name=branch, sonar_found=False)
            return ToolResult(
                name="local_fix",
                output={
                    "skipped": True,
                    "reason": "No Sonar findings available to fix.",
                    "workspace": workspace_path,
                    "branch": None,
                    "report_path": report_path,
                },
            )

        base_branch, branch = _resolve_branch(task, sonar_payload or {}, git_payload, workspace_path)
        logger.info("Base branch=%s Target branch=%s", base_branch, branch)

        # checkout/create/reset branch
        try:
            _run_cmd(["git", "fetch", "--all", "--prune"], cwd=workspace_path, check=False)
            _run_cmd(["git", "checkout", base_branch], cwd=workspace_path, check=False)
            if base_branch != branch:
                _run_cmd(["git", "checkout", "-B", branch, base_branch], cwd=workspace_path)
            else:
                _run_cmd(["git", "checkout", "-B", branch], cwd=workspace_path)
            current = _run_cmd(["git", "branch", "--show-current"], cwd=workspace_path, check=True).stdout.strip()
            logger.info("Current branch after checkout=%s", current)
            if _bool_env("CODEPULSE_CLEAN_BRANCH_BEFORE_FIX"):
                try:
                    _run_cmd(["git", "reset", "--hard"], cwd=workspace_path)
                    _run_cmd(["git", "clean", "-fd"], cwd=workspace_path)
                    logger.info("Workspace cleaned before applying edits (CODEPULSE_CLEAN_BRANCH_BEFORE_FIX enabled).")
                except subprocess.CalledProcessError as exc:
                    logger.warning("Branch clean failed: %s", exc.stderr or exc.stdout)
        except subprocess.CalledProcessError as exc:
            logger.exception("Branch checkout failed")
            _emit_run(
                status="failed",
                reason=f"Failed to create/checkout branch '{branch}': {exc.stderr}",
                duration_ms=_duration_ms(run_start),
                branch_name=branch,
                sonar_found=True,
            )
            return ToolResult(name="local_fix", output={"error": f"Failed to create/checkout branch '{branch}': {exc.stderr}", "workspace": workspace_path, "branch": branch})

        # LLM setup (Ollama)
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnableLambda
            base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
            model_name = os.getenv("CODEPULSE_FIX_MODEL", "deepseek-coder:6.7b")
            if model_name == "mock":
                mock_response = os.getenv("CODEPULSE_FIX_MOCK_RESPONSE")
                if not mock_response:
                    raise ValueError("CODEPULSE_FIX_MOCK_RESPONSE is required for mock mode")

                async def _mock_invoke(_: Dict[str, Any]) -> str:
                    return mock_response

                chain = RunnableLambda(_mock_invoke)
            else:
                from langchain_ollama import ChatOllama  # type: ignore

                llm = ChatOllama(model=model_name, temperature=0.1, base_url=base_url)
                chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
        except Exception as exc:
            logger.exception("Ollama init failed")
            _emit_run(
                status="failed",
                reason=f"Ollama unavailable: {exc}",
                duration_ms=_duration_ms(run_start),
                branch_name=branch,
                sonar_found=True,
            )
            return ToolResult(name="local_fix", output={"skipped": True, "reason": f"Ollama unavailable: {exc}", "workspace": workspace_path, "branch": branch})

        changed_files: List[str] = []
        applied: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        report_entries: List[Dict[str, Any]] = []
        commits: List[str] = []
        commit_errors: List[Dict[str, Any]] = []

        project_key = (sonar_payload or {}).get("project_key") or getattr(self.settings, "sonar_project_key", None)
        commit_mode = (commit_mode or "single").strip().lower()
        if commit_mode not in {"single", "per_issue"}:
            logger.warning("Unknown commit_mode=%s; defaulting to single", commit_mode)
            commit_mode = "single"

        for f in findings:
            item_start = time.monotonic()
            component = str(f.get("component") or "")
            rel_path = _normalize_component_to_path(component, project_key=project_key)
            abs_path = os.path.join(workspace_path, rel_path)

            tr = f.get("textRange") or {}
            start_line = int(tr.get("startLine") or 1)

            rule = f.get("rule") or f.get("ruleKey") or "SONAR_RULE"
            if not os.path.isfile(abs_path):
                logger.warning("File not found: %s", abs_path)
                skipped.append({"file": rel_path, "rule": rule, "reason": "File not found in workspace"})
                report_entries.append(
                    {"rule": rule, "file": rel_path, "status": "skipped", "reason": "File not found in workspace"}
                )
                _emit_item(
                    rule_key=rule,
                    rule_type=str(f.get("kind") or "unknown"),
                    action="skipped",
                    reason="File not found in workspace",
                    duration_ms=_duration_ms(item_start),
                    file_path=rel_path,
                )
                continue

            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
            win_start, win_end, window = _windowed_lines(content, start_line=start_line, radius=60)
            logger.info("RAG retrieval: rule_key='%s', namespace='sonar'", rule)
            rag_rule_key = rule
            rag_docs = self.rag.lookup_by_metadata(
                namespace="sonar",
                metadata_filter={"doc_type": "sonar_rule", "rule_key": rag_rule_key},
                query_text=rule,
            )
            if not rag_docs and ":" not in rag_rule_key:
                rag_rule_key = f"java:{rag_rule_key}"
                rag_docs = self.rag.lookup_by_metadata(
                    namespace="sonar",
                    metadata_filter={"doc_type": "sonar_rule", "rule_key": rag_rule_key},
                    query_text=rag_rule_key,
                )
            logger.info("RAG retrieved %d chunk(s) for rule %s", len(rag_docs), rule)
            if not rag_docs:
                skipped.append({"file": rel_path, "rule": rule, "reason": "RAG_EMPTY_RULE"})
                report_entries.append(
                    {"rule": rule, "file": rel_path, "status": "skipped", "reason": "RAG_EMPTY_RULE"}
                )
                continue
            ordered_chunks = _order_rule_chunks(list(rag_docs))
            rag_excerpt = "\n\n".join(d.page_content for d in ordered_chunks)
            for idx, doc in enumerate(ordered_chunks[:4]):
                logger.info("RAG chunk %d for rule %s: %s", idx + 1, rule, doc.page_content[:300].replace("\n", " "))

            prompt = (
                "You are a local code remediation agent running inside a developer workstation with read/write access to the repository.\n"
                "Your goal is to fix the provided Sonar findings by editing files locally and validating the changes.\n\n"
                "IMPORTANT CONSTRAINTS:\n"
                "- Output MUST be strict JSON only. No markdown, no code fences, no diff.\n"
                "- Use only the provided RAG content and code context. Do not invent APIs or rules.\n"
                "- Each edit must include a rag_quote copied verbatim from the provided RAG content.\n"
                "- Modify the original file in place; do NOT create new files.\n"
                "- Prefer minimal changes that compile and preserve behavior.\n"
                "- Avoid suppressions (NOSONAR / @SuppressWarnings) unless there is no safe alternative.\n\n"
                "OUTPUT JSON SCHEMA:\n"
                "{\n"
                "  \"rule\": \"java:S2119\",\n"
                "  \"plan\": \"...\",\n"
                "  \"edits\": [\n"
                "    {\n"
                "      \"file\": \"src/main/java/.../UserController.java\",\n"
                "      \"before\": \"<exact substring from current file>\",\n"
                "      \"after\": \"<replacement>\",\n"
                "      \"rag_quote\": \"<verbatim quote from RAG content>\"\n"
                "    }\n"
                "  ],\n"
                "  \"commands\": [\"mvn -q test\"],\n"
                "  \"notes\": \"...\"\n"
                "}\n\n"
                "SONAR FINDING:\n"
                f"- Kind: {f.get('kind')}\n- Rule: {rule}\n- Message: {f.get('message') or ''}\n- File: {rel_path}\n- Line: {start_line}\n\n"
                "RAG CONTENT (RULE GUIDANCE, USE VERBATIM QUOTES FROM THIS ONLY):\n"
                f"{rag_excerpt}\n\n"
                "CODE CONTEXT (CURRENT FILE CONTENT OR RELEVANT METHOD(S)):\n"
                f"{window}\n\n"
                "REPOSITORY ROOT PATH:\n"
                f"{workspace_path}\n\n"
                "Return ONLY the JSON object, no extra text."
            )
            logger.info("Prompt sent to LLM for rule %s: %s", rule, prompt[:500].replace("\n", " "))

            try:
                if telemetry.token_metrics_enabled():
                    callback = telemetry.TokenUsageCallback()
                    try:
                        llm_start_ms = telemetry.monotonic_ms() if telemetry.perf_metrics_enabled() else None
                        raw = await chain.ainvoke({"prompt": prompt}, config={"callbacks": [callback]})
                    except Exception:
                        llm_start_ms = telemetry.monotonic_ms() if telemetry.perf_metrics_enabled() else None
                        raw = await chain.ainvoke({"prompt": prompt})
                    if llm_start_ms is not None:
                        try:
                            telemetry.log_latency(
                                "rag_llm",
                                max(0, telemetry.monotonic_ms() - llm_start_ms),
                                telemetry.get_request_id(),
                            )
                        except Exception:
                            pass
                    if callback.usage:
                        try:
                            telemetry.log_token_usage(
                                model_name,
                                callback.usage["prompt_tokens"],
                                callback.usage["completion_tokens"],
                                callback.usage["total_tokens"],
                                telemetry.get_request_id(),
                            )
                        except Exception:
                            pass
                    else:
                        llm_start_ms = telemetry.monotonic_ms() if telemetry.perf_metrics_enabled() else None
                        raw = await chain.ainvoke({"prompt": prompt})
                        if llm_start_ms is not None:
                            try:
                                telemetry.log_latency(
                                    "rag_llm",
                                    max(0, telemetry.monotonic_ms() - llm_start_ms),
                                    telemetry.get_request_id(),
                                )
                            except Exception:
                                pass
                logger.info("Raw LLM response for rule %s: %s", rule, raw[:500].replace("\n", " "))
                payload = _parse_json_response(raw)
                if payload is None:
                    retry_prompt = prompt + "\n\nSTRICT JSON ONLY. No markdown, no diff, no extra text."
                    if telemetry.token_metrics_enabled():
                        callback = telemetry.TokenUsageCallback()
                        try:
                            llm_start_ms = telemetry.monotonic_ms() if telemetry.perf_metrics_enabled() else None
                            raw2 = await chain.ainvoke({"prompt": retry_prompt}, config={"callbacks": [callback]})
                        except Exception:
                            llm_start_ms = telemetry.monotonic_ms() if telemetry.perf_metrics_enabled() else None
                            raw2 = await chain.ainvoke({"prompt": retry_prompt})
                        if llm_start_ms is not None:
                            try:
                                telemetry.log_latency(
                                    "rag_llm",
                                    max(0, telemetry.monotonic_ms() - llm_start_ms),
                                    telemetry.get_request_id(),
                                )
                            except Exception:
                                pass
                        if callback.usage:
                            try:
                                telemetry.log_token_usage(
                                    model_name,
                                    callback.usage["prompt_tokens"],
                                    callback.usage["completion_tokens"],
                                    callback.usage["total_tokens"],
                                    telemetry.get_request_id(),
                                )
                            except Exception:
                                pass
                    else:
                        llm_start_ms = telemetry.monotonic_ms() if telemetry.perf_metrics_enabled() else None
                        raw2 = await chain.ainvoke({"prompt": retry_prompt})
                        if llm_start_ms is not None:
                            try:
                                telemetry.log_latency(
                                    "rag_llm",
                                    max(0, telemetry.monotonic_ms() - llm_start_ms),
                                    telemetry.get_request_id(),
                                )
                            except Exception:
                                pass
                    logger.info("Raw LLM retry response for rule %s: %s", rule, raw2[:500].replace("\n", " "))
                    payload = _parse_json_response(raw2)
            except Exception as exc:
                logger.exception("LLM invocation failed for %s", rel_path)
                skipped.append({"file": rel_path, "rule": rule, "reason": f"LLM failure: {exc}"})
                report_entries.append(
                    {"rule": rule, "file": rel_path, "status": "skipped", "reason": f"LLM failure: {exc}"}
                )
                _emit_item(
                    rule_key=rule,
                    rule_type=str(f.get("kind") or "unknown"),
                    action="failed",
                    reason=f"LLM failure: {exc}",
                    duration_ms=_duration_ms(item_start),
                    file_path=rel_path,
                )
                continue

            if payload is None:
                logger.warning("Invalid JSON response from model for %s", rel_path)
                skipped.append({"file": rel_path, "rule": rule, "reason": "MODEL_FORMAT_INVALID"})
                report_entries.append(
                    {"rule": rule, "file": rel_path, "status": "skipped", "reason": "MODEL_FORMAT_INVALID"}
                )
                _emit_item(
                    rule_key=rule,
                    rule_type=str(f.get("kind") or "unknown"),
                    action="skipped",
                    reason="MODEL_FORMAT_INVALID",
                    duration_ms=_duration_ms(item_start),
                    file_path=rel_path,
                )
                continue

            payload_rule = payload.get("rule") or rule
            plan = payload.get("plan") or ""
            edits = payload.get("edits") or []
            if not edits:
                skipped.append({"file": rel_path, "rule": rule, "reason": "MODEL_NO_EDITS"})
                report_entries.append(
                    {"rule": rule, "file": rel_path, "status": "skipped", "reason": "MODEL_NO_EDITS"}
                )
                _emit_item(
                    rule_key=rule,
                    rule_type=str(f.get("kind") or "unknown"),
                    action="skipped",
                    reason="MODEL_NO_EDITS",
                    duration_ms=_duration_ms(item_start),
                    file_path=rel_path,
                )
                continue

            rule_texts = [doc.page_content for doc in ordered_chunks]
            applied_files_for_finding: List[str] = []
            for edit in edits:
                edit_file = edit.get("file")
                before = edit.get("before")
                after = edit.get("after")
                rag_quote = edit.get("rag_quote")
                if not (edit_file and before and after and rag_quote):
                    skipped.append({"file": rel_path, "rule": rule, "reason": "EDIT_FIELDS_MISSING"})
                    report_entries.append(
                        {"rule": rule, "file": rel_path, "status": "skipped", "reason": "EDIT_FIELDS_MISSING"}
                    )
                    _emit_item(
                        rule_key=rule,
                        rule_type=str(f.get("kind") or "unknown"),
                        action="skipped",
                        reason="EDIT_FIELDS_MISSING",
                        duration_ms=_duration_ms(item_start),
                        file_path=rel_path,
                    )
                    continue
                if not any(rag_quote in text for text in rule_texts):
                    logger.warning(
                        "RAG quote not found for rule %s file %s; quote_len=%d quote_head=%s",
                        rule,
                        edit_file,
                        len(rag_quote),
                        rag_quote[:200].replace("\n", " "),
                    )

                    def _normalize_quote(text: str) -> str:
                        # Normalize whitespace/punctuation for tolerant matching.
                        return re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()

                    def _first_sentence(text: str, limit: int = 240) -> str:
                        if not text:
                            return ""
                        parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
                        sentence = parts[0] if parts else text.strip()
                        return sentence[:limit].strip()

                    def _overlap_score(needle: str, haystack: str) -> int:
                        needle_tokens = set(needle.split())
                        haystack_tokens = set(haystack.split())
                        if not needle_tokens or not haystack_tokens:
                            return 0
                        return len(needle_tokens & haystack_tokens)

                    normalized_quote = _normalize_quote(rag_quote)
                    normalized_hit = any(
                        normalized_quote and normalized_quote in _normalize_quote(text)
                        for text in rule_texts
                    )
                    logger.info(
                        "RAG quote normalized_match=%s normalized_len=%d",
                        normalized_hit,
                        len(normalized_quote),
                    )
                    if not normalized_hit:
                        best_text = ""
                        best_score = 0
                        for text in rule_texts:
                            score = _overlap_score(normalized_quote, _normalize_quote(text))
                            if score > best_score:
                                best_score = score
                                best_text = text
                        if best_score > 0 and best_text:
                            fallback_quote = _first_sentence(best_text)
                            if fallback_quote:
                                edit["rag_quote"] = fallback_quote
                                rag_quote = fallback_quote
                                logger.info(
                                    "RAG quote replaced with fallback sentence for rule %s file %s score=%d",
                                    rule,
                                    edit_file,
                                    best_score,
                                )
                                # Continue with fallback quote, do not skip.
                            else:
                                logger.warning(
                                    "RAG fallback sentence empty for rule %s file %s",
                                    rule,
                                    edit_file,
                                )
                        if not edit.get("rag_quote"):
                            for idx, text in enumerate(rule_texts[:4]):
                                logger.info(
                                    "RAG chunk %d len=%d head=%s",
                                    idx + 1,
                                    len(text),
                                    text[:200].replace("\n", " "),
                                )
                            skipped.append({"file": edit_file, "rule": rule, "reason": "RAG_QUOTE_NOT_FOUND"})
                            report_entries.append(
                                {"rule": rule, "file": edit_file, "status": "skipped", "reason": "RAG_QUOTE_NOT_FOUND"}
                            )
                            _emit_item(
                                rule_key=rule,
                                rule_type=str(f.get("kind") or "unknown"),
                                action="skipped",
                                reason="RAG_QUOTE_NOT_FOUND",
                                duration_ms=_duration_ms(item_start),
                                file_path=edit_file,
                            )
                            continue
                    else:
                        logger.info(
                            "RAG quote accepted after normalization for rule %s file %s",
                            rule,
                            edit_file,
                        )

                normalized = _normalize_edit_path(edit_file, workspace_path)
                if not normalized:
                    skipped.append({"file": edit_file, "rule": rule, "reason": "INVALID_EDIT_PATH"})
                    report_entries.append(
                        {"rule": rule, "file": edit_file, "status": "skipped", "reason": "INVALID_EDIT_PATH"}
                    )
                    _emit_item(
                        rule_key=rule,
                        rule_type=str(f.get("kind") or "unknown"),
                        action="skipped",
                        reason="INVALID_EDIT_PATH",
                        duration_ms=_duration_ms(item_start),
                        file_path=edit_file,
                    )
                    continue
                abs_target = os.path.join(workspace_path, normalized)
                if not os.path.isfile(abs_target):
                    skipped.append({"file": normalized, "rule": rule, "reason": "FILE_NOT_FOUND"})
                    report_entries.append(
                        {"rule": rule, "file": normalized, "status": "skipped", "reason": "FILE_NOT_FOUND"}
                    )
                    _emit_item(
                        rule_key=rule,
                        rule_type=str(f.get("kind") or "unknown"),
                        action="skipped",
                        reason="FILE_NOT_FOUND",
                        duration_ms=_duration_ms(item_start),
                        file_path=normalized,
                    )
                    continue

                with open(abs_target, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                occurrences = content.count(before)
                if occurrences == 0:
                    windowed_content = _replace_once_in_window(
                        content, before, after, start_line=win_start, end_line=win_end
                    )
                    if windowed_content is not None:
                        content = windowed_content
                        occurrences = 1
                if occurrences == 0:
                    normalized_content = _replace_once_with_flexible_whitespace(content, before, after)
                    if normalized_content is not None:
                        content = normalized_content
                        occurrences = 1
                if occurrences == 0:
                    skipped.append({"file": normalized, "rule": rule, "reason": "BEFORE_NOT_FOUND"})
                    report_entries.append(
                        {"rule": rule, "file": normalized, "status": "skipped", "reason": "BEFORE_NOT_FOUND"}
                    )
                    _emit_item(
                        rule_key=rule,
                        rule_type=str(f.get("kind") or "unknown"),
                        action="skipped",
                        reason="BEFORE_NOT_FOUND",
                        duration_ms=_duration_ms(item_start),
                        file_path=normalized,
                    )
                    continue
                if occurrences > 1:
                    skipped.append({"file": normalized, "rule": rule, "reason": "BEFORE_NOT_UNIQUE"})
                    report_entries.append(
                        {"rule": rule, "file": normalized, "status": "skipped", "reason": "BEFORE_NOT_UNIQUE"}
                    )
                    _emit_item(
                        rule_key=rule,
                        rule_type=str(f.get("kind") or "unknown"),
                        action="skipped",
                        reason="BEFORE_NOT_UNIQUE",
                        duration_ms=_duration_ms(item_start),
                        file_path=normalized,
                    )
                    continue

                content = content.replace(before, after, 1)
                with open(abs_target, "w", encoding="utf-8") as fh:
                    fh.write(content)

                if normalized not in changed_files:
                    changed_files.append(normalized)
                if normalized not in applied_files_for_finding:
                    applied_files_for_finding.append(normalized)
                applied.append({"file": normalized, "rule": payload_rule, "kind": f.get("kind"), "plan": plan})
                report_entries.append(
                    {"rule": payload_rule, "file": normalized, "status": "applied", "plan": plan}
                )
                logger.info(
                    "✅ Change applied to file: '%s'\n- Rule: %s\n- Type: %s\n- Workspace: %s\n- Branch: %s",
                    normalized,
                    payload_rule,
                    f.get("kind"),
                    workspace_path,
                    branch,
                )
                _emit_item(
                    rule_key=payload_rule,
                    rule_type=str(f.get("kind") or "unknown"),
                    action="applied",
                    duration_ms=_duration_ms(item_start),
                    file_path=normalized,
                )

            if commit_mode == "per_issue" and applied_files_for_finding:
                try:
                    status = _run_cmd(["git", "status", "--porcelain"], cwd=workspace_path, check=False).stdout.strip()
                    if status:
                        commit_start = time.monotonic()
                        _run_cmd(["git", "add", "-A"], cwd=workspace_path)
                        kind = str(f.get("kind") or "issue")
                        primary_file = applied_files_for_finding[0]
                        issue_key = f.get("issue_key")
                        commit_msg = f"Fix {payload_rule} ({kind}): {primary_file} (CodePulse)"
                        if issue_key:
                            commit_msg = f"{commit_msg} [{issue_key}]"
                        _run_cmd(["git", "commit", "-m", commit_msg], cwd=workspace_path)
                        commit_hash = _run_cmd(["git", "rev-parse", "HEAD"], cwd=workspace_path, check=True).stdout.strip()
                        commits.append(commit_hash)
                        logger.info(
                            "📝 Per-issue commit created: %s (rule=%s file=%s)",
                            commit_hash,
                            payload_rule,
                            primary_file,
                        )
                        _emit_stage("commit", "success", duration_ms=_duration_ms(commit_start))
                    else:
                        logger.info(
                            "Per-issue commit skipped; no changes after applying rule=%s file=%s",
                            payload_rule,
                            applied_files_for_finding[0],
                        )
                except subprocess.CalledProcessError as exc:
                    logger.warning(
                        "Per-issue commit failed for rule=%s file=%s: %s",
                        payload_rule,
                        applied_files_for_finding[0],
                        exc.stderr or exc.stdout,
                    )
                    commit_errors.append(
                        {
                            "rule": payload_rule,
                            "file": applied_files_for_finding[0],
                            "reason": exc.stderr or exc.stdout or "commit failed",
                        }
                    )
                    _emit_stage("commit", "failed", reason=exc.stderr or exc.stdout)
                    continue

        logger.info("Summary of changes:\n- Total files changed: %d\n- Total fixes applied: %d\n- Total skipped: %d", len(changed_files), len(applied), len(skipped))

        report_dir = os.path.join(workspace_path, ".codepulse", "reports")
        os.makedirs(report_dir, exist_ok=True)
        report_stamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        report_path = os.path.join(report_dir, f"sonar_fix_report_{report_stamp}.json")
        report_payload = {
            "workspace": workspace_path,
            "branch": branch,
            "applied_count": len(applied),
            "skipped_count": len(skipped),
            "applied": applied,
            "skipped": skipped,
            "entries": report_entries,
        }
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report_payload, fh, indent=2)

        commit_hash = commits[-1] if commits else None
        push_status = None
        try:
            # Only commit/push if there are actual changes staged by applied edits.
            status = _run_cmd(["git", "status", "--porcelain"], cwd=workspace_path, check=False).stdout.strip()
            if commit_mode == "single":
                if status:
                    commit_start = time.monotonic()
                    _run_cmd(["git", "add", "-A"], cwd=workspace_path)
                    _run_cmd(["git", "commit", "-m", "Fix Sonar issues (CodePulse)"], cwd=workspace_path)
                    commit_hash = _run_cmd(["git", "rev-parse", "HEAD"], cwd=workspace_path, check=True).stdout.strip()
                    commits.append(commit_hash)
                    logger.info("📝 All changes have been committed to branch '%s'.\n- Commit hash: %s", branch, commit_hash)
                    _emit_stage("commit", "success", duration_ms=_duration_ms(commit_start))
                else:
                    logger.info("No changes staged; skipping commit for branch '%s'", branch)

            if (commit_mode == "single" and commit_hash) or (commit_mode == "per_issue" and commits):
                allow_force_push = _bool_env("CODEPULSE_FORCE_PUSH", False)
                try:
                    push_start = time.monotonic()
                    push = _run_cmd(["git", "push", "-u", "origin", branch], cwd=workspace_path, check=False)
                    push_status = push.stdout.strip() or push.stderr.strip() or "pushed"
                    push_ok = push.returncode == 0
                    if not push_ok and allow_force_push:
                        logger.info("Non-fast-forward push detected; retrying with force push as CODEPULSE_FORCE_PUSH is enabled.")
                        try:
                            push_force = _run_cmd(["git", "push", "-f", "origin", branch], cwd=workspace_path, check=False)
                            push_status = push_force.stdout.strip() or push_force.stderr.strip() or "force pushed"
                            push_ok = push_force.returncode == 0
                        except subprocess.CalledProcessError as exc:
                            push_status = f"force push failed: {exc.stderr or exc.stdout}"
                    logger.info("Push attempt result: %s", push_status)
                    _emit_stage(
                        "push",
                        "success" if push_ok else "failed",
                        duration_ms=_duration_ms(push_start),
                        reason=None if push_ok else push_status,
                    )
                except subprocess.CalledProcessError as exc:
                    push_status = f"push failed: {exc.stderr or exc.stdout}"
                    logger.warning("Push failed: %s", push_status)
                    _emit_stage("push", "failed", duration_ms=_duration_ms(push_start), reason=push_status)
            else:
                logger.info("No changes staged; skipping push for branch '%s'", branch)
        except subprocess.CalledProcessError as exc:
            logger.exception("Commit failed")
            _emit_stage("commit", "failed", reason=exc.stderr or exc.stdout)
            _emit_run(
                status="failed",
                reason=f"Failed to commit changes: {exc.stderr}",
                duration_ms=_duration_ms(run_start),
                branch_name=branch,
                sonar_found=True,
            )
            return ToolResult(name="local_fix", output={"error": f"Failed to commit changes: {exc.stderr}", "workspace": workspace_path, "branch": branch})

        _emit_run(
            status="completed",
            duration_ms=_duration_ms(run_start),
            branch_name=branch,
            sonar_found=True,
        )
        return ToolResult(
            name="local_fix",
            output={
                "workspace": workspace_path,
                "branch": branch,
                "commit": commit_hash,
                "commits": commits,
                "push_status": push_status,
                "files_changed": changed_files,
                "applied": applied,
                "skipped": skipped,
                "commit_errors": commit_errors,
                "report_path": report_path,
            },
        )

    async def comment_pr_with_branch(
        self,
        task: str,
        git_payload: Dict[str, Any],
        local_fix_payload: Dict[str, Any],
    ) -> ToolResult:
        # Post results back to GitHub PR with branch/commit status when fixes requested.
        stage_start = time.monotonic()

        def _emit_stage(status: str, reason: Optional[str] = None) -> None:
            run_id = telemetry.get_sonar_run_id()
            if not run_id:
                return
            payload = {
                "event": "sonar_fix_stage",
                "run_id": run_id,
                "stage": "pr_comment",
                "status": status,
                "duration_ms": max(0, int((time.monotonic() - stage_start) * 1000)),
            }
            if reason:
                payload["reason"] = reason
            telemetry.emit_structured_log(payload)

        if not (
            ("fix" in task.lower())
            or ("apply the changes" in task.lower())
            or ("apply changes" in task.lower())
        ):
            _emit_stage("failed", "No fix requested")
            return ToolResult(
                name="pr_comment",
                output={"skipped": True, "reason": "No fix requested"},
            )

        pr = (git_payload or {}).get("pull_request") or {}
        repo_obj = (git_payload or {}).get("repo") or {}
        full_name = repo_obj.get("full_name")

        if not full_name or "/" not in full_name:
            _emit_stage("failed", "Repo full_name missing")
            return ToolResult(
                name="pr_comment",
                output={"skipped": True, "reason": "Repo full_name missing"},
            )

        owner, repo = full_name.split("/", 1)
        pull_number = pr.get("number") or git_payload.get("pull_number")

        if not pull_number:
            _emit_stage("failed", "Pull number missing")
            return ToolResult(
                name="pr_comment",
                output={"skipped": True, "reason": "Pull number missing"},
            )

        token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or os.getenv("GIT_TOKEN")
        branch = (local_fix_payload or {}).get("branch")
        commit = (local_fix_payload or {}).get("commit")
        applied = (local_fix_payload or {}).get("applied") or []
        skipped_raw = (local_fix_payload or {}).get("skipped") or []
        skipped = skipped_raw if isinstance(skipped_raw, list) else []
        report_path = (local_fix_payload or {}).get("report_path")
        error = (local_fix_payload or {}).get("error")

        body_lines = ["CodePulse Sonar AutoFix update:"]
        if branch:
            body_lines.append(f"- Branch: `{branch}`")
        if commit:
            body_lines.append(f"- Commit: `{commit}`")
            body_lines.append(f"- Branch link: https://github.com/{owner}/{repo}/tree/{branch}")
        else:
            body_lines.append("- Commit: none")
        body_lines.append(f"- Applied fixes: {len(applied)}")
        body_lines.append(f"- Skipped findings: {len(skipped)}")
        if error:
            body_lines.append(f"- Error: {error}")
        if report_path:
            body_lines.append(f"- Report: `{report_path}`")

        if applied:
            body_lines.append("")
            body_lines.append("Applied summary:")
            for item in applied[:10]:
                rule = item.get("rule") or "unknown-rule"
                file = item.get("file") or "unknown-file"
                plan = item.get("plan") or "no plan"
                body_lines.append(f"- {rule} | {file} | {plan}")
        if skipped:
            body_lines.append("")
            body_lines.append("Skipped reasons (first 3):")
            for s in skipped[:3]:
                body_lines.append(f"- {s.get('rule') or 'unknown-rule'} | {s.get('file')}: {s.get('reason')}")

        body = "\n".join(body_lines)

        if not token:
            workspace = (local_fix_payload or {}).get("workspace") or os.getcwd()
            report_dir = os.path.join(workspace, ".codepulse", "reports")
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, f"pr_{pull_number}_comment.md")
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write(body + "\n")
            logger.warning("GITHUB_TOKEN/GH_TOKEN missing; PR comment written to %s", report_path)
            _emit_stage("failed", "GITHUB_TOKEN/GH_TOKEN not set")
            return ToolResult(
                name="pr_comment",
                output={"skipped": True, "reason": "GITHUB_TOKEN/GH_TOKEN not set", "report_path": report_path},
            )

        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pull_number}/comments"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        }

        logger.info("Posting PR comment to %s", url)
        resp = requests.post(url, headers=headers, json={"body": body}, timeout=30)
        if resp.status_code >= 400:
            logger.warning("PR comment failed status=%s body=%s", resp.status_code, resp.text[:500])
            _emit_stage("failed", f"status={resp.status_code}")
        else:
            _emit_stage("success")

        return ToolResult(
            name="pr_comment",
            output={"status": resp.status_code},
        )

    async def comment_pr_with_summary(
        self,
        git_payload: Dict[str, Any],
        summary: str,
    ) -> ToolResult:
        # Post the generated summary back to the PR as a separate comment.
        stage_start = time.monotonic()

        def _emit_stage(status: str, reason: Optional[str] = None) -> None:
            run_id = telemetry.get_sonar_run_id()
            if not run_id:
                return
            payload = {
                "event": "sonar_fix_stage",
                "run_id": run_id,
                "stage": "pr_summary",
                "status": status,
                "duration_ms": max(0, int((time.monotonic() - stage_start) * 1000)),
            }
            if reason:
                payload["reason"] = reason
            telemetry.emit_structured_log(payload)

        token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or os.getenv("GIT_TOKEN")
        if not token:
            _emit_stage("failed", "GITHUB_TOKEN/GH_TOKEN/GIT_TOKEN not set")
            return ToolResult(
                name="pr_summary_comment",
                output={"skipped": True, "reason": "GITHUB_TOKEN/GH_TOKEN/GIT_TOKEN not set"},
            )

        if not summary or not summary.strip():
            _emit_stage("failed", "Empty summary")
            return ToolResult(
                name="pr_summary_comment",
                output={"skipped": True, "reason": "Empty summary"},
            )

        pr = (git_payload or {}).get("pull_request") or {}
        repo_obj = (git_payload or {}).get("repo") or {}
        full_name = repo_obj.get("full_name")

        if not full_name or "/" not in full_name:
            _emit_stage("failed", "Repo full_name missing")
            return ToolResult(
                name="pr_summary_comment",
                output={"skipped": True, "reason": "Repo full_name missing"},
            )

        owner, repo = full_name.split("/", 1)
        pull_number = pr.get("number") or git_payload.get("pull_number")

        if not pull_number:
            _emit_stage("failed", "Pull number missing")
            return ToolResult(
                name="pr_summary_comment",
                output={"skipped": True, "reason": "Pull number missing"},
            )

        body = f"CodePulse summary:\n\n{summary.strip()}"
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pull_number}/comments"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        }

        logger.info("Posting summary PR comment to %s", url)
        resp = requests.post(url, headers=headers, json={"body": body}, timeout=30)
        if resp.status_code >= 400:
            _emit_stage("failed", f"status={resp.status_code}")
        else:
            _emit_stage("success")

        return ToolResult(
            name="pr_summary_comment",
            output={"status": resp.status_code},
        )
