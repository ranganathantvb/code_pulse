import os
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

import requests

# Avoid tokenizer fork warnings/deadlocks when forking LLM workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# --- Unified diff helpers ---
def extract_unified_diff(text: str) -> str:
    """Extract unified diff from raw model output, respecting ```diff fences."""
    if not text:
        return ""
    fenced = re.search(r"```diff\s*(.*?)\s*```", text, flags=re.S | re.I)
    if fenced:
        return fenced.group(1).strip() + "\n"
    plain = re.search(r"(^---\s+.*)", text, flags=re.S | re.M)
    if plain:
        return text[plain.start():].strip() + "\n"
    return text.strip() + "\n"


def is_valid_unified_diff(diff_text: str) -> bool:
    return bool(diff_text and "--- " in diff_text and "+++ " in diff_text and "@@ " in diff_text)


def diff_creates_file(diff_text: str) -> bool:
    """Return True if diff attempts to create a new file (e.g., --- /dev/null)."""
    if not diff_text:
        return False
    return ("--- /dev/null" in diff_text) or ("new file mode" in diff_text)
from code_pulse.mcp.clients import client_factory
from code_pulse.mcp.git import GitClient
from code_pulse.mcp.sonar import SonarClient
from code_pulse.mcp.jira import JiraClient
from code_pulse.rag.service import RAGService
from code_pulse.config import get_settings, Settings
from code_pulse.logger import logger


@dataclass
class ToolResult:
    name: str
    output: Any


class Tooling:
    def __init__(self):
        self.settings: Settings = get_settings()
        clients = client_factory(self.settings)
        self.git: GitClient = clients["git"]  # type: ignore[assignment]
        self.sonar: SonarClient = clients["sonar"]  # type: ignore[assignment]
        self.jira: JiraClient = clients["jira"]  # type: ignore[assignment]
        self.rag = RAGService()

    def _parse_pull_url(self, url: str) -> Optional[Dict[str, str]]:
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
        if pull_url:
            parsed = self._parse_pull_url(pull_url)
            if parsed:
                owner = owner or parsed["owner"]
                repo = repo or parsed["repo"]
                if pull_number is None:
                    pull_number = parsed["number"]
        if not owner or not repo:
            raise ValueError("Git tool requires owner and repo")

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
        # Always try RAG for Sonar guidance.
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
        if project_key or project_search:
            async with self.sonar as client:
                if project_key:
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
                            pr_hotspots = await client.pr_hotspots(
                                project_key, pr_number, status=hotspot_status
                            )
                            pr_hotspots_count = pr_hotspots.get("paging", {}).get("total")
                        except Exception:  # noqa: BLE001
                            pr_hotspots = {"message": "Hotspots endpoint not available."}
                        try:
                            pr_quality_gate = await client.pr_quality_gate(project_key, pr_number)
                        except Exception:  # noqa: BLE001
                            pr_quality_gate = {"message": "Quality gate endpoint not available."}
                if project_search:
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
        message = None if excerpts else "No relevant documents found."
        return ToolResult(name="rag", output={"matches": excerpts, "message": message})

    async def fix_sonar_locally(
        self,
        task: str,
        sonar_payload: Dict[str, Any],
        workspace_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Attempt local Sonar remediations with guarded application + commit.
        - Supports BUG/VULNERABILITY/CODE_SMELL and SECURITY_HOTSPOT.
        - Prefers minimal hotspot changes (validation/safe API/annotations) over refactors.
        """
        import subprocess
        import tempfile
        from typing import List, Dict, Any, Optional, Tuple

        def _run_cmd(cmd: List[str], cwd: str, check: bool = True) -> subprocess.CompletedProcess:
            logger.info("CMD: %s (cwd=%s)", " ".join(cmd), cwd)
            return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)

        def _is_git_repo(path: str) -> bool:
            try:
                cp = _run_cmd(["git", "rev-parse", "--is-inside-work-tree"], cwd=path, check=True)
                return cp.stdout.strip().lower() == "true"
            except Exception:
                return False

        def _extract_branch_name(text: str) -> Optional[str]:
            m = re.search(r"\bbranch\b\s*[\"\']([^\"\']+)[\"\']", text, re.IGNORECASE)
            return m.group(1).strip() if m else None

        def _wants_fix(text: str) -> bool:
            t = text.lower()
            return ("fix" in t) or ("apply the changes" in t) or ("apply changes" in t)

        def _normalize_component_to_path(component: str, project_key: Optional[str] = None) -> str:
            if ":" in component:
                return component.split(":", 1)[1]
            if project_key and component.startswith(project_key):
                return component[len(project_key):].lstrip("/\\")
            return component

        def _windowed_lines(content: str, start_line: int, radius: int = 60) -> Tuple[int, int, str]:
            lines = content.splitlines()
            start_idx = max(0, start_line - 1 - radius)
            end_idx = min(len(lines), start_line - 1 + radius)
            return start_idx + 1, end_idx, "\n".join(lines[start_idx:end_idx])

        logger.info("ENTER fix_sonar_locally task=%s", task)

        if not _wants_fix(task):
            return ToolResult(name="local_fix", output={"skipped": True, "reason": "Task does not request fixes."})

        workspace_path = workspace_path or os.getenv("CODEPULSE_WORKSPACE_PATH")
        if not workspace_path:
            logger.error("CODEPULSE_WORKSPACE_PATH not set")
            return ToolResult(name="local_fix", output={"skipped": True, "reason": "CODEPULSE_WORKSPACE_PATH not set."})

        workspace_path = os.path.expanduser(workspace_path)
        logger.info("Workspace=%s", workspace_path)

        if not os.path.isdir(workspace_path):
            logger.error("Workspace path does not exist: %s", workspace_path)
            return ToolResult(name="local_fix", output={"skipped": True, "reason": f"Workspace path does not exist: {workspace_path}"})

        if not _is_git_repo(workspace_path):
            logger.error("Workspace is not a git repository: %s", workspace_path)
            return ToolResult(name="local_fix", output={"skipped": True, "reason": f"Workspace is not a git repository: {workspace_path}"})

        branch = _extract_branch_name(task) or "sonar_AI_fix"
        logger.info("Target branch=%s", branch)

        pr_issues = (sonar_payload or {}).get("pr_issues") or {}
        pr_hotspots = (sonar_payload or {}).get("pr_hotspots") or {}
        issues: List[Dict[str, Any]] = pr_issues.get("issues") or []
        hotspots: List[Dict[str, Any]] = pr_hotspots.get("hotspots") or []

        findings: List[Dict[str, Any]] = []
        for iss in issues:
            findings.append({
                "kind": (iss.get("type") or "issue").upper(),
                "rule": iss.get("rule"),
                "message": iss.get("message"),
                "component": iss.get("component"),
                "textRange": iss.get("textRange") or {},
            })
        for hs in hotspots:
            findings.append({
                "kind": "SECURITY_HOTSPOT",
                "rule": hs.get("rule") or hs.get("ruleKey"),
                "message": hs.get("message"),
                "component": hs.get("component") or hs.get("project"),
                "textRange": hs.get("textRange") or {},
            })

        logger.info("Findings total=%d (issues=%d, hotspots=%d)", len(findings), len(issues), len(hotspots))

        if not findings:
            return ToolResult(name="local_fix", output={"skipped": True, "reason": "No Sonar findings available to fix.", "workspace": workspace_path, "branch": branch})

        # checkout/create/reset branch
        try:
            _run_cmd(["git", "fetch", "--all", "--prune"], cwd=workspace_path, check=False)
            _run_cmd(["git", "checkout", "-B", branch], cwd=workspace_path)
            current = _run_cmd(["git", "branch", "--show-current"], cwd=workspace_path, check=True).stdout.strip()
            logger.info("Current branch after checkout=%s", current)
        except subprocess.CalledProcessError as exc:
            logger.exception("Branch checkout failed")
            return ToolResult(name="local_fix", output={"error": f"Failed to create/checkout branch '{branch}': {exc.stderr}", "workspace": workspace_path, "branch": branch})

        # LLM setup (Ollama)
        try:
            from langchain_ollama import ChatOllama  # type: ignore
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
            llm = ChatOllama(model="deepseek-coder:6.7b", temperature=0.1, base_url=base_url)
            chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
        except Exception as exc:
            logger.exception("Ollama init failed")
            return ToolResult(name="local_fix", output={"skipped": True, "reason": f"Ollama unavailable: {exc}", "workspace": workspace_path, "branch": branch})

        changed_files: List[str] = []
        applied: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []

        project_key = (sonar_payload or {}).get("project_key") or getattr(self.settings, "sonar_project_key", None)

        for f in findings:
            component = str(f.get("component") or "")
            rel_path = _normalize_component_to_path(component, project_key=project_key)
            abs_path = os.path.join(workspace_path, rel_path)

            tr = f.get("textRange") or {}
            start_line = int(tr.get("startLine") or 1)

            if not os.path.isfile(abs_path):
                logger.warning("File not found: %s", abs_path)
                skipped.append({"file": rel_path, "reason": "File not found in workspace"})
                continue

            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
            win_start, win_end, window = _windowed_lines(content, start_line=start_line, radius=60)

            rule = f.get("rule") or "SONAR_RULE"
            rag_docs = self.rag.query(f"How to fix Sonar rule {rule}", namespace="sonar")
            rag_excerpt = "\n\n".join(d.page_content[:800] for d in rag_docs[:3])

            prompt = (
                "You are an automated remediation agent.\n"
                "Output ONLY a unified diff (git-style) that applies cleanly with `git apply`.\n"
                "Do NOT create new files. Use headers --- a/<path> and +++ b/<path> with @@ hunks.\n"
                "Minimize changes; avoid functional regressions. Prefer minimal, safe edits for hotspots (validation, safe APIs, annotations).\n"
                "Avoid suppressions like NOSONAR unless unavoidable.\n\n"
                f"Finding:\n- Kind: {f.get('kind')}\n- Rule: {rule}\n- Message: {f.get('message') or ''}\n- File: {rel_path}\n- Line: {start_line}\n\n"
                f"Guidance:\n{rag_excerpt}\n\n"
                f"File excerpt (lines {win_start}-{win_end}):\n{window}\n"
            )

            try:
                raw = await chain.ainvoke({"prompt": prompt})
                diff_text = extract_unified_diff(raw)
                if not is_valid_unified_diff(diff_text):
                    retry_prompt = prompt + (
                        "\n\nIMPORTANT: Output ONLY a unified diff. Do NOT create new files. "
                        "Use headers exactly like '--- a/{path}' and '+++ b/{path}' and include '@@' hunk headers. No explanations."
                    )
                    raw2 = await chain.ainvoke({"prompt": retry_prompt})
                    diff_text = extract_unified_diff(raw2)
            except Exception as exc:
                logger.exception("LLM invocation failed for %s", rel_path)
                skipped.append({"file": rel_path, "reason": f"LLM failure: {exc}"})
                continue

            if not is_valid_unified_diff(diff_text):
                logger.warning("Invalid diff from model for %s", rel_path)
                skipped.append({"file": rel_path, "reason": "Model did not produce a valid unified diff"})
                continue

            if diff_creates_file(diff_text):
                logger.warning("Model diff attempts to create a file for %s; skipping", rel_path)
                skipped.append({"file": rel_path, "reason": "Model diff attempted to create a new file"})
                continue

            tf_path = None
            try:
                with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tf:
                    tf.write(diff_text)
                    tf_path = tf.name

                chk = subprocess.run(
                    ["git", "apply", "--check", "--whitespace=nowarn", tf_path],
                    cwd=workspace_path,
                    capture_output=True,
                    text=True,
                )
                if chk.returncode != 0:
                    logger.warning("git apply --check failed for %s: %s", rel_path, chk.stderr or chk.stdout)
                    ap = subprocess.run(
                        ["git", "apply", "--3way", "--whitespace=nowarn", tf_path],
                        cwd=workspace_path,
                        capture_output=True,
                        text=True,
                    )
                    if ap.returncode != 0:
                        skipped.append({"file": rel_path, "reason": f"git apply failed: {ap.stderr or ap.stdout}"})
                        continue
                else:
                    ap2 = subprocess.run(
                        ["git", "apply", "--whitespace=nowarn", tf_path],
                        cwd=workspace_path,
                        capture_output=True,
                        text=True,
                    )
                    if ap2.returncode != 0:
                        skipped.append({"file": rel_path, "reason": f"git apply failed: {ap2.stderr or ap2.stdout}"})
                        continue

                if rel_path not in changed_files:
                    changed_files.append(rel_path)
                applied.append({"file": rel_path, "rule": rule, "kind": f.get("kind")})
                logger.info("Applied diff to %s", rel_path)
            except Exception as exc:
                logger.warning("Patch apply failed for %s: %s", rel_path, exc)
                skipped.append({"file": rel_path, "reason": f"Patch apply failed: {exc}"})
            finally:
                if tf_path:
                    try:
                        os.unlink(tf_path)
                    except Exception:
                        pass

        logger.info("Apply summary: changed_files=%d applied=%d skipped=%d", len(changed_files), len(applied), len(skipped))

        commit_hash = None
        try:
            status = _run_cmd(["git", "status", "--porcelain"], cwd=workspace_path, check=False).stdout.strip()
            if status:
                _run_cmd(["git", "add", "-A"], cwd=workspace_path)
                _run_cmd(["git", "commit", "-m", "Fix Sonar issues (CodePulse)"], cwd=workspace_path)
                commit_hash = _run_cmd(["git", "rev-parse", "HEAD"], cwd=workspace_path, check=True).stdout.strip()
                logger.info("Commit created: %s", commit_hash)
        except subprocess.CalledProcessError as exc:
            logger.exception("Commit failed")
            return ToolResult(name="local_fix", output={"error": f"Failed to commit changes: {exc.stderr}", "workspace": workspace_path, "branch": branch})

        return ToolResult(
            name="local_fix",
            output={
                "workspace": workspace_path,
                "branch": branch,
                "commit": commit_hash,
                "files_changed": changed_files,
                "applied": applied,
                "skipped": skipped,
            },
        )

        # --- ADDITIVE: Comment on PR with branch/commit after sonar fixes are requested ---
    async def comment_pr_with_branch(
        self,
        task: str,
        git_payload: Dict[str, Any],
        local_fix_payload: Dict[str, Any],
    ) -> ToolResult:
        if not (
            ("fix" in task.lower())
            or ("apply the changes" in task.lower())
            or ("apply changes" in task.lower())
        ):
            return ToolResult(
                name="pr_comment",
                output={"skipped": True, "reason": "No fix requested"},
            )

        token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        if not token:
            return ToolResult(
                name="pr_comment",
                output={"skipped": True, "reason": "GITHUB_TOKEN/GH_TOKEN not set"},
            )

        pr = (git_payload or {}).get("pull_request") or {}
        repo_obj = (git_payload or {}).get("repo") or {}
        full_name = repo_obj.get("full_name")

        if not full_name or "/" not in full_name:
            return ToolResult(
                name="pr_comment",
                output={"skipped": True, "reason": "Repo full_name missing"},
            )

        owner, repo = full_name.split("/", 1)
        pull_number = pr.get("number") or git_payload.get("pull_number")

        if not pull_number:
            return ToolResult(
                name="pr_comment",
                output={"skipped": True, "reason": "Pull number missing"},
            )

        branch = (local_fix_payload or {}).get("branch")
        commit = (local_fix_payload or {}).get("commit")
        applied = (local_fix_payload or {}).get("applied") or []
        skipped = (local_fix_payload or {}).get("skipped") or []

        body_lines = ["CodePulse Sonar AutoFix update:"]
        if branch:
            body_lines.append(f"- Branch: `{branch}`")
        if commit:
            body_lines.append(f"- Commit: `{commit}`")
        else:
            body_lines.append("- Commit: none")
        body_lines.append(f"- Applied fixes: {len(applied)}")
        body_lines.append(f"- Skipped findings: {len(skipped)}")
        if skipped:
            body_lines.append("")
            body_lines.append("Skipped reasons (first 3):")
            for s in skipped[:3]:
                body_lines.append(f"- {s.get('file')}: {s.get('reason')}")

        body = "\n".join(body_lines)

        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pull_number}/comments"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        }

        logger.info("Posting PR comment to %s", url)
        resp = requests.post(url, headers=headers, json={"body": body}, timeout=30)

        return ToolResult(
            name="pr_comment",
            output={"status": resp.status_code},
        )
