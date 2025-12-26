import os
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

import requests

# Avoid tokenizer fork warnings/deadlocks when forking LLM workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# --- Unified diff helpers ---
# Pull out diff segments from LLM output so downstream git apply checks make sense.
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
    # Basic structural check so we do not attempt to apply malformed patches.
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

    # Heavyweight path: attempt to auto-apply Sonar fixes locally via LLM + git apply.
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

        # Extract explicit branch override from the task text if present.
        def _extract_branch_name(text: str) -> Optional[str]:
            m = re.search(r"\bbranch\b\s*[\"\']([^\"\']+)[\"\']", text, re.IGNORECASE)
            return m.group(1).strip() if m else None

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

        def _new_file_target(diff_text: str) -> Optional[str]:
            """Return target path if diff creates a new file (--- /dev/null to +++ b/path)."""
            if not diff_text:
                return None
            new_file_match = re.search(r"^\+\+\+\s+b/([^\n]+)", diff_text, flags=re.M)
            creates_file = re.search(r"^---\s+/dev/null", diff_text, flags=re.M)
            if new_file_match and creates_file:
                return new_file_match.group(1).strip()
            return None

        def _windowed_lines(content: str, start_line: int, radius: int = 60) -> Tuple[int, int, str]:
            lines = content.splitlines()
            start_idx = max(0, start_line - 1 - radius)
            end_idx = min(len(lines), start_line - 1 + radius)
            return start_idx + 1, end_idx, "\n".join(lines[start_idx:end_idx])

        logger.info("ENTER fix_sonar_locally task=%s", task)

        if not _wants_fix(task):
            # Guardrail: only run heavy remediation flow when explicitly requested.
            logger.info("Local fix skipped: task did not request fixes")
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

        # Combine issues and hotspots into a single list of findings to remediate.
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
            logger.info("Local fix skipped: Sonar payload contained no findings")
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
            rag_query = f"How to fix Sonar rule {rule}"
            logger.info("RAG retrieval: query='%s', namespace='sonar'", rag_query)
            rag_docs = self.rag.query(rag_query, namespace="sonar")
            rag_excerpt = "\n\n".join(d.page_content[:800] for d in rag_docs[:3])
            logger.info("RAG retrieved %d chunk(s) for rule %s", len(rag_docs), rule)
            for idx, doc in enumerate(rag_docs[:3]):
                logger.info("RAG chunk %d for rule %s: %s", idx + 1, rule, doc.page_content[:300].replace("\n", " "))

            prompt = (
                "You are a local code remediation agent running inside a developer workstation with read/write access to the repository.\n"
                "Your goal is to fix the provided Sonar findings by editing files locally and validating the changes.\n\n"
                "INPUTS YOU WILL RECEIVE:\n"
                "1) Sonar Findings (rule, message, file path, and ideally the line range)\n"
                "2) RAG Content (rule guidance and remediation patterns)\n"
                "3) Current code context (either the full file content or at minimum the exact method(s) and surrounding lines)\n\n"
                "IMPORTANT CONSTRAINTS:\n"
                "- Do NOT output a unified diff or patch in the narrative sections.\n"
                "- Create fixes in new files prefixed with 'AI_' (e.g., AI_<OriginalName>.java) instead of modifying the original file directly unless a minimal wiring change is unavoidable.\n"
                "- Prefer minimal changes that compile and preserve behavior.\n"
                "- Avoid suppressions (NOSONAR / @SuppressWarnings) unless there is no safe alternative.\n"
                "- Apply fixes for the actual rule intent; do not add unrelated validation unless it directly addresses the issue.\n"
                "- Keep edits localized; if you must touch the original file, only add minimal glue to use the new AI_ file.\n\n"
                "WORKFLOW YOU MUST FOLLOW:\n"
                "A) Identify exactly where each Sonar issue occurs in the provided code context.\n"
                "B) Choose the least risky remediation that satisfies the rule:\n"
                "   - java:S2119 “Save and re-use this Random”: avoid per-call new Random(); use ThreadLocalRandom OR reuse a static final Random if appropriate.\n"
                "   - java:S2245 “Pseudorandom generator safety”: if used for security, use SecureRandom; if not security-sensitive, use ThreadLocalRandom and explicitly keep it non-security.\n"
                "   - java:S4790 “Weak hash algorithm”: if used for passwords or security, replace with a stronger construct (bcrypt/PBKDF2/HMAC-SHA-256 depending on usage); if non-security integrity/dedup, prefer SHA-256.\n"
                "C) Edit the file(s) locally using direct file writes (you can describe the edits explicitly).\n"
                "D) Run local verification commands (compile/tests). If tests fail, fix and re-run.\n"
                "E) Report exactly what you changed and why, mapping each change to a Sonar rule.\n\n"
                "OUTPUT FORMAT (STRICT):\n"
                "1) PLAN\n"
                "   - Bullet list of issues to fix and chosen remediation per rule.\n"
                "2) EDITS TO APPLY (LOCAL)\n"
                "   For each file:\n"
                "   - File path (use AI_ prefixed files for new code; minimal glue in originals only if required)\n"
                "   - “Before” snippet (only the relevant lines/method)\n"
                "   - “After” snippet (only the updated lines/method)\n"
                "   - Explanation of why this satisfies the rule\n"
                "3) COMMANDS TO RUN\n"
                "   - Commands to format (if applicable), compile, and test (e.g., mvn -q -DskipTests=false test)\n"
                "4) VALIDATION RESULTS\n"
                "   - State what would be checked (build/tests/sonar scan), and any expected side-effects.\n"
                "   - If any uncertainty remains because code context is incomplete, state what additional context is needed (but still provide best-effort edits based on what was provided).\n\n"
                "NOW FIX THESE FINDINGS USING THE RAG CONTENT BELOW AND THE PROVIDED CODE CONTEXT.\n\n"
                "SONAR FINDINGS:\n"
                f"- Kind: {f.get('kind')}\n- Rule: {rule}\n- Message: {f.get('message') or ''}\n- File: {rel_path}\n- Line: {start_line}\n\n"
                "RAG CONTENT (RULE GUIDANCE):\n"
                f"{rag_excerpt}\n\n"
                "CODE CONTEXT (CURRENT FILE CONTENT OR RELEVANT METHOD(S)):\n"
                f"{window}\n\n"
                "REPOSITORY ROOT PATH:\n"
                f"{workspace_path}\n\n"
                "After producing the PLAN/EDITS/COMMANDS/VALIDATION sections above, output ONLY the final unified diff (git-style) that applies cleanly with `git apply`, using headers --- a/<path> and +++ b/<path> with @@ hunks, and nothing else after the diff. For new files, ensure the path starts with AI_ and is placed alongside the original."
            )
            logger.info("Prompt sent to LLM for rule %s: %s", rule, prompt[:500].replace("\n", " "))

            try:
                raw = await chain.ainvoke({"prompt": prompt})
                logger.info("Raw LLM response for rule %s: %s", rule, raw[:500].replace("\n", " "))
                diff_text = extract_unified_diff(raw)
                if not is_valid_unified_diff(diff_text):
                    # Retry once with a stricter prompt when the model does not emit a clean patch.
                    retry_prompt = prompt + (
                        "\n\nIMPORTANT: Output ONLY a unified diff. Do NOT create new files. "
                        "Use headers exactly like '--- a/{path}' and '+++ b/{path}' and include '@@' hunk headers. No explanations."
                    )
                    raw2 = await chain.ainvoke({"prompt": retry_prompt})
                    logger.info("Raw LLM retry response for rule %s: %s", rule, raw2[:500].replace("\n", " "))
                    diff_text = extract_unified_diff(raw2)
            except Exception as exc:
                logger.exception("LLM invocation failed for %s", rel_path)
                skipped.append({"file": rel_path, "reason": f"LLM failure: {exc}"})
                continue

            if not is_valid_unified_diff(diff_text):
                logger.warning("Invalid diff from model for %s", rel_path)
                skipped.append({"file": rel_path, "reason": "Model did not produce a valid unified diff"})
                continue

            new_file_target = _new_file_target(diff_text)
            if new_file_target:
                # Guardrail: AI-generated files must be prefixed to avoid overwriting real code.
                if not os.path.basename(new_file_target).startswith("AI_"):
                    logger.warning("Model diff attempts to create new file '%s' without AI_ prefix; skipping", new_file_target)
                    skipped.append({"file": new_file_target, "reason": "New files must use AI_ prefix"})
                    continue
            elif diff_creates_file(diff_text):
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
                    # If a clean apply fails, try 3-way merge to improve robustness on stale contexts.
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
                logger.info("✅ Change applied to file: '%s'\n- Rule: %s\n- Type: %s\n- Workspace: %s\n- Branch: %s\n- Patch Preview:\n%s", rel_path, rule, f.get("kind"), workspace_path, branch, diff_text[:500])
            except Exception as exc:
                logger.warning("Patch apply failed for %s: %s", rel_path, exc)
                skipped.append({"file": rel_path, "reason": f"Patch apply failed: {exc}"})
            finally:
                if tf_path:
                    try:
                        os.unlink(tf_path)
                    except Exception:
                        pass

        logger.info("Summary of changes:\n- Total files changed: %d\n- Total fixes applied: %d\n- Total skipped: %d", len(changed_files), len(applied), len(skipped))

        commit_hash = None
        push_status = None
        try:
            # Only commit/push if there are actual changes staged by applied patches.
            status = _run_cmd(["git", "status", "--porcelain"], cwd=workspace_path, check=False).stdout.strip()
            if status:
                _run_cmd(["git", "add", "-A"], cwd=workspace_path)
                _run_cmd(["git", "commit", "-m", "Fix Sonar issues (CodePulse)"], cwd=workspace_path)
                commit_hash = _run_cmd(["git", "rev-parse", "HEAD"], cwd=workspace_path, check=True).stdout.strip()
                logger.info("📝 All changes have been committed to branch '%s'.\n- Commit hash: %s", branch, commit_hash)
                try:
                    push = _run_cmd(["git", "push", "-u", "origin", branch], cwd=workspace_path, check=False)
                    push_status = push.stdout.strip() or push.stderr.strip() or "pushed"
                    logger.info("Push attempt result: %s", push_status)
                except subprocess.CalledProcessError as exc:
                    push_status = f"push failed: {exc.stderr or exc.stdout}"
                    logger.warning("Push failed: %s", push_status)
            else:
                logger.info("No changes staged; skipping commit/push for branch '%s'", branch)
        except subprocess.CalledProcessError as exc:
            logger.exception("Commit failed")
            return ToolResult(name="local_fix", output={"error": f"Failed to commit changes: {exc.stderr}", "workspace": workspace_path, "branch": branch})

        return ToolResult(
            name="local_fix",
            output={
                "workspace": workspace_path,
                "branch": branch,
                "commit": commit_hash,
                "push_status": push_status,
                "files_changed": changed_files,
                "applied": applied,
                "skipped": skipped,
            },
        )

    async def comment_pr_with_branch(
        self,
        task: str,
        git_payload: Dict[str, Any],
        local_fix_payload: Dict[str, Any],
    ) -> ToolResult:
        # Post results back to GitHub PR with branch/commit status when fixes requested.
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
