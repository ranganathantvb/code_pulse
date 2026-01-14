import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path

from code_pulse.agents.tools import Tooling
from code_pulse.rag.service import RAGService


def _run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def main() -> int:
    rules_path = os.getenv("CODEPULSE_SONAR_RULES_PATH") or ".data/ingest/sonar_rule_data.json"
    rules_file = Path(rules_path)
    if not rules_file.exists():
        print(f"Missing rules file: {rules_file}")
        return 1

    rag = RAGService()
    chunks = rag.ingest_sonar_rules(rules_file, namespace="sonar")
    print(f"Ingested {chunks} chunks from {rules_file}")

    payload = json.loads(rules_file.read_text())
    rules = payload.get("rules", [])
    if not rules:
        print("No rules found in sonar_rule_data.json")
        return 2
    rule_key = rules[0].get("key") or rules[0].get("rule_key")
    if not rule_key:
        print("First rule missing key")
        return 3

    matches = rag.lookup_by_metadata(
        namespace="sonar",
        metadata_filter={"doc_type": "sonar_rule", "rule_key": rule_key},
    )
    if not matches:
        print(f"Lookup failed for rule_key={rule_key}")
        return 4
    rag_quote = matches[0].page_content.splitlines()[0].strip()
    print(f"Verified retrieval for {rule_key}")

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir)
        target = repo / "src" / "demo.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("before_marker\n", encoding="utf-8")
        _run(["git", "init"], cwd=repo)
        _run(["git", "add", "-A"], cwd=repo)
        _run(["git", "commit", "-m", "init"], cwd=repo)

        mock_response = json.dumps(
            {
                "rule": rule_key,
                "plan": "Replace marker per rule guidance.",
                "edits": [
                    {
                        "file": "src/demo.txt",
                        "before": "before_marker",
                        "after": "after_marker",
                        "rag_quote": rag_quote,
                    }
                ],
                "commands": [],
                "notes": "self-check",
            }
        )

        os.environ["CODEPULSE_WORKSPACE_PATH"] = str(repo)
        os.environ["CODEPULSE_FIX_MODEL"] = "mock"
        os.environ["CODEPULSE_FIX_MOCK_RESPONSE"] = mock_response

        tooling = Tooling()
        sonar_payload = {
            "pr_issues": {
                "issues": [
                    {
                        "rule": rule_key,
                        "message": "self-check",
                        "component": f"project:{target.as_posix()}",
                        "textRange": {"startLine": 1},
                        "type": "CODE_SMELL",
                    }
                ]
            }
        }

        result = asyncio.run(
            tooling.fix_sonar_locally(
                "fix sonar issues",
                sonar_payload,
                workspace_path=str(repo),
            )
        )
        output = result.output
        branch = output.get("branch")
        if not branch or not branch.startswith("Fix_sonar_issues_"):
            print(f"Unexpected branch name: {branch}")
            return 5
        current = _run(["git", "branch", "--show-current"], cwd=repo).stdout.strip()
        if current != branch:
            print(f"Branch mismatch: expected {branch}, got {current}")
            return 6
        updated = target.read_text(encoding="utf-8")
        if "after_marker" not in updated:
            print("Edit did not apply")
            return 7

        print("Self-check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
