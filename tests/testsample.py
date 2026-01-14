# curl -X POST http://localhost:8000/agents/run \
#   -H "Content-Type: application/json" \
#   -d '{
#         "task": "Scan repo issues and summarize risks",
#         "tools": ["git", "sonar"],
#         "memory_key": "local-chat",
#         "tool_args": {
#           "git": {"owner": "ranganathantvb", "repo": "svc_catalog"},
#           "sonar": {"project_key": "ranganathantvb:svc_catalog"}
#         }
#       }'

## format: json below reponse

# GitHub Copilot
# Here is a human-readable summary of the scan results from your GitHub repository and SonarQube analysis:

# GitHub Repository: ranganathantvb/svc_catalog
# Language: Java
# Open Issues: 0
# Open Pull Requests: 0
# Forks: 0
# Watchers: 0
# Default Branch: main
# No open issues were found in the repository.

# SonarQube Analysis: Project svc_catalog
# Bugs: 0
# Vulnerabilities: 0
# Code Smells: 1
# Code Smell Details:
# File: src/test/java/com/codepulse/svc_catalog/SvcCatalogApplicationTests.java
# Line: 10
# Severity: CRITICAL
# Type: CODE_SMELL
# Message: Add a nested comment explaining why this method is empty, throw an UnsupportedOperationException or complete the implementation.
# Assignee: ranganathantvb-4YL4K@github
# Author: rangan.nath@gmail.com
# Tags: suspicious
# Impact: High severity on maintainability
# Status: OPEN
# Summary of Risks
# No open issues in GitHub.
# One critical code smell in SonarQube affecting maintainability. The issue is an empty method that should be documented, throw an exception, or be implemented.
# If you need a more detailed report or want to automate this summary, let me know!
### Validate the SonarCloud API access separately

# curl -i -u "$TOKEN:" \
#   "https://sonarcloud.io/api/authentication/validate"

# ### to get the list of projects for organization
# curl -s -u "$SONAR_TOKEN:" \
#   "https://sonarcloud.io/api/projects/search?organization=ranganathantvb" | jq 

# ### to get measures for a project

# curl --request GET \
#   --url "https://sonarcloud.io/api/measures/component?metricKeys=bugs,vulnerabilities,code_smells&component=ranganathantvb_svc_catalog" \
#   --header "Authorization: Bearer $SONAR_TOKEN"

### Generate a secure random string for GITHUB_WEBHOOK_SECRET and AGENT_API_KEY:
# python3 - <<EOF
# import secrets
# print(secrets.token_hex(32))
# EOF


# export GITHUB_WEBHOOK_SECRET="e102965ec2f74fb5ca721fb0fe843004401b4d6df8254208b1f3c554117a5054"   # same as in GitHub webhook
# export GITHUB_TOKEN="your-github-pat"                # PAT with repo access
# export AGENT_API_KEY="6cf488ac11da5c73fdd4e170401accea13e77762d7659e65f6d4874c2478d4f8"  # optional

# uvicorn webhook_server:app --host 0.0.0.0 --port 9000 --reload


#user_prompt,credentials
# How many SonarQube issues present in PR https://github.com/ranganathantvb/svc_catalog/pull/3 ?,
# What is validation status of the PR https://github.com/ranganathantvb/svc_catalog/pull/3?,
# How many files were changed in PR https://github.com/ranganathantvb/svc_catalog/pull/3?,
# Are there any security hotspots in PR https://github.com/ranganathantvb/svc_catalog/pull/3 ?,
# What SonarQube projects are linked to repository ranganathantvb/svc_catalog?,
# Fix SonarQube issues in PR https://github.com/ranganathantvb/ranganathantvb/pull/3 ?,


# When responding with summary, follow these formatting rules strictly:
# 1. Use markdown format.
# 2. Use "AI-Generated Summary:" as the main header.
# 3. Write a concise summary in one paragraph under the header about total SonarQube issues found in the given PR. Do not list issues individually. Specify additional statistics for each issue category (type) and severity (numbers only, no details).
# 4. When calculating totals, include both issues and security hotspots. For xample, if there are 3 BUG, 2 VULNERABILITY and 2 SECURITY_HOTSPOTs total number is 7.
# 5. Write a single, concise sentence that clearly states the main changes while fixing issues in the PR related to SonarQube, generalizing across all issues found and fixed. Add it only if the fixes were made.
# 6. Additionally, provide a link to the new branch created in the repository with proposed fixes.

# Summary output example is below (replace X, Y, Z... with actual numbers and <branch_link> with actual link): AI-Generated Summary: There were X SonarQube issues found in the provided PR, in the following categories:
# 1. B BUG(s)
# 2. V VULNERABILITY(ies)
# 3. S CODE_SMELL(s)
# 4. H SECURITY_HOTSPOT(s). The issues spanned various severity levels including:
# 5. P BLOCKER(s)
# 6. Q CRITICAL(s)
# 7. R MAJOR(s)
# 8. S MINOR(s)
# 9. T INFO(s).
# 10. K TO_REVIEW(s). 

# The main goal of the PR is to resolve all identified SonarQube issues by implementing recommended fixes and best practices to enhance code quality. New branch with proposed fixes: pr-<PR_number>-ai-sonarqube-fixes


#  ngrok http 9000  
 export GITHUB_TOKEN=" <github_token>"                            # PAT to post PR comments
 export AGENT_API_URL="http://127.0.0.1:8000/agents/run"  # your agent endpoint
 export AGENT_API_KEY="super-secret-agent-key"            # used to protect agents run
 export AGENT_API_TIMEOUT=30                              # seconds to wait on the agent call from webhook
 export CODEPULSE_WORKSPACE_PATH="Users/ranganathan/workspace/svc_catalog"
 export OLLAMA_HOST="http://localhost:11434"

# uvicorn code_pulse.app:app --reload
# CODEPULSE_WORKSPACE_PATH

='java:S2119', namespace='sonar'
2026-01-14 08:11:49,259 | INFO | code_pulse | RAG retrieved 0 chunk(s) for rule java:S2119
2026-01-14 08:11:49,259 | INFO | code_pulse | RAG retrieval: rule_key='java:S2245', namespace='sonar'
2026-01-14 08:11:49,262 | INFO | code_pulse | RAG retrieved 0 chunk(s) for rule java:S2245
2026-01-14 08:11:49,262 | INFO | code_pulse | RAG retrieval: rule_key='java:S4790', namespace='sonar'
2026-01-14 08:11:49,266 | INFO | code_pulse | RAG retrieved 0 chunk(s) for rule java:S4790
2026-01-14 08:11:49,266 | INFO | code_pulse | Summary of changes: