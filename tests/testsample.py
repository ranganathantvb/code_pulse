curl -X POST http://localhost:8000/agents/run \
  -H "Content-Type: application/json" \
  -d '{
        "task": "Scan repo issues and summarize risks",
        "tools": ["git", "sonar"],
        "memory_key": "local-chat",
        "tool_args": {
          "git": {"owner": "ranganathantvb", "repo": "svc_catalog"},
          "sonar": {"project_key": "ranganathantvb:svc_catalog"}
        }
      }'

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

curl -i -u "$TOKEN:" \
  "https://sonarcloud.io/api/authentication/validate"

# ### to get the list of projects for organization
curl -s -u "$SONAR_TOKEN:" \
  "https://sonarcloud.io/api/projects/search?organization=ranganathantvb" | jq 

# ### to get measures for a project

curl --request GET \
  --url "https://sonarcloud.io/api/measures/component?metricKeys=bugs,vulnerabilities,code_smells&component=ranganathantvb_svc_catalog" \
  --header "Authorization: Bearer $SONAR_TOKEN"
