from typing import Dict

from code_pulse.config import get_settings
from code_pulse.mcp.base import MCPClient
from code_pulse.mcp.git import GitClient
from code_pulse.mcp.sonar import SonarClient
from code_pulse.mcp.jira import JiraClient


def client_factory() -> Dict[str, MCPClient]:
    settings = get_settings()
    return {
        "git": GitClient(settings.git_base_url, settings.git_token),
        "sonar": SonarClient(settings.sonar_base_url, settings.sonar_token, settings.sonar_organization),
        "jira": JiraClient(settings.jira_base_url, settings.jira_api_token, settings.jira_user_email),
    }
