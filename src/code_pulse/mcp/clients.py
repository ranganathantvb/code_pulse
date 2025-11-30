from typing import Dict

from code_pulse.config import get_settings
from code_pulse.mcp.base import MCPClient
from code_pulse.mcp.jira import JiraClient


def client_factory() -> Dict[str, MCPClient]:
    settings = get_settings()
    return {
        "git": MCPClient(settings.git_base_url, settings.git_token),
        "sonar": MCPClient(settings.sonar_base_url, settings.sonar_token),
        "jira": JiraClient(settings.jira_base_url, settings.jira_api_token, settings.jira_user_email),
    }
