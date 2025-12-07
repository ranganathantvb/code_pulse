import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

from code_pulse.config import Settings, get_settings
from code_pulse.logger import setup_logging
from code_pulse.mcp.sonar import SonarClient

logger = setup_logging(__name__)

router = APIRouter(prefix="/sonar", tags=["sonar"])


def _client(settings: Settings) -> SonarClient:
    return SonarClient(settings.sonar_base_url, settings.sonar_token, settings.sonar_organization)


@router.get("/rules")
async def fetch_rules(
    query: str | None = Query(None, description="Free text filter on rule key/name"),
    languages: str | None = Query(None, description="Comma-separated language keys (e.g., 'py,js')"),
    severities: str | None = Query(None, description="Comma-separated severities (e.g., 'BLOCKER,CRITICAL')"),
    types: str | None = Query(None, description="Comma-separated rule types (e.g., 'BUG,VULNERABILITY')"),
    page: int = Query(1, ge=1, description="Results page to fetch"),
    page_size: int = Query(100, ge=1, le=500, description="Page size for results"),
    settings: Settings = Depends(get_settings),
):
    """Expose Sonar rules as JSON for downstream ingestion (e.g., RAG training)."""
    async with _client(settings) as client:
        try:
            return await client.rules(
                query=query,
                languages=languages,
                severities=severities,
                types=types,
                page=page,
                page_size=page_size,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = exc.response.text
            except Exception:  # noqa: BLE001
                detail = str(exc)
            raise HTTPException(status_code=exc.response.status_code, detail=detail)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to fetch Sonar rules")
            raise HTTPException(status_code=502, detail=str(exc))
