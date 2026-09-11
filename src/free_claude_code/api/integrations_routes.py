"""Local Admin transport for fixed, confirmed integration operations."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from free_claude_code.application.integrations import (
    IntegrationId,
)

from .admin_security import require_loopback_admin
from .dependencies import get_services
from .ports import ApiServices

router = APIRouter(
    prefix="/admin/api/integrations", dependencies=[Depends(require_loopback_admin)]
)


class PreviewPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ApplyPayload(PreviewPayload):
    revision: str = Field(pattern=r"^[0-9a-f]{64}$")


@router.get("")
async def inspect_integrations(services: ApiServices = Depends(get_services)):
    return await services.admin.inspect_integrations()


@router.post("/{item}/preview")
async def preview_integration(
    item: IntegrationId,
    payload: PreviewPayload,
    services: ApiServices = Depends(get_services),
):
    return await services.admin.preview_integration(item)


@router.post("/{item}/apply")
async def apply_integration(
    item: IntegrationId,
    payload: ApplyPayload,
    services: ApiServices = Depends(get_services),
):
    return await services.admin.apply_integration(item, payload.revision)
