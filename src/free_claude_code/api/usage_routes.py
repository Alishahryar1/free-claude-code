"""Admin API routes for Provider Usage and Observability."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from free_claude_code.application.usage import get_usage_service
from free_claude_code.application.usage.service import UsageService

from .admin_security import require_loopback_admin
from .dependencies import get_services
from .ports import ApiServices

router = APIRouter()


def _get_usage_service(services: ApiServices) -> UsageService:
    if services.usage is not None and isinstance(services.usage, UsageService):
        return services.usage
    return get_usage_service()


@router.get("/admin/api/usage/summary")
async def get_usage_summary(
    request: Request,
    time_range: str = Query("24h"),
    services: ApiServices = Depends(get_services),
) -> JSONResponse:
    require_loopback_admin(request)
    svc = _get_usage_service(services)
    summary = await svc.get_summary(time_range)
    return JSONResponse(asdict(summary), headers={"Cache-Control": "no-store"})


@router.get("/admin/api/usage/providers")
async def get_provider_usage(
    request: Request,
    time_range: str = Query("24h"),
    services: ApiServices = Depends(get_services),
) -> JSONResponse:
    require_loopback_admin(request)
    svc = _get_usage_service(services)
    providers = await svc.get_provider_breakdown(time_range)
    return JSONResponse(
        {"providers": [asdict(p) for p in providers]},
        headers={"Cache-Control": "no-store"},
    )


@router.get("/admin/api/usage/models")
async def get_model_usage(
    request: Request,
    time_range: str = Query("24h"),
    services: ApiServices = Depends(get_services),
) -> JSONResponse:
    require_loopback_admin(request)
    svc = _get_usage_service(services)
    models = await svc.get_model_breakdown(time_range)
    return JSONResponse(
        {"models": [asdict(m) for m in models]},
        headers={"Cache-Control": "no-store"},
    )


@router.get("/admin/api/usage/fallbacks")
async def get_fallback_analytics(
    request: Request,
    time_range: str = Query("24h"),
    services: ApiServices = Depends(get_services),
) -> JSONResponse:
    require_loopback_admin(request)
    svc = _get_usage_service(services)
    analytics = await svc.get_fallback_analytics(time_range)
    return JSONResponse(asdict(analytics), headers={"Cache-Control": "no-store"})


@router.get("/admin/api/usage/timeseries")
async def get_usage_timeseries(
    request: Request,
    time_range: str = Query("24h"),
    services: ApiServices = Depends(get_services),
) -> JSONResponse:
    require_loopback_admin(request)
    svc = _get_usage_service(services)
    points = await svc.get_timeseries(time_range)
    return JSONResponse(
        {"points": [asdict(pt) for pt in points]},
        headers={"Cache-Control": "no-store"},
    )


@router.post("/admin/api/usage/clear")
async def clear_usage_data(
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JSONResponse:
    require_loopback_admin(request)
    svc = _get_usage_service(services)
    await svc.clear_all()
    return JSONResponse({"status": "cleared"}, headers={"Cache-Control": "no-store"})
