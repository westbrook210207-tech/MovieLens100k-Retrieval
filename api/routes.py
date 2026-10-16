"""HTTP endpoints. Thin: validate input, call the service, shape the response."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from api.schemas import HealthResponse, RecommendationResponse
from api.service import RecommendationService, UserNotFoundError

logger = logging.getLogger("api")
router = APIRouter()


def get_service(request: Request) -> RecommendationService: # Returns the service object built in main.py
    return request.app.state.service


Service = Annotated[RecommendationService, Depends(get_service)]


@router.get("/health", response_model=HealthResponse)
def health(service: Service) -> HealthResponse:
    try:
        service.repository.ping() # healthcheck 2 databases
    except Exception:
        logger.exception("health check failed")
        raise HTTPException(status_code=503, detail="A backing service is unavailable")
    return HealthResponse(status="ok", model_run_id=service.model_run_id)


@router.get("/users/{user_id}/recommendations", response_model=RecommendationResponse)
def recommendations(
    service: Service,
    user_id: Annotated[int, Path(ge=1, le = 943)],
    k: Annotated[int, Query(ge=1, le=100)] = 10,
) -> RecommendationResponse:
    if user_id > service.max_user_id:
        raise HTTPException(status_code=422, detail=f"user_id must be between 1 and {service.max_user_id}")
    try:
        items = service.recommend(user_id, k)
    except UserNotFoundError:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return RecommendationResponse(user_id=user_id, items=items)
