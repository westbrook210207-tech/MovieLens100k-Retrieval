"""HTTP endpoints. Thin: validate input, call the recommender, shape the response."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from api.schemas import HealthResponse, RecommendationResponse
from api.recommender import RecommendationService, UserNotFoundError

logger = logging.getLogger("api")
router = APIRouter()

# This function returns the recommender object built in main.py
# Why do need a whole function just to return the recommender object but not import it?
# Because route.py is created before the recommender object, which mean recommender object was not yet created to be imported
# get_recommender() returns the recommender object after the object is created.
def get_recommender(request: Request) -> RecommendationService: 
    return request.app.state.recommender 


Recommender = Annotated[RecommendationService, Depends(get_recommender)] # recommender object is the main object that calls other files to do the retrieving
                                                                        

# Healthcheck the recommender object
@router.get("/health", response_model=HealthResponse)
def health(recommender: Recommender) -> HealthResponse:
    try:
        recommender.repository.ping() # healthcheck 2 databases
    except Exception:
        logger.exception("health check failed")
        raise HTTPException(status_code=503, detail="A backing service is unavailable")
    return HealthResponse(status="ok", model_run_id=recommender.model_run_id)


@router.get("/users/{user_id}/recommendations", response_model=RecommendationResponse)
def recommendations(
    recommender: Recommender,
    user_id: Annotated[int, Path(ge=1, le = 943)],
    k: Annotated[int, Query(ge=1, le=100)] = 10,
) -> RecommendationResponse:
    if user_id > recommender.max_user_id:
        raise HTTPException(status_code=422, detail=f"user_id must be between 1 and {recommender.max_user_id}")
    try:
        items = recommender.recommend(user_id, k)
    except UserNotFoundError:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return RecommendationResponse(user_id=user_id, items=items)
