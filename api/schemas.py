# Pydantic models

from pydantic import BaseModel


class RecommendedItem(BaseModel):
    item_id: int
    title: str
    release_year: int | None
    genres: list[str]
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    items: list[RecommendedItem]


class HealthResponse(BaseModel):
    status: str
    model_run_id: str | None
