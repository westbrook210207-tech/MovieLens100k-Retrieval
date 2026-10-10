"""API response contracts. All IDs are raw MovieLens IDs."""

from pydantic import BaseModel


class RecommendedItem(BaseModel):
    item_id: int
    title: str
    release_year: int | None
    genres: list[str]
    score: float  # cosine similarity between the user vector and the item vector


class RecommendationResponse(BaseModel):
    user_id: int
    items: list[RecommendedItem]


class HealthResponse(BaseModel):
    status: str
    model_run_id: str | None  # MLflow run of the model whose vectors are being served
