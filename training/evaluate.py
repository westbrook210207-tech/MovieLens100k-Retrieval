"""Offline top-K evaluation. Run: python -m training.evaluate --split val|test

Protocol
- Candidates: every item, minus the user's other known interactions
  (train for val; train + val for test).
- Relevant item: the user's single held-out positive, so Recall@K is hit rate and
  Precision@K = Recall@K / K (at most 1/K).
- Averaging: mean over users that have a held-out item.
- Ties are ranked pessimistically (the target is placed after equal-scored items).
"""

import argparse
import json
import math

import mlflow
import pandas as pd
import torch
from torch import Tensor

from config import get_settings
from training.models.two_tower import TwoTower, load_model

KS = (10, 20, 200)  # 200: candidate-set size if this retrieval feeds a ranker


def rank_metrics(scores: Tensor, target: Tensor, ks: tuple[int, ...] = KS) -> dict[str, float]:
    """scores: (U, n_items) with excluded items set to -inf; target: (U,) held-out item index."""
    target_score = scores.gather(1, target[:, None])          # (U, 1)
    rank = (scores >= target_score).sum(dim=1) - 1            # 0 = top of the list
    metrics = {}
    for k in ks:
        hit = (rank < k).float()
        metrics[f"recall@{k}"] = hit.mean().item()
        metrics[f"precision@{k}"] = hit.mean().item() / k
        # One relevant item -> ideal DCG is 1, so NDCG = 1 / log2(rank + 2) when hit.
        metrics[f"ndcg@{k}"] = (hit / torch.log2(rank.float() + 2)).mean().item()
    return metrics


def evaluate_scores(scores: Tensor, held_out: pd.DataFrame, seen: pd.DataFrame,
                    ks: tuple[int, ...] = KS) -> dict[str, float]:
    """scores: (n_users, n_items). `held_out` has one row per evaluated user; `seen`
    lists interactions to exclude. Both use zero-based user_idx / item_idx."""
    # NaN compares False with everything, which would rank every target first and report
    # perfect metrics for a model that has diverged.
    if not torch.isfinite(scores).all():
        raise ValueError("Scores contain NaN or infinity: the model has diverged (try a lower --lr).")
    scores = scores.clone()
    scores[seen["user_idx"].to_numpy(), seen["item_idx"].to_numpy()] = -math.inf
    users = torch.from_numpy(held_out["user_idx"].to_numpy())
    return rank_metrics(scores[users], torch.from_numpy(held_out["item_idx"].to_numpy()), ks)


@torch.no_grad()
def model_scores(model: TwoTower) -> Tensor:
    """Cosine score of every user against every item: (n_users, n_items)."""
    model.eval()
    device = model.user_categorical.device  # index tensors must sit on the model's device
    users = model.encode_users(torch.arange(len(model.user_categorical), device=device))
    items = model.encode_items(torch.arange(len(model.item_categorical), device=device))
    return users @ items.T


def popularity_scores(train: pd.DataFrame, n_users: int, n_items: int, positive_threshold: int) -> Tensor:
    """Baseline: same ranking for everyone, by number of training positives."""
    counts = train.loc[train["rating"] >= positive_threshold, "item_idx"].value_counts()
    scores = torch.zeros(n_items)
    scores[counts.index.to_numpy()] = torch.from_numpy(counts.to_numpy()).float()
    return scores.expand(n_users, n_items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    settings = get_settings()

    model, meta = load_model(settings.artifact_dir)
    data = settings.prepared_data_dir
    current_split = json.loads((data / "mappings.json").read_text())["split_hash"]
    if current_split != meta["split_hash"]:
        # Otherwise the held-out items of this split may be items the model trained on.
        raise SystemExit(
            f"The model was trained on split {meta['split_hash']} but {data} now holds split "
            f"{current_split}. Retrain on the current data, or restore the split the model used."
        )
    train = pd.read_csv(data / "train.csv")
    held_out = pd.read_csv(data / f"{args.split}.csv")
    seen = train if args.split == "val" else pd.concat([train, pd.read_csv(data / "val.csv")])
    n_users, n_items = len(meta["user_ids"]), len(meta["item_ids"])

    results = {
        "two_tower": evaluate_scores(model_scores(model), held_out, seen),
        "popularity": evaluate_scores(
            popularity_scores(train, n_users, n_items, meta["positive_threshold"]), held_out, seen),
    }
    print(f"split={args.split} users={len(held_out)} (precision@K is at most 1/K: one held-out item per user)")
    print(pd.DataFrame(results).T.round(4).to_string())

    # Attach the numbers to the training run that produced this artifact.
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with mlflow.start_run(run_id=meta["run_id"]):
        for name, metrics in results.items():
            mlflow.log_metrics({f"{args.split}_{name}_{k.replace('@', '_at_')}": v for k, v in metrics.items()})


if __name__ == "__main__":
    main()
