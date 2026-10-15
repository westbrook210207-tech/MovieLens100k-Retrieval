"""Offline top-K evaluation. Run: python -m training.evaluate --split val|test

Protocol
- Candidates: every item, minus the user's other known interactions
  (train for val; train + val for test).
- Relevant item: the user's single held-out positive, so Recall@K is hit rate and
  Precision@K = Recall@K / K (at most 1/K).
- Averaging: mean over users that have a held-out item.
- Ties are ranked pessimistically (the target is placed after equal-scored items).
"""

# evaluate.py — measure ranking quality
#
# - Job: answer "when the model ranks all films for a user, how high is the held-out film?"
# - Metrics: Recall, Precision and NDCG
# - Rule: films the user already rated in earlier parts of the split are removed from the
#   ranking first.
# - Used two ways: `train.py` calls its functions every epoch for validation, and you run it
#   yourself afterwards for the final validation or test score.
# - Baseline: it also scores "recommend the most popular films to everyone", so you can see
#   how much the model adds.

import argparse
import json
import math

import mlflow
import pandas as pd
import torch
from torch import Tensor

from config import get_settings
from training.models.two_tower import TwoTower, load_model

KS = (10, 20, 200)

# Define Recall, Precision, NDCG
def rank_metrics(scores: Tensor, target: Tensor, ks: tuple[int, ...] = KS) -> dict[str, float]:
    """scores: (U, n_items) with excluded items set to -inf; target: (U,) held-out item index."""
    target_score = scores.gather(1, target[:, None])
    rank = (scores >= target_score).sum(dim=1) - 1
    metrics = {}
    for k in ks:
        hit = (rank < k).float()
        metrics[f"recall@{k}"] = hit.mean().item()
        metrics[f"precision@{k}"] = hit.mean().item() / k
        metrics[f"ndcg@{k}"] = (hit / torch.log2(rank.float() + 2)).mean().item()
    return metrics


def evaluate_scores(scores: Tensor, held_out: pd.DataFrame, seen: pd.DataFrame,
                    ks: tuple[int, ...] = KS) -> dict[str, float]:
    """scores: (n_users, n_items). `held_out` has one row per evaluated user; `seen`
    lists interactions to exclude. Both use zero-based user_idx / item_idx."""
    if not torch.isfinite(scores).all():
        raise ValueError("Scores contain NaN or infinity: the model has diverged (try a lower --lr).")
    scores = scores.clone() # This makes a copy, so the next line's -inf doesn't change the caller's tensor.
    scores[seen["user_idx"].to_numpy(), seen["item_idx"].to_numpy()] = -math.inf # Score of rated movies by user have score = -inf so they're ranked last
                                                                                # DIFFERENT from -inf in info_nce: there exp(-inf) = 0, so the film doesn't contribute to the loss denominator,
                                                                                # so it's NOT considered as a negative. Here there's no softmax, it just drops the film's plain score for RANKING
                                                                                # (-inf is never >= the held-out film's score)
    
    users = torch.from_numpy(held_out["user_idx"].to_numpy()) # Turn all user in held_out from df to np to tensor
    return rank_metrics(scores[users], torch.from_numpy(held_out["item_idx"].to_numpy()), ks) # Pass score of all user in held_out, held_out tensor, Ks into rank_metrics()

# Eval loop
@torch.no_grad()
def model_scores(model: TwoTower) -> Tensor:
    """Cosine score of every user against every item: (n_users, n_items)."""
    model.eval()
    device = model.user_dense.device
    users = model.encode_users(torch.arange(len(model.user_dense), device=device)) # len(user_dense) is also the 943 ID, torch.arange returns a tensor starting from 0 so len() = 943, torch.arange(943) returns [0,1,...,942]
    items = model.encode_items(torch.arange(len(model.item_dense), device=device)) # same for item 
    return users @ items.T

# Baseline - popularity based, most positive rated movie ranked first
def popularity_scores(train: pd.DataFrame, n_users: int, n_items: int, positive_threshold: int) -> Tensor:
    """Baseline: same ranking for everyone, by number of training positives."""
    counts = train.loc[train["rating"] >= positive_threshold, "item_idx"].value_counts()
    scores = torch.zeros(n_items)
    scores[counts.index.to_numpy()] = torch.from_numpy(counts.to_numpy()).float()
    return scores.expand(n_users, n_items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["val", "test"], default="val") # Run eval on validation set or test set, default = validation
    args = parser.parse_args()
    settings = get_settings()

    model, meta = load_model(settings.artifact_dir) # Load model and meta from train.py
    data = settings.prepared_data_dir
    current_split = json.loads((data / "mappings.json").read_text())["split_hash"] # Take "split_hash" in mappings.json
    if current_split != meta["split_hash"]: # If the split on disk now differs from the one the model was trained on, it raises error.
        raise SystemExit(
            f"The model was trained on split {meta['split_hash']} but {data} now holds split "
            f"{current_split}. Retrain on the current data, or restore the split the model used."
        )
    train = pd.read_csv(data / "train.csv")
    held_out = pd.read_csv(data / f"{args.split}.csv")
    seen = train if args.split == "val" else pd.concat([train, pd.read_csv(data / "val.csv")]) # If we evaluate test set, validation set is considered as seen
    n_users, n_items = len(meta["user_ids"]), len(meta["item_ids"])

    results = {
        "two_tower": evaluate_scores(model_scores(model), held_out, seen),
        "popularity": evaluate_scores(
            popularity_scores(train, n_users, n_items, meta["positive_threshold"]), held_out, seen),
    }
    print(f"split={args.split} users={len(held_out)} (precision@K is at most 1/K: one held-out item per user)")
    print(pd.DataFrame(results).T.round(4).to_string())

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with mlflow.start_run(run_id=meta["run_id"]):
        for name, metrics in results.items():
            mlflow.log_metrics({f"{args.split}_{name}_{k.replace('@', '_at_')}": v for k, v in metrics.items()})


if __name__ == "__main__":
    main()
