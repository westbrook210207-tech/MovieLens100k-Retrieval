"""Train the hybrid two-tower model with listwise InfoNCE. Run: python -m training.train

Reads the prepared split, logs to MLflow, and exports the best checkpoint
(by validation Recall@200) to settings.artifact_dir as model.pt + meta.json.

One training example is a list (user, pos, {neg_1..neg_K}): one item the user rated
>= positive_threshold in the training period, plus K sampled negatives
(training/sampling.py), optionally extended with the other positives of the same batch
(--in-batch-negatives). The loss asks the model to pick the positive out of that list.

All indices in this file are zero-based model indices; raw MovieLens IDs only appear
in the exported meta.json.
"""

# train.py — the training loop
#
# - Job: bring the data from `prepare.py`, the model from `models/two_tower.py` and the
#   negatives from `sampling.py` together and train.
# - Each epoch: get fresh negatives, shuffle, go through batches of 256, compute the loss,
#   update the weights.
# - In-batch negatives: inside each batch, the other users' liked films are added as extra
#   negatives, with two corrections applied.
# - After each epoch: score the validation set and log to MLflow.
# - Early stopping: keep the weights with the best validation Recall@200 and stop after 5
#   epochs without improvement.
# - Output: the best `model.pt` and `meta.json` in `training/artifacts/two_tower/`. The test
#   set is never used here.

import argparse
import copy
import json

import mlflow
import pandas as pd
import torch

from config import get_settings
from training.evaluate import evaluate_scores, model_scores
from training.models.two_tower import TwoTower, info_nce, save_model
from training.sampling import NegativeSampler

SELECTION_METRIC = "recall@200"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--feature-dim", type=int, default=8, help="embedding size of each side feature")
    p.add_argument("--hidden-dim", type=int, default=256, help="width of each hidden layer")
    p.add_argument("--num-hidden-layers", type=int, default=2,
                   help="hidden layers per tower; 0 = a single linear projection")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num-negatives", type=int, default=20, help="K sampled negatives per positive")
    p.add_argument("--hard-negative-ratio", type=float, default=0.0,
                   help="share of the K negatives taken from the user's low-rated items")
    p.add_argument("--hard-negative-max-rating", type=int, default=2,
                   help="a training rating <= this counts as an explicit negative")
    p.add_argument("--in-batch-negatives", action=argparse.BooleanOptionalAction, default=True,
                   help="also use the other rows' positives as negatives (masked + logQ-corrected)")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100, help="upper limit; early stopping usually ends at ~20")
    p.add_argument("--patience", type=int, default=5, help="early-stopping patience in epochs")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu",
                   help="torch device to train on, e.g. mps or cpu")
    return p.parse_args(argv)


def build_model(users: pd.DataFrame, items: pd.DataFrame, mappings: dict, model_config: dict) -> TwoTower:
    """Turn the prepared feature tables into tensors (row i = user_idx / item_idx i) and build the model."""
    genres = mappings["genres"]
    user_categorical = torch.tensor(users[["user_idx"]].to_numpy())
    age_columns = [f"age_bucket_{i}" for i in range(len(mappings["age_bins"]) + 1)]
    occupation_columns = [f"occupation_{name}" for name in mappings["occupations"]]
    region_columns = [f"zip_region_{i}" for i in range(len(mappings["zip_regions"]))]
    user_dense = torch.from_numpy(users[["is_male"] + age_columns + occupation_columns + region_columns].to_numpy()).float()
    item_categorical = torch.from_numpy(items[["item_idx", "year_bucket"]].to_numpy())
    item_genres = torch.from_numpy(items[genres].to_numpy()).float()
    return TwoTower(
        user_categorical, user_dense, item_categorical, item_genres,
        user_cardinalities=(user_categorical.max(dim=0).values + 1).tolist(),
        item_cardinalities=(item_categorical.max(dim=0).values + 1).tolist(),
        **model_config,
    )


def main() -> None:
    args = parse_args()
    settings = get_settings()
    torch.manual_seed(args.seed)

    data = settings.prepared_data_dir
    mappings = json.loads((data / "mappings.json").read_text())
    train, val = pd.read_csv(data / "train.csv"), pd.read_csv(data / "val.csv")
    users, items = pd.read_csv(data / "users.csv"), pd.read_csv(data / "items.csv")
    n_items = len(items)

    model_config = {"embedding_dim": args.embedding_dim, "feature_dim": args.feature_dim,
                    "hidden_dim": args.hidden_dim, "dropout": args.dropout,
                    "num_hidden_layers": args.num_hidden_layers}
    sampling_config = {"num_negatives": args.num_negatives,
                       "hard_negative_ratio": args.hard_negative_ratio,
                       "hard_negative_max_rating": args.hard_negative_max_rating,
                       "in_batch_negatives": args.in_batch_negatives}
    device = torch.device(args.device)
    model = build_model(users, items, mappings, model_config).to(device)
    print(f"training on {device}")
    sampler = NegativeSampler(train, len(users), n_items, args.num_negatives,
                              args.hard_negative_ratio, mappings["positive_threshold"],
                              args.hard_negative_max_rating, seed=args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    seen = torch.from_numpy(sampler.seen).to(device)
    item_share = torch.bincount(torch.from_numpy(sampler.positives), minlength=n_items) / len(sampler.positives)
    item_share = item_share.to(device)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("two_tower")
    with mlflow.start_run() as run:
        mlflow.log_params({**vars(args), "split_hash": mappings["split_hash"],
                           "feature_hash": mappings["feature_hash"],
                           "positive_threshold": mappings["positive_threshold"]})
        best_score, best_state, best_metrics, stale = -1.0, None, {}, 0
        for epoch in range(args.epochs):
            model.train()
            user_idx, pos_idx, neg_idx = sampler.sample()
            item_lists = torch.cat([pos_idx[:, None], neg_idx], dim=1)
            user_idx, item_lists = user_idx.to(device), item_lists.to(device)
            total_loss = 0.0
            for batch in torch.randperm(len(user_idx)).to(device).split(args.batch_size):
                batch_users, batch_lists = user_idx[batch], item_lists[batch]
                false_negative = correction = None
                if args.in_batch_negatives:
                    batch_pos = batch_lists[:, 0]
                    false_negative = seen[batch_users][:, batch_pos]
                    expected_count = args.num_negatives / n_items + (len(batch) - 1) * item_share
                    log_expected_count = expected_count.clamp_min(1e-12).log()
                    candidates = torch.cat([batch_lists, batch_pos.expand(len(batch), -1)], dim=1)
                    correction = log_expected_count[candidates]
                loss = info_nce(model.encode_users(batch_users),
                                model.encode_items(batch_lists),
                                args.temperature, false_negative, correction)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)

            train_loss = total_loss / len(user_idx)
            metrics = evaluate_scores(model_scores(model).cpu(), val, seen=train)
            mlflow.log_metrics({"train_loss": train_loss,
                                **{f"val_{k.replace('@', '_at_')}": v for k, v in metrics.items()}},
                               step=epoch)
            print(f"epoch {epoch:2d} loss {train_loss:.4f} val_recall@200 {metrics['recall@200']:.4f} "
                  f"val_ndcg@10 {metrics['ndcg@10']:.4f} "
                  f"val_recall@10 {metrics['recall@10']:.4f}")

            if metrics[SELECTION_METRIC] > best_score:
                best_score, best_metrics, stale = metrics[SELECTION_METRIC], metrics, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= args.patience:
                    break

        model.load_state_dict(best_state)
        model.cpu()
        meta = {
            "run_id": run.info.run_id,
            "split_hash": mappings["split_hash"],
            "feature_hash": mappings["feature_hash"],
            "positive_threshold": mappings["positive_threshold"],
            "model_config": model_config,
            "sampling_config": sampling_config,
            "user_cardinalities": [e.num_embeddings for e in model.user_tower.embeddings],
            "user_dense_dim": model.user_dense.shape[1],
            "item_cardinalities": [e.num_embeddings for e in model.item_tower.embeddings],
            "user_ids": mappings["user_ids"],
            "item_ids": mappings["item_ids"],
            "genres": mappings["genres"],
            "val_metrics": best_metrics,
        }
        save_model(model, meta, settings.artifact_dir)
        mlflow.log_artifacts(str(settings.artifact_dir), artifact_path="two_tower")
        print(f"best val_recall@200 {best_score:.4f} val_ndcg@10 {best_metrics['ndcg@10']:.4f} "
              f"val_recall@10 {best_metrics['recall@10']:.4f} "
              f"-> exported to {settings.artifact_dir} (run {run.info.run_id})")


if __name__ == "__main__":
    main()
