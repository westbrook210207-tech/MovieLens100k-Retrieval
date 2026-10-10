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

# Validation metric that picks the checkpoint and drives early stopping. This model is the
# retrieval stage: what matters is whether the user's next film is among the ~200 candidates
# handed on, not its exact position near the top (that is NDCG@10's question).
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
    # Default 0: on validation, explicit low-rated negatives lowered NDCG@10 (see README).
    p.add_argument("--hard-negative-ratio", type=float, default=0.0,
                   help="share of the K negatives taken from the user's low-rated items")
    p.add_argument("--hard-negative-max-rating", type=int, default=2,
                   help="a training rating <= this counts as an explicit negative")
    # BooleanOptionalAction also creates --no-in-batch-negatives to switch it off.
    p.add_argument("--in-batch-negatives", action=argparse.BooleanOptionalAction, default=True,
                   help="also use the other rows' positives as negatives (masked + logQ-corrected)")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100, help="upper limit; early stopping usually ends at ~20")
    p.add_argument("--patience", type=int, default=5, help="early-stopping patience in epochs")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    # MPS = the Apple-silicon GPU. Default: use it when this machine has one.
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu",
                   help="torch device to train on, e.g. mps or cpu")
    return p.parse_args(argv)


def build_model(users: pd.DataFrame, items: pd.DataFrame, mappings: dict, model_config: dict) -> TwoTower:
    """Turn the prepared feature tables into tensors (row i = user_idx / item_idx i) and build the model."""
    genres = mappings["genres"]
    # Only the user ID is embedded; every side feature is a 0/1 column fed in as a plain number.
    user_categorical = torch.tensor(users[["user_idx"]].to_numpy())  # (n_users, 1); copies the array
    age_columns = [f"age_bucket_{i}" for i in range(len(mappings["age_bins"]) + 1)]
    occupation_columns = [f"occupation_{name}" for name in mappings["occupations"]]
    region_columns = [f"zip_region_{i}" for i in range(len(mappings["zip_regions"]))]
    user_dense = torch.from_numpy(users[["is_male"] + age_columns + occupation_columns + region_columns].to_numpy()).float()
    # Items: the ID and the release-year bucket are embedded; genres are a multi-hot vector
    # (a film can have several), so they go in as 0/1 numbers rather than one embedding lookup.
    item_categorical = torch.from_numpy(items[["item_idx", "year_bucket"]].to_numpy())
    item_genres = torch.from_numpy(items[genres].to_numpy()).float()
    return TwoTower(
        user_categorical, user_dense, item_categorical, item_genres,
        # Vocabulary size of each categorical column = largest index + 1.
        user_cardinalities=(user_categorical.max(dim=0).values + 1).tolist(),
        item_cardinalities=(item_categorical.max(dim=0).values + 1).tolist(),
        **model_config,
    )


def main() -> None:
    args = parse_args()
    settings = get_settings()
    torch.manual_seed(args.seed)  # weight init, dropout and batch order; the sampler has its own seeded RNG

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
    # .to(device) moves the weights AND the feature-table buffers; every tensor that meets
    # them in a computation below has to live on the same device.
    model = build_model(users, items, mappings, model_config).to(device)
    print(f"training on {device}")
    sampler = NegativeSampler(train, len(users), n_items, args.num_negatives,
                              args.hard_negative_ratio, mappings["positive_threshold"],
                              args.hard_negative_max_rating, seed=args.seed)
    # AdamW = Adam with weight decay applied directly to the weights (mild regularisation).
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Only needed for in-batch negatives; both come from training data alone.
    seen = torch.from_numpy(sampler.seen).to(device)  # (n_users, n_items) bool: any training interaction
    # item_share[i] = q_i = item i's share of the training positives = the probability that
    # a random row of a batch has item i as its positive.
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
            model.train()  # switches dropout on (model_scores below switches it off again)
            user_idx, pos_idx, neg_idx = sampler.sample()  # fresh negatives every epoch
            # pos_idx[:, None] is (N, 1); concatenating along dim 1 gives one list per row.
            item_lists = torch.cat([pos_idx[:, None], neg_idx], dim=1)  # (N, 1 + K), positive first
            # The sampler works in numpy on the CPU; move one epoch of indices over in one go.
            user_idx, item_lists = user_idx.to(device), item_lists.to(device)
            total_loss = 0.0
            # randperm shuffles the row numbers; split cuts them into batches of row numbers.
            for batch in torch.randperm(len(user_idx)).to(device).split(args.batch_size):
                batch_users, batch_lists = user_idx[batch], item_lists[batch]  # (B,), (B, 1 + K)
                false_negative = correction = None
                if args.in_batch_negatives:
                    batch_pos = batch_lists[:, 0]  # (B,) the positives of this batch
                    # Problem 1 - false negatives. A popular film is a positive of many users,
                    # so a batch-mate's positive is often an item this user also liked (about
                    # 1 in 4 on this dataset). false_negative[b, j] is True when user b has any
                    # training interaction with row j's positive; info_nce drops those candidates.
                    # The diagonal is always True (a user has seen their own positive), which
                    # is right: the positive already sits at position 0 of the list.
                    false_negative = seen[batch_users][:, batch_pos]  # (B, B)
                    # Problem 2 - popularity bias. Even after masking, item j shows up among
                    # the in-batch negatives in proportion to its popularity q_j, so popular
                    # items are pushed down far more often than rare ones. logQ correction
                    # (sampled softmax, Bengio & Senecal 2008; Yi et al. 2019): subtract from
                    # every candidate's logit the log of how often the negative sampling is
                    # expected to produce it, here K uniform draws + (B - 1) batch-mates:
                    #     expected_count[i] = K / n_items + (B - 1) * q_i
                    # Over-sampled items are discounted by exactly their over-sampling, and the
                    # learned cosine then estimates the preference itself, not the popularity.
                    # shortcut: explicit low-rated negatives are counted as uniform draws here;
                    # give them their own term if --hard-negative-ratio > 0 is combined with in-batch.
                    expected_count = args.num_negatives / n_items + (len(batch) - 1) * item_share
                    # clamp_min avoids log(0) for items that can never be a candidate.
                    log_expected_count = expected_count.clamp_min(1e-12).log()  # (n_items,)
                    # Candidate item of every logit column, in info_nce's order: the list, then
                    # the B in-batch positives (the same row vector repeated for every user).
                    candidates = torch.cat([batch_lists, batch_pos.expand(len(batch), -1)], dim=1)  # (B, 1 + K + B)
                    correction = log_expected_count[candidates]
                loss = info_nce(model.encode_users(batch_users),    # (B, D)
                                model.encode_items(batch_lists),    # (B, 1 + K, D)
                                args.temperature, false_negative, correction)
                optimizer.zero_grad()  # gradients accumulate by default: clear the previous step's
                loss.backward()        # backpropagate through both towers
                optimizer.step()
                total_loss += loss.item() * len(batch)  # weight by batch size: the last batch is smaller

            train_loss = total_loss / len(user_idx)
            # Full-catalogue ranking of each user's held-out validation positive, with the
            # items they already interacted with in training removed from the candidates.
            metrics = evaluate_scores(model_scores(model).cpu(), val, seen=train)  # metrics are computed on the CPU
            mlflow.log_metrics({"train_loss": train_loss,
                                **{f"val_{k.replace('@', '_at_')}": v for k, v in metrics.items()}},
                               step=epoch)
            print(f"epoch {epoch:2d} loss {train_loss:.4f} val_recall@200 {metrics['recall@200']:.4f} "
                  f"val_ndcg@10 {metrics['ndcg@10']:.4f} "
                  f"val_recall@10 {metrics['recall@10']:.4f}")

            # Early stopping on the validation selection metric: keep a copy of the best weights and stop
            # once `patience` epochs in a row fail to improve on them. The test set is never used here.
            if metrics[SELECTION_METRIC] > best_score:
                best_score, best_metrics, stale = metrics[SELECTION_METRIC], metrics, 0
                best_state = copy.deepcopy(model.state_dict())  # deepcopy: state_dict() returns live references
            else:
                stale += 1
                if stale >= args.patience:
                    break

        model.load_state_dict(best_state)
        model.cpu()  # export CPU tensors: the artifact must load on machines without this device
        meta = {
            "run_id": run.info.run_id,
            "split_hash": mappings["split_hash"],      # which train/val/test rows
            "feature_hash": mappings["feature_hash"],  # which user/item feature columns
            "positive_threshold": mappings["positive_threshold"],
            "model_config": model_config,
            "sampling_config": sampling_config,  # informational: not needed to load the model
            "user_cardinalities": [e.num_embeddings for e in model.user_tower.embeddings],
            "user_dense_dim": model.user_dense.shape[1],
            "item_cardinalities": [e.num_embeddings for e in model.item_tower.embeddings],
            "user_ids": mappings["user_ids"],  # raw ID at position user_idx
            "item_ids": mappings["item_ids"],  # raw ID at position item_idx
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
