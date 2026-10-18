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

# This function basically let us adjust the hyperparameters via terminal without editing the code
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])    # p is an object from Python's built-in argparse library. 
                                                                        # Its job is to read the words you type after a command. Right now it knows no options. 
                                                                        # The description is only the text shown when you type --help.
    # --... is the word that p looks for in what you typed
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[256, 256],
                   help="width of each hidden layer per tower, e.g. 256 128; no values = a single linear projection")
    p.add_argument("--dropouts", type=float, nargs="*", default=[0.1, 0.1],
                   help="dropout rate after each hidden layer; one value per --hidden-dims value")
    p.add_argument("--num-negatives", type=int, default=20, help="K sampled negatives per positive")
    p.add_argument("--in-batch-negatives", action=argparse.BooleanOptionalAction, default=True,
                   help="also use the other rows' positives as negatives (masked + logQ-corrected)") # This option takes no word after it. 
                                                                                                    #BooleanOptionalAction registers two words for the same name: typing --in-batch-negatives stores True, 
                                                                                                    # typing --no-in-batch-negatives stores False. Typing neither gives the default, True.
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100, help="upper limit; early stopping usually ends at ~20")
    p.add_argument("--patience", type=int, default=5, help="early-stopping patience in epochs")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu",
                   help="torch device to train on, e.g. mps or cpu")
    
    # The add_argument lines above list which hyperparameters exist. This line reads their values:
    # the typed value where you typed one, the default where you did not.
    args = p.parse_args(argv)
    if len(args.hidden_dims) != len(args.dropouts):
        p.error("--hidden-dims and --dropouts need the same number of values (one per hidden layer)")
    return args


def build_model(users: pd.DataFrame, items: pd.DataFrame, genres: list[str], model_config: dict) -> TwoTower:
    """Turn the prepared feature tables into tensors (row i = user_idx / item_idx i) and build the model."""
    # Age is one standard-scaled number (mean 0, std 1; see build_features in prepare.py),
    # the only user feature column that is not 0/1.
    age_columns = ["age_scaled"]
    occupation_columns = [c for c in users.columns if c.startswith("occupation_")] # All cols starting with "occupation_"
    region_columns = [c for c in users.columns if c.startswith("zip_region_")] # All cols starting with "zip_region_"
    user_dense = torch.from_numpy(users[["is_male"] + age_columns + occupation_columns + region_columns].to_numpy()).float() # Find those cols from df then turns it into numpy then turns it into tensor
    # Only the film ID is embedded. The release year is one standard-scaled number
    # (see build_features in prepare.py) fed in next to the 0/1 genre flags.
    item_dense = torch.from_numpy(items[genres + ["release_year_scaled"]].to_numpy()).float()  # (n_items, 20)
    return TwoTower(user_dense, item_dense, **model_config) # Plug user_dense and item_dense into TwoTower() in two_tower.py


def main() -> None:
    args = parse_args()
    settings = get_settings()
    torch.manual_seed(args.seed)

    data = settings.prepared_data_dir
    mappings = json.loads((data / "mappings.json").read_text())
    train, val = pd.read_csv(data / "train.csv"), pd.read_csv(data / "val.csv")
    users, items = pd.read_csv(data / "users.csv"), pd.read_csv(data / "items.csv")
    genres = pd.read_csv(settings.raw_data_dir / "u.genre", sep="|", names=["genre", "genre_idx"])["genre"].tolist()
    n_items = len(items)

    # Pass in output embedding dim, hidden layers dim, and dropout rates from args into model_config
    model_config = {"embedding_dim": args.embedding_dim, # embedding_dim : 64 (default)
                    "hidden_dims": args.hidden_dims, "dropouts": args.dropouts} # hidden_dims: [256,256] (default), dropouts: [0.1,0.1] (default)
    
    # Also pass in num_negatives and in_batch_negatives from args into sampling_config 
    sampling_config = {"num_negatives": args.num_negatives, #num_negatives: 20 (default)
                       "in_batch_negatives": args.in_batch_negatives} # in_batch_negatives : True (default)
    device = torch.device(args.device)
    # users comes from users.csv, items from items.csv, genres from ml-100k/u.genre, model_config from right above.
    # users.csv has 41 columns, but the model only gets 34 of them. The other 7 are things like
    # user_id, age, occupation (the text) and zip_code. build_model picks the 34 it wants:
    #   occupation_columns: every col whose name starts with "occupation_"   21 cols
    #   region_columns: every col whose name starts with "zip_region_"       11 cols (not zip_region, the text col)
    #   genres: the 19 names from u.genre, which are also the names of the genre cols in items.csv
    model = build_model(users, items, genres, model_config).to(device)
    print(f"training on {device}")

    # Create NegativeSampler instance called sampler
    sampler = NegativeSampler(train, len(users), n_items, args.num_negatives,
                              mappings["positive_threshold"], seed=args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # Using AdamW with weight decay (aka regularization)

    seen = torch.from_numpy(sampler.seen).to(device) # Turns seen list from NegativeSampler into tensor

    # item_share is to compute the popularity so that later we can resolve the in-batch popularity problem
    item_share = torch.bincount(torch.from_numpy(sampler.positives), minlength=n_items) / len(sampler.positives)# Count the popularity of appearence of all movies inside positive set
                                                                                                                # This is also the chance the movie gets picked as an in-batch negative
                                                                                                                # For example user 0 likes movie 0,1,2,3 and user 1 likes movie 2,3,4
                                                                                                                # Then the item_share = [1,1,2,2,1], which now movie 2 and 3 are considered a popular movie
                                                                                                                # with the probability is [1,1,2,2,1] / (1+1+2+2+1) = [0.143, 0.143, 0.286, 0.286, 0.143]
                                                                                                                # So item_share[i] is the probability of movie i in logQ(i)
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
            user_idx, pos_idx, neg_idx = sampler.sample() # Create (user,pos,{neg1,neg2,neg3,...}) for each positive rating in train set
            item_lists = torch.cat([pos_idx[:, None], neg_idx], dim=1) # Group pos and {neg1,neg2,neg3, ...} into 1 tesor, 
                                                                        # so later the model can give softmax probability for each movies in the tensor. 
                                                                        # Note that the pos movie must be the first index of the tensor
                                                                        # Since later, in-batch will take all item_list[0] for in-batch sampling
            user_idx, item_lists = user_idx.to(device), item_lists.to(device)
            total_loss = 0.0
            for batch in torch.randperm(len(user_idx)).to(device).split(args.batch_size): # Read all ratings in train set, split them with batch size = args.batch_size (default 256)
                                                                                        # torch.randperm() does the shuffling so every batch is random 256 rows
                batch_users, batch_lists = user_idx[batch], item_lists[batch] # Take 256 (user,pos,{negs}) rows
                false_negative = correction = None # Both start as None. If in-batch negatives are switched off, they stay None, and info_nce skips both of its if blocks. The row is then only the 21 films.
                if args.in_batch_negatives:
                    batch_pos = batch_lists[:, 0] # Takes all in-batch positive movie at batch_list[0] since all positive movies are at the first index of the list
                    false_negative = seen[batch_users][:, batch_pos] # for every user b and every liked film j in the batch, say whether user b has already rated film j. If they have, film j is not a negative for that user.
                                                                    # seen[batch_users] takes the seen row of the user of each of the 256 batch rows. That gives (256, 1682). A user with several rows in the batch gets their row several times
                                                                    # [:, batch_pos] keeps all rows (:) but only the 256 cols of the batch's liked films. That gives (256, 256). If 2 rows liked the same film, 2 cols are identical
                                                                    # So basically it's a T/F grid: rows = batch rows (users), cols = liked films in the batch. T = that user rated that film in training, so info_nce masks it to -inf

                    expected_count = args.num_negatives / n_items + (len(batch) - 1) * item_share # This is the Q equation in logQ(i), Q(i) is just an equation that takes in movie i probability as input
                    log_expected_count = expected_count.clamp_min(1e-12).log()  # This is the log in logQ(i)
                                                                                # clamp_min(1e-12) defines the min so the log is not 0, since ln(0) can't be computed
                                                                                # log() means ln()
                    candidates = torch.cat([batch_lists, batch_pos.expand(len(batch), -1)], dim=1) # Film IDs of all 277 logit cols of each row: the 21 of batch_lists (1st einsum) + the 256 batch_pos (2nd einsum)
                                                                                                    # batch_pos.expand(256, -1) repeats the 256 in-batch films for every row, giving (256, 256). -1 means keep that size
                                                                                                    # dim=1 glues them side by side, giving (256, 277), same order as the logits in info_nce
                                                                                                    # So basically candidate is a matrix of film IDs, with 256 rows as 256 ratings in 1 batch and 277 as 277 movies of 1 row
                    correction = log_expected_count[candidates] # Looks up logQ(i) of the film in each of the 277 cols, giving (256, 277). info_nce does logits - correction

                # Loss pass in all user ID and movie ID in 1 batch of training into encoders, compute embeddings from those IDs
                # Have those embeddings, temp, T/F false_negative grid and correction (logQ list for corresponding movie) plugged
                # into info_nce()
                loss = info_nce(model.encode_users(batch_users), # So batch_user is the user in (user,pos,{negs})
                                model.encode_items(batch_lists), # Batch_list is the (pos,{20 negs}) in (user,pos,{negs}), 
                                                                # note that we can find 256 in-batch negatives from this batch_list too since 256 
                                                                # rows has 256 pos which are the 256 in-batch
                                                                # So basically all 277 cols (1 pos + 20 negs + 256 in-batch) come from batch_list
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

            if metrics[SELECTION_METRIC] > best_score: # Select the best model based on Recall@200
                best_score, best_metrics, stale = metrics[SELECTION_METRIC], metrics, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= args.patience: # Patience = 5 (default)
                    break # Early stoppage

        model.load_state_dict(best_state)
        model.cpu()
        meta = {
            "run_id": run.info.run_id,
            "split_hash": mappings["split_hash"],
            "feature_hash": mappings["feature_hash"],
            "positive_threshold": mappings["positive_threshold"],
            "model_config": model_config,
            "sampling_config": sampling_config,
            "user_dense_dim": model.user_dense.shape[1],
            "item_dense_dim": model.item_dense.shape[1],
            "user_ids": mappings["user_ids"],
            "item_ids": mappings["item_ids"],
            "genres": genres,
            "val_metrics": best_metrics,
        }
        save_model(model, meta, settings.artifact_dir)
        mlflow.log_artifacts(str(settings.artifact_dir), artifact_path="two_tower")
        print(f"best val_recall@200 {best_score:.4f} val_ndcg@10 {best_metrics['ndcg@10']:.4f} "
              f"val_recall@10 {best_metrics['recall@10']:.4f} "
              f"-> exported to {settings.artifact_dir} (run {run.info.run_id})")


if __name__ == "__main__":
    main()
