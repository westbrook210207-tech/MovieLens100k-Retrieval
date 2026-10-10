"""Load, validate and split MovieLens 100K. Run: python -m training.prepare

Reads the raw files (never writes to them) and writes prepared CSVs plus
mappings.json to settings.prepared_data_dir.

ID convention: `user_id` / `item_id` are raw MovieLens IDs (1-based).
`user_idx` / `item_idx` are zero-based model indices, used only inside training.
"""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from config import get_settings

AGE_BINS = [18, 25, 35, 45, 50, 56]  # bucket edges (same cut points as MovieLens 1M)
YEAR_BINS = [1950, 1960, 1970, 1980, 1990, 1995, 1997]  # bucket 0 is reserved for "unknown"
# US zip codes are geographic: the first digit is a national area, numbered roughly east to west.
# Index = first digit; the last entry collects everything that is not a 5-digit US zip
# (in this dataset: 18 Canadian postal codes such as "V3N4P").
ZIP_REGIONS = [
    "Northeast (CT MA ME NH NJ RI VT)",
    "Mid-Atlantic (DE NY PA)",
    "South Atlantic (DC MD NC SC VA WV)",
    "Southeast (AL FL GA MS TN)",
    "Great Lakes (IN KY MI OH)",
    "Upper Midwest (IA MN MT ND SD WI)",
    "Central Plains (IL KS MO NE)",
    "South Central (AR LA OK TX)",
    "Mountain (AZ CO ID NM NV UT WY)",
    "Pacific (AK CA HI OR WA)",
    "Non-US / unknown",
]
ITEM_BASE_COLUMNS = ["item_id", "title", "release_date", "video_release_date", "imdb_url"]


def load_raw(raw_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (ratings, users, items, genre_names) with explicit column names."""
    genres = pd.read_csv(raw_dir / "u.genre", sep="|", names=["genre", "genre_idx"])["genre"].tolist()
    ratings = pd.read_csv(
        raw_dir / "u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
    )
    users = pd.read_csv(
        raw_dir / "u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"],
        dtype={"zip_code": str},
    )
    # u.item is ISO-8859-1 (accented titles), not UTF-8.
    items = pd.read_csv(
        raw_dir / "u.item", sep="|", names=ITEM_BASE_COLUMNS + genres, encoding="latin-1"
    )
    return ratings, users, items, genres


def validate(ratings: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame) -> None:
    """Raise ValueError if the raw data breaks an assumption the pipeline relies on."""
    problems = []
    if ratings.isna().any().any():
        problems.append("ratings contain nulls")
    for column in ["user_id", "item_id", "rating", "timestamp"]:
        if not pd.api.types.is_integer_dtype(ratings[column]):
            problems.append(f"ratings.{column} is not an integer column")
    if not ratings["rating"].between(1, 5).all():
        problems.append("ratings outside 1-5")
    if ratings.duplicated(["user_id", "item_id"]).any():
        problems.append("duplicate (user_id, item_id) pairs")
    if users["user_id"].duplicated().any() or items["item_id"].duplicated().any():
        problems.append("duplicate user or item IDs in metadata")
    if not ratings["user_id"].isin(users["user_id"]).all():
        problems.append("ratings reference users missing from u.user")
    if not ratings["item_id"].isin(items["item_id"]).all():
        problems.append("ratings reference items missing from u.item")
    if users[["age", "gender", "occupation"]].isna().any().any():
        problems.append("user features contain nulls")
    if not users["gender"].isin(["M", "F"]).all():
        problems.append("gender values other than M/F")  # is_male would silently encode them as 0
    if problems:
        raise ValueError("Invalid raw data: " + "; ".join(problems))


def split(ratings: pd.DataFrame, positive_threshold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user temporal leave-one-out over positives (rating >= positive_threshold).

    Per user, in time order: the latest positive is test, the second-latest
    positive is validation, and every interaction before the validation positive
    is training (any rating). Non-positive interactions after the validation
    positive are dropped, so training never sees anything later than validation.
    Users with fewer than 3 positives are kept entirely in training.
    """
    # The row order of u.data is shuffled, so it says nothing about time: sort explicitly.
    # Timestamp ties (a user rating many films in one sitting) are broken by item_id
    # so the split is deterministic.
    df = ratings.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    is_pos = df["rating"] >= positive_threshold
    # Number of positives at or after each row, within the user (reverse cumulative sum):
    # [::-1] reverses the rows, cumsum counts within the user, the second [::-1] restores the order.
    pos_remaining = is_pos[::-1].groupby(df["user_id"][::-1]).cumsum()[::-1]
    evaluable = is_pos.groupby(df["user_id"]).transform("sum") >= 3

    test_mask = evaluable & is_pos & (pos_remaining == 1)
    val_mask = evaluable & is_pos & (pos_remaining == 2)
    # A row is before the validation positive iff val+test are still ahead of it:
    # positives then count themselves (>= 3), non-positives do not (>= 2).
    before_val = (is_pos & (pos_remaining >= 3)) | (~is_pos & (pos_remaining >= 2))
    train_mask = ~evaluable | before_val
    return df[train_mask], df[val_mask], df[test_mask]


def build_features(users: pd.DataFrame, items: pd.DataFrame, genres: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Add zero-based indices and bucketed/encoded feature columns."""
    users = users.sort_values("user_id").reset_index(drop=True)
    items = items.sort_values("item_id").reset_index(drop=True)
    occupations = sorted(users["occupation"].unique())

    users["user_idx"] = np.arange(len(users))
    # np.digitize returns the index of the bin each value falls in: 0..len(bins)
    users["age_bucket"] = np.digitize(users["age"], AGE_BINS)
    # One-hot: age_bucket_<i> is 1 when age_bucket == i. The index column is kept for readability.
    for bucket in range(len(AGE_BINS) + 1):
        users[f"age_bucket_{bucket}"] = (users["age_bucket"] == bucket).astype(int)
    users["is_male"] = (users["gender"] == "M").astype(int)  # F -> 0, M -> 1
    # One-hot: one 0/1 column per occupation, named occupation_<name>, in `occupations` order.
    for occupation in occupations:
        users[f"occupation_{occupation}"] = (users["occupation"] == occupation).astype(int)
    # zip_code is read as text, so leading zeros ("01002") are intact.
    is_us_zip = users["zip_code"].str.fullmatch(r"\d{5}", na=False)  # a missing zip is "not a US zip"
    users["zip_region"] = np.where(is_us_zip, users["zip_code"].str[0].where(is_us_zip, "0").astype(int),
                                   len(ZIP_REGIONS) - 1)
    # One-hot: zip_region_<i> is 1 when zip_region == i. The index column is kept for readability.
    for region in range(len(ZIP_REGIONS)):
        users[f"zip_region_{region}"] = (users["zip_region"] == region).astype(int)

    items["item_idx"] = np.arange(len(items))
    # release_date looks like "01-Jan-1995"; a few items have none -> year NaN -> bucket 0.
    items["release_year"] = pd.to_numeric(items["release_date"].str[-4:], errors="coerce").astype("Int64")
    year = items["release_year"]
    items["year_bucket"] = np.where(year.isna(), 0, np.digitize(year.fillna(0), YEAR_BINS) + 1)

    mappings = {
        "user_ids": users["user_id"].tolist(),  # position = user_idx
        "item_ids": items["item_id"].tolist(),  # position = item_idx
        "genres": genres,
        "occupations": occupations,  # one-hot column order: occupation_<name>
        "zip_regions": ZIP_REGIONS,  # label at position i; one-hot column zip_region_<i>
        "age_bins": AGE_BINS,
        "year_bins": YEAR_BINS,
    }
    return users, items, mappings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--positive-threshold", type=int, default=4, help="rating >= this is a positive")
    args = parser.parse_args()
    settings = get_settings()

    ratings, users, items, genres = load_raw(settings.raw_data_dir)
    validate(ratings, users, items)
    users, items, mappings = build_features(users, items, genres)
    train, val, test = split(ratings, args.positive_threshold)

    out = settings.prepared_data_dir
    out.mkdir(parents=True, exist_ok=True)
    user_index = dict(zip(users["user_id"], users["user_idx"]))
    item_index = dict(zip(items["item_id"], items["item_idx"]))
    digest = hashlib.sha256()
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part = part.assign(user_idx=part["user_id"].map(user_index), item_idx=part["item_id"].map(item_index))
        part.to_csv(out / f"{name}.csv", index=False)
        digest.update((out / f"{name}.csv").read_bytes())
    users.to_csv(out / "users.csv", index=False)
    items.drop(columns=["video_release_date", "imdb_url"]).to_csv(out / "items.csv", index=False)
    feature_digest = hashlib.sha256((out / "users.csv").read_bytes() + (out / "items.csv").read_bytes())

    mappings["positive_threshold"] = args.positive_threshold
    mappings["split_hash"] = digest.hexdigest()[:16]  # identifies this exact split in MLflow
    # Changes whenever a feature column changes, even if the split does not.
    mappings["feature_hash"] = feature_digest.hexdigest()[:16]
    (out / "mappings.json").write_text(json.dumps(mappings))

    n_train_only = users["user_id"].nunique() - val["user_id"].nunique()
    print(f"ratings={len(ratings)} train={len(train)} val={len(val)} test={len(test)} "
          f"dropped={len(ratings) - len(train) - len(val) - len(test)}")
    print(f"users={len(users)} items={len(items)} train_only_users={n_train_only} "
          f"split_hash={mappings['split_hash']} feature_hash={mappings['feature_hash']}")


if __name__ == "__main__":
    main()
