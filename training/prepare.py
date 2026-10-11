"""Load, validate and split MovieLens 100K. Run: python -m training.prepare

Reads the raw files (never writes to them) and writes prepared CSVs plus
mappings.json to settings.prepared_data_dir.

ID convention: `user_id` / `item_id` are raw MovieLens IDs (1-based).
`user_idx` / `item_idx` are zero-based model indices, used only inside training.
"""

# prepare.py — turn raw data into training data
#
# - Job: read the four raw MovieLens files, check them, and write clean files for training.
# - Validates: no nulls, ratings between 1 and 5, no duplicate ratings, every rating points
#   to a known user and film.
# - Builds features: age buckets, gender, occupation and zip region for users; release-year
#   bucket and genres for films. It also gives every user and film a zero-based index
#   (`user_idx`, `item_idx`).
# - Splits by time, per user: the latest liked film (rating ≥ 4) is test, the one before is
#   validation, everything earlier is training.
# - Output: `train.csv`, `val.csv`, `test.csv`, `users.csv`, `items.csv`, `mappings.json`
#   in `training/data/`.

import argparse # read the --positive-threshold flag from the command line
import hashlib # compute a fingerprint of the output files
import json # write mappings.json

import numpy as np
import pandas as pd

from config import get_settings # pydantic setting in config.py for env var

# buckets for data pre-processing
AGE_BINS = [18, 25, 35, 45, 50, 56]
YEAR_BINS = [1950, 1960, 1970, 1980, 1990, 1995, 1997]
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


def load_raw(raw_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]: # This function reads 4 raw files "u.genre","u.data","u.user","u.item" and return ratings, users, items, genres in order. You may notice list[str] in response_model is genre and the other 3 df are 3 csv
    """Return (ratings, users, items, genre_names) with explicit column names."""
    genres = pd.read_csv(raw_dir / "u.genre", sep="|", names=["genre", "genre_idx"])["genre"].tolist() # turn into list[str] by .tolist()
    ratings = pd.read_csv(
        raw_dir / "u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
    )
    users = pd.read_csv(
        raw_dir / "u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"],
        dtype={"zip_code": str},
    )
    items = pd.read_csv(
        raw_dir / "u.item", sep="|", names=ITEM_BASE_COLUMNS + genres, encoding="latin-1" # ITEM_BASE_COLUMNS from above and genres from u.genre
    )
    return ratings, users, items, genres # Return in order of response_model

# This function checks that the raw data is what the rest of the pipeline assumes. It returns nothing: either it passes silently, or it stops the program with an error listing everything that is wrong.
def validate(ratings: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame) -> None: # take 3 df from load_raw
    """Raise ValueError if the raw data breaks an assumption the pipeline relies on."""
    problems = []
    if ratings.isna().any().any(): # .isna() returns True if that specific rating is null, 1st .any() return True for a col if that col contains any null, 2nd .any() return True for the df if the whole df contains any null. In conclusion, if there's any null inside rating.df, it's recognized
        problems.append("ratings contain nulls")
    for column in ["user_id", "item_id", "rating", "timestamp"]: # Check if values in 4 cols are int number
        if not pd.api.types.is_integer_dtype(ratings[column]): 
            problems.append(f"ratings.{column} is not an integer column")
    if not ratings["rating"].between(1, 5).all(): # Check if rating range from 1 to 5. Watch out for the tricky logic here, this line ask if not "all" are in 1-5 range, which has the same meaning as if there's any value not in 1-5 range.
        problems.append("ratings outside 1-5")
    if ratings.duplicated(["user_id", "item_id"]).any(): # Check for duplicate, did a user rate a movie more than once?
        problems.append("duplicate (user_id, item_id) pairs")
    if users["user_id"].duplicated().any() or items["item_id"].duplicated().any(): # Check duplicate of users df and items df
        problems.append("duplicate user or item IDs in metadata")
    if not ratings["user_id"].isin(users["user_id"]).all(): # Check if there's any user in rating df but not in users df
        problems.append("ratings reference users missing from u.user")
    if not ratings["item_id"].isin(items["item_id"]).all(): # Check if there's any item in rating df but not in items df
        problems.append("ratings reference items missing from u.item")
    if users[["age", "gender", "occupation"]].isna().any().any(): # Check if any col in users df is null, the .isnal().any().any() logic is the same as above
        problems.append("user features contain nulls")
    if not users["gender"].isin(["M", "F"]).all(): # Check if there's any different value beside M,F in gender col
        problems.append("gender values other than M/F")
    if problems: # If there's any problem above, raise error
        raise ValueError("Invalid raw data: " + "; ".join(problems)) # lay out all problems
# Key details of this function:
# "validate" collects every problem before raising, rather than stopping at the first.
# "validate" checks nothing about film titles or release dates. A missing release date is allowed and gets year bucket 0 later.


def split(ratings: pd.DataFrame, positive_threshold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: # Take in rating df and a rating threshold, return 3 df which are train/validation/test
    """Per-user temporal leave-one-out over positives (rating >= positive_threshold).

    Per user, in time order: the latest positive is test, the second-latest
    positive is validation, and every interaction before the validation positive
    is training (any rating). Non-positive interactions after the validation
    positive are dropped, so training never sees anything later than validation.
    Users with fewer than 3 positives are kept entirely in training.
    """
    df = ratings.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True) # groups each user's ratings together, oldest first. The raw file is shuffled, so its row order says nothing about time.

    # is_pos, pos_remaining AND evaluable ARE SERIES
    #
    # Each of the three is a Series: a standalone column with one value per rating row
    # (100,000 values), stored in its own variable. They are NOT added to df. df keeps its
    # four original columns (user_id, item_id, rating, timestamp) the whole time.
    #
    # What ties them to df is the row number (the index): value number 6 of is_pos belongs
    # to row number 6 of df.
    #
    # What each one holds for a row:
    #
    #   Series        | Value | Meaning for that row
    #   --------------+-------+------------------------------------------------------------
    #   is_pos        | bool  | True if this rating is >= positive_threshold (a liked film)
    #   pos_remaining | int   | how many of this user's liked films are at this row or later
    #   evaluable     | bool  | True if this user has at least 3 liked films in total
    #                 |       | (the same value on every row of one user)
    #
    # For the example user below (rows A to H):
    #
    #   is_pos        = [True, False, True, False, True, False, True, False]
    #   pos_remaining = [4,    3,     3,    2,     2,    1,     1,    0    ]
    #   evaluable     = [True, True,  True, True,  True, True,  True, True ]
    #
    # test_mask, val_mask, before_val and train_mask are bool Series too. They are used only
    # inside this function and are not written to the CSV files.
    is_pos = df["rating"] >= positive_threshold # Flag ratings >= threshold as positive, others as non-positive

    # WHY pos_remaining EXISTS
    #
    # The split is defined from the END of each user's history:
    #   - test       = the LAST liked film
    #   - validation = the SECOND-TO-LAST liked film
    #   - training   = everything BEFORE the validation film
    #
    # The problem is that a row does not know where it stands. Looking at one row, you see a
    # user, a film, a rating and a time, but not whether it is that user's last liked film.
    #
    # pos_remaining gives every row that missing information. It is a countdown:
    #   "from this row to the end of this user's history, how many liked films are there,
    #    counting this row if it is one?"
    #
    # It counts from the end because users have different amounts of history. Counting from
    # the start, the last liked film is number 4 for one user and number 200 for another.
    # Counting from the end, it is number 1 for everybody.
    #
    # Example: one made-up user with eight ratings, in time order.
    #
    #   Row | Rating | Liked? | pos_remaining | How to read it                             | Ends up in
    #   ----+--------+--------+---------------+--------------------------------------------+-----------
    #   A   |   5    |  yes   |       4       | A, C, E, G are still to come (counting A)  | train
    #   B   |   2    |  no    |       3       | C, E, G are still to come                  | train
    #   C   |   4    |  yes   |       3       | C, E, G                                    | train
    #   D   |   3    |  no    |       2       | E, G                                       | train
    #   E   |   5    |  yes   |       2       | E, G                                       | VALIDATION
    #   F   |   1    |  no    |       1       | G                                          | dropped
    #   G   |   4    |  yes   |       1       | G                                          | TEST
    #   H   |   2    |  no    |       0       | none                                       | dropped
    pos_remaining = is_pos[::-1].groupby(df["user_id"][::-1]).cumsum()[::-1]
    # Logic of pos_remaining:
    # is_pos[::-1] reverses the rows, newest first.
    # .groupby(df["user_id"][::-1]) handles each user separately.
    # .cumsum() is a running total, counting True as 1. Because the rows are reversed, it counts from the newest rating backwards.
    # The final [::-1] puts the rows back in the original order.

    # WHERE evaluable FITS
    #
    # evaluable appears in test_mask and val_mask so that a user with only one or two liked
    # films does not lose them to test and validation and end up with nothing to train on.
    # Those users go entirely to training (see train_mask below).
    evaluable = is_pos.groupby(df["user_id"]).transform("sum") >= 3 # Check if user has at least 3 positive ratings, if not then this user is in train df only

    # test_mask
    #
    # "The last liked film" means: only one liked film remains, AND it is this row.
    #   - pos_remaining == 1 is true for rows F and G.
    #   - F is not a liked film; its 1 refers to G, which comes after it.
    #   - is_pos removes F. Only G is left.
    #
    # So both conditions are needed: the count finds the right place, and is_pos makes sure
    # the row itself is the liked film.
    test_mask = evaluable & is_pos & (pos_remaining == 1)

    # val_mask
    #
    # Same idea, one step earlier: two liked films remain, and this row is one of them.
    #   - pos_remaining == 2 is true for rows D and E.
    #   - D is not a liked film. is_pos removes it. Only E is left.
    val_mask = evaluable & is_pos & (pos_remaining == 2)

    # before_val
    #
    # We want rows A, B, C, D: everything earlier than E. In words, a row is before the
    # validation film when the validation film and the test film are BOTH STILL AHEAD of it.
    #
    # The number that expresses "both still ahead" depends on whether the row counts itself:
    #
    #   - A liked row counts itself. So "itself + validation + test" is at least 3.
    #     Rows A (4) and C (3) pass. Row E (2) fails, correctly, because E is the
    #     validation film.
    #
    #   - A not-liked row does not count itself. So "validation + test" is at least 2.
    #     Rows B (3) and D (2) pass. Row F (1) fails, correctly, because only the test film
    #     is ahead of it.
    #
    # That is why there are two halves with different numbers. One number alone gets it wrong:
    #
    #   If the rule were only... | What goes wrong
    #   -------------------------+------------------------------------------------------------
    #   pos_remaining >= 3       | Row D (2) is left out of training, although it is before E.
    #   pos_remaining >= 2       | Row E (2) is put into training, so the validation film
    #                            | leaks into training.
    before_val = (is_pos & (pos_remaining >= 3)) | (~is_pos & (pos_remaining >= 2))
    train_mask = ~evaluable | before_val # For a rating to be in Train df, there are 2 cases. 1) user is not evaluable (< 3 pos), 2) ratings before validation rating
    return df[train_mask], df[val_mask], df[test_mask] # Return train/validation/test


def build_features(users: pd.DataFrame, items: pd.DataFrame, genres: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Add zero-based indices and bucketed/encoded feature columns."""
    users = users.sort_values("user_id").reset_index(drop=True)
    items = items.sort_values("item_id").reset_index(drop=True)
    occupations = sorted(users["occupation"].unique())

    users["user_idx"] = np.arange(len(users))
    users["age_bucket"] = np.digitize(users["age"], AGE_BINS)
    for bucket in range(len(AGE_BINS) + 1):
        users[f"age_bucket_{bucket}"] = (users["age_bucket"] == bucket).astype(int)
    users["is_male"] = (users["gender"] == "M").astype(int)
    for occupation in occupations:
        users[f"occupation_{occupation}"] = (users["occupation"] == occupation).astype(int)
    is_us_zip = users["zip_code"].str.fullmatch(r"\d{5}", na=False)
    users["zip_region"] = np.where(is_us_zip, users["zip_code"].str[0].where(is_us_zip, "0").astype(int),
                                   len(ZIP_REGIONS) - 1)
    for region in range(len(ZIP_REGIONS)):
        users[f"zip_region_{region}"] = (users["zip_region"] == region).astype(int)

    items["item_idx"] = np.arange(len(items))
    items["release_year"] = pd.to_numeric(items["release_date"].str[-4:], errors="coerce").astype("Int64")
    year = items["release_year"]
    items["year_bucket"] = np.where(year.isna(), 0, np.digitize(year.fillna(0), YEAR_BINS) + 1)

    mappings = {
        "user_ids": users["user_id"].tolist(),
        "item_ids": items["item_id"].tolist(),
        "genres": genres,
        "occupations": occupations,
        "zip_regions": ZIP_REGIONS,
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
    mappings["split_hash"] = digest.hexdigest()[:16]
    mappings["feature_hash"] = feature_digest.hexdigest()[:16]
    (out / "mappings.json").write_text(json.dumps(mappings))

    n_train_only = users["user_id"].nunique() - val["user_id"].nunique()
    print(f"ratings={len(ratings)} train={len(train)} val={len(val)} test={len(test)} "
          f"dropped={len(ratings) - len(train) - len(val) - len(test)}")
    print(f"users={len(users)} items={len(items)} train_only_users={n_train_only} "
          f"split_hash={mappings['split_hash']} feature_hash={mappings['feature_hash']}")


if __name__ == "__main__":
    main()
