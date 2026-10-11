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
# - Builds features: standard-scaled age, gender, occupation and zip region for users;
#   standard-scaled release year and genres for films. It also gives every user and film a zero-based index
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import get_settings # pydantic setting in config.py for env var

# buckets for data pre-processing
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
# "validate" checks nothing about film titles or release dates. A missing release date is allowed and gets the average year (scaled value 0) later.


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


# ==========================================================================================
# NOTES ON one_hot()
#
# one_hot is used for the two features that are labels: occupation and zip region. Age is
# NOT one-hot encoded: it is a number, so build_features standard-scales it into a single
# column, age_scaled (see the note at age_scaler in build_features).
#
# Every "without it" result below was tried on a tiny example: three users whose zip
# regions are 1, 1, 3, with allowed regions [0, 1, 2, 3]. (The real list has 11 regions,
# 0 to 10; four keeps the example small.)
#
#
# WHAT THE FUNCTION IS FOR
#
# You have a column where each user has one label out of a fixed set: an occupation, a zip
# region. The model cannot use a label directly, so you want one 0/1 column per possible
# label.
#
# You need this twice, for two different columns. So the function is written once and
# takes two things from you each time:
#
#   - column:     "here is the column I want encoded", for example users["zip_region"].
#   - categories: "here is the complete list of labels it can have, in the order I want
#                 the output columns", for example [0, 1, 2, 3].
#
# It hands you back a table of 0/1 columns, one row per user. For the example:
#
#   zip_region_0  zip_region_1  zip_region_2  zip_region_3
#        0             1             0             0
#        0             1             0             0
#        0             0             0             1
#
# How it knows which feature to encode: it doesn't decide; the caller does. build_features
# calls it twice:
#
#   Call | column               | categories            | Output
#   -----+----------------------+-----------------------+-----------
#   1    | users["occupation"]  | the 21 sorted names   | 21 columns
#   2    | users["zip_region"]  | [0, 1, ..., 10]       | 11 columns
#
# Without the function you would write the same two lines twice in build_features, once
# per feature. That works, but any fix would have to be made in two places.
#
#
# categories=[categories]: TELLING IT THE LABELS UP FRONT
#
# Why we need it. Left alone, the encoder looks at the data and makes a column only for
# labels it actually sees. In the example nobody is in region 0 or 2.
#
# Without it you get two columns, not four:
#
#   zip_region_1  zip_region_3
#
# Here is why that matters in this project. train.py asks for zip_region_0 through
# zip_region_10 by name. On the full dataset every region has users, so you would never
# notice. But run it on a small sample, or on new data where nobody lives in region 0, and
# zip_region_0 does not exist: training crashes, or worse, a model trained on 34 columns
# is fed 33.
#
# Giving the list means the output always has the same columns in the same order, whatever
# the data contains.
#
# Why the extra brackets. The encoder can encode several columns in one go, so it expects
# "a list of label-lists, one per column". We give one column, so it is a list holding one
# list. Without the brackets it fails with "Shape mismatch".
#
#
# sparse_output=False: GIVE ME A NORMAL TABLE
#
# Why we need it. A one-hot table is mostly zeros, so by default the encoder returns a
# compressed format that stores only the positions of the 1s. That saves memory on huge
# data, but pandas cannot attach it to the users table.
#
# Without it the function fails immediately: "Pandas output does not support sparse data".
#
#
# dtype=int: WHOLE NUMBERS
#
# Why we need it. By default the encoder produces decimals, 0.0 and 1.0.
#
# Without it nothing crashes and the model would train the same. But users.csv would
# contain 0.0 where it could contain 0, so the file gets bigger and its fingerprint (the
# feature hash) changes, for no gain. A flag is a whole number, so it is stored as one.
#
#
# column.to_frame(): WRAP THE COLUMN INTO A TABLE
#
# Why we need it. The encoder only accepts a table, meaning rows and columns.
# users["zip_region"] is a single column on its own, which pandas treats as
# one-dimensional. to_frame() turns it into a table with one column, keeping the name
# zip_region.
#
# Without it it fails: "Expected a 2-dimensional container but got Series".
#
# That kept name matters too: it is where the zip_region_ prefix in the output names
# comes from.
#
#
# .set_output(transform="pandas"): KEEP THE NAMES AND ROW NUMBERS
#
# Why we need it. By default the encoder returns a bare grid of numbers with no column
# names and no row numbers.
#
# Without it you get the right 0s and 1s but nothing saying which column is which region.
# You would have to build the names zip_region_0, ... yourself and make sure the order
# matches. With it, the encoder names the columns and keeps the row numbers, so
# users.join(...) attaches each row to the right user.
#
#
# .fit_transform(...): CHECK, THEN ENCODE
#
# It is two steps in one call.
#
#   - fit is normally "look at the data and learn the labels". We already gave the labels,
#     so here it learns nothing. What it still does is check: every value in the column
#     must be in the list. If a user had region 9, it stops with
#     "Found unknown categories [9]". That is a useful safety check.
#   - transform produces the 0/1 columns.
#
# Why not just transform? sklearn refuses: "This OneHotEncoder instance is not fitted
# yet". The encoder must be fitted once before use, even when there is nothing to learn.
#
#
# ABOUT LEAKAGE
#
# fit_transform on a whole dataset is where leakage usually hides. The classic case is
# fitting a scaler on all rows before splitting, so the mean it learns includes test rows.
#
# It is not leakage here, for two reasons:
#
#   - Nothing is learned. A user's output depends only on their own value and the list we
#     passed. No other row influences it, so nothing can leak from one row to another.
#   - It runs on the users table, not on ratings. The split is over ratings. Users are not
#     split, every user is in training, and occupation or zip region is not what the
#     model predicts.
#
# If we had let the encoder learn the labels from data, it would still not leak anything
# about ratings, but it would bring back the missing-column problem above.
#
# Compare with age: the StandardScaler in build_features is different, because it DOES
# learn something from the data (the mean and standard deviation of age). Only the second
# reason protects it. See the note at age_scaler.
#
#
# WHY THE ENCODER IS NOT KEPT
#
# The encoder is created, used once and thrown away; nothing saves it. That is fine only
# because the categories are explicit: training/index.py calls build_features again later,
# builds a fresh encoder from the same lists, and gets identical columns. If the categories
# were learned from data, the two runs could disagree and the encoder would have to be
# saved and reloaded.
# ==========================================================================================
def one_hot(column: pd.Series, categories: list) -> pd.DataFrame: 
    """One 0/1 column per category, named <column name>_<category>, in `categories` order.

    The categories are passed in rather than learned from the data, so a category that no
    row happens to have still gets its (all-zero) column and the model's input width is fixed.
    """
    encoder = OneHotEncoder(categories=[categories], sparse_output=False, dtype=int)
    return encoder.set_output(transform="pandas").fit_transform(column.to_frame())


# ==========================================================================================
# NOTES ON build_features(): NEW TABLE OR MODIFIED TABLE? AND WHY NOTHING IS DROPPED
#
#
# DOES IT BUILD A NEW DF OR MODIFY THE ONE PASSED IN?
#
# It builds new ones. The tables passed in are never modified.
#
# The reason is the first two lines of the function:
#
#   users = users.sort_values("user_id").reset_index(drop=True)
#   items = items.sort_values("item_id").reset_index(drop=True)
#
# sort_values and reset_index each return a NEW table and leave the original alone. The
# result is assigned to the name users, so from this line on, users inside the function
# refers to the new copy, not to the table the caller passed in.
#
# Everything after that works on the copy:
#
#   Line                                 | What it does to the copy
#   -------------------------------------+--------------------------------------------------
#   users["user_idx"] = ...              | adds a column to the copy in place
#   users = users.join(one_hot(...))     | join returns yet another new table, and users
#                                        | now refers to that one
#   return users, items, mappings        | hands the final tables back
#
# So "does it modify?" has two honest answers depending on which table you mean:
#
#   - The caller's original tables: no, untouched.
#   - The function's own copy: yes, columns are added to it step by step.
#
# At the call site the caller reuses the same names for the result:
#
#   users, items, mappings = build_features(users, items, genres)
#
# After this line, users in main refers to the new 41-column table. The raw 5-column table
# from load_raw still exists in memory for a moment, but no name points to it any more, so
# Python discards it. That reuse of the name is why it can look as if the table was
# modified in place.
#
#
# COPY-OR-IN-PLACE AND DROPPING COLUMNS ARE TWO SEPARATE QUESTIONS
#
#   - Copy or in-place is about whether the CALLER'S table is affected.
#   - Dropping columns is about WHICH COLUMNS THE RESULT HAS.
#
# One does not decide the other. The new table still contains the original columns.
#
# After build_features, the users table has all 5 raw columns plus the new ones, 41 in total:
#
#   user_id, age, gender, occupation, zip_code,        <- original, still there
#   user_idx, age_scaled, is_male,                     <- added
#   occupation_administrator, ...  (21 columns),       <- added
#   zip_region,                                        <- added
#   zip_region_0, ... zip_region_10 (11 columns)       <- added
#
# So age sits next to age_scaled, and occupation sits next to its 21 flag columns. Nothing
# was dropped, and it would be the same if the function had modified the original in place.
#
#
# WHY THE ORIGINAL COLUMNS DO NOT NEED DROPPING
#
#   - The model picks its columns by name. train.py builds the list is_male, age_scaled,
#     the occupation flags and the zip region flags, and selects exactly those. Extra
#     columns in the table are simply ignored, so leaving age or occupation there does no
#     harm.
#   - Another file needs the raw columns. training/index.py loads user_id, age, gender,
#     occupation and zip_code into Postgres. If build_features dropped them, that would
#     break.
#
# The only place columns are dropped is in main, when writing items.csv:
#
#   items.drop(columns=["video_release_date", "imdb_url"]).to_csv(out / "items.csv", index=False)
#
# Those two are removed because nothing in the project uses them.
#
# When dropping the originals DOES matter is the other common style, where you pass the
# WHOLE table to the model. Then a leftover text column like occupation would crash it or
# be fed in by mistake. This project avoids that by selecting columns by name.
# ==========================================================================================
def build_features(users: pd.DataFrame, items: pd.DataFrame, genres: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Add zero-based indices and bucketed/encoded feature columns."""
    users = users.sort_values("user_id").reset_index(drop=True)
    items = items.sort_values("item_id").reset_index(drop=True)
    occupations = sorted(users["occupation"].unique())

    users["user_idx"] = np.arange(len(users)) # Turn user_id into 0-based index

    # Age as ONE number per user instead of 7 bucket columns: age_scaled = (age - mean) / std,
    # so the column has mean 0 and standard deviation 1. A raw age (7 to 73) next to 0/1
    # columns would dominate the first layer; scaling puts it on the same footing.
    #   - fit learns the mean and std from the data; transform applies them.
    #   - The scaler only accepts a table, hence the double brackets users[["age"]];
    #     [:, 0] takes the single output column back out of the 2-D result.
    #   - The raw "age" column is kept untouched: training/index.py loads it into Postgres.
    # Leakage: unlike one_hot, this DOES learn from the data. It is still safe because users
    # are not split (every user has training ratings), so the mean and std come from training
    # users only. If users were ever held out, fit on the training users and transform the rest.
    age_scaler = StandardScaler()
    users["age_scaled"] = age_scaler.fit_transform(users[["age"]])[:, 0]
    users["is_male"] = (users["gender"] == "M").astype(int) # map 1 for M and 0 for F
    users = users.join(one_hot(users["occupation"], occupations)) # Apply one_hot() onto occupation
    is_us_zip = users["zip_code"].str.fullmatch(r"\d{5}", na=False) # takes in zipcode, see if the zipcode contains 5 numbers (\d for digit and 5 for 5 numbers), if the zipcode is not 5 numbers then it's marked as non-us (for example 9024 which is 4 num or 1254E which has a letter)
    users["zip_region"] = np.where(is_us_zip, users["zip_code"].str[0].where(is_us_zip, "0").astype(int), len(ZIP_REGIONS) - 1) # Take the first number of the is_us_zip and find its corresponding region on ZIP_REGION. The ones that are non-us, marked as 10 which is non-us/uknown in ZIP_REGION
    users = users.join(one_hot(users["zip_region"], list(range(len(ZIP_REGIONS))))) # Apply one_hot() onto zip regions (extract from zipcode)

    items["item_idx"] = np.arange(len(items)) # Turn item_id into 0-based index
    items["release_year"] = pd.to_numeric(items["release_date"].str[-4:], errors="coerce").astype("Int64") # Extract release_year form release_date

    # Release year as ONE number per film instead of a year bucket, same idea as age_scaled:
    # release_year_scaled = (year - mean) / std.
    #   - .astype("float64") turns the nullable Int64 column into plain floats, so the missing
    #     year becomes NaN, which the scaler accepts (pd.NA it does not).
    #   - One film (item 267, titled "unknown") has no release date. The scaler ignores NaN
    #     when learning the mean and std and leaves it NaN in the output; fillna(0.0) then
    #     gives that film the average year, because 0 is the mean after scaling.
    #   - The raw "release_year" column is kept untouched: training/index.py loads it into
    #     Postgres and Qdrant.
    #   - Years are bunched up (half the films are from 1995 or later, the oldest is 1922), so
    #     recent films land between about 0.2 and 0.6 and old ones go down to about -4.7.
    # Leakage: the scaler is fitted on the whole catalogue, including the few films with no
    # training rating. A release year is catalogue metadata known for every film up front,
    # not a held-out label, so nothing about validation or test ratings can leak through it.
    year_scaler = StandardScaler()
    items["release_year_scaled"] = year_scaler.fit_transform(items[["release_year"]].astype("float64"))[:, 0]
    items["release_year_scaled"] = items["release_year_scaled"].fillna(0.0)

    # mappings.json is the hand-over note from prepare.py to the later steps.
    # It records the decisions prepare.py made that CANNOT be read from the CSV files, so that train.py, evaluate.py and index.py use the same ones without redoing or guessing them.
    # Those one-hot cols are inside CSV so mapping doesn't need to have them, 
    # the scaled age and release year are also inside CSV 
    # but the mean and std that are used to compute it is not, that's why they're in mapping
    mappings = {
        "user_ids": users["user_id"].tolist(),
        "item_ids": items["item_id"].tolist(),
        "genres": genres,
        "occupations": occupations,
        "zip_regions": ZIP_REGIONS,
        "age_mean": float(age_scaler.mean_[0]),   # learned by the scaler; age = age_scaled * age_std + age_mean
        "age_std": float(age_scaler.scale_[0]),
        "release_year_mean": float(year_scaler.mean_[0]),  # release_year = release_year_scaled * std + mean
        "release_year_std": float(year_scaler.scale_[0]),
    }
    return users, items, mappings


def main() -> None:
    # NOTES ON main(): READING THE COMMAND LINE (THE NEXT THREE LINES)
    #
    # Use case: these three lines let you choose the "liked film" rating when you run the
    # script, without opening the file.
    #
    #   python -m training.prepare                          -> the threshold is 4
    #   python -m training.prepare --positive-threshold 5   -> the threshold is 5
    #
    # Without them, the 4 would be typed directly into the code, and trying 5 as an
    # experiment would mean editing the file every time.
    #
    # Line 1 creates the reader, line 2 tells it which option exists, line 3 does the reading.
    #
    # LINE 1, peeled from the inside out:
    #
    #   Piece                        | What it returns
    #   -----------------------------+----------------------------------------------------------
    #   __doc__                      | str: the docstring at the very top of this file
    #   __doc__.splitlines()         | list of str: that docstring cut into its lines
    #   __doc__.splitlines()[0]      | str: the first line only (shown below)
    #   argparse.ArgumentParser(...) | an ArgumentParser object: a reader that knows no options yet
    #
    # The first line of the docstring is:
    #   "Load, validate and split MovieLens 100K. Run: python -m training.prepare"
    #
    # description= is the text printed when you run: python -m training.prepare --help
    # This is the reason the docstrings were kept when the comments were deleted.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    # LINE 2: declares the one option this script accepts. It reads nothing yet.
    #
    #   Piece                  | What it means
    #   -----------------------+----------------------------------------------------------------
    #   "--positive-threshold" | the name you type on the command line
    #   type=int               | the command line is always text; this turns the text "5" into
    #                          | the number 5. Without it the threshold would be the text "5",
    #                          | and df["rating"] >= "5" in split() would raise an error.
    #   default=4              | the value used when you do not type the option
    #   help="..."             | the explanation shown next to the option in --help
    #
    # Returns: an object describing the option. Nobody needs it, so it is not stored in a
    # variable. What matters is the side effect: parser now knows the option.
    parser.add_argument("--positive-threshold", type=int, default=4, help="rating >= this is a positive")

    # LINE 3: reads what you typed after "python -m training.prepare" and checks it.
    #
    # Returns: a Namespace object, a small box whose values you read with a dot.
    #
    #   You typed                | args                            | args.positive_threshold
    #   -------------------------+---------------------------------+------------------------
    #   nothing                  | Namespace(positive_threshold=4) | 4
    #   --positive-threshold 5   | Namespace(positive_threshold=5) | 5
    #   --positive-threshold abc | the script stops with an error  | -
    #
    # The dash in the option name becomes an underscore, because a Python name cannot contain
    # a dash: --positive-threshold is read as args.positive_threshold.
    args = parser.parse_args()
    settings = get_settings() # Pass in the env var of prepared_data_dir from congif.py

    ratings, users, items, genres = load_raw(settings.raw_data_dir) # load raw data using load_raw()
    validate(ratings, users, items) # validate raw data using validate(), notice that validate() returns nothing so no variable needed. Besides that, validate() is above build_features() and split() so that if it sees a bad data, it raises error so build_features() and split() never run
    users, items, mappings = build_features(users, items, genres)
    train, val, test = split(ratings, args.positive_threshold)

    # out: THE OUTPUT FOLDER
    #
    # Returns: a Path object pointing at the folder training/data inside the project.
    # Nothing is created here. out is only a shorter name for settings.prepared_data_dir,
    # because it is used many times below.
    #
    # A Path is not plain text. It is an object that knows it is a location on disk, which
    # is what makes out.mkdir(...) and out / "users.csv" below possible.
    out = settings.prepared_data_dir

    # out.mkdir(...): CREATE THE FOLDER
    #
    #   Piece         | What it means                         | Without it
    #   --------------+---------------------------------------+----------------------------------
    #   out.mkdir()   | create the folder training/data       | to_csv below fails: no such folder
    #   parents=True  | also create missing folders above it  | error if training/ did not exist
    #   exist_ok=True | no error if the folder already exists | the SECOND run of the script
    #                 |                                       | crashes with FileExistsError
    #
    # Returns: None. Its effect is on the disk, not in a variable.
    out.mkdir(parents=True, exist_ok=True)

    # user_index AND item_index: LOOKUP TABLES FROM RAW ID TO 0-BASED INDEX
    #
    # Use case: the ratings tables (train, val, test) only have raw IDs. The model needs the
    # 0-based numbers. These two dictionaries are the translation used in the loop below.
    #
    # Peeled from the inside out:
    #
    #   Piece              | What it returns
    #   -------------------+--------------------------------------------------------
    #   users["user_id"]   | Series of 943 raw IDs:   1, 2, 3, ...
    #   users["user_idx"]  | Series of 943 indices:   0, 1, 2, ...
    #   zip(first, second) | pairs, row by row:       (1, 0), (2, 1), (3, 2), ...
    #   dict(those pairs)  | dictionary:              {1: 0, 2: 1, 3: 2, ..., 943: 942}
    #
    # zip works like a zipper: it takes value number 1 of both columns, then value number 2
    # of both, and so on. dict then uses the first of each pair as the key and the second as
    # the value.
    #
    # Reading them:   user_index[1] gives 0       item_index[168] gives 167
    #
    # user_index has 943 entries, item_index has 1,682.
    user_index = dict(zip(users["user_id"], users["user_idx"]))
    item_index = dict(zip(items["item_id"], items["item_idx"]))

    # digest: AN EMPTY FINGERPRINT CALCULATOR
    #
    # Returns: a sha256 hash object with nothing fed into it yet.
    #
    # What a fingerprint (hash) is: you feed it bytes, and at the end it gives a short text.
    # The same bytes always give the same text. Change one byte and the text is completely
    # different. So two fingerprints that match mean "these are the same files".
    #
    # It works in two steps, which is why it is created here, before the loop:
    #   digest.update(bytes)   feed it more data   (done three times, inside the loop)
    #   digest.hexdigest()     ask for the result  (done once, further down)
    digest = hashlib.sha256()

    # THE LOOP: WRITE train.csv, val.csv AND test.csv
    #
    # The list holds three pairs (tuples). Each pair is a text and a table. On every round,
    # Python unpacks one pair into the two names name and part:
    #
    #   Round | name    | part
    #   ------+---------+------------------------------
    #   1     | "train" | the train table (94,581 rows)
    #   2     | "val"   | the val table   (942 rows)
    #   3     | "test"  | the test table  (942 rows)
    #
    # name is needed because a table does not know what its own variable is called; the text
    # is used to build the file name. Without the loop, the three lines inside would have to
    # be written three times, once per table.
    for name, part in [("train", train), ("val", val), ("test", test)]:
        # LOOP LINE 1: ADD user_idx AND item_idx TO THE TABLE
        #
        #   Piece                           | What it returns
        #   --------------------------------+---------------------------------------------------
        #   part["user_id"]                 | Series of raw user IDs, one per rating row
        #   part["user_id"].map(user_index) | Series of the same length: every ID looked up in
        #                                   | the dictionary, so 1 becomes 0
        #   part["item_id"].map(item_index) | the same for films, so 168 becomes 167
        #   part.assign(user_idx=...,       | a NEW table: part plus two columns. The words
        #               item_idx=...)       | before the = signs become the column names.
        #
        # One real row, before and after:
        #
        #   user_id  item_id  rating  timestamp
        #         1      168       5  874965478
        #
        #   user_id  item_id  rating  timestamp  user_idx  item_idx
        #         1      168       5  874965478         0       167
        #
        # .map and an ID that is not in the dictionary: the result is NaN (empty), not an
        # error. That cannot happen here, because validate() already stopped the program if a
        # rating referred to a user or film missing from the metadata.
        #
        # part = ... : assign does not change the table, it returns a new one, and the name
        # part is pointed at the new one. The variables train, val and test outside the loop
        # still have 4 columns. Only the files get 6.
        #
        # Without these two columns, train.py would only have raw IDs, and it needs the
        # 0-based numbers to pick rows out of the embedding tables.
        part = part.assign(user_idx=part["user_id"].map(user_index), item_idx=part["item_id"].map(item_index))

        # LOOP LINE 2: WRITE THE TABLE TO A FILE
        #
        #   Piece                          | What it returns
        #   -------------------------------+----------------------------------------------------
        #   f"{name}.csv"                  | str: "train.csv" on round 1. The f in front lets
        #                                  | {name} be replaced by the value of the variable.
        #   out / f"{name}.csv"            | Path: training/data/train.csv. On a Path, the /
        #                                  | sign is not division; it means "inside this folder".
        #   part.to_csv(path, index=False) | None. It writes the file.
        #
        # index=False: do not write pandas' own row numbers. Without it the file would get an
        # extra first column with no name, holding 0, 1, 2, ...
        part.to_csv(out / f"{name}.csv", index=False)

        # LOOP LINE 3: FEED THE FILE INTO THE FINGERPRINT
        #
        #   Piece                      | What it returns
        #   ---------------------------+--------------------------------------------------------
        #   out / f"{name}.csv"        | Path to the file written one line above
        #   (...).read_bytes()         | bytes: the whole file exactly as stored on disk
        #                              | (2,606,353 bytes for train.csv)
        #   digest.update(those bytes) | None. It changes digest itself: the bytes are added to
        #                              | what it has seen so far.
        #
        # The brackets around out / f"{name}.csv" are needed so that the path is built first
        # and .read_bytes() is then called on the whole path. Without them, Python would try
        # .read_bytes() on the text "train.csv" alone, which fails.
        #
        # The file is written and then read back, instead of fingerprinting the table in
        # memory, so the fingerprint describes exactly what is on disk, which is what train.py
        # will read.
        #
        # After the three rounds, digest has seen train.csv, then val.csv, then test.csv.
        digest.update((out / f"{name}.csv").read_bytes())

    # WRITE users.csv
    #
    # The same to_csv as in the loop. Returns None. The file gets all 41 columns: nothing is
    # dropped (see the note above build_features).
    users.to_csv(out / "users.csv", index=False)

    # WRITE items.csv, WITHOUT TWO COLUMNS
    #
    #   Piece                           | What it returns
    #   --------------------------------+---------------------------------------------------
    #   items.drop(columns=[..., ...])  | a NEW table without those two columns: 25 columns
    #                                   | left out of 27. items itself still has both.
    #   .to_csv(out / "items.csv", ...) | None. Called on that new table, it writes the file.
    #
    # The two calls are chained: the table returned by drop is used at once and never gets a
    # name of its own.
    #
    # Why these two: video_release_date is empty in all 1,682 rows of the raw file, and
    # imdb_url is not read by any file in training/ or api/.
    items.drop(columns=["video_release_date", "imdb_url"]).to_csv(out / "items.csv", index=False)

    # feature_digest: A SECOND FINGERPRINT, FOR THE TWO FEATURE FILES
    #
    #   Piece                            | What it returns
    #   ---------------------------------+--------------------------------------------------
    #   (out / "users.csv").read_bytes() | bytes of users.csv
    #   (out / "items.csv").read_bytes() | bytes of items.csv
    #   first + second                   | bytes: the two joined end to end. On bytes, as on
    #                                    | text, + means "join", not "add numbers".
    #   hashlib.sha256(joined bytes)     | a sha256 hash object that has ALREADY been fed
    #
    # Difference from digest above: there the calculator was created empty and fed three
    # times with update(). Here the data is handed over at creation, in one go. Both ways
    # give the same kind of result.
    feature_digest = hashlib.sha256((out / "users.csv").read_bytes() + (out / "items.csv").read_bytes())

    # THREE MORE ENTRIES FOR mappings (THE NEXT THREE LINES)
    #
    # mappings["key"] = value adds a new entry to the dictionary (or overwrites it if the key
    # already exists). It returns nothing. build_features() made 9 entries; after these three
    # lines there are 12.
    #
    # They are added here and not in build_features() because only main() knows them: the
    # threshold comes from the command line, and the fingerprints exist only after the files
    # are written.
    #
    # Line 1: args.positive_threshold is an int, 4 unless you typed something else.
    #
    # Lines 2 and 3:
    #
    #   Piece              | What it returns
    #   -------------------+-----------------------------------------------------------------
    #   digest.hexdigest() | str of 64 characters: the fingerprint written with 0-9 and a-f
    #   [:16]              | str: the first 16 characters only
    #
    # With the current data:
    #   digest.hexdigest()       "f8386e5170af9700d7278930c2884610e06a3c6b9e8bd4f2cdd09ede79d315a5"
    #   digest.hexdigest()[:16]  "f8386e5170af9700"
    #
    # 16 characters are enough to tell two versions of the data apart, and are easier to read
    # and compare than 64.
    #
    # What the two fingerprints are for: train.py copies both into meta.json, next to the
    # model, so the model carries a record of the exact data it was trained on.
    #   - split_hash:   evaluate.py compares the one in mappings.json with the one in meta.json
    #                   and stops if they differ. Without it you could re-run prepare.py with
    #                   another threshold, forget to retrain, and get a test score on films
    #                   the model was trained on.
    #   - feature_hash: recorded in meta.json and MLflow only. No code compares it.
    mappings["positive_threshold"] = args.positive_threshold
    mappings["split_hash"] = digest.hexdigest()[:16]
    mappings["feature_hash"] = feature_digest.hexdigest()[:16]

    # WRITE mappings.json
    #
    #   Piece                  | What it returns
    #   -----------------------+-----------------------------------------------------------
    #   json.dumps(mappings)   | str: the whole dictionary as JSON text, 14,711 characters,
    #                          | starting {"user_ids": [1, 2, 3, ...
    #   out / "mappings.json"  | Path to the file
    #   (...).write_text(text) | int: the number of characters written. It is not stored,
    #                          | because nobody needs it.
    #
    # dumps means "dump to string". A dictionary lives only in memory and a file can only hold
    # text, so the dictionary is turned into text first. train.py does the reverse with
    # json.loads.
    (out / "mappings.json").write_text(json.dumps(mappings))

    # n_train_only: HOW MANY USERS HAVE NO VALIDATION FILM
    #
    #   Piece                      | What it returns
    #   ---------------------------+----------------------------------------------------------
    #   users["user_id"].nunique() | int: number of different user IDs in the users table, 943
    #   val["user_id"].nunique()   | int: number of different users with a row in val, 942
    #   first - second             | int: 1
    #
    # Those are the users with fewer than 3 liked films (evaluable is False in split()). All
    # of their ratings are in train.
    n_train_only = users["user_id"].nunique() - val["user_id"].nunique()

    # THE TWO print LINES: A REPORT ON THE SCREEN
    #
    # They change no data and write no file. print returns None.
    #
    #   Piece                    | What it means
    #   -------------------------+-------------------------------------------------------------
    #   f"...{x}..."             | f-string: each {...} is replaced by the value of what is
    #                            | inside the braces
    #   len(table)               | int: the number of rows of the table
    #   two f"..." in a row,     | Python joins them into ONE text. It is only a way to keep
    #   nothing between them     | each line of code short.
    #   mappings['split_hash']   | single quotes here, because the text around it already
    #                            | uses double quotes
    #
    # dropped = all ratings minus the ones that landed in the three files. Those are the
    # non-liked ratings that come after the validation film (rows F and H in the split()
    # example).
    #
    # With the current data the two lines print:
    #   ratings=100000 train=94581 val=942 test=942 dropped=3535
    #   users=943 items=1682 train_only_users=1 split_hash=f8386e5170af9700 feature_hash=69aedaf36d5e4b50
    print(f"ratings={len(ratings)} train={len(train)} val={len(val)} test={len(test)} "
          f"dropped={len(ratings) - len(train) - len(val) - len(test)}")
    print(f"users={len(users)} items={len(items)} train_only_users={n_train_only} "
          f"split_hash={mappings['split_hash']} feature_hash={mappings['feature_hash']}")
# In conclusion, besides calling the 4 functions above, main() is mostly I/O syntax.

if __name__ == "__main__":
    main()
