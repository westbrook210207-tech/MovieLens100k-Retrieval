-- Applied by `python -m training.index` inside the same transaction that reloads the data,
-- so the tables always match this file and readers never see a half-loaded state.
-- All IDs are raw MovieLens IDs (1-based), not model indices.
--
-- shortcut: tables are dropped and rebuilt on every index run, which is only safe while
-- Postgres holds nothing but data derived from ml-100k/. Switch to migrations (e.g. Alembic)
-- before the API starts writing its own rows here.

DROP TABLE IF EXISTS interactions, items, users;

CREATE TABLE users (
    user_id    integer PRIMARY KEY,
    age        integer NOT NULL,
    gender     text    NOT NULL,
    occupation text    NOT NULL,
    zip_code   text
);

CREATE TABLE items (
    item_id      integer PRIMARY KEY,
    title        text    NOT NULL,
    release_year integer,              -- NULL when the dataset has no release date
    genres       text[]  NOT NULL
);

-- One row per rating (100,000 rows): "this user rated this film".
--
-- Holds ALL ratings, including the ones held out for validation/test during training.
-- The API reads this table to exclude films a user has already rated, and serving must
-- never recommend something already seen, whichever split the rating fell into.
CREATE TABLE interactions (
    -- REFERENCES = foreign key: a rating for a user or film that does not exist is rejected.
    user_id  integer     NOT NULL REFERENCES users (user_id),
    item_id  integer     NOT NULL REFERENCES items (item_id),
    rating   smallint    NOT NULL CHECK (rating BETWEEN 1 AND 5),
    -- u.data stores Unix seconds; training/index.py converts them with to_timestamp().
    rated_at timestamptz NOT NULL,
    -- Composite primary key: a user can rate a film only once. Postgres builds an index
    -- on it with user_id first, which is exactly what the API's per-request query needs:
    --     SELECT item_id FROM interactions WHERE user_id = %s
    -- so no separate index is required.
    PRIMARY KEY (user_id, item_id)
);
