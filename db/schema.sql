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
    release_year integer,
    genres       text[]  NOT NULL
);

CREATE TABLE interactions (
    user_id  integer     NOT NULL REFERENCES users (user_id),
    item_id  integer     NOT NULL REFERENCES items (item_id),
    rating   smallint    NOT NULL CHECK (rating BETWEEN 1 AND 5),
    rated_at timestamptz NOT NULL,
    PRIMARY KEY (user_id, item_id)
);
