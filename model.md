# The two-tower retrieval model

**Who this is for.** You have built a content-based (CBF) retriever on MovieLens 100K in a
notebook and reached 55–60% Recall@200. You know the dataset, cosine similarity, Recall@K,
and what a user profile vector is. This document does not re-explain those. It explains
what is *different* here: vectors that are learned instead of constructed, the loss that
learns them, where negatives come from, and the evaluation discipline around it.

I assume your CBF looks roughly like this: an item vector built from features (genres,
maybe year or title terms), a user vector aggregated from the items they liked, cosine to
rank. If yours differs, section 1 still maps over.

**Where this model ends up** (temporal split, one held-out film per user, full catalogue):

| | Recall@10 | Recall@200 |
|---|---|---|
| Two-tower, validation | 0.191 | 0.792 |
| Two-tower, test | 0.126 | 0.722 |
| Most-popular baseline, validation | 0.072 | 0.521 |
| Most-popular baseline, test | 0.068 | 0.464 |

Do not put your 55–60% next to these yet. Section 2 explains why, and how to score your
CBF under the same protocol in five lines.

---

## 1. From your CBF to this model

Same retrieval geometry, different source for the vectors.

| | Your CBF | This model |
|---|---|---|
| Item vector | Constructed from features | **Learned**: ID embedding + features through an MLP |
| User vector | Aggregate of liked items' vectors | **Learned**: ID embedding + demographics through an MLP |
| User and item live in the same space because | The user vector is *made of* item vectors | The loss forces it (section 4) |
| Score | Cosine | Cosine |
| Training | None (or a fitted TF-IDF) | Gradient descent on a contrastive loss |
| Signal used | Content only | Content **and** co-rating behaviour |
| Needs negatives | No | **Yes** (section 5; this is the biggest new idea) |
| Serving | Nearest neighbours of the user vector | Identical: nearest neighbours in Qdrant |

### Why CBF plateaus on this dataset

The item features in ML-100K are 19 genre flags and a release date. Many films share the
same genre vector, so a content model literally cannot tell them apart: every "Drama" from
the 1990s is the same point. Whatever separates *Schindler's List* from a forgotten drama
is not in the features. It is in **who rated them**.

An ID embedding gives each film (and each user) its own free vector, trained so that users
end up near the films they liked. That is the collaborative signal. Two films with identical
genres get different vectors because different people liked them.

### Why keep the features at all ("hybrid")

A free vector per ID is only as good as the amount of data behind it. In the training split,
259 of the 1,682 films have **no** positive rating at all, so their ID embedding never gets
a useful gradient. For those, the genre and year inputs are all the tower has, which is
exactly your CBF, but with the feature weighting learned instead of hand-set.

So think of this model as: **your CBF, plus a per-ID correction learned from behaviour, with
the model deciding how much to trust each.**

---

## 2. Comparing numbers honestly

Recall@200 depends heavily on the protocol. Four things change it by tens of points:

| Protocol choice | Here |
|---|---|
| How the held-out item is chosen | The user's **latest** liked film by timestamp (test) and the one before (validation). Not random. |
| What "relevant" means | Rating >= 4. One relevant item per user, so Recall@K is a hit rate. |
| Candidate set | **All 1,682 films**, minus what the user rated in earlier parts of the split. No sampled candidates. |
| What training may see | Only ratings made *before* the validation film. |

Two reference points for how much the protocol matters:

- **Popularity alone scores 0.52 / 0.46 Recall@200 here.** On ML-100K, recommending the same
  200 most-liked films to everyone is a strong baseline. If your 55–60% was measured on a
  comparable protocol, it is close to popularity, and that is the first thing to check.
- **The same two-tower model scored 0.896 on a random hold-out and 0.765 on the temporal
  one** (section 6). Thirteen points from the split alone.

### Score your CBF under this protocol

`evaluate_scores` in `training/evaluate.py` takes any `(n_users, n_items)` score matrix. The
popularity baseline goes through the same function.

```python
import pandas as pd, torch
from config import get_settings
from training.evaluate import evaluate_scores

data = get_settings().prepared_data_dir          # training/data/, written by training.prepare
train, val = pd.read_csv(data / "train.csv"), pd.read_csv(data / "val.csv")

# Build your user profiles from `train` ONLY, then:
scores = torch.tensor(my_cbf_scores, dtype=torch.float32)   # (943, 1682)
print(evaluate_scores(scores, val, seen=train))             # recall/ndcg/precision @ 10, 20, 200
```

Rows and columns are the zero-based `user_idx` / `item_idx` columns of those CSVs (raw
MovieLens ID minus 1). If your profiles were built from ratings that are in `val.csv` or
`test.csv`, the number is leaked and will look better than it is.

---

## 3. Architecture

Code: `Tower` and `TwoTower` in `training/models/two_tower.py`.

### Inputs

| Tower | Piece | Width | Enters as |
|---|---|---|---|
| User | User ID | 64 | embedding, 943 rows |
| User | Gender (1), standard-scaled age (1), occupation (21), zip region (11) | 34 | raw numbers: 0/1 columns plus the scaled age |
| User | **Total** | **98** | |
| Item | Film ID | 64 | embedding, 1,682 rows |
| Item | Release-year bucket | 8 | embedding, 9 rows |
| Item | Genres (multi-hot) | 19 | raw 0/1 columns |
| Item | **Total** | **91** | |

Note what the user tower does **not** get: the user's rating history. Unlike your CBF profile,
the user vector here is not computed from liked items at inference time. The user's taste is
stored in their ID embedding, learned during training. Consequence: a user who was not in
training has no useful vector (section 13).

### Layers

```text
User:   98 -> Linear 256 -> ReLU -> Dropout 0.1 -> Linear 256 -> ReLU -> Dropout 0.1 -> Linear 64 -> L2-normalise
Item:   91 -> Linear 256 -> ReLU -> Dropout 0.1 -> Linear 256 -> ReLU -> Dropout 0.1 -> Linear 64 -> L2-normalise
```

About 381K parameters; 168K of them are the ID embedding tables.

The literal numbers are not in the model file because the MLP is built in a loop:

```python
width = sum(sizes) + dense_dim           # 98 for users, 91 for items
for _ in range(num_hidden_layers):       # --num-hidden-layers, default 2
    layers += [nn.Linear(width, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]   # --hidden-dim, default 256
    width = hidden_dim
layers.append(nn.Linear(width, embedding_dim))                                  # --embedding-dim, default 64
```

### Choices that are not obvious

| Choice | Reason |
|---|---|
| No activation on the output layer | ReLU before normalising would put every vector in the positive orthant: no cosine below 0, so a negative can never be pushed further than "orthogonal". |
| L2-normalised output | Score = cosine, same as Qdrant's search. Also removes vector length as a shortcut: without it the model lowers the loss by making popular films' vectors long, which raises their score for everyone. |
| Feature tables stored in the model as buffers | The checkpoint carries the exact features it was trained on; encoding needs only an index. |
| Two separate towers, no cross-features | A user and an item never pass through a layer together, so all item vectors can be precomputed. That is the constraint that makes this a retrieval model. |

---

## 4. The objective: listwise InfoNCE

Code: `info_nce` in `training/models/two_tower.py`.

Your CBF has no objective: the vectors are whatever the features say. Here the vectors are
free parameters, so something has to say what "good" means.

One training example is a list built around one liked film:

```text
(user, positive, negative_1, ..., negative_K)
```

The model scores every item in the list against the user and is trained as a classifier
that must pick the positive:

```text
loss = -log( exp(s_pos / t) / sum_j exp(s_j / t) ),     s = cosine(user, item)
```

Minimising this pulls the user vector toward the positive and pushes it away from the
negatives, in proportion to how wrong the current scores are. Because both towers receive
gradients from the same loss, the two spaces are aligned by construction.

**Temperature `t` = 0.1.** Cosines live in [-1, 1]. With 20 negatives and no temperature,
even a perfect arrangement gives the positive a probability of about 0.27, so the loss has
a floor of 1.31 and weak gradients. Dividing by 0.1 widens the logits to [-10, 10] and
removes that ceiling. Lower `t` concentrates the gradient on the hardest negatives;
too low and training chases noise (0.05 was measurably worse here).

**The absolute loss value is not interpretable across settings.** It depends on the number
of candidates (chance level is `log(list size)`) and on the logQ correction below. Compare
validation metrics, never losses.

**Optimizer.** AdamW, learning rate 0.003, weight decay 0.0001. InfoNCE is the loss; AdamW
is what applies its gradients. No learning-rate schedule, no batch norm, no L1.

---

## 5. Negative sampling

Code: `NegativeSampler` in `training/sampling.py`; in-batch logic in `training/train.py`.

This is the part with no CBF equivalent, and the part that most determines the result.
The data says which films a user liked. It never says which films they would *not* like.
The loss needs "wrong answers", so we have to manufacture them.

### The options

| Source | What it is | The catch |
|---|---|---|
| **Uniform** | Random films the user has no training rating for | "Unrated" is not "disliked". But a random film is rarely one the user loves: 0.12% of draws hit the user's own held-out film. |
| **Explicit low ratings** ("hard") | Films the user rated 1–2 (or <= 3) | The user *chose to watch* them, so they resemble the user's taste. Pushing them away teaches "avoid things like what this user picks". |
| **In-batch** | The positives of the other rows in the same batch | Free (already encoded). But drawn in proportion to popularity, and full of false negatives. |

### What is used

**20 uniform negatives per positive, plus in-batch negatives with two corrections.**
Explicit negatives are off.

### What "uniform" means exactly

For one training row (a user and one film they rated >= 4), `NegativeSampler.sample()`:

1. draws 20 film indices at random from all 1,682 films, every film equally likely;
2. rejects any draw the user has a training rating for (at any score) and redraws it;
3. repeats until all 20 are films the user has not rated in training.

"Uniform" is about the probability: an obscure film is as likely to be drawn as *Star Wars*.
That is the opposite of in-batch negatives, which arrive in proportion to popularity.
The 20 are redrawn every epoch, so over ~20 epochs a positive meets roughly 400 different
films. Validation and test films are deliberately *not* excluded from the pool: excluding
them would use held-out labels to build training examples.

### How many negatives one row really has

The sampler stores only `(user, pos, 20 negatives)`. The in-batch negatives are not stored;
they are added inside the loss by reusing the other rows' positives from the same batch.

| Part | Count per row |
|---|---|
| Positive | 1 |
| Uniform negatives | 20, always |
| In-batch candidates | 255 (the other rows of a batch of 256) |
| Removed by the false-negative mask | about 85 (33% on average) |
| In-batch negatives actually used | about 170, varies per user |
| **Effective negatives** | **about 190** (upper bound 275) |

So the positive-to-negative ratio is at most 1:275 and about 1:190 in practice. It is
lower for heavy raters (more of their batch-mates' films are masked) and in the last batch
of an epoch (243 rows). The negatives are also not equally weighted: the logQ correction
below discounts popular in-batch films.

### Why raw in-batch negatives are a trap

Measured on this training split:

| Fact | Value |
|---|---|
| A batch-mate's positive is a film this user also rated | 33% (23% rated it >= 4) |
| Share of all positives held by the top 10% of films | 52% |
| Validation targets that are among the top 10% most popular films | 40% |

So without care, a quarter of the "negatives" are films the user likes, and the films most
often punished are the popular ones, which are exactly the ones most likely to be the right
answer.

**Correction 1: false-negative mask.** For user *b* and batch-mate positive *j*, if *b* has
any training rating for *j*, that logit is set to -inf and drops out of *b*'s softmax.
It reuses the sampler's `seen[user, item]` matrix.

**Correction 2: logQ.** After masking, item *j* still appears as a negative in proportion to
its popularity. Sampled-softmax theory says to subtract from each logit the log of how often
the sampler is expected to produce that item:

```text
expected_count(i) = K / n_items  +  (B - 1) * q_i        q_i = item i's share of training positives
logit_i           = s_i / t  -  log(expected_count(i))
```

Over-sampled items are discounted by exactly their over-sampling, so the learned cosine
estimates preference, not popularity. With no in-batch term the correction is a constant
and changes nothing, so one formula covers every variant.

Both corrections use training data only.

### Evidence

First comparison (one hidden layer of 128, three seeds, validation Recall@200):

| Variant | Recall@200 |
|---|---|
| Uniform K=20 | 0.736 |
| Uniform K=20 + explicit <= 2 | 0.693 |
| Uniform K=20 + explicit <= 3 | 0.658 |
| In-batch only (masked + logQ) | 0.711 |
| Uniform K=20 + in-batch (masked + logQ) | 0.751 |

Repeated later with the final tower and learning rate (section 10): explicit negatives
without in-batch are still clearly worst (0.733 against 0.795), in-batch alone is weak
(0.764), and the remaining variants are tied within seed noise.

**Takeaways.** (1) For retrieval, a user's low ratings are bad negatives; the more of them
you include (<= 3 instead of <= 2), the worse it gets. (2) In-batch alone under-covers the
rare films, because they almost never appear in a batch. (3) Uniform negatives carry most
of the value; in-batch on top is a tie-level addition on a catalogue this small. Its real
advantage is cost on catalogues too large to sample well.

---

## 6. The split, and a leak worth remembering

Code: `split` in `training/prepare.py`.

Per user, ratings ordered by timestamp (ties broken by item ID):

```text
[ ............ training ............ ] [validation positive] [test positive]
                                          second-latest          latest
```

Low ratings made after the validation positive are dropped, so nothing in training is later
than validation. Users with fewer than three positives are train-only (one user).

| Part | Rows |
|---|---|
| Training | 94,581 (53,491 of them positives) |
| Validation | 942 |
| Test | 942 |
| Dropped | 3,535 |

### The leak

Midway through, the split was switched to "last positive in the row order of `u.data`".
That file is shuffled, so this is a random hold-out: training then contains ratings the
user made *after* the held-out film.

| Same model, same settings | NDCG@10 | Recall@200 | Popularity Recall@200 |
|---|---|---|---|
| Row-order (random) hold-out | 0.240 | 0.896 | 0.696 |
| Temporal hold-out | 0.099 | 0.765 | 0.521 |

Nothing about the model changed. The task became "fill a gap in a history you have seen
both sides of" instead of "predict forward". Since the API's job is "what next", the
temporal split is the honest one and is what everything below uses.

If your notebook used `train_test_split` or a random leave-one-out, this is the most likely
reason its number is not comparable.

Caveat: ML-100K timestamps are *rating* times, and many users rated dozens of films in one
sitting. "Latest" is a weak notion of "next", which adds noise.

---

## 7. Evaluation protocol

Code: `training/evaluate.py`.

- **Candidates:** all items, minus the user's interactions in earlier parts of the split
  (train for validation; train + validation for test).
- **Relevant:** the single held-out positive. Recall@K is a hit rate; Precision@K is
  Recall@K / K.
- **Ties:** ranked pessimistically (the target is placed after equal-scored items), so a
  constant score matrix cannot look good.
- **Diverged model:** NaN scores raise an error instead of silently ranking everything first.
- **Reported:** Recall, NDCG, Precision at 10, 20, 200, averaged over users.

**Selection metric: validation Recall@200** (`SELECTION_METRIC` in `training/train.py`).
It picks the checkpoint within a run, drives early stopping, and decides hyperparameters.
This is the retrieval stage: the question is whether the right film reaches the ranker, not
where it sits in the top 10.

**Noise.** With 942 validation users, the same configuration varies by about ±0.005
Recall@200 across random seeds. Every comparison here uses three seeds, and differences
below that spread are treated as ties. A single run cannot rank two close configurations.

---

## 8. The training loop

Code: `main` in `training/train.py`. Run: `python -m training.train`.

Per epoch:

1. `sampler.sample()` draws **fresh** uniform negatives for all 53,491 positives. Over
   training each positive is contrasted with many different negatives.
2. Shuffle; batches of 256.
3. Per batch: encode users `(B, 64)` and item lists `(B, 1+K, 64)`; append the `B`
   in-batch positives as extra candidates; apply mask and logQ; InfoNCE; AdamW step.
4. Score all users against all items; compute validation metrics; log to MLflow.
5. If validation Recall@200 improved, snapshot the weights. Five epochs without improvement
   ends the run.

Then the best snapshot is exported to `training/artifacts/two_tower/` (`model.pt`,
`meta.json`) and uploaded to MLflow together with every setting, the split hash and the
feature hash.

### Defaults

| Setting | Flag | Value |
|---|---|---|
| Output / ID embedding size | `--embedding-dim` | 64 |
| Side-feature embedding size | `--feature-dim` | 8 |
| Hidden width | `--hidden-dim` | 256 |
| Hidden layers per tower | `--num-hidden-layers` | 2 |
| Dropout | `--dropout` | 0.1 |
| Uniform negatives per positive | `--num-negatives` | 20 |
| In-batch negatives | `--in-batch-negatives` / `--no-in-batch-negatives` | on |
| Share of explicit negatives | `--hard-negative-ratio` | 0 |
| Explicit-negative threshold | `--hard-negative-max-rating` | 2 |
| Temperature | `--temperature` | 0.1 |
| Learning rate | `--lr` | 0.003 |
| Weight decay | `--weight-decay` | 0.0001 |
| Batch size | `--batch-size` | 256 |
| Epoch cap | `--epochs` | 100 (runs end near 20) |
| Patience | `--patience` | 5 |
| Device | `--device` | `mps` when available, else `cpu` |

---

## 9. Does the MLP matter?

Compared on the temporal split, three seeds, before the hyperparameter search
(learning rate 0.001, batch 1024):

| Hidden layers | Recall@200 | NDCG@10 |
|---|---|---|
| none (one linear projection) | 0.741 | 0.086 |
| 128 | 0.740 | 0.087 |
| 256 | 0.755 | 0.094 |
| 128 -> 128 | 0.757 | 0.095 |
| **256 -> 256** | **0.771** | 0.096 |
| 128 -> 256 -> 128 -> 128 | 0.747 | 0.091 |

Reading:

- A plain linear projection is within noise of a 128-unit hidden layer. Most of the model
  is the ID embeddings; the MLP is a mixing layer for ID and a handful of feature bits.
- 256 -> 256 has the best Recall@200 and is the default. The gain over the smallest tower
  is about three points.
- A deeper, narrower "funnel" was worse at the same parameter count. With few features to
  cross, depth adds optimisation difficulty, not capacity that is used.

If you come from CBF expecting the network to be where the power is: on a dataset this
size it is not. The embeddings and the negative sampling are.

---

## 10. Hyperparameter search

*This search was run with the earlier 7-bucket age one-hot (user input 104 wide). The
scaled-age model was not re-searched.*

**Method.** One setting at a time, three seeds each, temporal split, MPS. A value replaced
the current one only if it raised mean validation Recall@200 by at least 0.005. About 130
runs. Each row compares against the value in force at that stage.

| Setting | Tried | Best (Recall@200) | In force (Recall@200) | Decision |
|---|---|---|---|---|
| Learning rate | 0.0003, 0.001, 0.003 | 0.003 (0.7948) | 0.001 (0.7820) | **-> 0.003** |
| Patience | 5, 10, 20 | 10 (0.7969) | 5 (0.7948) | keep 5 |
| Temperature | 0.05, 0.07, 0.1, 0.15, 0.2 | 0.1 = 0.15 (0.7948) | 0.1 | keep 0.1 |
| Batch size | 256, 512, 1024, 2048 | 256 (0.8001) | 1024 (0.7948) | **-> 256** |
| Dropout | 0, 0.1, 0.2, 0.3 | 0.3 (0.8015) | 0.1 (0.8001) | keep 0.1 |
| Weight decay | 0, 0.0001, 0.001, 0.01 | 0.01 (0.8043) | 0.0001 (0.8001) | keep 0.0001 |
| Embedding size | 32, 64, 128 | 64 (0.8001) | 64 | keep 64 |
| Side-feature size | 4, 8, 16 | 8 (0.8001) | 8 | keep 8 |
| Hidden width | 128, 256, 512 | 128 (0.8022) | 256 (0.8001) | keep 256 |
| Hidden layers | 1, 2, 3 | 3 (0.8015) | 2 (0.8001) | keep 2 |

Negative sampling inside the same search (learning rate 0.003, batch 1024):

| Variant | Recall@200 | NDCG@10 |
|---|---|---|
| **Uniform K=20 + in-batch (default)** | 0.7948 | 0.0929 |
| Uniform K=50 + in-batch | 0.7930 | 0.0934 |
| Uniform K=100 + in-batch | 0.7941 | 0.0959 |
| Uniform K=20 only | 0.7987 | 0.0904 |
| Uniform K=100 only | 0.7962 | 0.0931 |
| Uniform + in-batch + explicit <= 2 | 0.7926 | 0.0945 |
| In-batch only | 0.7640 | 0.0903 |
| Uniform + explicit <= 2, no in-batch | 0.7325 | 0.0692 |

**What it shows.**

- The model is close to flat. Only learning rate and batch size cleared the noise threshold.
- **Winner's-curse check.** Choosing the best of many noisy runs overstates the gain. Tuned
  and original settings were both retrained on three *new* seeds: 0.7951 ± 0.0015 against
  0.7895 ± 0.0039. Real, but about half a point.
- Switching checkpoint selection from NDCG@10 to Recall@200 was worth about a point on its
  own (0.771 -> 0.782 for the same settings), as much as the learning-rate change: the
  metric you stop on is the metric you get.

---

## 11. Final results

Default settings, seed 0, age as one standard-scaled column, MLflow run
`f3bcb2c38026448c80231a003703a73c`.

| Split | Model | Recall@10 | NDCG@10 | Recall@20 | Recall@200 | NDCG@200 |
|---|---|---|---|---|---|---|
| Validation | Two-tower | 0.1911 | 0.0978 | 0.3025 | 0.7919 | 0.2074 |
| Validation | Popularity | 0.0722 | 0.0385 | 0.1253 | 0.5212 | 0.1164 |
| Test | Two-tower | 0.1263 | 0.0650 | 0.2038 | 0.7219 | 0.1708 |
| Test | Popularity | 0.0679 | 0.0348 | 0.1083 | 0.4639 | 0.1017 |

Caveats:

- **Validation is optimistic.** The checkpoint, the stopping epoch and every hyperparameter
  were chosen on it. Quote the test row.
- **Test has been scored four times** in this work, for four successive models. No choice
  was made from it, but it is not a perfectly untouched estimate.
- **Test is lower than validation** by about seven points. The test film is one step further
  into the future, and 17 test targets (against 7 validation targets) are films with no
  training positive at all.

---

## 12. From trained model to served recommendations

```text
python -m training.prepare    ml-100k/ -> training/data/        split, features, hashes
python -m training.train      -> training/artifacts/two_tower/  model.pt, meta.json, MLflow run
python -m training.evaluate   -> metrics (refuses a model trained on a different split)
python -m training.index      -> 943 user vectors + 1,682 item vectors into Qdrant; metadata into Postgres
uvicorn api.main:app          -> per request: fetch user vector, nearest items excluding rated ones, titles from Postgres
```

The API never imports PyTorch and never runs the model. This is your notebook's
"nearest neighbours of the user vector", moved into a vector database.

- Re-run `training.index` after every retrain. `/health` reports the MLflow run id being served.
- Raw MovieLens IDs are used in Postgres, Qdrant and the API. Zero-based indices exist only
  inside the model; `meta.json` holds the lists that translate.

---

## 13. What was built and changed

The pipeline code existed before this work but had never been trained.

| # | Change | Where | Outcome |
|---|---|---|---|
| 1 | Negative-sampling comparison | experiments | Explicit negatives rejected; uniform + in-batch chosen |
| 2 | In-batch negatives with false-negative mask and logQ correction | `info_nce`, `train.py` | Section 5 |
| 3 | K=200 added to metrics | `evaluate.py` | Section 7 |
| 4 | Configurable depth; tower comparison | `two_tower.py`, `train.py` | 256 -> 256 default |
| 5 | Temporal split restored after a row-order split leaked | `prepare.py` | Section 6 |
| 6 | Training on MPS; artifact still saved for CPU | `train.py`, `evaluate.py` | `--device` |
| 7 | 128 -> 256 -> 128 -> 128 tested | experiment | Worse; code removed |
| 8 | Checkpoint selection and early stopping on Recall@200 | `train.py` | Was NDCG@10 |
| 9 | Hyperparameter search | experiments | Learning rate 0.003, batch 256 |
| 10 | Tests for split, sampler, loss, metrics, save/load | `tests/test_training.py` | passing |
| 11 | Age: 7-bucket one-hot replaced by one `StandardScaler` column | `prepare.py`, `train.py` | Tie: Recall@200 0.7983 ± 0.0057 against 0.8001 ± 0.0044 (seeds 0/1/2); user input 104 -> 98 |

---

## 14. Limitations and what to try next

| Limitation | Next step |
|---|---|
| **No features-only ablation.** How much of the result is the ID embeddings versus the features has not been measured. | Train with the ID columns removed. That is a *learned CBF* and the direct comparison with your notebook. |
| **User history is not an input.** New users, and new ratings from existing users, need a retrain. | Feed the user tower an aggregate of the embeddings of recently liked films, as your CBF profile does. |
| **Item features are thin** (genres, year). | Add your CBF item vectors (title terms or similar) as item-tower inputs. |
| Hyperparameters were searched one at a time. | Joint search; test the three near-misses together (weight decay 0.01, dropout 0.3, patience 10). |
| No learning-rate schedule; validation is noisy epoch to epoch. | Add decay and compare. |
| 942 validation users: differences under 0.005 are unreadable. | More seeds, or MovieLens 1M. |
| The search script is not in the repository. | Add it as `training/tune.py`. |
| Offline metrics only. | Online test. |

---

## 15. Terms that may be new

| Term | Meaning here |
|---|---|
| **Two-tower** | Separate user and item encoders whose outputs are compared by dot product, so item vectors can be precomputed |
| **Collaborative signal** | Information from who rated what, as opposed to item or user attributes |
| **InfoNCE** | Softmax cross-entropy over one positive and a set of negatives |
| **Listwise** | The loss is computed over a whole candidate list per user, not per pair |
| **In-batch negatives** | Other rows' positives reused as negatives |
| **False negative** | A sampled negative the user actually likes |
| **logQ correction** | Subtracting the log sampling frequency from each logit to undo sampling bias |
| **Temperature** | Divisor on the logits; controls how sharply the loss focuses on hard negatives |
| **Winner's curse** | The best of many noisy runs looks better than it is; re-test on fresh seeds |
| **Leakage** | Held-out or future information reaching training |
