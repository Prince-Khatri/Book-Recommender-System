import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

print("Loading data...")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
ratings = pd.read_csv("Books_rating.csv")
ratings = ratings[:20_000]
ratings = ratings[["User_id", "Title", "review/score"]]
ratings.columns = ["user_id", "book_id", "rating"]

ratings = ratings.dropna()

ratings["rating"] = pd.to_numeric(
    ratings["rating"],
    errors="coerce"
)

ratings = ratings.dropna()

# -------------------------------------------------
# FILTER USERS & BOOKS
# -------------------------------------------------
user_counts = ratings["user_id"].value_counts()
book_counts = ratings["book_id"].value_counts()

ratings = ratings[
    ratings["user_id"].isin(
        user_counts[user_counts >= 5].index
    )
]

ratings = ratings[
    ratings["book_id"].isin(
        book_counts[book_counts >= 5].index
    )
]

# -------------------------------------------------
# ENCODE
# -------------------------------------------------
u_map = {
    u: i for i, u in enumerate(
        ratings["user_id"].unique()
    )
}

b_map = {
    b: i for i, b in enumerate(
        ratings["book_id"].unique()
    )
}

ratings["u"] = ratings["user_id"].map(u_map)
ratings["b"] = ratings["book_id"].map(b_map)

n_users = len(u_map)
n_books = len(b_map)

# -------------------------------------------------
# SPLIT
# -------------------------------------------------
test = ratings.groupby("user_id").sample(
    frac=0.2,
    random_state=42
)

train = ratings.drop(test.index)

# -------------------------------------------------
# MATRIX
# -------------------------------------------------
def build_matrix(df):
    return csr_matrix(
        (df["rating"], (df["u"], df["b"])),
        shape=(n_users, n_books)
    )

# -------------------------------------------------
# RMSE
# -------------------------------------------------
def rmse(actual, pred):
    return np.sqrt(
        np.mean(
            (np.array(actual) - np.array(pred)) ** 2
        )
    )

# -------------------------------------------------
# USER BASED PREDICTION
# -------------------------------------------------
def predict_user_cf(
    u, b,
    knn_user,
    user_matrix,
    user_ratings,
    train
):

    distances, indices = knn_user.kneighbors(
        user_matrix[u]
    )

    neighbors = indices.flatten()[1:]
    sims = 1 - distances.flatten()[1:]

    num = 0
    den = 0

    for neighbor, sim in zip(neighbors, sims):

        if sim <= 0:
            continue

        if b in user_ratings.get(neighbor, {}):

            r = user_ratings[neighbor][b]

            num += sim * r
            den += sim

    if den > 0:
        return num / den

    return train[train["u"] == u]["rating"].mean()

# -------------------------------------------------
# ITEM BASED PREDICTION
# -------------------------------------------------
def predict_item_cf(
    u, b,
    knn_item,
    item_matrix,
    user_rated,
    train
):

    distances, indices = knn_item.kneighbors(
        item_matrix[b]
    )

    neighbors = indices.flatten()[1:]
    sims = 1 - distances.flatten()[1:]

    rated_items = set(
        user_rated.get(u, [])
    )

    num = 0
    den = 0

    for neighbor, sim in zip(neighbors, sims):

        if sim <= 0:
            continue

        if neighbor in rated_items:

            r = train[
                (train["u"] == u)
                &
                (train["b"] == neighbor)
            ]["rating"].mean()

            num += sim * r
            den += sim

    if den > 0:
        return num / den

    return train[train["u"] == u]["rating"].mean()

# -------------------------------------------------
# ENSEMBLE TRAINING
# -------------------------------------------------
def train_ensemble(
    train,
    test,
    k=10,
    alpha=0.5
):

    print("\nTraining Ensemble...")

    user_matrix = build_matrix(train)
    item_matrix = user_matrix.T

    # USER KNN
    knn_user = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=k
    )

    knn_user.fit(user_matrix)

    # ITEM KNN
    knn_item = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=k
    )

    knn_item.fit(item_matrix)

    user_ratings = train.groupby("u").apply(
        lambda x: dict(zip(x["b"], x["rating"]))
    ).to_dict()

    user_rated = train.groupby("u")["b"].apply(list).to_dict()

    preds = []
    actuals = []

    for _, row in test.iterrows():

        u = int(row["u"])
        b = int(row["b"])
        actual = row["rating"]

        # USER CF
        pred_user = predict_user_cf(
            u, b,
            knn_user,
            user_matrix,
            user_ratings,
            train
        )

        # ITEM CF
        pred_item = predict_item_cf(
            u, b,
            knn_item,
            item_matrix,
            user_rated,
            train
        )

        # ENSEMBLE
        pred = (
            alpha * pred_user
            +
            (1 - alpha) * pred_item
        )

        preds.append(np.clip(pred, 1, 5))
        actuals.append(actual)

    score = rmse(actuals, preds)

    return score

# -------------------------------------------------
# RUN
# -------------------------------------------------
best_k = 0
best_score = float("inf")
k_arr = [5, 10, 15, 20]
for k in k_arr:
    print(k)
    score = train_ensemble(
        train,
        test,
        k=k,
        alpha=0.5
    )
    best_score = min(score, best_score)
    if best_score==score:
        best_k=k
    print(f"K:{k} score:{score}")


print(f"\nENSEMBLE with k={best_k} RMSE:", round(best_score, 4))