import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

print("Loading data...")

# Load datasets
ratings = pd.read_csv("Books_rating.csv")
ratings = ratings[:20_000]
# Keep required columns
ratings = ratings[["User_id", "Title", "review/score"]]
ratings.columns = ["user_id", "book_id", "rating"]

# Clean data
ratings = ratings.dropna()
ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
ratings = ratings.dropna()

print("Original shape:", ratings.shape)

# -------------------------------------------------
# FILTER ACTIVE USERS & BOOKS
# -------------------------------------------------
user_counts = ratings["user_id"].value_counts()
book_counts = ratings["book_id"].value_counts()

ratings = ratings[
    ratings["user_id"].isin(user_counts[user_counts >= 5].index)
]

ratings = ratings[
    ratings["book_id"].isin(book_counts[book_counts >= 5].index)
]

print("Filtered shape:", ratings.shape)

# -------------------------------------------------
# ENCODE USERS & BOOKS
# -------------------------------------------------
u_map = {u: i for i, u in enumerate(ratings["user_id"].unique())}
b_map = {b: i for i, b in enumerate(ratings["book_id"].unique())}

ratings["u"] = ratings["user_id"].map(u_map)
ratings["b"] = ratings["book_id"].map(b_map)

n_users = len(u_map)
n_books = len(b_map)

print("Users:", n_users)
print("Books:", n_books)

# -------------------------------------------------
# TRAIN TEST SPLIT
# -------------------------------------------------
print("Splitting data...")

test = ratings.groupby("user_id").sample(frac=0.2, random_state=42)
train = ratings.drop(test.index)

# -------------------------------------------------
# BUILD USER-ITEM MATRIX
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
    return np.sqrt(np.mean((np.array(actual) - np.array(pred)) ** 2))

# -------------------------------------------------
# USER BASED COLLABORATIVE FILTERING
# -------------------------------------------------
def train_user_cf(train, test, k_values):

    print("\nRunning User-Based CF...")

    # User-item matrix
    user_matrix = build_matrix(train)

    # Store user ratings
    user_ratings = train.groupby("u").apply(
        lambda x: dict(zip(x["b"], x["rating"]))
    ).to_dict()

    best_k = None
    best_rmse = float("inf")

    for k in k_values:

        print(f"\nTesting K = {k}")

        # KNN on users
        knn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=k
        )

        knn.fit(user_matrix)

        preds = []
        actuals = []

        # Predict every test rating
        for _, row in test.iterrows():

            u = int(row["u"])
            b = int(row["b"])
            actual = row["rating"]

            # Find similar users
            distances, indices = knn.kneighbors(user_matrix[u])

            neighbors = indices.flatten()[1:]
            sims = 1 - distances.flatten()[1:]

            num = 0
            den = 0

            # Weighted average from neighbors
            for neighbor, sim in zip(neighbors, sims):

                if sim <= 0:
                    continue

                # Check if neighbor rated this book
                if b in user_ratings.get(neighbor, {}):

                    r = user_ratings[neighbor][b]

                    num += sim * r
                    den += sim

            # Prediction
            if den > 0:
                pred = num / den
            else:
                # fallback = user mean
                pred = train[train["u"] == u]["rating"].mean()

            preds.append(np.clip(pred, 1, 5))
            actuals.append(actual)

        score = rmse(actuals, preds)

        print(f"RMSE for K={k}: {round(score, 4)}")

        if score < best_rmse:
            best_rmse = score
            best_k = k

    return best_k, best_rmse

# -------------------------------------------------
# RUN
# -------------------------------------------------
k_values = [5, 10, 15, 20]

best_k, best_rmse = train_user_cf(train, test, k_values)

print("\nBEST RESULT")
print("Best K:", best_k)
print("Best RMSE:", round(best_rmse, 4))

# -------------------------------------------------
# SAVE RESULT
# -------------------------------------------------
with open("output.txt", "w") as f:
    f.write("RMSE: " + str(round(best_rmse, 4)))