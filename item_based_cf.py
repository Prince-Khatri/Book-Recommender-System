import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

print("Loading data...")

ratings = pd.read_csv("Books_rating.csv")
ratings = ratings[:20_000]
ratings = ratings[["User_id", "Title", "review/score"]]
ratings.columns = ["user_id", "book_id", "rating"]

ratings = ratings.dropna()
ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
ratings = ratings.dropna()

# ratings = ratings.sample(200000, random_state=42)

print("Data shape:", ratings.shape)

print("Preprocessing...")

# Filter active users/books
user_counts = ratings["user_id"].value_counts()
book_counts = ratings["book_id"].value_counts()

ratings = ratings[ratings["user_id"].isin(user_counts[user_counts >= 5].index)]
ratings = ratings[ratings["book_id"].isin(book_counts[book_counts >= 5].index)]

# Encode users/items
u_map = {u:i for i,u in enumerate(ratings["user_id"].unique())}
b_map = {b:i for i,b in enumerate(ratings["book_id"].unique())}

ratings["u"] = ratings["user_id"].map(u_map)
ratings["b"] = ratings["book_id"].map(b_map)

n_users = len(u_map)
n_books = len(b_map)

print("Users:", n_users, "| Books:", n_books)

# ─────────────────────────────
# 3. TRAIN-TEST SPLIT
# ─────────────────────────────
print("Splitting data...")

test = ratings.groupby("user_id").sample(frac=0.2, random_state=42)
train = ratings.drop(test.index)

# ─────────────────────────────
# 4. BUILD MATRIX
# ─────────────────────────────
def build_matrix(df):
    return csr_matrix((df["rating"], (df["u"], df["b"])),
                      shape=(n_users, n_books))

# ─────────────────────────────
# 5. RMSE FUNCTION
# ─────────────────────────────
def rmse(a, p):
    return np.sqrt(np.mean((np.array(a) - np.array(p))**2))

# ─────────────────────────────
# 6. KNN WITH AUTO K
# ─────────────────────────────
def train_knn_with_k(train, test, k_values):
    print("\nRunning KNN with different K values...")

    item_matrix = build_matrix(train).T
    user_rated = train.groupby("u")["b"].apply(list).to_dict()

    best_k = None
    best_rmse = float("inf")

    for k in k_values:
        print(f"\n🔍 Testing K = {k}")

        knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k)
        knn.fit(item_matrix)

        preds, actuals = [], []

        for _, row in test.iterrows():
            u, b, actual = int(row["u"]), int(row["b"]), row["rating"]

            distances, indices = knn.kneighbors(item_matrix[b])
            neighbors = indices.flatten()[1:]
            sims = 1 - distances.flatten()[1:]

            rated_items = set(user_rated.get(u, []))

            num, den = 0, 0
            for n, s in zip(neighbors, sims):
                if n in rated_items and s > 0:
                    r = train[(train["u"] == u) & (train["b"] == n)]["rating"].mean()
                    num += s * r
                    den += s

            if den > 0:
                pred = num / den
            else:
                pred = train[train["u"] == u]["rating"].mean()

            preds.append(np.clip(pred, 1, 5))
            actuals.append(actual)

        score = rmse(actuals, preds)
        print(f"RMSE for K={k}: {round(score,4)}")

        if score < best_rmse:
            best_rmse = score
            best_k = k

    return best_k, best_rmse

# ─────────────────────────────
# 7. RUN
# ─────────────────────────────
k_values = [5, 10, 15, 20]
# k_values = [10]

best_k, best_rmse = train_knn_with_k(train, test, k_values)

print("\n🏆 BEST RESULT")
print("Best K:", best_k)
print("Best RMSE:", round(best_rmse, 4))
