def build_popularity_baseline(train_df, min_ratings=5):
    """Movies are ranked by average rating, with a minimum rating-count
    threshold to reduce noisy high averages from rarely rated movies.
    """

    movie_stats = (
        train_df
        .groupby("movieId")
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index())

    movie_stats = movie_stats[movie_stats["rating_count"] >= min_ratings]

    movie_stats = movie_stats.sort_values(by=["avg_rating", "rating_count"], ascending=False)

    return movie_stats

def recommend_popular_movies(user_id, train_df, popularity_df, n=10):
    """
    Recommend the top-N globally popular movies that the user has not already rated.
    """

    seen_movies = set(train_df.loc[train_df["userId"] == user_id, "movieId"])

    recommendations = popularity_df[~popularity_df["movieId"].isin(seen_movies)].head(n)

    return recommendations["movieId"].tolist()


from sklearn.neighbors import NearestNeighbors
import pandas as pd

def build_user_item_matrix(train_df):
    """Build a user-item ratings matrix.
    Rows are users, columns are movies, values are ratings.
    Missing ratings are filled with 0 for kNN.
    """

    return train_df.pivot_table(
        index="userId",
        columns="movieId",
        values="rating").fillna(0)


def fit_item_knn(train_df, k_neighbors=20):
    """Fit item-item kNN model using cosine distance.
    Returns the user-item matrix and item-neighbor dictionary.
    """

    user_item_matrix = build_user_item_matrix(train_df)
    item_user_matrix = user_item_matrix.T

#inilaize kNN model w/ cosine distance and brute-force search
    knn_model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=k_neighbors + 1)

    knn_model.fit(item_user_matrix) #fit the model on the item-user matrix

    distances, indices = knn_model.kneighbors(item_user_matrix) #comput nn for all items
    item_ids = item_user_matrix.index.to_list()

    item_neighbors = {}

#construct dict that maps each item to list of (neighbor_id, similarity) tuples k nearest neighbors
    for item_pos, item_id in enumerate(item_ids):
        neighbors = []

        for distance, neighbor_pos in zip(distances[item_pos], indices[item_pos]):
            neighbor_id = item_ids[neighbor_pos]

            if neighbor_id == item_id:
                continue

            similarity = 1 - distance

            if similarity > 0:
                neighbors.append((neighbor_id, similarity))

        item_neighbors[item_id] = neighbors[:k_neighbors]

    return user_item_matrix, item_neighbors


def score_user_items_knn(user_id, user_item_matrix, item_neighbors):
    """
    Score unseen movies for a user using item-item kNN.
    """

    if user_id not in user_item_matrix.index: #no ratings, empty scores
        return pd.Series(dtype=float)

    user_ratings = user_item_matrix.loc[user_id]

    rated_items = user_ratings[user_ratings > 0]
    seen_items = set(rated_items.index)

    scores = {}
    similarity_sums = {}
    
#score unseen items based on ratings of seen & similarity
    for rated_item_id, rating in rated_items.items():
        neighbors = item_neighbors.get(rated_item_id, [])

        for neighbor_id, similarity in neighbors:
            if neighbor_id in seen_items:
                continue

            scores[neighbor_id] = scores.get(neighbor_id, 0) + similarity * rating
            similarity_sums[neighbor_id] = similarity_sums.get(neighbor_id, 0) + similarity

#normalize scores by total similarity
    normalized_scores = {
        item_id: scores[item_id] / similarity_sums[item_id]
        for item_id in scores
        if similarity_sums[item_id] > 0
    }

    return pd.Series(normalized_scores).sort_values(ascending=False)


def recommend_item_knn(user_id, user_item_matrix, item_neighbors, n=10):
    """
    Return Top-N movie IDs for one user using item-item kNN.
    """

    scores = score_user_items_knn(
        user_id=user_id,
        user_item_matrix=user_item_matrix,
        item_neighbors=item_neighbors)

    return scores.head(n).index.tolist()

from surprise import Dataset, Reader, SVD

#default hyperparams for SVD model
def fit_svd_model(
    train_df,
    n_factors=50, #default # latent factors
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42):
    """
    Fit an SVD matrix factorization model using Surprise.
    """

    reader = Reader(rating_scale=(1, 5))

    surprise_data = Dataset.load_from_df(
        train_df[["userId", "movieId", "rating"]], reader) #load data into surprise formatting

    trainset = surprise_data.build_full_trainset()

#initialize SVD w/ specified hyperparameters, fit to training data
    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state)

    model.fit(trainset)

    return model


def recommend_svd(user_id, train_df, movies_df, model, n=10):
    """
    Recommend Top-N unseen movies for a user using a trained SVD model.
    """

    all_movie_ids = set(movies_df["movieId"].unique())

    seen_movies = set(train_df.loc[train_df["userId"] == user_id, "movieId"])

    unseen_movies = list(all_movie_ids - seen_movies)

    predictions = [
        (movie_id, model.predict(user_id, movie_id).est)
        for movie_id in unseen_movies]

    predictions.sort(key=lambda x: x[1], reverse=True) #sort by predicted rating

    return [movie_id for movie_id, _ in predictions[:n]]