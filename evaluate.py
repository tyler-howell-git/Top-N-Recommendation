import numpy as np


def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k


def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant) if len(relevant) > 0 else 0


def hit_rate_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    return int(len(set(recommended_k) & set(relevant)) > 0)


def evaluate_recommender(train_df, test_df, recommend_func, model_name, k=10):
    """
    Generic Top-K evaluator for leave-one-out testing.
    """

    precisions = []
    recalls = []
    hit_rates = []

    for _, row in test_df.iterrows():
        user_id = row["userId"]
        test_movie = row["movieId"]

        recommendations = recommend_func(user_id, k)
        relevant = [test_movie]

        precisions.append(precision_at_k(recommendations, relevant, k))
        recalls.append(recall_at_k(recommendations, relevant, k))
        hit_rates.append(hit_rate_at_k(recommendations, relevant, k))

    return {
        "Model": model_name,
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}": np.mean(recalls),
        f"HitRate@{k}": np.mean(hit_rates)
    }