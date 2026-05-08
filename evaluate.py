'''______Generic evaluation against all data______'''
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


def eval_recommender(train_df, test_df, recommend_func, model_name, k=10):
    """Generic Top-K evaluator for leave-one-out testing."""

    precisions, recalls, hit_rates = [], [], []

#test interactions, gen recommendations, calc metrics
    for _, row in test_df.iterrows():
        user_id = row["userId"]
        test_movie = row["movieId"]

        recommendations = recommend_func(user_id, k)
        relevant = [test_movie]

        precisions.append(precision_at_k(recommendations, relevant, k))
        recalls.append(recall_at_k(recommendations, relevant, k))
        hit_rates.append(hit_rate_at_k(recommendations, relevant, k))

    return {"Model": model_name, f"Precision@{k}": np.mean(precisions), f"Recall@{k}": np.mean(recalls), f"HitRate@{k}": np.mean(hit_rates)}
    
    
'''______Sampled evaluation______'''

def get_sampled_candidates(user_id, test_movie, train_df, movies_df, n_neg=100, random_state=42):
    """create candidate set for sampled eval:
    - 1 held-out test movie (positive)
    - randomly sampled unseen (negative)"""

    rng = np.random.default_rng(random_state + int(user_id)) #seed

    all_movies = set(movies_df["movieId"].unique())
    seen_movies = set(train_df.loc[train_df["userId"] == user_id, "movieId"])

    negative_pool = list(all_movies - seen_movies - {test_movie}) #exclude seen movies / test movies
    n_sample = min(n_neg, len(negative_pool)) #case where user has fewer than n_neg unseen movies

    sampled_negatives = rng.choice(negative_pool, size=n_sample, replace=False).tolist()

    candidates = sampled_negatives + [test_movie] #combine positives / negatives
    rng.shuffle(candidates) #avoid position bias

    return candidates

def eval_recommender_sampled(train_df, test_df, movies_df, score_func,
    model_name, k=10, n_neg=100, random_state=42):
    
    """Evals recommender using leave-one-out sampled candidate ranking"""
    precisions, recalls, hit_rates = [], [], []

#test interactions, gen candidates, score/rank, calc metrics
    for _, row in test_df.iterrows():
        user_id, test_movie  = row["userId"], row["movieId"]

        candidates = get_sampled_candidates(user_id=user_id,
            test_movie=test_movie,
            train_df=train_df,
            movies_df=movies_df,
            n_neg=n_neg,
            random_state=random_state)

        ranked_items = score_func(user_id, candidates)
        relevant = [test_movie]

        precisions.append(precision_at_k(ranked_items, relevant, k))
        recalls.append(recall_at_k(ranked_items, relevant, k))
        hit_rates.append(hit_rate_at_k(ranked_items, relevant, k))

    return {"Model": model_name,
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}": np.mean(recalls),
        f"HitRate@{k}": np.mean(hit_rates),
        "Evaluation": f"Sampled Candidates ({n_neg} negatives)"}