import pandas as pd

def load_data(ratings_path="dataset/ratings.csv",
              movies_path="dataset/movies.csv"):
    """
    Load MovieLens ratings and movie metadata.
    """

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    return ratings, movies


def leave_one_out_split(ratings_df, random_state=42):
    """
    Leave-one-out split:
    Reserve one interaction per user for testing.
    """

    shuffled = ratings_df.sample(frac=1, random_state=random_state)

    test_df = shuffled.groupby("userId").head(1)
    train_df = shuffled.drop(test_df.index)

    return train_df, test_df