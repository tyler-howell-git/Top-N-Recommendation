
def load_movielens_stats():
    """Loads the MovieLens dataset and computes basic statistics"""
    with open("dataset/ratings.csv", "r") as f:
        stats = f.read()
        return stats
    return None

if __name__ == "__main__":
    stats = load_movielens_stats()
    print(stats)