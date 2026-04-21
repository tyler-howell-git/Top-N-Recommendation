import pandas as pd

def load_movielens(x:str):
    """Loads the MovieLens dataset"""
    if x == 'ratings':
        rating_data = pd.read_csv('dataset/ratings.csv')
        return rating_data
    else:
        return None
    
# rating_data = load_movielens('ratings')
# print(rating_data.head())
# baseline = rating_data.sort_values('rating', ascending=True).head(10)
#print(load_movielens('ratings').describe())


def compute_global_avg_rating(rating_data:pd.DataFrame):
    """Computes the global average rating"""
    global_avg_ratings = [] #index is movie id, value is average rating for that movie
    
    #sort by movie id
    rating_data_sorted = rating_data.sort_values('movieId')
    current_movie_id = rating_data_sorted['movieId'].iloc[0]
    print(current_movie_id)
    current_rating_num = 0  
    current_rating_sum = 0.0
    
    for row in rating_data_sorted.itertuples():
        if row.movieId == current_movie_id:
            current_rating_num += 1
            current_rating_sum += row.rating
        else:
            avg_rating = current_rating_sum / current_rating_num #compute average rating for the previous movie
            global_avg_ratings.append(round(avg_rating, 2))
            current_movie_id = row.movieId
            current_rating_num = 1
            current_rating_sum = row.rating
            
    return global_avg_ratings

rating_data = load_movielens('ratings')
global_avg_ratings = compute_global_avg_rating(rating_data)
print(global_avg_ratings)
print('number of movies:', len(global_avg_ratings) + 1) #movie id starts from 1, so we need to add 1 to get the total number of movies

print('Top 50 movies with highest average ratings:\n')
top_50_movies = sorted(enumerate(global_avg_ratings, start=1), key=lambda x: x[1], reverse=True)[:50]
for movie_id, avg_rating in top_50_movies:
    print(f'Movie ID: {movie_id}, Average Rating: {avg_rating}') 


        