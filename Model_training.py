import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import numpy as np
data_path = 'ml-100k/'

# Read the u.data file into a pandas DataFrame
udata = pd.read_csv(data_path+'u.data', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
genre= pd.read_csv(data_path+'u.genre', sep='|', header=None, names=['genre', 'id'])
moviesinfo = pd.read_csv(data_path+'u.item', sep='|', header=None,encoding='latin-1', names=[ 'movie id' , 'movie title' , 'release date' , 'video release date' ,
            "IMDb UR'L" , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
            "Children's" , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
            'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
            'Thriller' , 'War' , 'Western' ,])

userd = pd.read_csv(data_path+'u.user', sep='|', header=None, names=['user id' , 'age', 'gender' , 'occupation' , 'zip code'])


movieeval = udata.merge(moviesinfo, left_on='item_id', right_on='movie id')
wholedb = movieeval.merge(userd, left_on='user_id', right_on='user id')

dummiesdb = pd.concat([wholedb, pd.get_dummies(wholedb['occupation']),pd.get_dummies(wholedb.gender)], axis = 1)
dummiesdb = dummiesdb.drop(columns=['release date','video release date','occupation', 'gender','timestamp','movie id', "IMDb UR'L",'user id', 'zip code','movie title'])


# Crear las variables de entrenamiento para la clasificación.
user_features = dummiesdb.pivot(index=dummiesdb.drop(columns='item_id').columns, columns='item_id', values='rating').fillna(0).values
    
# Create the nearest neighbors model
nn_model = NearestNeighbors(algorithm='brute', metric='manhattan')

# Fit the model with all user features
nn_model.fit(user_features)
#print(dummiesdb)

def get_recommendations(user_id, num_recs=10):
   
    user_idx = dummiesdb.index.get_loc(user_id)
    
    
    distances, indices = nn_model.kneighbors([user_features[user_idx]], n_neighbors=num_recs+1)
    
    
    indices = indices[0][1:]
    
    rec_user_ids = dummiesdb.iloc[indices].index.tolist()
    
    rec_movies = []
    for rec_user_id in rec_user_ids:
        top_movies = dummiesdb.loc[dummiesdb.index == rec_user_id].sort_values('rating', ascending=False)['item_id'].head(5).tolist()
        rec_movies.extend(top_movies)
    
    rec_movies = list(set(rec_movies))[:num_recs]
    
    return moviesinfo[moviesinfo['movie id'].isin(rec_movies)]['movie title'].tolist()