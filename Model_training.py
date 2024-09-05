import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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
x = dummiesdb.drop(columns ='item_id')
y = dummiesdb.item_id


# Creación y entrenamiento del modelo.
nn_model = KNeighborsClassifier()
nn_model.fit(x,y)
#print(dummiesdb)

def get_recommendations(user_id, num_recs=10):
    #Utilizamos el indice en la tabla inicial para encontrar la posición del ID
    user_idx = dummiesdb.index.get_loc(user_id)
    
    # Luego encontramso las vecinos similares.
    # dado que iloc retorna una pandas series, covnierte en dataframe y se transpone para mantener el formato inicial. 
    distances, indices = nn_model.kneighbors(pd.DataFrame(dummiesdb.iloc[dummiesdb.index.get_loc(user_id)].drop('item_id')).T, n_neighbors=num_recs+1)
    # Aunque no es necesario se guardan las distancias 
    #Se Guarda el índice que contiene las peliculas recomendadas
    
    indices = indices[0][1:]
    
    #Se utiliza la base con los nomrbes de las películas para retornar las lista con las 10 recomendaciones.

    return moviesinfo[moviesinfo['movie id'].isin(dummiesdb.iloc[indices]['item_id'])]['movie title'].tolist()