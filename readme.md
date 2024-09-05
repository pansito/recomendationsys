# Solution for a reccomendation system
This solution uses multiple python packages/libraries to work:
    - pandas
    - datetime
    - numpy
    - sklearn
    - flask
    - plotly


## How it works? 

The jupyter notebook [Data Review](https://github.com/pansito/recomendationsys/blob/main/Data%20Review.ipynb) was used for testing and data exploration to review the values. 


The main file is the [backend.py](https://github.com/pansito/recomendationsys/blob/main/backend.py) this defines 2 functios 

- home: Compiles the main home html file to construct the website
- recomendacion: this creates a api which receives a id request and sends back the list of recomendations given by the ML model.

The [Model_training.py](https://github.com/pansito/recomendationsys/blob/main/Model_training.py) computes a nearest neighbor model from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors) to compute the distances between the points uses the **kneighbors** feature to bring back the closets 10 recommendations. 

Insite the html function there's a javascript code which executes when the **Get Recommendations** buttom is pressed and adds the list of reccomendations. 


## How to use it? 

Just install the python requeriments for libraries and run the [backend.py](https://github.com/pansito/recomendationsys/blob/main/backend.py) file, it will create a small web interface, which can be open through any browser.


