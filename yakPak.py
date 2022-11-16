import yfinance as yf
import bs4 as bs
import pickle
import requests
import pandas as pd
import json
from dateutil import parser
from datetime import datetime
import os
import time
from pandas.io.json import json_normalize
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import wrds
from datetime import datetime as dt
import numpy as np
import glob, os
import csv
from matplotlib import pyplot as plt

import hdbscan
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from numpy import inf

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

import pandas_datareader.data as web
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas import ExcelWriter
from pandas import ExcelFile
import os
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from matplotlib.pyplot import figure

import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, silhouette_samples
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,MaxAbsScaler
from sklearn.metrics import silhouette_score
import random


from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from kneed import KneeLocator

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from sklearn.datasets import make_blobs
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import itertools


def save_SP500_tickers():
    '''
    Gets list of SP500 companies and their CIKs
    '''
    resp = requests.get('https://web.archive.org/web/20200411084847/https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    ciks = []
    
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text[:-1]
        cik = row.findAll('td')[7].text[:]
        
        tickers.append(ticker)
        ciks.append(cik)
    
    #drops '\n' from certain entries in CIK list
    ciks = list(map(lambda x:x.strip(),ciks))
    
    tickers_ciks_df = pd.DataFrame({'ticker': tickers, 'cik': ciks})

        
    return tickers_ciks_df

def loop_files(string_to_find):
    '''
    Returns list of file paths containing certain string
    '''
    filepaths = []
    for root, dirs, files in os.walk('Export/'):
        for name in files:
            if name.endswith('.csv'):

                filepath = os.path.join(root, name)
                if f'{string_to_find}' in name:
                    filepaths.append(filepath)
    return filepaths

def read_returns(file_path):
    """
    Returns a dataframe when given a file path to csv file of returns
    """
    movements = pd.read_csv(f'{file_path}',
                              header=0,
                              index_col=0,
                              parse_dates=True)
    
    return movements
    
    
def find_knee(movements, min_samples):
    '''
    Finds knee for data set. AKA the epsilon, or distance
    points need to be from centroid to be considered a cluster
    '''
    
    nearest_neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors = nearest_neighbors.fit(movements)
    distances, indices = neighbors.kneighbors(movements)

    distances = np.sort(distances[:,1], axis=0)



    i = np.arange(len(distances))
    knee = KneeLocator(i, distances,
                       S=1, curve='convex',
                       direction='increasing',
                       interp_method='polynomial')
    '''
    fig = plt.figure(figsize=(5, 3))

    knee.plot_knee()

    plt.xlabel("Points")
    plt.ylabel("Distance")
    '''
    return distances[knee.knee]


def tune_n_clusters(movements, upper_bound = 21, random_state = 10, init = 'k-means++'):
    
    '''
    returns optimal cluster number
    '''
    
    n_clusters = list(range(2,upper_bound))

    sil_dict = {}

    for cluster in n_clusters:
        clusterer = KMeans(n_clusters = cluster, random_state=random_state, init=init)
        cluster_labels = clusterer.fit(movements)

        sil_avg = silhouette_score(movements, clusterer.labels_)

        sil_dict[cluster] = sil_avg
        
    #return max(sil_dict, key = sil_dict.get)
    return sil_dict

def get_n_clusters(movements, random_state=10, max_iter=1000):
    '''
    Solves for n_clusters progrimatically using
    SSE elbow method, and Shilouette method.
    
    Uses both random init and k-means++ init.
    '''
    n_clusters = []
    #------
    kmeans_kwargs = {
        "init": 'k-means++',
        "max_iter": max_iter,
        "random_state": random_state,
    }

    # A list holds the SSE values for each k
    sse = []
    for k in range(1, 15):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(movements)
        sse.append(kmeans.inertia_)

    kl = KneeLocator(range(1, 15), sse, curve="convex", direction="decreasing")

    n_clusters.append(kl.elbow)
    #------
    kmeans_kwargs = {
        "init": "random",
        "max_iter": 1000,
        "random_state": random_state,
    }

    # A list holds the SSE values for each k
    sse = []
    for k in range(1, 15):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(movements)
        sse.append(kmeans.inertia_)

    kl = KneeLocator(range(1, 15), sse, curve="convex", direction="decreasing")

    n_clusters.append(kl.elbow)
    #------
    kmeans_kwargs = {
        "init": "k-means++",
        "max_iter": 1000,
        "random_state": random_state,
    }

    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []

    # Notice you start at 2 clusters for silhouette coefficient
    for k in range(2, 15):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(movements)
        score = silhouette_score(movements, kmeans.labels_)
        silhouette_coefficients.append(score)
        
    #return len(silhouette_coefficients), len(list(range(2, 15)))
    zipped = pd.DataFrame({
        '0': list(range(2, 15)),
        '1': silhouette_coefficients
    })

    n_clusters.append(int(zipped.loc[zipped['1'].idxmax()][0]))

    kmeans_kwargs = {
        "init": "random",
        "max_iter": 1000,
        "random_state": 42,
    }

    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []

    # Notice you start at 2 clusters for silhouette coefficient
    for k in range(2, 15):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(movements)
        score = silhouette_score(movements, kmeans.labels_)
        silhouette_coefficients.append(score)

    zipped = pd.DataFrame({
        '0': list(range(2, 15)),
        '1': silhouette_coefficients
    })
    
    n_clusters.append(int(zipped.loc[zipped['1'].idxmax()][0]))
    #------

    return list(set(n_clusters))


def opti_cluster(movements, eps, upper_bound = 50, DBS = True):
    '''
    Returns number of cluster for certain number of min_samples
    '''
    labels_dict = {}
    ms = list(range(1,upper_bound))
    
    if DBS:
        for i in ms:
            db = DBSCAN(eps = eps, min_samples=i)
            labels = db.fit_predict(movements)


            labels = set([label for label in labels if label >= 0])
            labels_dict[i] = len(labels)
        return labels_dict
        
    if not DBS:
        for i in ms:
            db = OPTICS(eps = eps, min_samples=i)
            labels = db.fit_predict(movements)


            labels = set([label for label in labels if label >= 0])

            labels_dict[i] = len(labels)
        return labels_dict
    
    
def KMEANS_DS(movements,
              n_clusters=3,
              init='k-means++',
              n_init=50,
              max_iter=1000,
              random_state=10,
              algorithm='lloyd',
              PCA_components = 3,
              scaler = MinMaxScaler()):
    '''
    Returns tuple of Silhoutte Score and Predicted Labels
    '''
    preprocessor = Pipeline([
        ("scaler", scaler),
        ("pca", PCA(n_components=PCA_components, random_state=random_state)),
    ])

    clusterer = Pipeline([
        (
            "kmeans",
            KMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                random_state=random_state,
            ),
        ),
    ])

    pipe = Pipeline([("preprocessor", preprocessor), ("clusterer", clusterer)])

    fitted_movements = pipe.fit(movements)

    labels = pipe.predict(movements)

    preprocessed_data = pipe["preprocessor"].transform(movements)

    predicted_labels = pd.DataFrame(index=range(nrow),
                                    columns=['ticker', 'group'])

    i = 0
    for ticker in list(movements.index):
        current_prediction = fitted_movements.predict(
            [list(movements.iloc[i])])[0]
        predicted_labels.loc[i, 'ticker'] = ticker
        predicted_labels.loc[i, 'group'] = current_prediction
        i = i + 1
    #predicted_labels = predicted_labels[predicted_labels.group > -1]
    sil_score = silhouette_score(preprocessed_data, predicted_labels['group'])

    return sil_score, predicted_labels


def DBSCAN_DS(
        movements,
        eps=0.5,
        min_samples=5,
        metric='euclidean',
        metric_params=None,
        algorithm='auto',
        leaf_size=30,
        p=None,
        n_jobs=None,
        scaler=MinMaxScaler(),
        PCA_components=3,
        random_state=10):

    preprocessor = Pipeline([
        ("scaler", scaler),
        ("pca", PCA(n_components=PCA_components, random_state=random_state)),
    ])

    clusterer = Pipeline([
        (
            "DBSCAN",
            DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                algorithm=algorithm,
                p=p,
                leaf_size=leaf_size,
            ),
        ),
    ])

    pipe = Pipeline([("preprocessor", preprocessor), ("clusterer", clusterer)])
    
    
    labels = pipe.fit_predict(movements)
    #preprocessed_data = pipe["preprocessor"].transform(movements)
    d = {'ticker': movements.index.values, 'group': labels}
    predicted_labels = pd.DataFrame(data=d,
                                    index=range(nrow),
                                    columns=['ticker', 'group'])
    #sil_score = silhouette_score(preprocessed_data, predicted_labels['group'])

    return predicted_labels

def OPTICS_DS(movements,
              scaler=MinMaxScaler(),
              PCA_components=3,
              random_state=10,
              min_samples=5,
              max_eps=inf,
              metric='minkowski',
              p=2,
              metric_params=None,
              cluster_method='xi',
              eps=None,
              xi=0.05,
              predecessor_correction=True,
              min_cluster_size=None,
              algorithm='auto',
              leaf_size=30,
             **kwargs):

    preprocessor = Pipeline([
        ("scaler", scaler),
        ("pca", PCA(n_components=PCA_components, random_state=10)),
    ])

    clusterer = Pipeline([
        (
            "OPTICS",
            OPTICS(
                min_samples=5,
                max_eps=inf,
                metric='minkowski',
                p=2,
                metric_params=None,
                cluster_method='xi',
                eps=eps,
                xi=0.05,
                predecessor_correction=True,
                min_cluster_size=None,
                algorithm='auto',
                leaf_size=30,
                memory=None,
                n_jobs=None,
            ),
        ),
    ])

    pipe = Pipeline([("preprocessor", preprocessor), ("clusterer", clusterer)])
    
    preprocessed_data = pipe["preprocessor"].transform(movements)
    
    labels = pipe.fit_predict(movements)

    #preprocessed_data = pipe["preprocessor"].transform(movements)
    d = {'ticker': movements.index.values, 'group': labels}
    predicted_labels = pd.DataFrame(data=d,
                                    index=range(nrow),
                                    columns=['ticker', 'group'])
    
    sil_score = silhouette_score(preprocessed_data, predicted_labels['group'])
    
    return sil_score, predicted_labels

def OPTICS_DS(movements,
              scaler=MinMaxScaler(),
              PCA_components=3,
              random_state=10,
              min_samples=5,
              max_eps=inf,
              metric='minkowski',
              p=2,
              metric_params=None,
              cluster_method='xi',
              eps=None,
              xi=0.05,
              predecessor_correction=True,
              min_cluster_size=None,
              algorithm='auto',
              leaf_size=30,
             **kwargs):

    preprocessor = Pipeline([
        ("scaler", scaler),
        ("pca", PCA(n_components=PCA_components, random_state=10)),
    ])

    clusterer = Pipeline([
        (
            "OPTICS",
            OPTICS(
                min_samples=5,
                max_eps=inf,
                metric='minkowski',
                p=2,
                metric_params=None,
                cluster_method='xi',
                eps=eps,
                xi=0.05,
                predecessor_correction=True,
                min_cluster_size=None,
                algorithm='auto',
                leaf_size=30,
                memory=None,
                n_jobs=None,
            ),
        ),
    ])

    pipe = Pipeline([("preprocessor", preprocessor), ("clusterer", clusterer)])
    
    preprocessed_data = pipe["preprocessor"].transform(movements)
    
    labels = pipe.fit_predict(movements)

    #preprocessed_data = pipe["preprocessor"].transform(movements)
    d = {'ticker': movements.index.values, 'group': labels}
    predicted_labels = pd.DataFrame(data=d,
                                    index=range(nrow),
                                    columns=['ticker', 'group'])
    
    sil_score = silhouette_score(preprocessed_data, predicted_labels['group'])
    
    return sil_score, predicted_labels

def SUMMARYSTATS_DS(predicted_labels, returns_file_name, sharpes_file_name):
    '''
    Returns Constituent Companies of Portfolio and their return
    in during covid in excess of SP500 avg return during same time frame
    '''
    #Portfolio Construction
    sharpes = pd.read_csv(f"{sharpes_file_name}",
                          header=0,
                          index_col=0,
                          parse_dates=True)

    merged = sharpes.merge(predicted_labels, on=['ticker'])

    max_sharpes = merged.groupby('group')['sharpes', 'ticker'].max()

    # Testing Portfolio Returns During Covid

    movements = pd.read_csv(f'Export/returns_covid_PC.csv',
                            header=0,
                            index_col=0,
                            parse_dates=True)

    portfolio = movements[movements.index.isin(max_sharpes['ticker'])]

    portfolio = portfolio.T

    overall = (portfolio.iloc[:, :] + 1).prod() - 1
    average = overall.mean()

    #Excess Return


    excess = average - spExcess


    return portfolio.columns, excess