import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

"""
KMeans Cluster Algorithm steps:

1.Read data from csv
2.Prepare data - clean data and rename
3.Scaling/normalize data - min max scaling
4.Initialize random CENTROIDS
5.Label each data point - Calculate geometric mean distance between all data-points and CENTROIDS
6.Update CENTROIDS
7.Repeat step 5 and 6 until CENTROIDS stop changing
"""


def k_means_algorithm(data, max_iterations=75, centroid_count=6):
    centroids = create_random_centroids(data, centroid_count)
    st.write(f'First Centroid:\n', centroids)
    old_centroids = pd.DataFrame()
    iteration = 1
    max_i = 0

    while iteration < max_iterations and not centroids.equals(old_centroids):
        old_centroids = centroids

        labels = find_labels(data, centroids)
        centroids = new_centroids(data, labels)
        plot_clusters(data, labels, centroids, iteration)  # Visualization function of how clusters changing
        max_i = iteration
        iteration += 1

    st.write(f'Last Centroids:\n', centroids)
    st.write(f'after {max_i} iterations')
    st.write(f'\nCount of cars going to every cluster category:\n', labels.value_counts())

    return labels


"""
    # Categories
    #1 - Best deal car (balanced stats, good year, middle price, low kms, good engine cc) 
    #2 - Normal city every day car (balanced stats, low consume)
    #3 - Normal city every day car (balanced stats, low consume)
    #4 - Big old car(High engine cc, high consume, 5+ seats = Vans, bus, mini-vans etc...)
    #5 - Fast Sport luxury car(very high stats - year, engine, horsepower)
    #6 - Cheap and old and small, last chance cars, bad stats
"""


# CENTROIDS ACTIONS
def create_random_centroids(data, k):
    random_centroids = []
    for _ in range(k):
        centroid = data.apply(lambda x: float(x.sample().iloc[0]))  # sample() get a random value from each column
        # Type of centroid <class 'pandas.core.series.Series'>
        random_centroids.append(centroid)

    # Convert into <class 'pandas.core.frame.DataFrame'>
    return pd.concat(random_centroids, axis=1)


def find_labels(data, centroids):  # Finding distance between data points and centroid using Pythagorean theorem
    distances = centroids.apply(lambda x: np.sqrt((data - x) ** 2).sum(axis=1))
    return distances.idxmin(axis=1)  # idxmin returns the index of the minimum value


def new_centroids(data, labels):  # Create new centroids updated with GEOMETRIC mean data
    return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T


# Visualisation
def plot_clusters(data, labels, centroids, iteration):
    """"
    # PCA - Principle Components Analysis.
    PCA Class Help us to transform the 5 dimensional features data into 2 dimensional data.
    It's much easier to display 2 instead of 5 or more.
    """
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    plt.figure()
    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=labels)
    plt.scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1], color='red', s=75)
    st.pyplot(plt)
