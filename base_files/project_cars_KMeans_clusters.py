from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output

import pandas as pd
import numpy as np

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


# Elbow method is to find the best count of the clusters we need for the KMeans Cluster Algorithm
def elbow_method1(scaled_data):
    means = range(1, 12)
    inertias = []

    for k in means:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(scaled_data)
        inertias.append(km.inertia_)

    plt.title('Elbow Method for Optimal Clusters')
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()


def rename_data(df):  # Rename columns
    df.rename(
        columns={
            'selling_price': 'price',
            'engine': 'engine_cc',
            'km_driven': 'kms',
            'max_power': 'horsepower'
        }, inplace=True)
    return df


def clean_car_data(df):  # Clean data
    cleaned_df = df.copy()
    re_pattern = f'([0-9]+)'

    cleaned_df['price'] = cleaned_df['price'].apply(lambda x: x / 10)
    cleaned_df['engine_cc'] = cleaned_df['engine_cc'].str.extract(re_pattern).astype(float)
    cleaned_df['horsepower'] = cleaned_df['horsepower'].str.extract(re_pattern).astype(float)
    cleaned_df['mileage'] = cleaned_df['mileage'].str.extract(re_pattern).astype(float)

    return cleaned_df


# Read data
cars = pd.read_csv('your_csv_file.csv')
features = ['year', 'price', 'kms', 'engine_cc', 'horsepower', 'seats', 'mileage']
cars = rename_data(cars)
cars = cars.dropna(subset=features)
data = cars[features].copy()
car_data = clean_car_data(data)

"""
                        Cleaned data: 
    year         price        kms        engine         hp 
------------------------------------------------------------
0  2014    |    45000.0   |   145500   |   1248.0   |    74.0
1  2014    |    37000.0   |   120000   |   1498.0   |    103.0
2  2006    |    15800.0   |   140000   |   1497.0   |    78.0
3  2010    |    22500.0   |   127000   |   1396.0   |    90.0
4  2007    |    13000.0   |   120000   |   1298.0   |    88.0
-------------------------------------------------------------
"""

"""
(data - data.min()): This subtracts the minimum value from each element in the respective column. 
This shifts the minimum value to 0.
(data.max() - data.min()): This calculates the range (difference between maximum and minimum) for each column.
# (No negative values or 0) the values must be in range (0.1 to 1) or (1 to 10) or (10 to 100) etc...
"""

# Scaling data, Min Max Scaling
# (No negative value or 0) the values must be between (0.1 to 1) or (1 to 10) or (10 to 100) etc...
data = ((car_data - car_data.min()) / (car_data.max() - car_data.min())) * 9 + 1
# data = ((car_data - car_data.min()) / (car_data.max() - car_data.min())) * 0.9 + 0.1
elbow_method1(car_data)
"""
                        Scaled data: 
    year            price            kms          engine          hp 
------------------------------------------------------------------------
0  7.923077   |    1.379138   |   1.554762   |   2.884564   |   2.027174
1  7.923077   |    1.306922   |   1.457535   |   3.639597   |   2.736413
2  5.153846   |    1.115548   |   1.533791   |   3.636577   |   2.125000
3  6.538462   |    1.176029   |   1.484225   |   3.331544   |   2.418478
4  5.500000   |    1.090272   |   1.457535   |   3.035570   |   2.369565
------------------------------------------------------------------------
"""
print(f'\nCleaned data: \n', car_data.head())
print(f'\nScaled data: \n', data.head())


def create_random_centroids(data, k):
    random_centroids = []
    for _ in range(k):
        centroid = data.apply(lambda x: float(x.sample().iloc[0]))  # sample() get a random value from each column
        # Type of centroid <class 'pandas.core.series.Series'>
        random_centroids.append(centroid)

    # Convert into <class 'pandas.core.frame.DataFrame'>
    convert_to_df = pd.concat(random_centroids, axis=1)
    return convert_to_df


# For testing logic
# K = Number of Clusters
# k = 6
# centroids = create_random_centroids(data, k)
# print(f'\nCentroids: ')
# print(centroids)

# Finding distance between data points and centroid using Pythagorean theorem
# distance = np.sqrt((data - centroids.iloc[:, 0]) ** 2).sum(axis=1)
def find_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt((data - x) ** 2).sum(axis=1))
    return distances.idxmin(axis=1)


# labels = find_labels(data, centroids)


# Create new centroids updated with geometric mean data
def new_centroids(data, labels):
    return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T


# Visualisation
""""
# PCA - Principle Components Analysis.
PCA Class Help us to transform the 5 dimensional features data into 2 dimensional data.
It's much easier to display 2 instead of 5 or more.
"""


def plot_clusters(data, labels, centroids, iteration):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)
    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=labels)
    plt.scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1], color='red', s=75)
    plt.show()


def k_means_algorithm(max_iterations=50, centroid_count=5):
    centroids = create_random_centroids(data, centroid_count)
    print(f'\nFirst Centroid:\n', centroids)
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

    print(f'\nLast Centroids:\n', centroids)
    print(f'after {max_i} iterations')
    print(f'\nCount of cars going to every cluster category:\n', labels.value_counts())
    return labels


labels = k_means_algorithm(50, 6)

top_clusters = []
for i in range(len(labels.value_counts())):
    top_clusters.append(int(labels.value_counts().index[i]))

"""
# Categories
#1 - Best deal car (balanced stats, good year, middle price, low kms, good engine cc) 
#2 - Normal city every day car (balanced stats, low consume)
#3 - Normal city every day car (balanced stats, low consume)
#4 - Big old car(High engine cc, high consume, 5+ seats = Vans, bus, mini-vans etc...)
#5 - Fast Sport luxury car(very high stats - year, engine, horsepower)
#6 - Cheap and old and small, last chance cars, bad stats
"""

df_list1, df_list2, df_list3, df_list4, df_list5 = [], [], [], [], []

for car_index, cluster in labels.items():
    row = cars.loc[car_index].copy()

    if cluster == top_clusters[0]:
        row['cluster'] = 1
        df_list1.append(row)

    elif cluster == top_clusters[1] or cluster == top_clusters[2]:
        row['cluster'] = 2
        df_list2.append(row)

    elif cluster == top_clusters[3]:
        row['cluster'] = 3
        df_list3.append(row)

    elif cluster == top_clusters[4]:
        row['cluster'] = 4
        df_list4.append(row)

    elif cluster == top_clusters[5]:
        row['cluster'] = 5
        df_list5.append(row)

new_cars_cluster1 = clean_car_data(pd.DataFrame(df_list1))
new_cars_cluster2 = clean_car_data(pd.DataFrame(df_list2))
new_cars_cluster3 = clean_car_data(pd.DataFrame(df_list3))
new_cars_cluster4 = clean_car_data(pd.DataFrame(df_list4))
new_cars_cluster5 = clean_car_data(pd.DataFrame(df_list5))

# Visualisation as Pie plot
y = np.array([len(new_cars_cluster1), len(new_cars_cluster2), len(new_cars_cluster3), len(new_cars_cluster4),
              len(new_cars_cluster5)])
my_labels = ["Best offers", "Balanced cars", "Big cars/jeeps/4x4", "Fast luxury cars", "Cheap and old cars", ]
my_explode = [0.2, 0, 0, 0, 0]
plt.pie(y, labels=my_labels, explode=my_explode)
plt.show()

print('\nBest deal car (balanced stats, good year, middle price, low kms, good engine cc) \n', new_cars_cluster1)
new_cars_cluster1.to_csv('data/saved_data/new_cars_cluster1.csv')

print('\nNormal city every day car (balanced stats)\n', new_cars_cluster2)
new_cars_cluster2.to_csv('data/saved_data/new_cars_cluster2.csv')

print('\nBig old car(High engine cc,5+ seats = Vans, bus,jeep, 4x4, mini-vans etc...)\n', new_cars_cluster3)
new_cars_cluster3.to_csv('data/saved_data/new_cars_cluster3.csv')

print('\nFast Sport luxury car(new year, very high price, very fast)\n', new_cars_cluster4)
new_cars_cluster4.to_csv('data/saved_data/new_cars_cluster4.csv')

print('\nCheap and old and small, last chance cars\n', new_cars_cluster5)
new_cars_cluster5.to_csv('data/saved_data/new_cars_cluster5.csv')
