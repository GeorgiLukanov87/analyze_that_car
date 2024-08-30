# ui_kmeans_cluster_analyze.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Function definitions (cleaned for brevity)
def rename_data(df):
    df.rename(
        columns={
            'selling_price': 'price',
            'engine': 'engine_cc',
            'km_driven': 'kms',
            'max_power': 'horsepower'
        }, inplace=True)
    return df


def clean_car_data(df):
    cleaned_df = df.copy()
    re_pattern = f'([0-9]+)'

    cleaned_df['price'] = cleaned_df['price'].apply(lambda x: x / 10)
    cleaned_df['engine_cc'] = cleaned_df['engine_cc'].str.extract(re_pattern).astype(float)
    cleaned_df['horsepower'] = cleaned_df['horsepower'].str.extract(re_pattern).astype(float)
    cleaned_df['mileage'] = cleaned_df['mileage'].str.extract(re_pattern).astype(float)

    return cleaned_df


def elbow_method(scaled_data):
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
    st.pyplot(plt.gcf())  # Display the plot in Streamlit


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


def create_random_centroids(data, k):
    random_centroids = []
    for _ in range(k):
        centroid = data.apply(lambda x: float(x.sample().iloc[0]))  # sample() get a random value from each column
        random_centroids.append(centroid)

    return pd.concat(random_centroids, axis=1)


def find_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt((data - x) ** 2).sum(axis=1))
    return distances.idxmin(axis=1)


def new_centroids(data, labels):
    return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T


def plot_clusters(data, labels, centroids, iteration):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    plt.figure()
    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=labels)
    plt.scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1], color='red', s=75)
    st.pyplot(plt.gcf())  # Display the plot in Streamlit


# Streamlit Interface
st.title('KMeans Clustering Algorithm')

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    cars = pd.read_csv(uploaded_file)
    features = ['year', 'price', 'kms', 'engine_cc', 'horsepower', 'seats', 'mileage']
    cars = rename_data(cars)
    cars = cars.dropna(subset=features)
    data = cars[features].copy()
    car_data = clean_car_data(data)

    st.write(f'Cleaned Data:\n', car_data.head())

    # Scaling data, Min Max Scaling
    data = ((car_data - car_data.min()) / (car_data.max() - car_data.min())) * 9 + 1
    st.write(f'Scaled Data:\n', data.head())

    # Elbow method visualization
    elbow_method(data)

    # Run KMeans Algorithm
    if st.button('Run KMeans'):
        labels = k_means_algorithm(data)

        # Categorization of clusters
        top_clusters = []
        for i in range(len(labels.value_counts())):
            top_clusters.append(int(labels.value_counts().index[i]))

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
            else:
                pass

        # Convert lists to DataFrames
        new_cars_cluster1 = clean_car_data(pd.DataFrame(df_list1))
        new_cars_cluster2 = clean_car_data(pd.DataFrame(df_list2))
        new_cars_cluster3 = clean_car_data(pd.DataFrame(df_list3))
        new_cars_cluster4 = clean_car_data(pd.DataFrame(df_list4))
        new_cars_cluster5 = clean_car_data(pd.DataFrame(df_list5))

        st.write(f"Category 1 cars: {len(new_cars_cluster1)}")
        st.write(f"Category 2 cars: {len(new_cars_cluster2)}")
        st.write(f"Category 3 cars: {len(new_cars_cluster3)}")
        st.write(f"Category 4 cars: {len(new_cars_cluster4)}")
        st.write(f"Category 5 cars: {len(new_cars_cluster5)}")

        # Display and save each categorized cluster
        st.write('Best deal cars(Balanced stats, good year, middle price, low kms, high engine cc):')
        st.dataframe(new_cars_cluster1)

        st.write('Normal city every day cars(Balanced stats, low consume):')
        st.dataframe(new_cars_cluster2)

        st.write('Huge old cars(High engine cc, 5+ seats = Vans, bus, mini-vans, etc.):')
        st.dataframe(new_cars_cluster3)

        st.write('Fast Sport luxury cars(New year, very high price, very fast):')
        st.dataframe(new_cars_cluster4)

        st.write('Cheap,old and small, last chance cars(Cheap solution):')
        st.dataframe(new_cars_cluster5)

        # Remove comment if you want to local save
        # new_cars_cluster1.to_csv('new_cars_cluster1.csv')
        # new_cars_cluster2.to_csv('new_cars_cluster2.csv')
        # new_cars_cluster3.to_csv('new_cars_cluster3.csv')
        # new_cars_cluster4.to_csv('new_cars_cluster4.csv')
        # new_cars_cluster5.to_csv('new_cars_cluster5.csv')
