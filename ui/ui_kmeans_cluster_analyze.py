# ui/ui_kmeans_cluster_analyze.py

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from utils.data_processing import prepare_kmeans_data, clean_car_data
from algorithms.kmeans_clustering import k_means_algorithm
from config import MAX_ITERATIONS, CENTROID_COUNT
from sklearn.cluster import KMeans


def run_kmeans_ui():
    st.title('KMeans Clustering Algorithm')

    # Add the download link at the top of the main screen
    st.markdown("""
    To get started, you can download the Car-DataSet CSV file from this link:
    [Download CarDataSet     CSV](https://github.com/GeorgiLukanov87/analyze_that_car/tree/main/data/raw_data)
    """)

    st.write("Once you have downloaded the CSV file, you can upload or dropdown it below:")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        cars, data = prepare_kmeans_data(uploaded_file)

        st.write(f'Cleaned Data:\n', data.head())

        """
        (data - data.min()): This subtracts the minimum value from each element in the respective column.
        This shifts the minimum value to 0.
        (data.max() - data.min()): This calculates the range (difference between maximum and minimum) for each column.
        # (No negative values or 0) the values must be in range (0.1 to 1) or (1 to 10) or (10 to 100) etc...
        """

        # Scaling data, Min Max Scaling
        scaled_data = ((data - data.min()) / (data.max() - data.min())) * 9 + 1
        st.write(f'Scaled Data:\n', scaled_data.head())

        plot_elbow_method(scaled_data)

        if st.button('Run KMeans'):
            labels = k_means_algorithm(scaled_data, MAX_ITERATIONS, CENTROID_COUNT)
            display_cluster_results(cars, labels)


def plot_elbow_method(scaled_data):
    # Elbow method is to find the best count of the clusters we need for the KMeans Cluster Algorithm
    means = range(1, 12)
    inertias = []

    for k in means:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(scaled_data)
        inertias.append(km.inertia_)

    plt.figure()
    plt.title('Elbow Method for Optimal Clusters')
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    st.pyplot(plt)


def display_cluster_results(cars, labels):
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

    # automate this
    # TODO

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

    # Saving locally the results for every category
    new_cars_cluster1.to_csv('data/saved_data/new_cars_cluster1.csv')
    new_cars_cluster2.to_csv('data/saved_data/new_cars_cluster2.csv')
    new_cars_cluster3.to_csv('data/saved_data/new_cars_cluster3.csv')
    new_cars_cluster4.to_csv('data/saved_data/new_cars_cluster4.csv')
    new_cars_cluster5.to_csv('data/saved_data/new_cars_cluster5.csv')
