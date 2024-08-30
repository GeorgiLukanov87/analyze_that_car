# utils/data_processing.py

import pandas as pd
import numpy as np
from config import TOPSIS_FEATURES, KMEANS_FEATURES


def prepare_topsis_data(uploaded_file):
    car_data = pd.read_csv(uploaded_file, index_col=0)
    car_data.reset_index(inplace=True)

    cars_df = pd.DataFrame(car_data)
    initial_cars_data = cars_df.copy()

    cars_df = cars_df[['name'] + TOPSIS_FEATURES]

    car_array = cars_df[TOPSIS_FEATURES].values
    matrix = np.array(car_array)

    return car_data, cars_df, initial_cars_data, matrix


def prepare_kmeans_data(uploaded_file):
    cars = pd.read_csv(uploaded_file)
    cars = rename_data(cars)
    cars = cars.dropna(subset=KMEANS_FEATURES)
    data = cars[KMEANS_FEATURES].copy()
    car_data = clean_car_data(data)
    return cars, car_data


def rename_data(df):  # Rename colums
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
    re_pattern = r'([0-9]+)'

    cleaned_df['price'] = cleaned_df['price'].apply(lambda x: x / 10)
    cleaned_df['engine_cc'] = cleaned_df['engine_cc'].str.extract(re_pattern).astype(float)
    cleaned_df['horsepower'] = cleaned_df['horsepower'].str.extract(re_pattern).astype(float)
    cleaned_df['mileage'] = cleaned_df['mileage'].str.extract(re_pattern).astype(float)

    return cleaned_df
