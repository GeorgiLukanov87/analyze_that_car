import numpy as np
import pandas as pd


def topsis(matrix, weights):
    norm_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))
    weighted_matrix = norm_matrix * weights

    # 'year', 'price', 'kms', 'engine_cc', 'horsepower'
    ideal_solution = np.max(weighted_matrix, axis=0)
    ideal_solution[1] = np.min(weighted_matrix[:, 1])  # price
    ideal_solution[2] = np.min(weighted_matrix[:, 2])  # kms

    # 'year', 'price', 'kms', 'engine_cc', 'horsepower'
    anti_ideal_solution = np.min(weighted_matrix, axis=0)
    anti_ideal_solution[1] = np.max(weighted_matrix[:, 1])  # price
    anti_ideal_solution[2] = np.max(weighted_matrix[:, 2])  # kms

    # Calculate the distance to the ideal and anti-ideal solutions
    dist_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution) ** 2, axis=1))
    dist_to_anti_ideal = np.sqrt(np.sum((weighted_matrix - anti_ideal_solution) ** 2, axis=1))

    # Calculate the closeness coefficient to the ideal solution
    closeness_coefficient = dist_to_anti_ideal / (dist_to_ideal + dist_to_anti_ideal)

    # Rank the alternatives
    ranking = closeness_coefficient.argsort()[::-1]  # argsort returns indices in ascending order

    return closeness_coefficient, ranking


def print_top_results(count, initial_cars_data, ranking):  # Prepare data for printing Top car offers
    saved_cars = []
    car_counter = 0
    for rank in ranking:
        car_counter += 1
        if car_counter == count + 1:
            break

        car_row = initial_cars_data.iloc[rank]
        saved_cars.append(car_row)

    return pd.DataFrame(saved_cars)
