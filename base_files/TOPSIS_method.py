import numpy as np
import pandas as pd


def print_top_results(count):  # Print the details of cars in ranked order
    saved_cars = []
    car_counter = 0
    for rank in ranking:
        car_counter += 1
        if car_counter == count + 1:
            break

        car_row = initial_cars_data.iloc[rank]
        saved_cars.append(car_row)

        print(f"Rank {car_counter}: {car_row['name']}")
        print(f"Year: {int(car_row['year'])}")
        print(f"Selling Price: {car_row['price']} $")
        print(f"Kms driven: {car_row['kms']} kms")
        print(f"Engine: {car_row['engine_cc']} cc")
        print(f"Horsepower: {car_row['horsepower']} hp")
        print(f"Mileage: {car_row['mileage']} kms per liter")
        print(f"Seats: {car_row['seats']}")
        print(f"Fuel Type: {car_row['fuel']}")
        print(f"Seller: {car_row['seller_type']}")
        print(f"Transmission Type: {car_row['transmission']}")
        print(f"Owner: {car_row['owner']}")
        print()

    return pd.DataFrame(saved_cars)


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
    anti_ideal_solution[2] = np.max(weighted_matrix[:, 2])  # price

    print("\n Ideal solution:", ideal_solution)
    print("\n Anti-ideal solution:", anti_ideal_solution)

    # Step 4: Calculate the distance to the ideal and anti-ideal solutions
    dist_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution) ** 2, axis=1))
    dist_to_anti_ideal = np.sqrt(np.sum((weighted_matrix - anti_ideal_solution) ** 2, axis=1))

    # print("\n Distance to IDEAL solution:", dist_to_ideal) #long print
    # print("\n Distance to ANTI-IDEAL solution:", dist_to_anti_ideal) #long print

    # Step 5: Calculate the closeness coefficient to the ideal solution
    closeness_coefficient = dist_to_anti_ideal / (dist_to_ideal + dist_to_anti_ideal)

    # Step 6: Rank the alternatives
    ranking = closeness_coefficient.argsort()[::-1]  # argsort returns indices in ascending order
    return closeness_coefficient, ranking


car_data = pd.read_csv('data/saved_data/new_cars_cluster4.csv', index_col=0)
features = ['year', 'price', 'kms', 'engine_cc', 'horsepower']

cars_df = pd.DataFrame(car_data)
initial_cars_data = cars_df.copy()
cars_df = cars_df[['name'] + features]

car_array = cars_df[features].values
matrix = np.array(car_array)

# Weights: 'year', 'price', 'kms', 'engine_cc', 'horsepower'
weights = np.array(
    [
        0.3,  # year
        0.2,  # price
        0.3,  # kms
        0.2,  # engine_cc
        0.4,  # horsepower
    ]
)

closeness, ranking = topsis(matrix, weights)

# print("\nCloseness coefficients:", closeness)
# print("\nRanking of alternatives:", ranking)
# print("\nCloseness coefficients:", closeness)
# print("\nRanking of alternatives:", ranking)

print(initial_cars_data)
print('\nSaved cars:')
search_count = 5
result = print_top_results(search_count)
print(result.to_string())
result.to_string(f'data/saved_data/last_top_{search_count}_search_cars.csv')
