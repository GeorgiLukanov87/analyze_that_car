# ui_topsis_search.py
import streamlit as st
import numpy as np
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder


# Your custom functions
def print_top_results(count, initial_cars_data, ranking):
    saved_cars = []
    car_counter = 0
    for rank in ranking:
        car_counter += 1
        if car_counter == count + 1:
            break

        car_row = initial_cars_data.iloc[rank]
        saved_cars.append(car_row)

    return pd.DataFrame(saved_cars)


def topsis(matrix, weights):
    norm_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))
    weighted_matrix = norm_matrix * weights

    ideal_solution = np.max(weighted_matrix, axis=0)
    ideal_solution[1] = np.min(weighted_matrix[:, 1])  # price
    ideal_solution[2] = np.min(weighted_matrix[:, 2])  # kms

    anti_ideal_solution = np.min(weighted_matrix, axis=0)
    anti_ideal_solution[1] = np.max(weighted_matrix[:, 1])  # price
    anti_ideal_solution[2] = np.max(weighted_matrix[:, 2])  # kms

    dist_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution) ** 2, axis=1))
    dist_to_anti_ideal = np.sqrt(np.sum((weighted_matrix - anti_ideal_solution) ** 2, axis=1))

    closeness_coefficient = dist_to_anti_ideal / (dist_to_ideal + dist_to_anti_ideal)
    ranking = closeness_coefficient.argsort()[::-1]

    return closeness_coefficient, ranking


# Streamlit app starts here
st.title("Car Selection using TOPSIS")

# 1. File upload and data display
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    car_data = pd.read_csv(uploaded_file, index_col=0)
    car_data.reset_index(inplace=True)  # Reset index
    features = ['year', 'price', 'kms', 'engine_cc', 'horsepower']

    cars_df = pd.DataFrame(car_data)
    initial_cars_data = cars_df.copy()
    cars_df = cars_df[['name'] + features]

    car_array = cars_df[features].values
    matrix = np.array(car_array)

    st.write("### Initial Cars Data")
    st.dataframe(initial_cars_data)

    # Constants values for search sliders
    MIN_SLIDER_VALUE = 0.1
    MAX_SLIDER_VALUE = 0.5
    INITIAL_VALUES = 0.2
    STEP_VALUES = 0.1

    # 2. Dynamic weight selection by the user
    st.write("### Select Weights for the Criteria")
    weight_year = st.slider("Weight for Year",
                            MIN_SLIDER_VALUE, MAX_SLIDER_VALUE, INITIAL_VALUES, STEP_VALUES)
    weight_price = st.slider("Weight for Price",
                             MIN_SLIDER_VALUE, MAX_SLIDER_VALUE, INITIAL_VALUES, STEP_VALUES)
    weight_kms = st.slider("Weight for Kms Driven",
                           MIN_SLIDER_VALUE, MAX_SLIDER_VALUE, INITIAL_VALUES, STEP_VALUES)
    weight_engine_cc = st.slider("Weight for Engine Capacity",
                                 MIN_SLIDER_VALUE, MAX_SLIDER_VALUE, INITIAL_VALUES, STEP_VALUES)
    weight_horsepower = st.slider("Weight for Horsepower",
                                  MIN_SLIDER_VALUE, MAX_SLIDER_VALUE, INITIAL_VALUES, STEP_VALUES)

    weights = np.array([
        weight_year,  # year
        weight_price,  # price
        weight_kms,  # kms
        weight_engine_cc,  # engine_cc
        weight_horsepower  # horsepower
    ])

    # 3. Apply TOPSIS and display the results
    closeness, ranking = topsis(matrix, weights)

    # Let the user input how many top results to display
    search_count = st.slider("Select number of top cars to display", 1, len(cars_df), 10)

    # Display the top results using your function
    st.write("### Top Results")
    result = print_top_results(search_count, initial_cars_data, ranking)

    # Display the result in a searchable and filterable table
    gb = GridOptionsBuilder.from_dataframe(result)
    gb.configure_pagination(paginationAutoPageSize=True)  # Add pagination
    gb.configure_side_bar()  # Add a sidebar for filtering
    grid_options = gb.build()

    AgGrid(
        result,
        gridOptions=grid_options,
        enable_enterprise_modules=True,  # Enables advanced features like filtering and search
        theme='streamlit',  # Theme for the table
    )

    # Option to download the results as CSV
    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download top results as CSV",
        data=csv,
        file_name=f'top_{search_count}_cars.csv',
        mime='text/csv',
    )
