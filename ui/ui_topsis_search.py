# ui/ui_topsis_search.py

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from utils.data_processing import prepare_topsis_data
from algorithms.topsis import topsis, print_top_results
from config import TOPSIS_FEATURES, MIN_SLIDER_VALUE, MAX_SLIDER_VALUE, INITIAL_VALUES, STEP_VALUES


def run_topsis_ui():
    st.title("Car Selection using TOPSIS")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        car_data, cars_df, initial_cars_data, matrix = prepare_topsis_data(uploaded_file)

        st.write("### Initial Cars Data")
        st.dataframe(initial_cars_data)

        weights = get_user_weights()

        closeness, ranking = topsis(matrix, weights)

        search_count = st.slider("Select number of top cars to display", 1, len(cars_df), 10)

        st.write("### Top Results")
        result = print_top_results(search_count, initial_cars_data, ranking)

        display_topsis_results(result)


def get_user_weights():
    st.write("### Select Weights for the Criteria")
    weights = []
    for feature in TOPSIS_FEATURES:
        weight = st.slider(
            f"Weight for {feature.capitalize()}",
            MIN_SLIDER_VALUE,
            MAX_SLIDER_VALUE,
            INITIAL_VALUES,
            STEP_VALUES
        )
        weights.append(weight)

    return weights


def display_topsis_results(result):
    gb = GridOptionsBuilder.from_dataframe(result)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    AgGrid(
        result,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        theme='streamlit',
    )

    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download top results as CSV",
        data=csv,
        file_name=f'top_{len(result)}_cars.csv',
        mime='text/csv',
    )
