# main.py

import streamlit as st
from ui.ui_topsis_search import run_topsis_ui
from ui.ui_kmeans_cluster_analyze import run_kmeans_ui


# TODO
# Error handling
# separate logic
# improve categorizing car
# improve read csv(working with all type of dataset)


def main():
    st.title("Analyze That Car!")

    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ("K-Means Clustering", "TOPSIS")
    )

    if analysis_type == "TOPSIS":
        run_topsis_ui()
    else:
        run_kmeans_ui()


if __name__ == "__main__":
    main()
