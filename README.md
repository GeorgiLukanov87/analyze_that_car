# Project: Analyze that car!

### Deployment Link:
#### https://analyze-that-car.streamlit.app/

### Download Cars-Dataset:
#### https://github.com/GeorgiLukanov87/analyze_that_car/tree/main/data/raw_data
#

### Steps test the App:

1. Upload or dropdown the Car-Dataset("selling_cars_list".csv" from the link above).
2. Then you will see automatically the cleaned and scaled data. Then you are ready to execute the program.
3. Click on the "Run KMeans" button to run the algorithm and see the process - how clusters are changing and visualizations from the process.
4. After executing the KMeans process, you can save one or more of the results as CSV file on your computer(new generated cluster/category you wish) and perform the next step.
5. The Topsis method -> clicking on the left side of the panel to choose the dropdown menu(TOPSIS).
6. Now upload or dropdown the saved CSV of your category you saved and perform the Topsis method.
7. Set preferences/weight to your search for more detailed and best matches for you.
#


### Car Analysis and Recommendation System using Pandas, Numpy, Matplotlib and Streamlit for UI-visualization.
### This project consists of two main components:

# 1. KMeans Clustering Algorithm:

Performs data preprocessing and clustering on a car dataset.
Uses the KMeans algorithm to group cars into 2-6 clusters based on
features like year, price, kilometers driven, engine capacity, and horsepower.
Implements the elbow method to determine the optimal number of clusters.
Visualizes the clustering results using PCA for dimensionality reduction.

## Examples:

![clusters_changing2](https://github.com/user-attachments/assets/1dd2e0f8-d451-4a6b-8a20-1a6f9b647579)

![elbow1](https://github.com/user-attachments/assets/28990845-e789-4ff3-8e2e-7567e9c94bb0)

![clusters_changing3](https://github.com/user-attachments/assets/2e12fd85-c408-48d0-83aa-db85f863fd9c)

![topsis1](https://github.com/user-attachments/assets/2337f7c6-82d7-4f3c-b015-e3b66580fae5)

![clusters_changing](https://github.com/user-attachments/assets/286e45a7-c795-4ec6-b996-86e27afecdc5)





    Categorizes cars into clusters:

1. "Best offers(the best stats)",
2. "Balanced cars",
3. "Huge sized vans/mini-vans/jeeps/4x4/5+ seats etc...",
4. "Fast luxury cars(expensive, but fast and new)",
5. "Cheap and old cars(worst offer, worst stats)".

or

1. Good - best offers
2. Normal - balanced
3. Huge sized
4. Fast luxury
5. Cheap, budget

### 2. TOPSIS Method (TOPSIS_method.py):

Implements the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) for car recommendation.
Reads the clustered car data from KMeans Clustering Algorithm.
Applies the TOPSIS method to rank cars based on multiple criteria:

1. year
2. price
3. kilometers driven
4. engine
5. capacity

Provides a function to print the top-ranked cars with their details.

The project aims to analyze a dataset of cars, group them into meaningful clusters,
and then provide recommendations for
the best cars within a specific cluster based on multiple criteria.
This system could be useful for car dealerships or consumers
looking for specific types of vehicles that best match their preferences.

## App structure:

    analyze_that_car/
    ├── ui/
    │   ├── ui_kmeans_cluster_analyze.py
    │   └── ui_topsis_search.py
    │
    ├── utils/
    │   ├── __init__.py
    │   └── data_processing.py
    │
    ├── algorithms/
    │   ├── __init__.py
    │   ├── kmeans_clustering.py
    │   └── topsis.py
    │
    ├── data/
    │   ├── raw_data/
    │   └── saved_data/
    |
    ├── main.py
    └── requirements.txt

## Installation Steps

1. Clone the repository:
   git clone https://github.com/GeorgiLukanov87/analyze_that_car.git
   cd analyze_that_car/

2. Create a virtual environment (recommended):
   python -m venv venv

3. Activate the virtual environment:
    - On Windows:
      ```
      venv\Scripts\activate
      ```
    - On macOS and Linux:
      ```
      source venv/bin/activate
      ```

4. Install the required packages:
   pip install -r requirements.txt

5. Ensure you have the necessary data files:
    - Place your car dataset CSV file in the `data/raw_data` directory, for easy access.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- AgGrid (for Streamlit)

For a complete list of dependencies, refer to the `requirements.txt` file.

## Run the Streamlit apps:

    streamlit run analyze_that_cars\main.py

You can now view your Streamlit app locally in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.100.12:8501

