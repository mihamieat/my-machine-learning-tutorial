#!/usr/bin/env python
# pylint: disable=E1101
"""Tutorial from https://www.data-transitionnumerique.com/scikit-learn-python/."""

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris


def create_plants_data_frame():
    """Create the iris_plants data frame from the sklearn datasets iris library."""
    iris_plants = load_iris()
    df_iris_plants = pd.DataFrame(iris_plants.data, columns=iris_plants.feature_names)

    return df_iris_plants


def plot_plants_data_frame(df_iris_plants):
    """Plot the iris_plants data frame."""
    scatter_matrix(df_iris_plants, figsize=(10, 10))
    plt.show()


plant_iris = create_plants_data_frame()
plot_plants_data_frame(plant_iris)
