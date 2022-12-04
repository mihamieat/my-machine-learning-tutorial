#!/usr/bin/env python
# pylint: disable=E1101
"""Tutorial from https://www.data-transitionnumerique.com/scikit-learn-python/."""

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris


def create_plants_data_frame():
    """Create the plantes_iris data frame from the sklearn datasets iris library."""
    plantes_iris = load_iris()
    df_plantes_iris = pd.DataFrame(
        plantes_iris.data, columns=plantes_iris.feature_names
    )

    return df_plantes_iris


def plot_plants_data_frame(df_plantes_iris):
    """Plot the plantes_iris data frame."""
    scatter_matrix(df_plantes_iris, figsize=(10, 10))
    plt.show()


plant_iris = create_plants_data_frame()
plot_plants_data_frame(plant_iris)
