import analysis_tools as tools
import pandas as pd
import numpy as np


# TODO: Implement analysis & GUI communication in this file. The .ipynb file can be for displaying results.
if __name__ == '__main__':
    # Import dataset of reduced representation to perform clustering, sampling, and/or KNN.
    recipe_data = pd.read_csv('Data/final_recipes_normalized_varFS_incrementalPCA100.csv')
