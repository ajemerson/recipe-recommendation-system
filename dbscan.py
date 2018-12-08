import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN as dbizzle


def dbscan(data):
    """
    Perform DBSCAN density-based clustering on the given dataset
    :param data: a pandas dataframe to be clustered
    :return: the same pandas dataframe with clusters attached to each row
    """

    print('Clustering input data with DBSCAN...')
    clustering = dbizzle(eps=5, min_samples=10).fit(data)
    clusters = clustering.labels_
    print('DBSCAN complete.')
    print(clusters)
    print("The unique clusters:", np.unique(clusters))
    data['clusters'] = clusters
    return data


if __name__ == "__main__":
    path = "../Datasets/"
    print('Reading data...')
    df100 = pd.read_csv(path + 'recipes_encoded100.csv')
    df1000 = pd.read_csv(path + 'recipes_encoded1000.csv')
    recipe_info = pd.read_csv(path + 'recipe_info.csv')
    print('Data reading complete.')
    data = dbscan(df100)
