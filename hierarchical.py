from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd


def hierarchical_cluster(d, n_clusters):
    """
    :param d: a pandas dataframe of data
    :param n_clusters: number of clusters to find
    :return: a pandas dataframe with the clustering appended as the last column to the input
    """
    print('Performing hierarchical clustering with Ward...')
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    ward.fit(d)
    print('Hierarchical clustering complete.')
    labels = ward.labels_
    print(labels)
    print("Unique clusters:", np.unique(labels))
    d['clusters'] = labels
    return d


if __name__ == "__main__":
    path = "../Datasets/"
    print('Reading data...')
    df100 = pd.read_csv(path + 'recipes_encoded100.csv')
    df100 = df100.iloc[:, 1:]
    df1000 = pd.read_csv(path + 'recipes_encoded1000.csv')
    df1000 = df1000.iloc[:, 1:]
    recipe_info = pd.read_csv(path + 'recipe_info.csv')
    print('Data reading complete.')
    data = hierarchical_cluster(df100, 54)
    print('Saving data...')
    data.iloc[:, -1].to_csv(path + 'dbclusters_min25.csv')