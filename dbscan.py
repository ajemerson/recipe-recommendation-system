import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN as dbizzle
from sklearn.metrics import silhouette_score as silhouette


def dbscan(data, eps, min_samps):
    """
    Perform DBSCAN density-based clustering on the given dataset
    :param data: a pandas dataframe to be clustered
    :param eps: hyperparameter of DBSCAN --- distance between points to be considered members of same neighborhood
    :param min_samps: hyperparameter of DBSCAN --- number of neighbors for a observation to be considered core point
    :return: the same pandas dataframe with clusters attached to each row
    """

    print('Clustering input data with DBSCAN...')
    clustering = dbizzle(eps=eps, min_samples=min_samps).fit(data)
    clusters = clustering.labels_
    print('DBSCAN complete.')
    print(clusters)
    print("The unique clusters:", np.unique(clusters))

    # count the number of noise points in the clustering
    noise_count = 0
    for i in range(len(clusters)):
        if clusters[i] == -1:
            noise_count += 1
    print("Number of noise points:", noise_count)
    # silhouette_coef(data, clustering)

    # append the clustering to the end of the input dataframe
    data['clusters'] = clusters
    return data


def silhouette_coef(d, clustering):
    """
    Measure the performance of any clustering with Silhouette coefficient. Outputs a coefficient in [-1, 1]
    We prefer coefficients close to 1
    :param d: the pandas dataframe before appending cluster labels
    :param clustering: the clustering of the data
    :return coef: the Silhouette Coefficient
    """
    print("Calculating Silhouette Coefficient...")
    labels = clustering.labels_
    coef = silhouette(d, labels)
    print('Silhouette Coefficient of clustering:', coef)
    return coef


if __name__ == "__main__":
    path = "../Datasets/"
    print('Reading data...')
    df100 = pd.read_csv(path + 'recipes_encoded100.csv')
    df100 = df100.iloc[:, 1:]
    df1000 = pd.read_csv(path + 'recipes_encoded1000.csv')
    df1000 = df1000.iloc[:, 1:]
    recipe_info = pd.read_csv(path + 'recipe_info.csv')
    print('Data reading complete.')
    data = dbscan(df100, 0.6, 25)
    print('Saving data...')
    data.iloc[:, -1].to_csv(path + 'dbclusters_min25.csv')
