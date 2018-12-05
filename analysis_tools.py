import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    Draws an ellipse with a given position and covariance (for clustering)
    """
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2*np.sqrt(s)
    else:
        angle = 0
        width, height = 2*np.sqrt(covariance)
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig*width, nsig*height, angle, **kwargs))


def plot_gmm(gmm, x, label=True, ax=None):
    """
    Ex:
    gmm = GaussianMixture(n_components=3).fit(X)
    plot_gmm(gmm, X)
    """
    ax = ax or plt.gca()
    labels = gmm.fit(x).predict(x)
    if label:
        ax.scatter(x[0], x[1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(x[0], x[1], s=40, zorder=2)
    ax.axis('equal')
    w_factor = 0.2/gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w*w_factor)


def get_pca_components(x, num_components=2, incremental=False, batch_size=100):
    """
    Converts input data into its reduced PCA components. Can be used to visualize in 2D.
    Set incremental to true if memory is an issue (i.e., large dataset)
    :param x: n-dimensional input data to be reduced to d-dimensional
    :param num_components: default=2, d-dimensions
    :param incremental: true for large datasets, memory issues
    :param batch_size: refers to the number of rows processed per iteration
    :return: Dataframe of the resulting components.
    """
    x_standardized = StandardScaler().fit_transform(x)
    if incremental:
        pca = IncrementalPCA(n_components=num_components, batch_size=batch_size)
    else:
        pca = PCA(n_components=num_components)
    x_pca = pca.fit_transform(x_standardized)
    x_pca = pd.DataFrame(data=x_pca)
    return x_pca


def get_cluster_assignments(x, num_clusters):
    """
    Quick way to retrieve cluster labels directly using k_means, passing value k
    :param x: type dataframe
    :param num_clusters: can be based on the visualization of first 2 principal components
    :return: labels/cluster assignments
    """
    km = KMeans(n_clusters=num_clusters).fit(x)
    return km.labels_


def scale(df, scale_type="standardize"):
    """
    :param df: contains only the columns of numerical data, non identifier data
    :param scale_type: (default) 'standardize' = standard scaler (i.e., subtract by mean, divide by std deviation OR
            'normalize' = min max scaler (i.e., on a scale from 0 to 1)
    :return: standardized columns as a dataframe
    """
    scaler = None
    if scale_type == "normalize":
        scaler = preprocessing.MinMaxScaler()
    elif scale_type == "standardize":
        scaler = preprocessing.StandardScaler()
    scaler.fit_transform(df)
    return scaler.transform(df)


def variance_threshold_fs(x, threshold=0.0):
    """
    Removes all features below certain variance level (default is 0 variance)
    :param x: all numerical data features
    :param threshold: variance level
    :return: remaining features after removing those under threshold
    """
    fs = VarianceThreshold(threshold=threshold)
    fs.fit(x)
    return x[x.columns[fs.get_support(indices=True)]]
