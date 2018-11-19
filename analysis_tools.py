import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    Draws an ellipse with a given position and covariance (for clustering)
    :param position:
    :param covariance:
    :param ax:
    :param kwargs:
    :return:
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


def plot_gmm(gmm, X, label=True, ax=None):
    """
    Ex:
    gmm = GaussianMixture(n_components=3).fit(X)
    plot_gmm(gmm, X)
    """
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[0], X[1], c=labels, s=40, cmap='viridis', zorder =2)
    else:
        ax.scatter(X[0], X[1], s=40, zorder=2)
    ax.axis('equal')
    w_factor = 0.2/gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w*w_factor)


def get_pca_components(X, num_components=2):
    """
    Converts input data into its reduced PCA components. Can be used to visualize in 2D.
    :param X: n-dimensional input data to be reduced to d-dimensional
    :param num_components: default=2, d-dimensions
    :return: Dataframe of the resulting components.
    """
    X_standardized = StandardScaler().fit_transform(X)
    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X_standardized)
    X_pca = pd.DataFrame(X_pca)
    return X_pca


def cluster(X, num_clusters):
    """
    Quick way to retrieve cluster labels directly using k_means, passing value k
    :param X: type dataframe
    :param num_clusters: can be based on the visualization of first 2 principal components
    :return: labels/cluster assignments
    """
    km = KMeans(n_clusters=num_clusters).fit(X)
    return km.labels_

