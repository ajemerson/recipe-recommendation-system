import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from keras.layers import Input, Dense
from keras.models import Model
from keras import losses
from sklearn.cluster import DBSCAN as dbizzle
from sklearn.neighbors import NearestNeighbors


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


def extra_trees_fs(x, y, n_estimators=50):
    """
    Ensemble tree-based method to choose most useful features (feature importance)
    evaluated by predicting a given label
    :param x: all numerical data features
    :param y: label wished to predict (will be encoded)
    :param n_estimators: number of ensemble trees to use
    :return: remaining features after removing irrelevant features (using SelectFromModel)
    """
    le = preprocessing.LabelEncoder()
    le.fit(y)
    labels = le.transform(y)
    clf = ExtraTreesClassifier(n_estimators=n_estimators)
    clf = clf.fit(x, labels)
    model = SelectFromModel(clf, prefit=True)
    return x[x.columns[model.get_support(indices=True)]]


def autoencoder(data, dim_enc1, dim_enc2):
    """
    Trains a stacked autoencoder with two hidden layers. Reduces data
    to the size dim_enc2 and saves the reduced dataset to a csv
    :param data: a pandas dataframe 
    :param dim_enc1: dimension of first hidden layer of autoencoder. First dimensionality reduction
    :param dim_enc2: dimension of last hidden layer of autoencoder. Last dimensionality reduction and output data size
    :return: reduced, encoded dataset
    """
    in_size = data.shape[1]
    print('Number of attributes in dataset:', in_size)

    # placeholder for input vector
    input_vect = Input(shape=(in_size, ))
    # placeholder for encoding
    encoded = Dense(dim_enc1, activation='relu')(input_vect)
    # encoded = Dense(dim_enc2, activation='relu')(encoded)
    # placeholder for decoding
    # decoded = Dense(dim_enc1, activation='relu')(encoded)
    decoded = Dense(in_size, activation='sigmoid')(encoded)

    # full autoencoder placeholder
    auto_model = Model(input_vect, decoded)
    # encoder portion of autoencoder
    encoder = Model(input_vect, encoded)

    # the reconstructed representation at the end of the autoencoder

    auto_model.compile(optimizer='adadelta', loss=losses.mean_squared_error)

    # Training
    auto_model.fit(data, data, epochs=15, batch_size=256, shuffle=True)
    # Extract encoded data
    enc_data = pd.DataFrame(encoder.predict(data))

    print('Saving encoded data...')
    enc_data.to_csv('recipes_encoded1000.csv')
    print('Encoded data was successfully saved to a csv')
    return enc_data


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

    # append the clustering to the end of the input dataframe
    data['clusters'] = clusters
    return data


def cluster_sampling(data, c_type, w=[], c={}):
    """
    Based on the clustering type that is input, a uniform distribution among
    the clustering is returned.
    :param c_type: an integer representing the clustering type. For simplicity,
        0: 'e100_gmm54'
        1: 'e100_gmm25'
        2: 't186_gmm4'
    :return c: a list of 10 cluster numbers based on the choice of a uniform distribution
    :return clusters: a dictionary of clusters where each key is the cluster number and each value is a dataframe
        containing all datapoints from only that cluster
    """
    choices = ['e100_gmm54', 'e100_gmm25', 't186_gmm4']
    # retrieve the corresponding column of the dataset
    col = data[choices[c_type]]
    num_clusters = np.max(col)
    if len(w) == 0:
        weights = []
        # calculate the weights and separate the clusters
        for j in range(num_clusters + 1):
            weights.append(1 / (num_clusters + 1))
    else:
        weights = w
    if len(c) == 0:
        clusters = {}
        for j in range(num_clusters + 1):
            # keys correspond to cluster number, values correspond to dataframe with only observations of that cluster
            clusters[str(j)] = data.loc[data[choices[c_type]] == j]
    else:
        clusters = c

    # Now we can sample a cluster number based on the initial weights
    print("Keys:", list(clusters.keys()))
    print("Weights:", weights)
    sampled_clusters = np.random.choice(list(clusters.keys()), 10, p=weights)
    print("Clusters to sample from:", sampled_clusters)
    return weights, sampled_clusters, clusters


def sample_from_cluster(choices, clusters):
    """
    Performs random sampling for each given cluster
    :param choices: the cluster numbers (chosen by a weighted sampling) on which to perform a random sampling
    :param clusters: a dictionary where each key is the cluster number and each value is a dataframe with
        observations from only that cluster
    :return: len(choices) number of observations
    """
    size = len(choices)
    rlist = []
    index_list = []
    for i in range(size):
        cluster = clusters[str(choices[i])]  # the dataframe of that cluster only
        sample = cluster.sample(1)
        recipe = sample.name.values
        index = cluster.loc[cluster['name'] == recipe[0]].index[0]
        # make sure that there aren't any repeats in that sample
        if recipe[0] not in rlist:
            rlist.append(recipe[0])
            index_list.append(index)
            print(recipe[0], ':', choices[i])
        else:
            i -= 1
    return rlist, index_list


def reweight(w, rate_dict, choices, clusters):
    """
    Helps the system converge to a preference by the user
    :param w: the vector of weights corresponding to each cluster
    :param rate_dict: a dictionary with 10 recipes as keys and 10 ratings as values
    :param choices: a list of 10 values each corresponding to a cluster. Recipe i is from the cluster corresponding to
        the ith index of this list
    :param clusters: a dictionary where each key is the cluster number and each value is the dataframe of only that
        cluster's observations
    :return: a new vector of weights based upon the ratings given by the user
    """
    ratings = np.array(list(rate_dict.values()))
    print("Ratings:", ratings)
    print("Clusters corresponding to recipe:", choices)
    choices = np.array([int(i) for i in choices])  # convert entries to integers
    tot_clusts = len(clusters)
    print('Total Number of Clusters:', tot_clusts)
    divs = np.zeros(tot_clusts)  # dividends used to average scores of clusters that have more than one recipe present
    tot_scores = np.zeros(tot_clusts)  # the total scores before averaging
    for i in range(tot_clusts):
        divs[i] = np.sum(choices == i)  # allows us to average the ratings corresponding to the recipes' clusters
        indices = choices == i  # a 1 will be at the index where the recipe belongs to the ith cluster
        tot_scores[i] = np.dot(indices, ratings)
    # Perform an element-wise division
    avg_scores = np.divide(tot_scores, divs, out=np.zeros_like(tot_scores), where=divs != 0)
    print("Dividends:", divs)
    print("Total Scores:", tot_scores)
    print("Average Scores:", avg_scores)

    # This will need to be re-normalized
    new_w = avg_scores * w  # element-wise product of average_scores and the old weight vector will give us the new w

    # Establish a vector based on presence of clusters in the recipe sample
    presence = np.zeros(tot_clusts)
    for i in range(tot_clusts):
        if i in choices:
            presence[i] = 1

    new_w *= presence  # insures that clusters that are not present will not be re-weighted
    norm_to = np.dot(w, presence)  # normalize only with respect to the present clusters' weights
    print("Clusters Present:", presence)
    new_w = (new_w / np.sum(new_w)) * norm_to
    new_w += w * (presence == 0)  # add back in the old weights of absent clusters (these are unaffected)

    print(new_w)
    print(np.sum(new_w))

    return new_w


def find_info(rec_rate, ind_list, clustering, recipe_info, clusters):
    """
    Get the info corresponding the the highest-rated recipe. Cluster number and index of recipe
    :param rec_rate: a dictionary where each key is a recipe and each value is the rating. The final pass.
    :param ind_list: the list of indices of each recipe in the original dataset
    :param clustering: the clustering that is being used --- again indexed by integer for simplicity
        0: e100_gmm54
        1: e100_gmm25
        2: t186_gmm4
    :param recipe_info: the recipe_info.csv dataframe with recipe names, types, subtypes, and corresponding clusters
    :param clusters: dictionary of subsetted clusters. Key corresponding to cluster number
    :return: the subset of data that contains FULL observations corresponding to only that cluster number
    :return index: the index in the original dataset of the highest rated recipe
    """
    # Find the highest rated recipe
    max_rating = 0
    max_recipe = None
    max_index = None
    size = len(rec_rate)
    ratings = list(rec_rate.values())
    recipes = list(rec_rate.keys())
    for i in range(size):
        if ratings[i] > max_rating:
            max_rating = ratings[i]
            max_recipe = recipes[i]
            max_index = ind_list[i]

    # Now, we have the highest rated recipe and its index in the original dataset
    # Find its corresponding cluster
    c_list = ['e100_gmm54', 'e100_gmm25', 't186_gmm4']
    clustering = c_list[clustering]  # convert from integer interpretation of clustering to string
    print("Clustering being used:", clustering)
    print('Max Index:', max_index)
    cluster = recipe_info.loc[max_index, clustering]
    print("The highest-rated recipe:", max_recipe)
    print("The recipe's cluster:", cluster)
    print("The recipe's index in the dataset:", max_index)

    if clustering.startswith('e'):
        data = pd.read_csv('Data/recipes_encoded100.csv').iloc[:, 1:]
    else:
        data = pd.read_csv('Data/recipes_normalized_varFS_extraTrees186.csv').iloc[:, 1:]
    # subset the data
    # get relevant rows in list format
    row_list = list(clusters[str(cluster)].index)
    print(recipe_info.loc[row_list, ['name', clustering]])
    # return a dataframe of only those rows
    subset = data.loc[row_list, :]

    return subset, max_index


def get_closest_neighbors(recipe_index, recipe_cluster, n_neighbors=10):
    """
    Will find the closest neighbors to the given recipe as a final recommendation
    :param recipe_index: the single highest rated recipe in the last round of ratings
    :param recipe_cluster: all recipes belonging to the same cluster as the highest rated recipe
    :param n_neighbors: number of closest recipes desired
    :return: the list of unique recipes closest to the highest rated recipe
    """
    recipe_info = pd.read_csv("Data/recipe_info.csv").iloc[:, 1:]
    knn = NearestNeighbors(n_neighbors=n_neighbors)

    if len(recipe_cluster) <= 10:
        return np.unique(recipe_info.name[list(recipe_cluster.index)])

    cluster_recipe_info = recipe_info.iloc[list(recipe_cluster.index)]
    cluster_recipe_info = cluster_recipe_info.reset_index()
    print("Cluster subset: ", cluster_recipe_info)
    print("Fitting KNN")
    knn.fit(recipe_cluster)
    print("Fitting complete")
    print("Index: ", recipe_index)
    neighbor_indices = knn.kneighbors(recipe_cluster.loc[recipe_index].values.reshape(1, -1))[1][0]
    print("Neighbor indices: ", neighbor_indices)
    recommendations = cluster_recipe_info.name[neighbor_indices].values
    return np.unique(recommendations)

