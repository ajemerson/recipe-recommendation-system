from tkinter import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

global subdict

class RateGui:
    choices = {}
    vars = {}

    def __init__(self, master, recipe_list, counter):
        global subdict
        subdict = {}
        self.master = master
        self.label = Label(master, text="Of the following, what are you in the mood for?")
        self.label.grid()
        self.frm = Frame(root, bd=16, relief='sunken')
        self.frm.grid()
        # create the rating gui
        self.recipe_list = recipe_list
        self.rate_gui(self.frm, self.recipe_list)
        self.close_button = Button(master, text="Done", command=self.save_choice)
        self.close_button.grid()
        self.counter = counter

    def rate_gui(self, frm, recipe_list):

        num = len(recipe_list)  # number of recipes in the list to rate

        label_place = 0
        for i in range(num):
            rec = Label(frm, text=recipe_list[i])
            rec.grid(row=label_place, column=0)
            # self.vars["0" + str(i)] = StringVar()
            self.vars["1" + str(i)] = StringVar()
            self.vars["2" + str(i)] = StringVar()
            self.vars["3" + str(i)] = StringVar()
            self.vars["4" + str(i)] = StringVar()
            self.vars["5" + str(i)] = StringVar()

            # zero = Radiobutton(frm, text='0', variable=self.vars["0" + str(i)])
            # zero.config(indicatoron=0, bd=4, width=5, value='0')
            # zero.grid(row=label_place+1, column=0)

            one = Radiobutton(frm, text='1', variable=self.vars["1" + str(i)])
            one.config(indicatoron=0, bd=4, width=5, value='1')
            one.grid(row=label_place+1, column=1)

            two = Radiobutton(frm, text='2', variable=self.vars["2" + str(i)])
            two.config(indicatoron=0, bd=4, width=5, value='2')
            two.grid(row=label_place+1, column=2)

            three = Radiobutton(frm, text='3', variable=self.vars["3" + str(i)])
            three.config(indicatoron=0, bd=4, width=5, value='3')
            three.grid(row=label_place+1, column=3)

            four = Radiobutton(frm, text='4', variable=self.vars["4" + str(i)])
            four.config(indicatoron=0, bd=4, width=5, value='4')
            four.grid(row=label_place+1, column=4)

            five = Radiobutton(frm, text='5', variable=self.vars["5" + str(i)])
            five.config(indicatoron=0, bd=4, width=5, value='5')
            five.grid(row=label_place+1, column=5)
            label_place += 2

    def save_choice(self):
        global subdict
        num = len(self.recipe_list)
        for i in range(num):
            for j in range(5):  # for each button
                if self.vars[str(j+1) + str(i)].get():
                    self.choices[self.recipe_list[i]] = int(self.vars[str(j+1) + str(i)].get())
        # Prints the choices --- the recipes checked by the user
        keys = list(self.choices.keys())[4*self.counter:]
        subdict = {x: self.choices[x] for x in keys if x in self.choices}
        self.master.quit()


def init_sampling(data, c_type):
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
    weights = []
    clusters = {}
    # calculate the weights and separate the clusters
    for j in range(num_clusters + 1):
        weights.append(1 / (num_clusters + 1))
    for j in range(num_clusters + 1):
        # keys correspond to cluster number, values correspond to dataframe with only observations of that cluster
        clusters[str(j)] = data.loc[data[choices[c_type]] == j]

    # Now we can sample a cluster number based on the initial weights
    print("Keys:", list(clusters.keys()))
    print("Weights:", weights)
    c = np.random.choice(list(clusters.keys()), 10, p=weights)
    print("Clusters to sample from:", c)
    return weights, c, clusters


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
    for i in range(size):
        sample = clusters[str(choices[i])].sample(1)
        recipe = sample.name.values
        # make sure that there aren't any repeats in that sample
        if recipe[0] not in rlist:
            rlist.append(recipe[0])
            print(recipe[0], ':', choices[i])
        else:
            i -= 1
    return rlist


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
    avg_scores = np.divide(tot_scores, divs, out=np.zeros_like(tot_scores), where=divs!=0)
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


if __name__ == "__main__":
    recipe_data = pd.read_csv('Data/recipe_info.csv').iloc[:, 1:]
    # obtain the weights of sampling from each cluster based on the clustering we want to use
    init_w, choices, clusters = init_sampling(recipe_data, 2)
    init_rlist = sample_from_cluster(choices, clusters)
    # for i in range(3):
    root = Tk()
    rlist = init_rlist
    # rlist = recipe_data.sample(10).name.values
    my_gui = RateGui(root, rlist, 0)  # once we figure out how to update weights based on rankings, change 0 to i because we will loop some number of times
    root.mainloop()
    root.destroy()

    # begin the resampling
    w = init_w
    # print('\n', subdict)
    for i in range(1):
        w = reweight(w, subdict, choices, clusters)
