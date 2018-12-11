from tkinter import *
import pandas as pd
import analysis_tools as tools
import numpy as np

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
        keys = list(self.choices.keys())[-10:]  # subsets dictionary to yield only the current recipes
        subdict = {x: self.choices[x] for x in keys if x in self.choices}
        self.master.quit()


class DisplayGUI:
    def __init__(self, master, recommendations):
        self.label = Label(master, text="Recommended Recipes!")
        self.label.grid()
        self.frm = Frame(root, bd=16, relief='sunken')
        self.frm.grid()
        self.recommendations = recommendations
        self.display_gui(self.frm, self.recommendations)
        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.grid()

    def display_gui(self, frm, recommendations):
        label_place = 0
        for i in range(len(recommendations)):
            rec = Label(frm, text=recommendations[i])
            rec.grid(row=label_place, column=0)
            label_place += 2


if __name__ == "__main__":
    recipe_data = pd.read_csv('Data/recipe_info.csv').iloc[:, 1:]
    w = []
    c = {}
    for gui_iteration in range(1):
        # obtain the weights of sampling from each cluster based on the clustering we want to use
        w, choices, clusters = tools.cluster_sampling(recipe_data, 1, w, c)
        rlist, index_list = tools.sample_from_cluster(choices, clusters)
        root = Tk()
        my_gui = RateGui(root, rlist, gui_iteration)
        root.mainloop()
        root.destroy()
        w = tools.reweight(w, subdict, choices, clusters)

    # find the info on the highest rated recipe in the last iteration through the GUI
    cluster_data, max_index = tools.find_info(subdict, index_list, 1, recipe_data, clusters)
    recommendations, neighbor_distances = tools.get_closest_neighbors(max_index, cluster_data)
    print("Recommendations: ", recommendations)
    print("Distances: ", neighbor_distances)
    root = Tk()
    display_gui = DisplayGUI(root, np.unique(recommendations))
    root.mainloop()
    root.destroy()

