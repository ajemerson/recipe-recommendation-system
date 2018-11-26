from tkinter import *
import pandas as pd
import analysis_tools as at


class RateGui:
    choices = {}
    vars = {}

    def __init__(self, master, recipe_list):
        self.master = master
        self.label = Label(master, text="Rate each recipe")
        self.label.grid()
        self.frm = Frame(root, bd=16, relief='sunken')
        self.frm.grid()
        # create the rating gui
        self.recipe_list = recipe_list
        self.rate_gui(self.frm, self.recipe_list)
        self.close_button = Button(master, text="Done", command=self.save_choice)
        self.close_button.grid()

    def rate_gui(self, frm, recipe_list):

        num = len(recipe_list)  # number of recipes in the list to rate

        label_place = 0
        for i in range(num):
            rec = Label(frm, text=recipe_list[i])
            rec.grid(row=label_place, column=0)
            self.vars["0" + str(i)] = StringVar()
            self.vars["1" + str(i)] = StringVar()
            self.vars["2" + str(i)] = StringVar()
            self.vars["3" + str(i)] = StringVar()
            self.vars["4" + str(i)] = StringVar()
            self.vars["5" + str(i)] = StringVar()

            zero = Radiobutton(frm, text='0', variable=self.vars["0" + str(i)])
            zero.config(indicatoron=0, bd=4, width=5, value='0')
            zero.grid(row=label_place+1, column=0)

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
            label_place+=2

    def save_choice(self):
        num = len(self.recipe_list)
        for i in range(num):
            for j in range(6): # for each button
                if self.vars[str(j) + str(i)].get():
                    self.choices[self.recipe_list[i]] = int(self.vars[str(j) + str(i)].get())
        # Prints the choices --- the recipes checked by the user
        print(self.choices)
        # In the future, trigger next sampling of recipes and create new gui
        self.master.quit()


if __name__ == "__main__":
    recipe_data = pd.read_csv('cleaned_recipe_data.csv')
    root = Tk()
    # rlist = ["Rec1", "Rec2", "Rec3", "Rec4", "Rec5"]
    rlist = recipe_data.sample(10).name.values
    my_gui = RateGui(root, rlist)
    root.mainloop()
