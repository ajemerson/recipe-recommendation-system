from tkinter import *

class choice_gui:
    recipe_list = []
    choices = []
    vars = {}
    checks = {}
    def __init__(self, master, recipe_list):
        self.master = master
        self.recipe_list = recipe_list
        master.title("A simple GUI")

        self.label = Label(master, text="Choose which recipes you like!")
        self.label.pack()

        num = len(recipe_list)
        for i in range(num):
            self.vars["var" + str(i)] = IntVar()
            self.checkbox = Checkbutton(self.master, text=recipe_list[i], variable=self.vars["var" + str(i)],
                                        onvalue=1, offvalue=0)
            self.checkbox.pack()
            self.checks["check" + str(i)] = self.checkbox

        self.close_button = Button(master, text="Done", command=self.save_choice)
        self.close_button.pack()

    def save_choice(self):
        num = len(self.vars)
        for i in range(num):
            if self.vars["var" + str(i)].get():
                self.choices.append(self.recipe_list[i])
        # Prints the choices --- the recipes checked by the user
        print(self.choices)
        # In the future, trigger next sampling of recipes and create new gui
        self.master.quit()

root = Tk()

# An example list of recipes
# Eventually, sample list of recipes
rlist = ["Rec1", "Rec2", "Rec3"]
my_gui = choice_gui(root, rlist)
root.mainloop()