# -*- coding: utf-8 -*-
"""
Spyder Editor

This is will combine all data.csv files
"""

def combine_data():
    fout=open("preparsed_data.csv","a")
    # first file:
    for line in open("data.csv"):
        fout.write(line)
    # now the rest:    
    for num in range(2, 6):
        f = open("data"+str(num)+".csv", "r+")
        f.readline() # skip the header
        for line in f:
             fout.write(line)
        f.close()
    fout.close()

if __name__ == '__main__':
    combine_data()
