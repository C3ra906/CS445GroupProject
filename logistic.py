#CS545 Group Project. Logistic Regression

import pandas as pd
import numpy as np

#Data Path
path = './healthcare-dataset-stroke-data.csv'

#Function that loads csv data files
def csv_load(file_path):
    with open(file_path, 'r') as csv_file:
        dt = np.genfromtxt(file_path, delimiter=',',skip_header=1, dtype=None, encoding='ascii')

    return dt

#TODO Function to parse data into training and test set


#TODO Logisitc function
def logistic():

   #F = 1/(1+np.exp(-w dot x))
   return

#TODO Maximum Likelihood Estimate for parameters
def MLE():
    #wi = wi + n (sum (yi - sigma(w dot xi))xi
    return 


def main():
    data = csv_load(path)
    print(data)

if __name__ == '__main__':
    main()