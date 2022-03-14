#CS545 Group Project. Logistic Regression

from numpy.lib.shape_base import row_stack
import pandas as pd
import numpy as np
from pandas.core.base import DataError
import scipy.stats as sc
from sklearn import preprocessing

#Data Path
path = './healthcare-dataset-stroke-data.csv'

#Function that loads csv data files
def csv_load(file_path):
   data = pd.read_csv(file_path)

   # replace categorical values with numerical
   data['gender'].replace(['Male', 'Female', 'Other'], [0,1,2], inplace = True)
   data['ever_married'].replace(['No', 'Yes'], [0,1], inplace = True)
   data['work_type'].replace(['Private', 'Self-employed','Govt_job', 'children', 'Never_worked'],[0,1,2,3,4], inplace = True)
   data['Residence_type'].replace(['Rural','Urban'], [0,1], inplace = True)
   data['smoking_status'].replace(['never smoked', 'Unknown', 'formerly smoked', 'smokes'],[0,1,2,3], inplace = True)

   # remove rows with nan
   data.dropna(inplace = True) #This gives us 4909 rows 

   # remove id column
   data = data.drop(columns = 'id')

   #scale data to range 0 to 1
   min_max_scaler = preprocessing.MinMaxScaler()
   data[['gender','age', 'work_type', 'smoking_status', 'avg_glucose_level', 'bmi', 'smoking_status']] = min_max_scaler.fit_transform(data[['gender','age', 'work_type', 'smoking_status', 'avg_glucose_level', 'bmi', 'smoking_status']])

   #save labels as its own vector
   labels = np.copy(data['stroke'])
   data['stroke'].replace([0], [1], inplace = True) #I'm just replacing this entire row with 1s for weight x input multiplication


   return data, labels

#TODO Split test and training data 

#Function that initializes the weights for the input x weight matrix and change matrix to keep track of changes in weights
def initialize_weights():
    #We have 10 features + b_0 and 4909 rows of data
    w = np.random.randint(-5, 5, size=(1, 10))/100 #set weights to random number between -0.05 to 0.05
    b_0 = np.ones((1, 1))

    #add b_0 to the end of weight matrix since we changed the stroke column to 1s
    weights = np.append(w, b_0, axis = 1) 
    w_changes = np.zeros((1, 11))

    return weights, w_changes

#TODO Logisitc function
def logistic(weights, inputs):

   sigma = 1/(1 + np.exp(-1 * np.dot(weights, inputs.T)))

   return sigma

#TODO Maximum Likelihood Estimate for parameters
def MLE(data, labels, weights, w_changes):
    learn = 0.001 #learning rate
    sigma = logistic(weights, data) #gives 1x4909 sigma values
    labels = np.reshape(labels, (4909, 1))
    runs = 0
    comparison = w_changes == weights

    #while (comparison.all() == False): #This loop runs forever
    for n in range(500):
        w_changes = np.copy(weights)
        weights += learn * np.dot((labels.T - sigma), data) #This code may not be working correctly
        sigma = logistic(weights, data)
        comparison = w_changes == weights
        runs += 1
        #print(f"previous weight: {w_changes}")
        #print(f"weights:{weights}")
        #print(f"Run {runs + 1 } Weight change: {weights - w_changes}")
    
    return weights

def predict(data, weights, labels):
    confusion = np.zeros((2,2)) #[TP, FN][FP, TN]
    index = 0
    results = np.dot(data, weights.T)
    for row in results:
        #print(row)
        #print(labels[index])
        if row >= 0:
            if labels[index] == 1:
               confusion[0,0] += 1
               print("stroke - correct")
            else:
               confusion[1,0] += 1
               print("stroke - false")                   
        else:
            if labels[index] == 0:
               confusion[1,1] += 1
               print("Non-stroke - correct")
            else:
               confusion[0,1] += 1
               print("Non-stroke - false") 
        index+= 1

    tp = confusion[0,0]
    print(tp)
    fp = confusion[1,0]
    print(fp)
    tn = confusion[1,1]
    print(tn)
    fn = confusion[0,1]
    print(fn)
    #accuracy = tp/(tp + fp)
    #print(f"Acurracy: {accuracy}")

def main():
    data, labels = csv_load(path)
    #print(data)
    weights, w_changes = initialize_weights()
    #print(weights)
    final_weights = MLE(data, labels, weights,w_changes)
    predict(data, final_weights, labels)


if __name__ == '__main__':
    main()