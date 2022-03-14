import pandas as pd
import numpy as np
import scipy.stats as sc

# load data
data = pd.read_csv("healthcare-dataset-stroke-data.csv")

# replace categorical values with numerical
data['gender'].replace(['Male', 'Female', 'Other'], [0,1,2], inplace = True)
data['ever_married'].replace(['No', 'Yes'], [0,1], inplace = True)
data['work_type'].replace(['Private', 'Self-employed','Govt_job', 'children', 'Never_worked'],[0,1,2,3,4], inplace = True)
data['Residence_type'].replace(['Rural','Urban'], [0,1], inplace = True)
data['smoking_status'].replace(['never smoked', 'Unknown', 'formerly smoked', 'smokes'],[0,1,2,3], inplace = True)

# remove rows with nan
data.dropna(inplace = True)

# remove id column
data.drop(columns = 'id')

# create weights (will be replaced with actual when available)
weights = np.random.uniform(size = (data.shape[1]-1))

# create probabilities of success (will be replaced with actual when available)
probs = np.random.uniform(size = data.shape[0])

# calculate covariance matrix of regression coefficients
X = np.array(data.iloc[:,:-1]) # design matrix
V = np.diag(probs*(1-probs))
cov_mat = np.linalg.inv(np.dot(np.dot(X.transpose(),V),X))

# extract the square root of the diagonal of the covariance matrix
st_errors = np.sqrt(np.diag(cov_mat))

# divide coefficients by their standard errors
test_stats = weights/st_errors

# test stats on standard normal distribution to get p-values
sc.norm.cdf(abs(test_stats))
