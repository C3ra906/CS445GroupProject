import numpy as np
import scipy.stats as sc
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import sklearn.metrics

# to run this script, the training data and weights are needed from 
# the logistic_oversample script
init_weights, w_changes = initialize_weights()
train_set, test_set, train_labels, test_labels = preprocess_data(path, 0)
weights = MLE(train_set, train_labels, init_weights, w_changes)

# calculate probabiities of positive class
probs = logistic(weights, train_set).T

# calculate covariance matrix of regression coefficients
X = train_set # design matrix
V = np.diag((probs*(1-probs)).flatten())
cov_mat = np.linalg.inv(np.dot(np.dot(X.transpose(),V),X))

# extract the square root of the diagonal of the covariance matrix
# these are the standard errors of the coefficients
st_errors = np.sqrt(np.diag(cov_mat))

# divide coefficients by their standard errors
test_stats = weights/st_errors

# test stats on standard normal distribution to get p-values
p_values = 1-sc.norm.cdf(abs(test_stats))

# calculate 95% confidence interval for the coefficients
lbound = weights - 1.96*st_errors
ubound = weights + 1.96*st_errors
intervals = np.round(np.column_stack((lbound.T, ubound.T)),2)

### Plots for ROC and DET #####################################################
t_probs = logistic(weights, test_set).T
# do model selection
sklearn.metrics.RocCurveDisplay.from_predictions(test_labels, t_probs, pos_label = 1)
plt.title("Logistic Regression ROC")
plt.savefig("LR ROC.png", dpi = 150)

sklearn.metrics.DetCurveDisplay.from_predictions(test_labels, t_probs, pos_label = 1)
plt.title("Logistic Regression DET")
plt.savefig("LR DET.png", dpi = 150)

### Plots for confusion matrices before and after model selection #############
# plot original confusion matrix before model selection
res = np.dot(test_set, weights.T)
pred = np.ones(res.shape)
pred[res <= 0] = 0

cmat = confusion_matrix(test_labels, pred)

# plot confusion matrix
cmat_plot = sn.heatmap(cmat, annot = True, fmt = ".4g",
            xticklabels = ["Non stroke", "Stroke"],
            yticklabels = ["Non stroke", "Stroke"])

plt.title("Logistic Regression Confusion Matrix")
plt.rc('font', size = 10)
plt.rc('xtick', labelsize = 10)
cmat_plot.set_xlabel("Predicted")
cmat_plot.set_ylabel("Reference")

plt.savefig("cmat.png", dpi = 150)

# do model selection by removing columns that were insignificant
sig_columns = np.where(p_values <= 0.05)[1]
train_set_sub = train_set.iloc[:,sig_columns]
test_set_sub = test_set.iloc[:,sig_columns]

# train weights on subset data
# initialize weights and weight changes arrays
np.random.seed(34)
w = np.random.randint(-5, 5, size=(1, len(sig_columns) -1))/100 #set weights to random number between -0.05 to 0.05
b_0 = np.ones((1, 1))

#Add b_0 to the end of weight matrix since we changed the stroke column to 1s
init_weights_sub = np.append(w, b_0, axis = 1) 
w_changes = np.zeros((1, len(init_weights_sub)))

# train LR on subset data
weights_sub = MLE(train_set_sub, train_labels, init_weights_sub, w_changes)

# plot confusion matrix after model selection
res_sub = np.dot(test_set_sub, weights_sub.T)
pred_sub = np.ones(res_sub.shape)
pred_sub[res_sub <= 0] = 0

cmat_sub = confusion_matrix(test_labels, pred_sub)

cmat_plot_sub = sn.heatmap(cmat_sub, annot = True, fmt = ".4g",
            xticklabels = ["Non stroke", "Stroke"],
            yticklabels = ["Non stroke", "Stroke"])
plt.title("Confusion Matrix after Model Selection")
plt.rc('font', size = 10)
plt.rc('xtick', labelsize = 10)
cmat_plot_sub.set_xlabel("Predicted")
cmat_plot_sub.set_ylabel("Reference")

plt.savefig("cmat_sub.png", dpi = 150)