import numpy as np
import torch
import pandas as pd
from sklearn import preprocessing


# name
name = 'Boston house prices'
name_saving = 'bostonprices'

df=pd.read_csv('datasets/Boston.csv', sep=',')

# system and simulation parameters
X_tot = pd.DataFrame(data=df[list(df.columns)[1:-1]]).values
y_tot = pd.DataFrame(data=df['medv']).values
y_tot = np.reshape(y_tot, [506,])

nmc = 1000
train_full_bayes = False
dimx = X_tot.shape[1]
ndata_max = 450
ndata_min = 404
num_samples = 100 #100
warmup_steps = 100 #100
datasizes = torch.linspace(ndata_min, ndata_max, 2, dtype=int)

mean_inputs = X_tot.mean(0)
std_inputs = X_tot.std(0)
mean_targets = y_tot.mean(0)
std_targets = y_tot.std(0)

def get_data(ndata):
    perm = list(np.random.permutation(list(range(506))))

    X_data = X_tot[perm[0:ndata]]
    y_data = y_tot[perm[0:ndata]]
    dy = 0.1 * np.random.random(y_data.shape)
    noise = np.random.normal(0, dy)
    y_data += noise

    X_test = X_tot[perm[ndata + 1:-1]]
    f_true = y_tot[perm[ndata + 1:-1]]

    X_data_processed = (X_data - mean_inputs)/std_inputs
    X_test_processed = (X_test - mean_inputs) / std_inputs
    y_data_processed = (y_data - mean_targets) / std_targets
    f_true_processed = (f_true - mean_targets) / std_targets

    return X_data_processed, y_data_processed, X_test_processed, f_true_processed


# ----------------------------------------------------------------------
# GP kernel hyperparameter bounds
ub = 7*torch.ones(15)
lb = 1e-9*torch.ones(15)
lb[-1] = 1e-3
lb[0] = 1e-3
ub[0] = 40