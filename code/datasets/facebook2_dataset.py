from scipy.io import arff
import numpy as np
import torch
import pandas as pd
from sklearn import preprocessing



# name
name = 'facebook2'
name_saving = 'facebook2'
training_iterations = 400 # 10000
# retrain GP using calibratoin data set?
retrain = False
ratio = 0.75
# size of subset used to train log likelihood maximum hyperparameters
training_subset_size = 100000
sparse = True
n_inducing_points = 400
# deltas = [0, 0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975, 1] #


data = arff.loadarff('datasets/FacebookCommentDataset/Dataset/Training/Features_Variant_2.arff')
df = pd.DataFrame(data[0])

# system and simulation parameters
X_tot = df.to_numpy()[:,:-1]
y_tot = df.to_numpy()[:,-1]
# # X_tot, y_tot = load_boston(return_X_y=True)
# # preprocessing
scaler = preprocessing.StandardScaler().fit(X_tot)
X_tot = scaler.transform(X_tot)
y_tot= np.array(y_tot-y_tot.mean(), dtype=float)
y_tot = y_tot/y_tot.std()

dimx = X_tot.shape[1]
ndata_max = X_tot.shape[0]
ndata_min = np.multiply(X_tot.shape[0], 0.1).__int__()
datasizes = torch.linspace(ndata_min, ndata_max, 2, dtype=int)
chol_jitter = 1e-2 # higher value required for Cholesky jitter during fully Bayesian training

def get_data(ndata):
    perm = list(np.random.permutation(list(range(X_tot.shape[0]))))
    X_data = X_tot[perm[0:ndata]]
    y_data = y_tot[perm[0:ndata]]
    # dy = 0.1 * np.random.random(y_data.shape)
    #noise = np.random.normal(0, dy)
    #y_data += noise

    X_test = X_tot[perm[ndata + 1:-1]]
    f_true = y_tot[perm[ndata + 1:-1]]
    return X_data, y_data, X_test, f_true


# ----------------------------------------------------------------------
# GP kernel hyperparameter bounds
ub = 1 * torch.ones(dimx + 2)
lb = 1e-12 * torch.ones(dimx + 2)
lb[-1] = 1e-1
# ub[-1] = 10
ub[-1] = 10
lb[0] = 1e-1
ub[0] = 5