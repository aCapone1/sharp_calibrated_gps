import numpy as np
import torch
import pandas as pd
from sklearn import preprocessing


# name
name = 'Cement'
name_saving = 'cement'

cementdata = pd.read_excel('datasets/Concrete_Data.xls').to_numpy()
# system and simulation parameters
# car/model names are ignored
X_tot = cementdata[:,:-1]
y_tot = cementdata[:,-1]
# X_tot, y_tot = load_boston(return_X_y=True)
# preprocessing
scaler = preprocessing.StandardScaler().fit(X_tot)
X_tot = scaler.transform(X_tot)
y_tot= np.array(y_tot-y_tot.mean(), dtype=float)
y_tot = y_tot/y_tot.std()
dimx = X_tot.shape[1]
ndata_max = X_tot.shape[0]
ndata_min = np.multiply(X_tot.shape[0],0.8).__int__()
num_samples = 100 #100
warmup_steps = 100 #100
datasizes = torch.linspace(ndata_min, ndata_max, 2, dtype=int)
chol_jitter = 1e-5 # higher value required for Cholesky jitter during fully Bayesian training
ratiosubset = 0.5 # ratio of subset used to train and choose calibration variances

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
# ub = 30*torch.ones(dimx+2)
# lb = 1e-6*torch.ones(dimx+2)
# lb[-1] = 1e-1
# lb[0] = 1e-1
# ub[0] = 1e2
ub = 20 * torch.ones(dimx + 2)
lb = 1e-2 * torch.ones(dimx + 2)
lb[-1] = 1e-1
# ub[-1] = 10
lb[0] = 1e-5
ub[0] = 25
