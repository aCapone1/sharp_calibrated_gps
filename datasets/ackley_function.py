import numpy as np
import torch
import os
from botorch.test_functions import Ackley
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

# name
name = 'Ackley function'
name_saving = 'ackley'

# system and simulation parameters
dimx = 2
noisestd = 0.1
ndata0 = 20
n_bo_iter = 1500
max_val=0  # Known maximum objective for negative Ackley function.

ackley = Ackley().to(**tkwargs)
def eval_fun(x):
    """x is assumed to be in [0, 1]^d"""
    lb, ub = ackley.bounds
    return - ackley(lb + (ub - lb) * x[..., :2])

def get_noisy_data(x):
    """x is assumed to be in [0, 1]^d"""
    lb, ub = ackley.bounds
    output = - ackley(lb + (ub - lb) * x[..., :2])
    noise = noisestd * np.random.normal(size=output.size())
    return output + noise

def get_starting_data():
    dy = noisestd * np.random.normal(size=ndata0)
    X0 = np.random.rand(ndata0,2)
    y0 = eval_fun(X0).numpy() + dy
    return X0, y0


# ----------------------------------------------------------------------
# GP kernel hyperparameter bounds
ub = 1*torch.ones(2+dimx)
lb = 1e-9*torch.ones(2+dimx)
lb[-1] = 1e-2
ub[-1] = 1e-1
lb[0] = 1e-1
ub[0] = 50

# set new deltas
deltas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
          0.5, 0.55, 0.6, 0.65, 0.7, 0.75]