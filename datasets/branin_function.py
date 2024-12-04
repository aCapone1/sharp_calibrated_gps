import numpy as np
import torch
import os
from botorch.test_functions import Branin
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

# name
name = 'Branin function'
name_saving = 'branin'

# system and simulation parameters
dimx = 2
noisestd = 0.1
ndata0 = 10
n_bo_iter = 2000
max_val=0.397887  # Known maximum objective for negative Branin function.

branin = Branin().to(**tkwargs)
def eval_fun(x):
    """x is assumed to be in [0, 1]^d"""
    lb, ub = branin.bounds
    return - branin(lb + (ub - lb) * x[..., :2])

def get_noisy_data(x):
    """x is assumed to be in [0, 1]^d"""
    lb, ub = branin.bounds
    output = - branin(lb + (ub - lb) * x[..., :2])
    noise = noisestd * np.random.normal(size=output.size())
    return output + noise

def get_starting_data():
    dy = noisestd * np.random.normal(size=ndata0)
    X0 = np.random.rand(ndata0,2)
    y0 = eval_fun(X0).numpy() + dy
    return X0, y0


# ----------------------------------------------------------------------
# GP kernel hyperparameter bounds
ub = 10*torch.ones(4)
lb = 1e-2*torch.ones(4)
lb[-1] = 1e-1
lb[0] = 1e-1
ub[0] = 50