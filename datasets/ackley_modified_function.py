import numpy as np
import torch
import os
from botorch.test_functions import Ackley
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

# name
name = 'Ackley modified function'
name_saving = 'ackley_modified'

# system and simulation parameters
dimx = 1
noisestd = 0.1
ndata0 = 20
n_bo_iter = 1500
offset = 12 # normalizing offset
max_val=offset  # Known maximum objective for negative Ackley function.

ackley = Ackley(dim=dimx).to(**tkwargs)
def eval_fun(x):
    """x is assumed to be in [0, 1]^d"""
    lb, ub = ackley.bounds
    return offset - ackley(lb + (ub - lb) * x[..., :2])

def get_noisy_data(x):
    """x is assumed to be in [0, 1]^d"""
    lb, ub = ackley.bounds
    output = offset - ackley(lb + (ub - lb) * x[..., :2])
    noise = noisestd * np.random.normal(size=output.size())
    return output + noise

def get_starting_data():
    dy = noisestd * np.random.normal(size=ndata0)
    X0 = np.random.rand(ndata0,dimx)
    y0 = eval_fun(X0).numpy() + dy
    return X0, y0


# ----------------------------------------------------------------------
# GP kernel hyperparameter bounds
ub = 1*torch.ones(2+dimx)
lb = 1e-9*torch.ones(2+dimx)
lb[-1] = 1e-3
ub[-1] = 1e-1
lb[0] = 1e-1
ub[0] = 50