import numpy as np
import torch
import os
from botorch.test_functions import Rosenbrock
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

SMOKE_TEST = os.environ.get("SMOKE_TEST")

WARMUP_STEPS = 512 if not SMOKE_TEST else 32
NUM_SAMPLES = 256 if not SMOKE_TEST else 16
THINNING = 16


# name
name = 'Rosenbrock function'
name_saving = 'rosenbrock'

# system and simulation parameters
dimx = 2
noisestd = 0.01
ndata0 = 100
n_bo_iter = 1500
offset = 133000
max_val= offset  # Known maximum objective for negative Rosenbrock function.

rosenbrock = Rosenbrock().to(**tkwargs)
def eval_fun(x):
    """x is assumed to be in [0, 1]^d"""
    lb, ub = rosenbrock.bounds
    f_unscaled = offset - rosenbrock(lb + (ub - lb) * x[..., :2])
    return f_unscaled

def get_noisy_data(x):
    """x is assumed to be in [0, 1]^d"""
    lb, ub = rosenbrock.bounds
    f_unscaled = offset - rosenbrock(lb + (ub - lb) * x[..., :2])
    noise = noisestd * np.random.normal(size=f_unscaled.size())
    return f_unscaled + noise

def get_starting_data():
    dy = noisestd * np.random.normal(size=ndata0)
    X0 = np.random.rand(ndata0,2)
    y0 = eval_fun(X0).numpy() + dy
    return X0, y0


# ----------------------------------------------------------------------
# GP kernel hyperparameter bounds
ub = 1e-1*torch.ones(4)
lb = 1e-9*torch.ones(4)
lb[-1] = 1e-9
lb[0] = 1e-1
ub[-1] = 1e-1
ub[0] = 1e10


# set new deltas
deltas = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]