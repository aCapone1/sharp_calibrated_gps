import torch
import os
from botorch.test_functions import Ackley
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}


# name
name = 'Ackeley one dimensional problem'
name_saving = 'ackleyonedim'
train_full_bayes = True

# system and simulation parameters
dimx = 1
noisestd = 0.1
offset = 12 # normalizing offset
ndata = torch.tensor(40)
ntest = torch.tensor(90)
num_samples = 20
warmup_steps = 20
nreps = 1
chol_jitter = 1e-4 # default cholesky jitter used

ntot = 120
x_tot = torch.linspace(0,1,ntot)
deltas = [0.005, 0.025, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995]
nreps = 1
# ----------------------------------------------------------------------
# set upper and lower bounds for uniform hyperprior
# ub = 7*torch.ones(dimx+2)
# ub[0] = 7
# ub[-1] = 0.13
# lb = 1e-2*torch.ones(dimx+2)
# lb[-1] = 1e-4
# ----------------------------------------------------------------------
# GP kernel hyperparameter bounds
ub = 1*torch.ones(2+dimx)
lb = 1e-9*torch.ones(2+dimx)
lb[-1] = 1e-3
ub[-1] = 1e-1
lb[0] = 1e-1
ub[0] = 50



# perm = list(np.random.permutation(list(range(x_tot.shape[0]))))
# perm[0:9] = 4*[3, 19,  225, 269, 101, 140, 350, 352,              355, 356]
#
ackley = Ackley(dim=dimx).to(**tkwargs)
def eval_fun(x):
    """x is assumed to be in [0, 1]^d"""
    lb_ack, ub_ack = ackley.bounds
    return - offset + ackley(lb_ack + (ub_ack - lb_ack) * x[..., :2])


def get_data(ndata):
    # Create test set and condition GP model on it
    f_true = eval_fun(x_tot.reshape(ntot,1)).to(dtype=torch.float32)
    # torch.sigmoid(100*X_true) - 2*torch.exp(-X_true**2) + torch.sin(5*X_true)
    randperm = torch.randperm(f_true.shape[0])
    # Load training set
    X_data = x_tot[randperm[:ndata]]
    y_data = f_true[randperm[:ndata]]

    X_test = x_tot[randperm[ndata:ndata + ntest].sort()[0]]
    f_true = f_true[randperm[ndata:ndata + ntest].sort()[0]]

    return X_data, y_data, X_test, f_true
