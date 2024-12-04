import torch

# name
name = 'Toy problem'
name_saving = 'toyproblem'


# system and simulation parameters
dimx = 1
ndata = torch.tensor(30)
ntest = torch.tensor(120)
num_samples = 20
warmup_steps = 20
nreps = 1
chol_jitter = 1e-4 # default cholesky jitter used


x_tot = torch.linspace(-2.5,2.5,120)


# ----------------------------------------------------------------------
# set upper and lower bounds for uniform hyperprior
# ub = 7*torch.ones(dimx+2)
# ub[0] = 7
# ub[-1] = 0.13
# lb = 1e-2*torch.ones(dimx+2)
# lb[-1] = 1e-4
ub = 20 * torch.ones(dimx +2)
lb = 1e-6 * torch.ones(dimx+2)

# perm = list(np.random.permutation(list(range(x_tot.shape[0]))))
# perm[0:9] = 4*[3, 19,  225, 269, 101, 140, 350, 352, 355, 356]
#


def get_data(ndata):
    # Create test set and condition GP model on it
    X_test = x_tot
    f_true = 2 * torch.sigmoid(10 * x_tot) - 3*torch.exp(-4*(x_tot-1)**2)
    # torch.sigmoid(100*X_true) - 2*torch.exp(-X_true**2) + torch.sin(5*X_true)
    randperm = torch.randperm(f_true.shape[0])
    # Load training set
    X_data = x_tot[randperm[:ndata]]
    y_data = f_true[randperm[:ndata]]

    X_test = x_tot[randperm[ndata:ndata + ntest].sort()[0]]
    f_true = f_true[randperm[ndata:ndata + ntest].sort()[0]]

    return X_data, y_data, X_test, f_true
