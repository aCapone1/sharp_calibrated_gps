import gpytorch
import torch
import tqdm
import gc
import numpy as np
import scipy.stats as st
from torch.autograd import Variable
from copy import copy
from functions.LBFGS import FullBatchLBFGS
from gpregression.ExactGPModel import ExactGPModel
from gpregression.calibration_metrics import *
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal


def train(train_x, train_y, model0, likelihood0, n_training_iter):
    # Use the adam optimizer
    optimizer = torch.optim.SGD(model0.parameters(), lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll0 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood0, model0)
    for i in range(n_training_iter):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output0 = model0(train_x)
        # Calc loss and backprop derivatives
        loss = -mll0(output0, train_y)
        loss.backward()

        # print('Iter %d/%d - Loss: %.3f \r' % (i + 1, training_iterations, loss.item()))
        optimizer.step()
        torch.cuda.empty_cache()
    return model0, likelihood0, mll0(output0, train_y)


def train_approximate(train_x, train_y, train_loader, model0, likelihood0, n_training_iter, num_epochs = 4):

    optimizer = torch.optim.Adam([
        {'params': model0.parameters()},
        {'params': likelihood0.parameters()},
    ], lr=0.01)

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood0, model0, num_data=train_y.size(0))

    epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm.notebook.tqdm(train_loader, desc="Minibatch", leave=False)
        for x_batch, y_batch in minibatch_iter:
            optimizer.zero_grad()
            output = model0(x_batch)
            loss = -mll(output, y_batch)
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

    return model0, likelihood0


def train_covar(calib_x, calib_y, boundinggp, n_training_iter, delta = 0.1,
                method=0, nrestarts = 1, sign_beta=1, sparse = False):

    beta_start_kuleshov = (calib_y / boundinggp(calib_x).stddev).quantile(1 - delta, 0).detach().item()
    beta_start_varfree = (calib_y).quantile(1 - delta, 0).detach().item()

    # METHODS:
    # -1 = Vovk et al., 2020
    # 0 = Ours
    # 1 = Kuleshov et al., 2018, Vovk et al., 2020
    # 2 = Marx et al., 2022 variance-free

    if method==1:
        best_model = deepcopy(boundinggp)
        # sign is inverted later again
        best_beta = torch.tensor(beta_start_kuleshov)
    elif method==2:
        best_model = deepcopy(boundinggp)
        # sign is inverted later again
        best_beta = torch.tensor(beta_start_varfree)
    elif method==0:

        # initialize best value obtained with optimizer
        best_model = deepcopy(boundinggp)
        best_beta = torch.tensor(beta_start_kuleshov)
        best_val = np.inf

        if sparse:
            lengthscales0 = copy(boundinggp.covar_module.base_kernel.base_kernel.lengthscale.detach())
        else:
            lengthscales0 = copy(boundinggp.covar_module.base_kernel.lengthscale.detach())
        learningrate = 0.5

        for nrestart in range(nrestarts):
            if nrestart>0:
                if nrestart == 1:
                    if sparse:
                        lengthscales0 = boundinggp.covar_module.base_kernel.base_kernel.lengthscale_prior.low
                    else:
                        lengthscales0 = boundinggp.covar_module.base_kernel.lengthscale_prior.low
                elif nrestart == 2:
                    if sparse:
                        lengthscales0 = boundinggp.covar_module.base_kernel.base_kernel.lengthscale_prior.high
                    else:
                        lengthscales0 = boundinggp.covar_module.base_kernel.lengthscale_prior.high
                else:
                    if sparse:
                        lengthscales0 = boundinggp.covar_module.base_kernel.base_kernel.lengthscale_prior.low + \
                                        torch.rand(boundinggp.covar_module.base_kernel.base_kernel.lengthscale.size()) * \
                                        (boundinggp.covar_module.base_kernel.base_kernel.lengthscale_prior.high -
                                         boundinggp.covar_module.base_kernel.base_kernel.lengthscale_prior.low)
                    else:
                        lengthscales0 = boundinggp.covar_module.base_kernel.lengthscale_prior.low +\
                                    torch.rand(boundinggp.covar_module.base_kernel.lengthscale.size())*\
                                    (boundinggp.covar_module.base_kernel.lengthscale_prior.high -
                                     boundinggp.covar_module.base_kernel.lengthscale_prior.low)
            if sparse:
                boundinggp.covar_module.base_kernel.base_kernel.lengthscale = \
                    Variable(lengthscales0.clone().detach().requires_grad_(True), requires_grad=True)
                params = list(boundinggp.covar_module.base_kernel.base_kernel.parameters())
            else:
                boundinggp.covar_module.base_kernel.lengthscale = \
                    Variable(lengthscales0.clone().detach().requires_grad_(True), requires_grad=True)
                params = list(boundinggp.covar_module.base_kernel.parameters())
            # Use the adam optimizer: torch.optim.torch.optim.Adam
            optimizer = torch.optim.SGD(params, lr=learningrate)
            i=0
            # Get output from model
            while i < 100:
                try:
                    # Zero backprop gradients
                    optimizer.zero_grad()
                    output0 = boundinggp(calib_x)

                    # Calc loss and backprop derivatives
                    def closure():
                        optimizer.zero_grad()
                        beta = (calib_y / output0.stddev).quantile(1 - delta, 0)
                        loss = ((beta * output0.stddev) ** 2).mean()
                        loss.backward(retain_graph=True)
                        return loss
                    # print('Iter %d/%d - Loss: %.3f \r' % (i + 1, training_iterations, loss.item()))
                    # check for improvement before and after performing optimization step
                    if closure().isnan():
                        break
                    elif closure().item() < best_val:
                        # check if value is still improved after model change. this is not necessarily the case when using
                        # sparse GPS, most likely due to a bug
                        new_model = deepcopy(boundinggp)
                        new_model.eval()
                        new_beta = (calib_y / new_model(calib_x).stddev).quantile(1 - delta, 0)
                        new_val = ((new_beta * new_model(calib_x).stddev) ** 2).mean()
                        boundinggp.train()
                        boundinggp.eval()
                        if new_val < best_val:
                            best_val = new_val.item()
                            best_model = new_model
                            best_beta = new_beta.detach()
                        # best_val = closure().item()
                        # #best_model = deepcopy(boundinggp)
                        # best_beta = (calib_y / output0.stddev).quantile(1 - delta, 0).detach()
                    optimizer.step(closure)

                    torch.cuda.empty_cache()
                    i+=1
                except:
                    break

    best_model.eval()
    return best_model, best_beta