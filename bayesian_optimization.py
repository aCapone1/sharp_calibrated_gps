"This code is partially based on the Exact GP Regression with Multiple GPUs and \
Kernel Partitioning Example, which can be found at \
https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/Simple_MultiGPU_GP_Regression.html"

import os
import pickle
import torch
import random
import gpytorch
import pyro
import numpy as np
import scipy.stats as st
from itertools import product
from functions.plottoyproblem import plottoyproblem

from pyro.infer.mcmc import NUTS, MCMC
from gpytorch.priors import UniformPrior

from gpregression.calibration_metrics import *
from plot_calibration_sharpness import plot_calibration_sharpness
from functions import *
from functions.plot1D import plot1D
from gpregression.ExactGPModel import ExactGPModel

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

ratiosubset = 1 # default ratio of subset used to train and choose calibration variances
chol_jitter = 1e-6 # default cholesky jitter used
nreps = 100 # change to higher number to perform more repetitions
training_iterations = 1800  # number of gp training interations for training regressor (posterior mean)
train_iter_covar = 100 # iterations used to train covariance
# calibration levels that we want to achieve. 0.5 means that the error bound needs to be correct 50 percent of the time
deltas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
          0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
# deltas = [0.1, 0.5, 0.9]
#
kernel = False


## TRAIN HYPERPARAMETERS OF FIRST AND SECOND GP

print ("Which experiment would you like to run? \n [B]ranin function, [R]osenbrock function, "
       "[A]ckley function, [AM] Ackley modified function")
while True:
    user_input = input()
    if user_input == "B":
        print ("You have chosen to run the Branin function experiment.")
        from datasets.branin_function import *
        break
    elif user_input == "A":
        print("You have chosen to run the Ackley function experiment.")
        from datasets.ackley_function import *
        break
    elif user_input == "AM":
        print("You have chosen to run the modified Ackley function experiment.")
        from datasets.ackley_modified_function import *
        break
    elif user_input == "R":
        print("You have chosen to run the Rosenbrock function experiment.")
        from datasets.rosenbrock_function import *
        break


# some initializations
regret_base_total = [None] * nreps
regret_calib_total = [None] * nreps


# check if GPU is available for large datasets and Laplace approximation
gpu = torch.cuda.is_available()
n_devices = 1 #torch.cuda.device_count()
print('Planning to run on {} GPUs.'.format(n_devices))
if gpu:
    output_device = torch.device('cuda:0')
else:
    output_device = torch.device('cpu')
# transfer bounding tensors of uniform distribution to output device (gpu)
lb = lb.to(output_device)
ub = ub.to(output_device)

for rep in list(range(nreps)):

    X0, y0 = get_starting_data()

    X_data = X0
    y_data = y0

    # set up likelihood
    likelihood0 = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Interval(lb[-1], ub[-1]),
        noise_prior=UniformPrior(lb[-1], ub[-1]))

    if gpu:
        base_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=dimx,
                                       lengthscale_prior=UniformPrior(lb[1:-1],
                                                                      ub[1:-1]),
                                       lengthscale_constraint=gpytorch.constraints.Interval(
                                           lb[1:-1], ub[1:-1])),
            outputscale_prior=UniformPrior(lb[0], ub[0]),
            outputscale_constraint=gpytorch.constraints.Interval(lb[0], ub[0]))

        kernel = gpytorch.kernels.MultiDeviceKernel(base_kernel, device_ids=range(n_devices),
                                                    output_device=output_device)

    print('Training GP model with (constrained) log-likelihood optimization...')
    if gpu:
        # make continguous
        train_x = torch.Tensor(X_data).detach().contiguous().to(output_device)
        train_y = torch.Tensor(y_data).detach().contiguous().to(output_device)

        # generate and train vanilla gp with log-likelihood optimization
        model0 = ExactGPModel(train_x, train_y, likelihood0, dimx, lb, ub, kernel).to(output_device)

        # use multiple GPUs if specified
        from gpregression.train import traingpu, find_best_gpu_setting, train, train_covar_gpu

        likelihood0 = likelihood0.to(output_device)
        model0 = model0.to(output_device)
        preconditioner_size = 100
        checkpoint_size = find_best_gpu_setting(train_x, train_y, model0=model0, likelihood0=likelihood0,
                                                n_devices=n_devices, output_device=output_device,
                                                preconditioner_size=preconditioner_size)

        # train hyperparameters at zeroth Bayesian optimization iteration
        model0, likelihood0, _ = \
            traingpu(train_x, train_y, model0, likelihood0, checkpoint_size=checkpoint_size,
                     preconditioner_size=100, n_training_iter=training_iterations)
    else:
        #
        train_x = torch.tensor(X_data).detach()
        train_y = torch.Tensor(y_data).detach()

        # generate and train vanilla gp with log-likelihood optimization
        model0 = ExactGPModel(train_x, train_y, likelihood0, dimx, lb, ub, kernel)

        from gpregression.train import train, train_covar

        # train hyperparameters before running Bayesian optimization
        model0, likelihood0, _ = \
            train(train_x, train_y, model0, likelihood0, training_iterations)



    # ratio of train over total data for computing bounding covariance. Remaining data set used to optimize
    # bounding covariance
    ratio = 0.8
    deltabo = 0.1

    # new training method, platt scaling-based
    calibrated_gps = get_calibrated_gp(ratiosubset, ratio, model0, likelihood0, gpu, lb, ub,
                                       kernel, train_x, train_y, deltas, train_iter_covar, kuleshov=False)
    cal_gp_delta, beta_gp_delta = calibrated_gps.get_calibrated_gaussian_processes([deltabo])
    # run Bayesian optimization
    from bo_tools import run_bo
    beta = 2

    regret_calibrated = run_bo(gpu, X_data, y_data, n_bo_iter, model0, cal_gp_delta[0], beta_gp_delta[0],
                         likelihood0, get_noisy_data, eval_fun, max_val)

    regret_base = run_bo(gpu, X_data, y_data, n_bo_iter, model0, model0, beta,
                         likelihood0, get_noisy_data, eval_fun, max_val)


    regret_calib_total[rep] = regret_calibrated
    regret_base_total[rep] = regret_base




    print('Training calibrated GP posterior variances...')
    # generate random permutation for splitting data into training and test data for bounding covariance

    print('Name: ' + name_saving + '. Repetition number %d' % (rep))




    with open('bayesianoptimizationresults/boresults' + name_saving + '.pkl', 'wb') as f:
        pickle.dump([regret_calib_total[:rep+1], regret_base_total[:rep+1]], f)

    # plot_calibration_sharpness(name_saving, datastr)
