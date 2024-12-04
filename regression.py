"This code is partially based on the Exact GP Regression with Multiple GPUs and \
Kernel Partitioning Example, which can be found at \
https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/Simple_MultiGPU_GP_Regression.html"

import pickle
import torch
import random
import gpytorch
import pyro
import numpy as np
import scipy.stats as st
from itertools import product

from pyro.infer.mcmc import NUTS, MCMC
from gpytorch.priors import UniformPrior

from gpregression.calibration_metrics import *
from print_calibration_results import print_calibration_results
from functions import *
from functions.plottoyproblem import plottoyproblem

# use multiple GPUs if specified
from gpregression.train import train, train_covar, train_approximate
from gpregression.ExactGPModel import ExactGPModel, ApproximateGPModel

np.random.seed(2)
random.seed(2)
torch.manual_seed(0)

# ratio of train over total data for computing bounding covariance. Remaining data set used to optimize
# bounding covariance
ratio = 0.8
ratiosubset = 1 # default ratio of subset used to train and choose calibration variances
chol_jitter = 1e-3 # default cholesky jitter used
nreps = 100 # change to higher number to perform more repetitions
training_iterations = 20  # number of gp training interations for training regressor (posterior mean)
train_iter_covar = 10 # iterations used to train covariance 800
train_full_bayes = False  # number of data points after which Laplace approximation is used
# calibration levels that we want to achieve. 0.5 means that the error bound needs to be correct 50 percent of the time
# deltas = np.linspace(0,1,20)#
deltas = [0, 0.005 , 0.025, 0.05 , 0.1  , 0.15 , 0.2, 0.25 , 0.3, 0.35, 0.4, 0.45, 0.5,
          0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95 , 0.975, 0.995, 1] #
training_subset_size = np.int64(1e12)
n_inducing_points = []
# deltas = [0.1, 0.5, 0.9]
#
kernel = False
retrain = False
sparse = False
approximategp = False

# choose data set
print("Which experiment would you like to run? \n [T]oy problem, [B]oston house prices,"
       "[Y]acht, [W]ine, [FB2] Facebook Cooments 2, [A]uto MPG, [K]in8nm or [C]oncrete")
while True:
    user_input = input()
    if user_input == "B":
        print ("You have chosen to run the Boston house prices experiment.")
        from datasets.boston_house_prices import *
        break
    elif user_input == "Y":
        print ("You have chosen to run the yacht hydrodynamics experiment.")
        from datasets.yacht_hydrodynamics import *
        break
    elif user_input == "W":
        print ("You have chosen to run the wine quality experiment.")
        from datasets.wine import *
        break
    elif user_input == "FB2":
        print ("You have chosen to run the Facebook comments 2 experiment.")
        from datasets.facebook2_dataset import *
        break
    elif user_input == "K":
        print("You have chosen to run the kin8nm experiment.")
        from datasets.kin8nm_dataset import *
        break
    elif user_input == "A":
        print("You have chosen to run the auto MPG experiment.")
        from datasets.auto_mpg import *
        break
    elif user_input == "T":
        print("You have chosen to run the one dimensional toy (Ackley) experiment.")
        from datasets.ackley_one_dim import *
        break
    elif user_input == "C":
        print("You have chosen to run the concrete experiment.")
        from datasets.cement import *
        break
    else:
        print("Please enter T, B, M, Y, W, A, or S and press Enter.")
        continue


# some initializations
CI95_cal_total = [None] * nreps
CI95_kuleshov_total = [None] * nreps
CI95_varfree_total = [None] * nreps
CI95_random_total = [None] * nreps
CI95_vanilla_total = [None] * nreps
CI95_fullbayes_total = [None] * nreps

NLL_cal_total = [None] * nreps
NLL_kuleshov_total = [None] * nreps
NLL_varfree_total = [None] * nreps
NLL_random_total = [None] * nreps
NLL_vanilla_total = [None] * nreps
NLL_fullbayes_total = [None] * nreps

sigma_cal_total = [None] * nreps
sigma_kuleshov_total = [None] * nreps
sigma_varfree_total = [None] * nreps
sigma_random_total = [None] * nreps

perc_robust = [None] * nreps
perc_cal_total = [None] * nreps
perc_vanilla_naive_total = [None] * nreps
perc_robust_kuleshov = [None] * nreps
perc_robust_random = [None] * nreps
perc_robust_varfree = [None] * nreps
perc_fullbayes_naive_total = [None] * nreps

try:
    ndata = datasizes[0]
    print('Delete data sizes vector. Using smallest data size.')
except:
    ndata = ndata

datastr = np.array2string(ndata.detach().numpy())

if train_full_bayes:
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
else:
    gpu = False
    output_device = torch.device('cpu')

for rep in list(range(nreps)):
    print('Name: ' + name + '. Data set size N=%d, Repetition number %d' % (ndata, rep))
    X_data, y_data, X_test, f_true = get_data(ndata)

    # set up likelihood
    likelihood0 = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Interval(lb[-1], ub[-1]),
        noise_prior=UniformPrior(lb[-1], ub[-1]))

    if gpu:
        output_device = torch.device('cuda:0')

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

        # make continguous
        train_x = torch.Tensor(X_data).detach().contiguous().to(output_device)
        train_y = torch.Tensor(y_data).detach().contiguous().to(output_device)
        test_x = torch.Tensor(X_test).detach().contiguous().to(output_device)
        test_y = torch.Tensor(f_true).detach().contiguous().to(output_device)

    else:
        output_device = torch.device('cpu')
        #
        train_x = torch.Tensor(X_data).detach()
        train_y = torch.Tensor(y_data).detach()

        perm = np.random.permutation(range(ndata))

        test_x = torch.Tensor(X_test).detach()
        test_y = torch.Tensor(f_true).detach()


    print('Training GP model with (constrained) log-likelihood optimization...')


    if gpu:

        # generate and train vanilla gp with log-likelihood optimization
        model0 = ExactGPModel(train_x, train_y, likelihood0, dimx, lb, ub, kernel, sparse = sparse).to(output_device)
        likelihood0 = likelihood0.to(output_device)
        model0 = model0.to(output_device)
        preconditioner_size = 100
        checkpoint_size = find_best_gpu_setting(train_x, train_y, model0=model0, likelihood0=likelihood0,
                                                n_devices=n_devices, output_device=output_device,
                                                preconditioner_size=preconditioner_size)
        model0, likelihood0, _ = \
            traingpu(train_x, train_y, model0, likelihood0, checkpoint_size=checkpoint_size,
                     preconditioner_size=100, n_training_iter=training_iterations)

        model0.eval()
        print('Training calibrated GP posterior variances...')

    else:
        # training without gpu

        train_x_subset = train_x[:training_subset_size]
        train_y_subset = train_y[:training_subset_size]

        # generate and train vanilla gp with log-likelihood optimization
        if approximategp:
            train_x = train_x.contiguous()
            train_y = train_y.contiguous()
            test_x = test_x.contiguous()
            test_y = test_y.contiguous()
            from torch.utils.data import TensorDataset, DataLoader

            train_dataset = TensorDataset(train_x, train_y)
            train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

            test_dataset = TensorDataset(test_x, test_y)
            test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

            inducing_points = train_x[:n_inducing_points, :]
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model0 = ApproximateGPModel(inducing_points = inducing_points)
            model0.train()
            likelihood0.train()

            model0, likelihood0 = train_approximate(train_x_subset, train_y_subset, model0, likelihood0, training_iterations)
            model0 = ExactGPModel(train_x, train_y, likelihood0, dimx, [], [], model0.covar_module).to(output_device)
        else:
            model0 = ExactGPModel(train_x_subset, train_y_subset, likelihood0,
                                  dimx, lb, ub, kernel, sparse=sparse, n_inducing_points=n_inducing_points)
            model0, likelihood0, _ = train(train_x_subset, train_y_subset, model0, likelihood0, training_iterations)
            model0 = ExactGPModel(train_x, train_y, likelihood0, dimx, [], [], model0.covar_module).to(output_device)
        model0.eval()
        print('Training calibrated GP posterior variances...')

        if not kernel == False:
            if kernel._get_name() == 'SpectralMixtureKernel':
                print('Spectral mixture kernel being used. \
                Hyperparameter bounds not being reset for covariance estimation. ')


        # METHODS:
        # -1 = Vovk et al., 2020
        # 0 = Ours
        # 1 = Kuleshov et al., 2018


        calibrated_gps = get_calibrated_gp(ratiosubset, ratio, model0, likelihood0, gpu, lb, ub,
                                           kernel, train_x, train_y, deltas, train_iter_covar,
                                           method=0, perm=perm, retrain = retrain, sparse = sparse)

        kuleshov_calibrated_gps = get_calibrated_gp(ratiosubset, ratio, model0, likelihood0, gpu, lb, ub,

                                            kernel, train_x, train_y, deltas, train_iter_covar,
                                                    method=1, perm=perm, retrain = retrain, sparse = sparse)

        varfree_calibrated_gps = get_calibrated_gp(ratiosubset, ratio, model0, likelihood0, gpu, lb, ub,
                                                    kernel, train_x, train_y, deltas, train_iter_covar,
                                                   method=2, perm=perm, retrain = retrain, sparse = sparse)


        checkpoint_size = 0
        preconditioner_size = 100


    # check if data set is too large for fully Bayesian approach. If so, Laplace approximation will be used
    if train_full_bayes:
        # increase jitter for numerical stability
        with gpytorch.settings.cholesky_jitter(chol_jitter):
            # set up likelihood
            likelihood = deepcopy(likelihood0)

            # Use a positive constraint instead of usual GreaterThan(1e-4) so that LogNormal has support over full range.
            fullbmodel = ExactGPModel(train_x, train_y, likelihood, dimx, lb, ub, deepcopy(kernel))

            def pyro_model(x, y):
                with gpytorch.settings.fast_computations(False, False, False):
                    sampled_fullbmodel = fullbmodel.pyro_sample_from_prior()
                    output = sampled_fullbmodel.likelihood(sampled_fullbmodel(x))
                    pyro.sample("obs", output, obs=y)
                return y


            nuts_kernel = NUTS(pyro_model)
            mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, disable_progbar=False)
            print('Generating GP samples for fully Bayesian GP...')
            mcmc_run.run(train_x, train_y)

            fullbmodel.pyro_load_from_samples(mcmc_run.get_samples())

            # get marginal log likelihood of models used for MC integration
            if dimx > 1:
                expanded_test_x = test_x.unsqueeze(0).repeat(num_samples, 1, 1)
                expanded_test_y = test_y.unsqueeze(0).repeat(num_samples, 1)
            else:
                expanded_test_x = deepcopy(test_x)
                expanded_test_y = deepcopy(test_y)
            fullbmodel.eval()
            outputfb = fullbmodel(expanded_test_x)

    else:
        fullbmodel = []
        torch.cuda.empty_cache()


    if retrain:
        model_regressor = deepcopy(model0)
    else:
        model_regressor = ExactGPModel(train_x[perm[0:np.int64(ratiosubset * ratio * ndata)]],
                     train_y[perm[0:np.int64(ratiosubset * ratio * ndata)]], likelihood0, dimx,
                     [], [], model0.covar_module, sparse = sparse)
    del model0
    model_regressor.eval()

    with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
            gpytorch.settings.max_preconditioner_size(preconditioner_size), \
            gpytorch.settings.max_cg_iterations(1000000), \
            gpytorch.settings.fast_computations(log_prob=False):
        try:
            mean0 = model_regressor(test_x).mean.detach()
            stddev0 = model_regressor(test_x).stddev.detach()
        except:
            # compute each entry of mean and covariance individually if memory limitations are reached
            mean0, stddev0 = torch.as_tensor([model_regressor(x.reshape(1, dimx)).mean.detach()
                                     for x in test_x]).detach().to(output_device), \
                             torch.as_tensor([model_regressor(x.reshape(1, dimx)).stddev.detach()
                                       for x in test_x]).detach().to(output_device)

        if not (name_saving == 'toyproblem' or name_saving == 'ackleyonedim'):
            del model_regressor

        perc_vanilla_naive = []
        perc_fullbayes_naive = []
        for delta in deltas:

            beta_bayesian = st.norm.ppf(1 - delta)
            beta_bayesian_plus = st.norm.ppf(1 - delta)
            beta_bayesian_minus = st.norm.ppf(delta)

            if np.isinf(beta_bayesian):
                if beta_bayesian<0:
                   beta_bayesian = -100
                else:
                    beta_bayesian = 100
            perc_vanilla_naive_delta = evaleceandsharpness(test_y, mean0, stddev0, beta_bayesian)

            if delta == 0.025 and (name_saving == 'toyproblem' or name_saving =='ackleyonedim'):
                plottoyproblem(X_data, y_data, f_true, X_test, gp_mean=model_regressor,
                               gp_stddev_minus=model_regressor, gp_stddev_plus=model_regressor,
                               beta_minus=beta_bayesian, beta_plus=beta_bayesian_minus,
                               delta=delta, color='m', name='vanilla')

            perc_vanilla_naive.append(perc_vanilla_naive_delta)

            if fullbmodel:
                with gpytorch.settings.cholesky_jitter(chol_jitter):
                    # transpose means and standard deviations of fully Bayesian model for Gaussian mixture model
                    fbmeans = torch.transpose(outputfb.mean, 0, 1)
                    fbstddevs = torch.transpose(outputfb.stddev, 0, 1)

                    # create set of normally distributed variables with means fbmeans and standard deviations fbstddevs
                    fbN = torch.distributions.Normal(fbmeans, fbstddevs)

                    # generate weights (ones) for Gaussian mixture models
                    fbgmmweights = torch.distributions.Categorical(torch.ones(num_samples, ))

                    # create Gaussian mixture model using fully Bayesian GP evaluations
                    fbgmm = torch.distributions.mixture_same_family.MixtureSameFamily(fbgmmweights, fbN)
                    fb_mean = fbgmm.mean
                    fb_stddev = fbgmm.stddev

                    perc_fullbayes_naive_delta = evaleceandsharpness(test_y, fb_mean, fb_stddev, beta_bayesian)

                if delta == 0.025 and (name_saving == 'toyproblem' or name_saving =='ackleyonedim'):
                    from functions.plottoyproblem import plottoyproblem
                    plottoyproblem(X_data, y_data, f_true, X_test, gp_mean=fullbmodel,
                                   gp_stddev_minus=fullbmodel, gp_stddev_plus=fullbmodel,
                                   beta_minus=beta_bayesian, beta_plus=beta_bayesian_minus,
                                   delta=delta, color='g', name='fullbayes',
                                   chol_jitter=chol_jitter)
                # del fullbmodel
                # torch.cuda.empty_cache()
            else:
                fb_mean = []
                fb_stddev = []
                perc_fullbayes_naive_delta = np.nan

            perc_fullbayes_naive.append(perc_fullbayes_naive_delta)
        ## eliminate models to free up memory
        # del model_regressor
        # torch.cuda.empty_cache()
        #
        # if fullbmodel:
        #     del fullbmodel
        #     torch.cuda.empty_cache()

        perc_cal = []
        perc_kuleshov = []
        perc_varfree = []
        perc_random = []

        cal_gp_interp, betas_cal_interp = calibrated_gps.get_calibrated_gaussian_processes(deltas)
        _, betas_kuleshov_interp = kuleshov_calibrated_gps.get_calibrated_gaussian_processes(deltas)
        _, betas_varfree_interp = varfree_calibrated_gps.get_calibrated_gaussian_processes(deltas)
        betas_random_interp = kuleshov_calibrated_gps.get_randomly_interpolated_betas(deltas,test_y.size()[0])

        CI95_cal, CI95_kuleshov, CI95_varfree, CI95_random, CI95_vanilla, CI95_fullbayes \
            = get_CI95(calibrated_gps, kuleshov_calibrated_gps, varfree_calibrated_gps, stddev0, fb_stddev, test_x, test_y)

        NLL_cal, NLL_kuleshov, NLL_varfree, NLL_random, NLL_vanilla, NLL_fullbayes, \
        sigma_cal, sigma_kuleshov, sigma_varfree, sigma_random \
            = get_NLL(calibrated_gps, kuleshov_calibrated_gps, varfree_calibrated_gps, mean0, stddev0,
                      fb_mean, fb_stddev, test_x, test_y, deltas)

        # evaluate performance
        for i in range(len(deltas)):

            cal_gp_delta = cal_gp_interp[i]
            beta_gp_delta = betas_cal_interp[i]
            beta_gp_kuleshov = betas_kuleshov_interp[i]
            beta_gp_varfree = betas_varfree_interp[i]

            # divide test_x into chunks to avoid memory overloading
            chunk_size = 200
            n_chunks = np.int64(test_x.size()[0] / chunk_size + 1)
            test_x_chunks = [test_x[j * chunk_size:(j + 1) * chunk_size] for j in range(n_chunks)]
            stddev_bound_cal = torch.cat([cal_gp_delta(test_x_chunk).stddev.detach() for test_x_chunk in test_x_chunks])

            perc_cal_delta = evaleceandsharpness(test_y, mean0, stddev_bound_cal, beta_gp_delta)
            del stddev_bound_cal
            perc_kuleshov_delta = evaleceandsharpness(test_y, mean0, stddev0, beta_gp_kuleshov)
            perc_random_delta = evaleceandsharpness(test_y, mean0, betas_random_interp[i]*stddev0, 1)
            perc_varfree_delta = evaleceandsharpness(test_y, mean0, torch.ones(stddev0.size()), beta_gp_varfree)

            perc_kuleshov.append(perc_kuleshov_delta)

            perc_varfree.append(perc_varfree_delta)

            perc_random.append(perc_random_delta)

            if deltas[i]==0.025 and (name_saving == 'toyproblem' or name_saving =='ackleyonedim'):

                cal_gp_interp_minus, betas_cal_interp_minus \
                    = calibrated_gps.get_calibrated_gaussian_processes([1-deltas[i]])
                _, betas_kuleshov_interp_minus \
                    = kuleshov_calibrated_gps.get_calibrated_gaussian_processes([1-deltas[i]])
                _, betas_varfree_interp_minus \
                    = varfree_calibrated_gps.get_calibrated_gaussian_processes([1-deltas[i]])
                betas_random_interp_minus = \
                    kuleshov_calibrated_gps.get_randomly_interpolated_betas([1-deltas[i]],
                                                                            y_data.size()[0]+test_y.size()[0])
                betas_random_interp_plus = kuleshov_calibrated_gps.\
                    get_randomly_interpolated_betas([deltas[i]], y_data.size()[0]+test_y.size()[0])

                plottoyproblem(X_data, y_data, f_true, X_test, model_regressor, gp_stddev_minus = cal_gp_interp_minus[0],
                               gp_stddev_plus = cal_gp_delta, beta_minus=betas_cal_interp_minus[0],
                               beta_plus = beta_gp_delta, delta=deltas[i], color='b', name='ours')
                plottoyproblem(X_data, y_data, f_true, X_test, model_regressor, gp_stddev_minus = model_regressor,
                               gp_stddev_plus = model_regressor, beta_minus=betas_kuleshov_interp_minus[0],
                               beta_plus=beta_gp_kuleshov, delta=deltas[i], color='r', name='kuleshov')
                plottoyproblem(X_data, y_data, f_true, X_test, model_regressor, gp_stddev_minus=model_regressor,
                               gp_stddev_plus=model_regressor, beta_minus=betas_varfree_interp_minus[0],
                               beta_plus=beta_gp_varfree, delta=deltas[i], color='c', name='varfree')
                plottoyproblem(X_data, y_data, f_true, X_test, model_regressor, gp_stddev_minus=model_regressor,
                               gp_stddev_plus=model_regressor, beta_minus=betas_random_interp_minus[0],
                               beta_plus=betas_random_interp_plus[0], delta=deltas[i], color='c', name='random')

            perc_cal.append(perc_cal_delta)

        CI95_cal_total[rep] = CI95_cal
        CI95_kuleshov_total[rep] = CI95_kuleshov
        CI95_varfree_total[rep] = CI95_varfree
        CI95_random_total[rep] = CI95_random
        CI95_vanilla_total[rep] = CI95_vanilla
        CI95_fullbayes_total[rep] = CI95_fullbayes

        NLL_cal_total[rep] = NLL_cal
        NLL_kuleshov_total[rep] = NLL_kuleshov
        NLL_varfree_total[rep] = NLL_varfree
        NLL_random_total[rep] = NLL_random
        NLL_vanilla_total[rep] = NLL_vanilla
        NLL_fullbayes_total[rep] = NLL_fullbayes

        sigma_cal_total[rep] = sigma_cal
        sigma_kuleshov_total[rep] = sigma_kuleshov
        sigma_varfree_total[rep] = sigma_varfree
        sigma_random_total[rep] = sigma_random

        perc_cal_total[rep] = perc_cal
        perc_vanilla_naive_total[rep] = perc_vanilla_naive
        perc_fullbayes_naive_total[rep] = perc_fullbayes_naive
        perc_robust_kuleshov[rep] = perc_kuleshov
        perc_robust_random[rep] = perc_random
        perc_robust_varfree[rep] = perc_varfree


        torch.cuda.empty_cache()


    with open('regressionresults/percanderrs' + name_saving + datastr + '.pkl', 'wb') as f:
        pickle.dump([perc_robust[:rep+1], perc_cal_total[:rep+1], perc_vanilla_naive_total[:rep+1],
                     perc_fullbayes_naive_total[:rep+1], perc_robust_kuleshov[:rep+1],
                     perc_robust_random[:rep + 1], perc_robust_varfree[:rep + 1], perc_fullbayes_naive_total[:rep+1],
                     NLL_cal_total[:rep+1], NLL_kuleshov_total[:rep+1], NLL_varfree_total[:rep+1],
                     NLL_random_total[:rep+1], NLL_vanilla_total[:rep+1], NLL_fullbayes_total[:rep+1],
                     sigma_cal_total[:rep + 1], sigma_kuleshov_total[:rep + 1], sigma_varfree_total[:rep + 1],
                     sigma_random_total[:rep + 1], CI95_cal_total[:rep + 1], CI95_kuleshov_total[:rep + 1], CI95_varfree_total[:rep + 1],
                     CI95_random_total[:rep + 1], CI95_vanilla_total[:rep + 1], CI95_fullbayes_total[:rep + 1] , deltas], f)

    print_calibration_results(name_saving, datastr)
