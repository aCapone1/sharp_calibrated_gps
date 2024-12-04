from copy import deepcopy
import torch
import gpytorch
import pickle
import random
import pyro
import numpy as np
import scipy.stats as st
from torch.autograd import Variable
from itertools import product
from functions.plottoyproblem import plottoyproblem

from pyro.infer.mcmc import NUTS, MCMC
from gpytorch.priors import UniformPrior

from functions import *
from gpregression.ExactGPModel import ExactGPModel


def get_calibrated_gp(ratiosubset, ratio, model0, likelihood0, gpu, lb, ub, kernel,
                      train_x, train_y, deltas_desired, train_iter_covar, method=0,
                      perm = False, retrain = False, sparse =False):
    # METHODS:
    # -1 = Vovk et al., 2020
    # 0 = Ours
    # 1 = Kuleshov et al., 2018
    # 2 = Marx et al., 2022, variance-free

    if gpu:
        output_device = torch.device('cuda:0')
    else:
        output_device = torch.device('cpu')

    ndata = train_x.shape[0]
    dimx = model0.train_inputs[0].shape[1]
    if not torch.tensor(perm).any():
        # generate random permutation for splitting data into training and test data for bounding covariance
        perm = np.random.permutation(range(ndata))

    # data used to condition the bounding gp. Kuleshov's method uses full available dat
    train_covar_x = train_x[perm[0:np.int64(ratiosubset * ratio * ndata)]]
    train_covar_y = train_y[perm[0:np.int64(ratiosubset * ratio * ndata)]]

    model0.eval()
    if gpu:

        from gpregression.train import traingpu, find_best_gpu_setting, train, train_covar_gpu

        # generate predictors conditioned on calibration training data only. This model will be used exclusively to
        # determine the test targets used for calibrating the second GP
        model0_train = ExactGPModel(train_covar_x, train_covar_y, likelihood0, dimx,
                                    lb, ub, model0.covar_module).to(output_device)
        model0_train.eval()
        calib_covar_x = train_x[perm[np.int64(ratiosubset*ratio*ndata):]]
        calib_covar_y = train_y[perm[np.int64(ratiosubset*ratio*ndata):]] - model0_train(calib_covar_x).mean

    else:

        from gpregression.train import train, train_covar

        # generate predictors conditioned on calibration training data only. This model will be used exclusively to
        # determine the test targets used for calibrating the second GP
        model0_train = ExactGPModel(train_covar_x, train_covar_y, likelihood0, dimx, lb, ub, model0.covar_module)
        model0_train.eval()
        calib_covar_x = train_x[perm[np.int64(ratiosubset*ratio*ndata):]]
        calib_covar_y = train_y[perm[np.int64(ratiosubset*ratio*ndata):]] - model0_train(calib_covar_x).mean
        # determine how much of the data is above the mean of the base GP
        delta_split = (calib_covar_y>0).sum()/calib_covar_y.size()[0]
        deltas_negative = [delta for delta in deltas_desired if delta > delta_split]
        deltas_positive = [delta for delta in deltas_desired if delta < delta_split]
        sign_beta = 1

        calibrated_gps = []
        deltas_cal_gp = []

        for deltas in [deltas_negative, deltas_positive]:

            # change sign beta
            sign_beta *=-1

            if not kernel == False:
                if kernel._get_name() == 'SpectralMixtureKernel':
                    print('Spectral mixture kernel being used. \
                    Hyperparameter bounds not being reset for covariance estimation. ')

            # # establish confidence level for basic GP
            # f_std = model0_train(calib_covar_x).stddev
            # if sign_beta < 0:
            #     viol_rate0 = np.inf
            # else:
            #     viol_rate0 = -np.inf
            # viol_rate0 = (calib_covar_y > sign_beta*f_std).sum()/calib_covar_y.size()[0]
            # deltas_larger_lengthscales = [delta for delta in deltas if sign_beta * delta > sign_beta * viol_rate0]
            # deltas_smaller_lengthscales = [delta for delta in deltas if sign_beta * delta < sign_beta * viol_rate0]
            deltas_lengthscales = deepcopy(deltas)
            if sign_beta == -1:
                deltas_lengthscales.reverse()
            #     deltas_larger_lengthscales.reverse()
            # else:
            #     deltas_smaller_lengthscales.reverse()

            ub_lengthscale0 = ub[1:-1]
            lb_lengthscale0 = lb[1:-1] * 1e-3

            # smaller_lengthscales = True
            # perform an extensive search for the first set of hyperparameters
            nrestarts = 4
            # for deltas_lengthscales in [deltas_smaller_lengthscales, deltas_larger_lengthscales]:
            # if smaller_lengthscales:
            #     beta_gp_delta_old = torch.tensor(0)
            # else:
            beta_gp_delta_old = torch.tensor(np.inf)
            if sparse:
                lengthscales_delta_old = model0.covar_module.base_kernel.base_kernel.lengthscale[0].detach()
            else:
                lengthscales_delta_old = model0.covar_module.base_kernel.lengthscale[0].detach()
            # new training method, platt scaling-based
            first_delta = True
            for delta in deltas_lengthscales:
                # set new bounds for hyperparameters based on regressor. The point of the following lines is to avoid
                # having equal lower and upper bounds on the lengthscale constraints
                if sign_beta < 0:
                    ub_add_multiplier = 1-delta
                else:
                    ub_add_multiplier = delta
                ub_lengthscales = deepcopy(ub[1:-1] * (1+0.01*ub_add_multiplier)).detach()
                lb_lengthscales = lb_lengthscale0
                lengthscales0 = torch.max(lengthscales_delta_old, lb_lengthscales)
                if method==0:
                    # get bounding gp with first calibration metric
                    kernel_delta_base = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dimx,
                                   lengthscale_prior=UniformPrior(lb_lengthscales, ub_lengthscales),
                                                   lengthscale_constraint=gpytorch.constraints.Interval(
                                                       lb_lengthscales, ub_lengthscales)),
                                                    outputscale_prior=UniformPrior(lb[0], ub[0]),
                                                    outputscale_constraint=gpytorch.constraints.Interval(lb[0], ub[0]))
                    if sparse:
                        inducing_points = model0.covar_module.inducing_points.detach()
                        kernel_delta_base.outputscale = model0.covar_module.base_kernel.outputscale.clone().detach()
                        kernel_delta = gpytorch.kernels.InducingPointKernel(kernel_delta_base,
                                                                      inducing_points=inducing_points,
                                                                      likelihood=likelihood0)
                        kernel_delta.base_kernel.base_kernel.lengthscale = \
                            Variable(lengthscales0.clone().detach().requires_grad_(True), requires_grad=True)
                    else:
                        kernel_delta_base.outputscale = model0.covar_module.outputscale.clone().detach()
                        kernel_delta = kernel_delta_base
                        kernel_delta.base_kernel.lengthscale = \
                            Variable(lengthscales0.clone().detach().requires_grad_(True), requires_grad=True)
                    cal_gp_delta = ExactGPModel(train_covar_x, train_covar_y, model0.likelihood, \
                                                [], [], [], kernel=kernel_delta)
                else:
                    cal_gp_delta = deepcopy(model0_train)

                cal_gp_delta.eval()
                cal_gp_delta, beta_gp_delta = train_covar(calib_covar_x, calib_covar_y, cal_gp_delta,
                                                          train_iter_covar, delta= delta, sign_beta=sign_beta,
                                                          nrestarts = nrestarts, method = method, sparse = sparse)
                nrestarts = 1
                if sign_beta == torch.sign(beta_gp_delta):
                    beta_gp_delta = sign_beta * torch.min(beta_gp_delta.abs(), beta_gp_delta_old.abs())
                    if torch.min(beta_gp_delta.abs(), beta_gp_delta_old.abs()) == beta_gp_delta.abs():
                        delta_cal_gp = delta
                    elif sparse:
                        raise ValueError('Optimization resulted in betas that do not satisfy monotonicity property. '
                                         'This is only supported for non-sparse GPs.')
                    else:
                        f_std = cal_gp_delta(calib_covar_x).stddev
                        y_diff = calib_covar_y - beta_gp_delta*f_std
                        argsort_y_diff = torch.argsort(y_diff)
                        y_diff_sort = y_diff[argsort_y_diff].flip(0)
                        if not (y_diff_sort<0).any():
                            delta_cal_gp = 1
                        elif not (y_diff_sort>0).any():
                            delta_cal_gp = 0
                        else:
                            for i in range(y_diff_sort.size()[0]-1):
                                if y_diff_sort[i]>0>=y_diff_sort[i+1]:
                                    i_negative = i
                                    i_positive = i+1
                                    break
                            delta_cal_gp = (i_negative - (y_diff_sort[i_negative])
                                            /(y_diff_sort[i_positive] - y_diff_sort[i_negative]))/(y_diff_sort.size()[0]-1)

                    deltas_cal_gp.append(delta_cal_gp)
                    calibrated_gps.append([cal_gp_delta, beta_gp_delta])
                    beta_gp_delta_old = deepcopy(beta_gp_delta)
                    if sparse:
                        lb_lengthscale0 = cal_gp_delta.covar_module.base_kernel.base_kernel.lengthscale[0].detach()
                        lengthscales_delta_old = \
                            deepcopy(cal_gp_delta.covar_module.base_kernel.base_kernel.lengthscale[0].detach())
                    else:
                        lb_lengthscale0 = cal_gp_delta.covar_module.base_kernel.lengthscale[0].detach()
                        lengthscales_delta_old = deepcopy(cal_gp_delta.covar_module.base_kernel.lengthscale[0].detach())
            # smaller_lengthscales = False

        checkpoint_size = 0
        preconditioner_size = 100

    deltas_cal_gp = torch.tensor(deltas_cal_gp)
    # if method<0:
    #     deltas_cal_gp = torch.tensor(deltas)
    # else:
    #     deltas_cal_gp = torch.tensor(deltas_cal_gp)
    z_score_calib_data = calib_covar_y/model0_train(calib_covar_x).stddev
    if retrain:
        cal_train_x, cal_train_y = train_x, train_y
    else:
        cal_train_x, cal_train_y = train_covar_x, train_covar_y

    calib_gaussian_process = calibrated_gaussian_process(calibrated_gps, deltas_cal_gp, cal_train_x, cal_train_y, kernel,
                                                         z_score_calib_data = z_score_calib_data,
                                                         gpu = gpu, checkpoint_size=checkpoint_size,
                                                         preconditioner_size=preconditioner_size, sparse = sparse)
    return calib_gaussian_process



class calibrated_gaussian_process:

    # constructor method
    def __init__(self, cal_gps, delta_gps, train_x, train_y, kernel, z_score_calib_data = [], gpu = False,
                 checkpoint_size = 0, preconditioner_size = 100, sparse = False):
        # object attributes

        self.cal_gps = cal_gps
        self.delta_gps = delta_gps
        self.train_x = train_x
        self.train_y = train_y
        self.kernel = kernel
        self.z_score_calib_data = z_score_calib_data
        self.checkpoint_size = checkpoint_size
        self.preconditioner_size = preconditioner_size
        self.gpu = gpu
        self.sparse = sparse

    def get_calibrated_gaussian_processes(self, deltas):

        checkpoint_size = self.checkpoint_size
        preconditioner_size = self.preconditioner_size
        dimx = self.cal_gps[0][0].train_inputs[0].shape[1]

        train_x = self.train_x
        train_y = self.train_y
        kernel = self.kernel
        gpu = self.gpu
        sparse = self.sparse

        if gpu:
            output_device = torch.device('cuda:0')
        else:
            output_device = torch.device('cpu')

        with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
                gpytorch.settings.max_preconditioner_size(preconditioner_size), \
                gpytorch.settings.max_cg_iterations(1000000), \
                gpytorch.settings.fast_computations(log_prob=False):

            calibrated_gps = self.cal_gps
            delta_gps = self.delta_gps

            hyps_calgp = []
            betas_calgp = []

            for [cal_gp_delta, beta_gp_delta] in calibrated_gps:
                hyp = torch.zeros(dimx + 2)
                if sparse:
                    hyp[1:-1] = cal_gp_delta.covar_module.base_kernel.base_kernel.lengthscale.detach()
                    hyp[0] = cal_gp_delta.covar_module.base_kernel.outputscale.detach()
                else:
                    hyp[1:-1] = cal_gp_delta.covar_module.base_kernel.lengthscale.detach()
                    hyp[0] = cal_gp_delta.covar_module.outputscale.detach()
                hyp[-1] = cal_gp_delta.likelihood.noise.detach()
                hyps_calgp.append(hyp.numpy())
                betas_calgp.append(beta_gp_delta.detach().numpy())
            betas_calgp = np.array(betas_calgp)
            hyps_calgp = np.array(hyps_calgp)

            argsort_cal = torch.argsort(delta_gps)

            # warning: interpolation only available for vanilla GPs, not for sparse GPs due to complications with interpolating for sparse GPs
            # the following line checks if interpolations are necessary. if so, the code stops.
            delta_eq_deltagps = torch.tensor([(torch.tensor(deltas) == delta_gp).any().detach().item() for delta_gp in delta_gps])
            if delta_eq_deltagps.sum() == len(deltas):
                betas_cal_interp = []
                calibrated_gps_deltas = []
                nums = np.arange(0,len(delta_gps))
                for delta in deltas:
                    num = nums[delta==delta_gps].item()
                    betas_cal_interp.append(betas_calgp[num])
                    calibrated_gps_deltas.append(calibrated_gps[num][0])
            else:

                hyp_cal_interp = np.array([np.interp(deltas, delta_gps[argsort_cal], hyps_calgp[:,i][argsort_cal])
                                           for i in range(dimx+2)])
                betas_cal_interp = np.interp(deltas, delta_gps[argsort_cal], betas_calgp[argsort_cal])
                calibrated_gps_deltas = []

                for i in range(len(deltas)):
                    hyps_cal = hyp_cal_interp[:,i]
                    kernel_delta_base = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dimx))
                    kernel_delta_base.outputscale = hyps_cal[0]
                    kernel_delta_base.base_kernel.lengthscale = hyps_cal[1:-1]
                    likelihood_delta = gpytorch.likelihoods.GaussianLikelihood()
                    likelihood_delta.noise = hyps_cal[-1]
                    kernel_delta = kernel_delta_base
                    cal_gp_delta = ExactGPModel(train_x, train_y, likelihood_delta, dimx, [], [], kernel_delta).to(output_device)
                    # cal_gp_delta = ExactGPModel(train_x, train_y, likelihood_cal, dimx, torch.tensor(hyps_cal),
                    #              torch.tensor(hyps_cal+1e-5), kernel).to(output_device)
                    cal_gp_delta.eval()
                    calibrated_gps_deltas.append(cal_gp_delta)

        return calibrated_gps_deltas, betas_cal_interp

    def get_randomly_interpolated_betas(self, deltas, ntest):

        checkpoint_size = self.checkpoint_size
        preconditioner_size = self.preconditioner_size
        dimx = self.cal_gps[0][0].train_inputs[0].shape[1]

        train_x = self.train_x
        train_y = self.train_y
        kernel = self.kernel
        z_score_calib_data = self.z_score_calib_data
        gpu = self.gpu

        if gpu:
            output_device = torch.device('cuda:0')
        else:
            output_device = torch.device('cpu')

        with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
                gpytorch.settings.max_preconditioner_size(preconditioner_size), \
                gpytorch.settings.max_cg_iterations(1000000), \
                gpytorch.settings.fast_computations(log_prob=False):

            betas_rand_interp = []
            arg_sort_zscore = torch.argsort(z_score_calib_data).flip(0)
            z_score_sorted = z_score_calib_data[arg_sort_zscore]

            for delta in deltas:

                if 0<delta<1:
                    # for lower, upper in zip(deltas_z_score[:-1], deltas_z_score[1:]):
                    #     if lower <= delta <= upper:
                    #         lower_delta = lower
                    #         upper_delta = upper
                    #         break
                    i_delta = delta*(z_score_sorted.size()[0]-1)
                    i_low = np.int64(np.floor(i_delta))
                    i_high = np.int64(np.ceil(i_delta))
                    betas_rand_interp_delta = z_score_sorted[i_low]*torch.ones(ntest) \
                                              + (z_score_sorted[i_high]-z_score_sorted[i_low])*torch.rand(ntest)
                elif delta ==0:
                    betas_rand_interp_delta = max(z_score_calib_data)*torch.ones(ntest)
                elif delta == 1:
                    betas_rand_interp_delta = min(z_score_calib_data) * torch.ones(ntest)
                betas_rand_interp.append(betas_rand_interp_delta)

        return betas_rand_interp