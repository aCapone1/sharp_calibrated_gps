import torch
from gpregression.ExactGPModel import ExactGPModel
from gpytorch.means import ZeroMean
from gpytorch.distributions import MultivariateNormal
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound, AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf


from abc import ABC
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions import UnsupportedError
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from torch import Tensor
from torch.distributions import Normal

import gpytorch
import numpy as np

def run_bo(gpu, X_data, y_data, n_bo_iter, first_gp, second_gp,
           beta, likelihood, get_noisy_data, eval_fun, max_val, decision_set = False):

    dimx = X_data.shape[1]

    if gpu:
        output_device = torch.device('cuda:0')
    else:
        output_device = torch.device('cpu')

    regret = []
    regret_it = 0
    for bo_iter in range(n_bo_iter):
        if gpu:
            # make continguous
            train_x = torch.Tensor(X_data).detach().contiguous().to(output_device)
            train_y = torch.Tensor(y_data).detach().contiguous().to(output_device)

            # generate gps with new data
            first_gp = ExactGPModel(train_x, train_y, likelihood, kernel=first_gp.covar_module).to(output_device)
            second_gp = ExactGPModel(train_x, train_y, likelihood, kernel=second_gp.covar_module).to(output_device)

        else:
            #
            train_x = torch.tensor(X_data).detach()
            train_y = torch.Tensor(y_data).detach()

            # generate gps with new data
            first_gp = ExactGPModel(train_x, train_y, likelihood, kernel=first_gp.covar_module)
            second_gp = ExactGPModel(train_x, train_y, likelihood, kernel=second_gp.covar_module)

        # get sequence of zeros to use as training data for GP - the posterior mean is then defined uniquely by the
        # prior mean
        train_zeros = torch.zeros(y_data.shape).reshape(y_data.size, 1)
        #
        # gp = DoubleGP(train_x, train_zeros, likelihood, covar_module=second_gp.covar_module)
        # gp.mean_module = gpmean(first_gp)
        train_y_reshaped = train_y.reshape(y_data.size, 1)
        gp1 = SingleTaskGP(train_x, train_y_reshaped, likelihood, first_gp.covar_module)
        gp2 = SingleTaskGP(train_x, train_zeros, likelihood, second_gp.covar_module)
        # gp = SimpleCustomGP(train_x, train_zeros, likelihood, first_gp,second_gp)
        # mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
        # fit_gpytorch_model(mll)

        UCBD = DoubleGPUpperConfidenceBound(model=gp1, model2=gp2, beta=beta)

        bounds = torch.stack([torch.zeros(dimx), torch.ones(dimx)])
        if decision_set:
            candidate, acq_value = optimize_acqf(
                UCBD, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
            )
        else:
            candidate, acq_value = optimize_acqf(
                UCBD, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
            )

            # collect noisy measurement
        y_candidate = get_noisy_data(candidate).detach()

        # update data
        X_data = np.concatenate([X_data, candidate])
        y_data = np.concatenate([y_data, y_candidate])

        regret_it += max_val - eval_fun(candidate)

        regret.append(regret_it.item())

    return regret


class gpmean(gpytorch.means.Mean):
    def __init__(self, gpmodel):
        super(gpmean, self).__init__()
        gpmodel.eval()
        self.gpmodel = gpmodel

    def forward(self, x):
        gpmean = self.gpmodel(x.double()).mean
        return gpmean.squeeze()

from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model

class DoublePosteriorGP(Model):

    def __init__(self, first_gp, second_gp):
        super().__init__()
        self.first_gp = first_gp
        self.second_gp = second_gp

    def posterior(self, x, posterior_transform):
        mean_x = self.first_gp.mean_module(x)
        covar_x = self.second_gp.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SimpleCustomGP(SingleTaskGP):

    def __init__(self, train_X, train_Y, likelihood, first_gp, second_gp):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y, likelihood)
        self.first_gp = first_gp
        self.second_gp = second_gp
        self.mean_module = ZeroMean()
        self.covar_module = second_gp.covar_module
        self.to(train_X)  # make sure we're on the right device/dtype

    def posterior(self, X, posterior_transform):
        mean_x = self.first_gp.mean_module(X)
        covar_x = self.second_gp.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)


class DoubleGPUpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound using two different GPs (UCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation of a second GP weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu1(x) + sqrt(beta) * sigma2(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> UCB = UpperConfidenceBound(model, beta=0.2)
        >>> ucb = UCB(test_X)
    """

    def __init__(
        self,
        model: Model,
        model2: Model,
        beta: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.maximize = maximize
        self.model2 = model2
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """
        self.beta = self.beta.to(X)
        posterior1 = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        posterior2 = self.model2.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior1.mean
        view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
        mean = mean.view(view_shape)
        variance = posterior2.variance.view(view_shape)
        delta = (self.beta.expand_as(mean) * variance).sqrt()
        if self.maximize:
            return mean + delta
        else:
            return -mean + delta


#
# class UpperConfidenceBound_DoubleGP(UpperConfidenceBound):
#
#     def __init__(
#         self,
#         first_model: Model,
#         second_model: Model,
#         beta: Union[float, Tensor],
#         **kwargs):
#         super().__init__(first_model, beta=beta)
#         r"""Single-outcome Upper Confidence Bound.
#
#         Args:
#             model: A fitted single-outcome GP model (must be in batch mode if
#                 candidate sets X will be)
#             beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
#                 representing the trade-off parameter between mean and covariance
#             posterior_transform: A PosteriorTransform. If using a multi-output model,
#                 a PosteriorTransform that transforms the multi-output posterior into a
#                 single-output posterior is required.
#             maximize: If True, consider the problem a maximization problem.
#         """
#         super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
#         self.maximize = maximize
#         if not torch.is_tensor(beta):
#             beta = torch.tensor(beta)
#         self.register_buffer("beta", beta)
#
#     @t_batch_mode_transform(expected_q=1)
#     def forward(self, X: Tensor) -> Tensor:
#         r"""Evaluate the Upper Confidence Bound on the candidate set X.
#
#         Args:
#             X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
#
#         Returns:
#             A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
#             given design points `X`.
#         """
#         self.beta = self.beta.to(X)
#         posterior = self.model.posterior(
#             X=X, posterior_transform=self.posterior_transform
#         )
#         mean = posterior.mean
#         view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
#         mean = mean.view(view_shape)
#         variance = posterior.variance.view(view_shape)
#         delta = (self.beta.expand_as(mean) * variance).sqrt()
#         if self.maximize:
#             return mean + delta
#         else:
#             return -mean + delta
