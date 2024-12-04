import torch
from torch.autograd import Variable
import gpytorch
from gpytorch.priors import UniformPrior


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dimx=False, lb=False, ub=False,
                 kernel=False, hyp0=[False], sparse = False, n_inducing_points = 100):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        if not kernel==False:
            if kernel._get_name()=='SpectralMixtureKernel':
                kernel.initialize_from_data(train_x, train_y)
        else:
            kernel_base = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=dimx,
                                           lengthscale_prior=UniformPrior(lb[1:-1],
                                                                          ub[1:-1]),
                                           lengthscale_constraint=gpytorch.constraints.Interval(
                                               lb[1:-1], ub[1:-1])),
                outputscale_prior=UniformPrior(lb[0], ub[0]),
                outputscale_constraint=gpytorch.constraints.Interval(lb[0], ub[0])
            )
            if any(hyp0):
                kernel_base.outputscale = Variable(hyp0[0].clone().detach().requires_grad_(True), requires_grad=True)
                kernel_base.base_kernel.lengthscale = Variable(hyp0[1:-1].clone().detach().requires_grad_(True), requires_grad=True)
                self.likelihood.noise = Variable(hyp0[-1].clone().detach().requires_grad_(True), requires_grad=True)
            if sparse:
                self.base_covar_module  = kernel_base
                kernel = gpytorch.kernels.InducingPointKernel(self.base_covar_module,
                                                     inducing_points=train_x[:n_inducing_points, :], likelihood=likelihood)
            else:
                kernel = kernel_base
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, lb=False, ub=False, kernel=False):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(ApproximateGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        if kernel == False:
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=dimx,
                                           lengthscale_prior=UniformPrior(lb[1:-1],
                                                                          ub[1:-1]),
                                           lengthscale_constraint=gpytorch.constraints.Interval(
                                               lb[1:-1], ub[1:-1])),
                outputscale_prior=UniformPrior(lb[0], ub[0]),
                outputscale_constraint=gpytorch.constraints.Interval(lb[0], ub[0])
            )
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)