import gpytorch
import torch
import gc
import numpy as np
from copy import deepcopy
from functions.LBFGS import FullBatchLBFGS
from math import pi
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal


def calibration_metric_0(delta,
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor, beta=1
):
    """
    Inspired by Mean Standardized Log Loss, with the exception of only using test_y and dividing by delta
    Reference: Page No. 23,
    Gaussian Processes for Machine Learning,
    Carl Edward Rasmussen and Christopher K. I. Williams,
    The MIT Press, 2006. ISBN 0-262-18253-X
    """
    combine_dim = -2 if isinstance(pred_dist, MultitaskMultivariateNormal) else -1
    # f_mean = pred_dist.mean
    f_var = beta*pred_dist.variance
    # return 0.5 * (torch.log(2 * pi * f_var) + torch.square(test_y - f_mean) / (2 * f_var)).mean(dim=combine_dim)
    return 0.5 * (torch.log(2 * pi * f_var) + torch.square(test_y) / (2 * delta * f_var)).mean(dim=combine_dim)


def calibration_metric_1(delta,
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor, beta=1, tightening=1, eps=1e-6
):
    """
    Inspired by Mean Standardized Log Loss, with the exception of only using test_y and dividing by delta
    Reference: Page No. 23,
    Gaussian Processes for Machine Learning,
    Carl Edward Rasmussen and Christopher K. I. Williams,
    The MIT Press, 2006. ISBN 0-262-18253-X
    """
    combine_dim = -2 if isinstance(pred_dist, MultitaskMultivariateNormal) else -1
    # f_mean = pred_dist.mean
    f_std = beta*pred_dist.stddev
    viol_rate = torch.sigmoid((test_y.abs() - f_std)*tightening).mean() # approximate violation rate
    # cal_metric  = (f_std + 10000*(viol_rate - des_rate)**2).mean(dim=combine_dim)
    cal_metric = (viol_rate - delta) ** 2 + eps*(f_std**2).mean()
    log_cal_metric = torch.log(cal_metric)
    # return 0.5 * (torch.log(2 * pi * f_var) + torch.square(test_y - f_mean) / (2 * f_var)).mean(dim=combine_dim)
    return log_cal_metric


def calibration_metric_2(delta,
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor, beta=1, sign_beta=1, tightening=1, eps=1e-6
):
    """
    Inspired by Mean Standardized Log Loss, with the exception of only using test_y and dividing by delta
    Reference: Page No. 23,
    Gaussian Processes for Machine Learning,
    Carl Edward Rasmussen and Christopher K. I. Williams,
    The MIT Press, 2006. ISBN 0-262-18253-X
    """
    combine_dim = -2 if isinstance(pred_dist, MultitaskMultivariateNormal) else -1
    # f_mean = pred_dist.mean
    f_std = beta*pred_dist.stddev
    viol_rate = torch.sigmoid((test_y - sign_beta*f_std)*tightening).mean() # approximate violation rate
    # cal_metric  = (f_std + 10000*(viol_rate - des_rate)**2).mean(dim=combine_dim)
    cal_metric = (viol_rate - delta) ** 2 + eps*(f_std**2).mean()
    log_cal_metric = torch.log(cal_metric)
    # return 0.5 * (torch.log(2 * pi * f_var) + torch.square(test_y - f_mean) / (2 * f_var)).mean(dim=combine_dim)
    return log_cal_metric


def sharpness(delta,
    pred_dist: MultivariateNormal,beta=1, sign_beta=1):
    """
    Inspired by Mean Standardized Log Loss, with the exception of only using test_y and dividing by delta
    Reference: Page No. 23,
    Gaussian Processes for Machine Learning,
    Carl Edward Rasmussen and Christopher K. I. Williams,
    The MIT Press, 2006. ISBN 0-262-18253-X
    """
    # f_mean = pred_dist.mean
    f_std = beta*pred_dist.stddev # approximate violation rate
    # cal_metric  = (f_std + 10000*(viol_rate - des_rate)**2).mean(dim=combine_dim)
    sharpness_metric = (f_std**2).mean()
    log_sharpness_metric = torch.log(sharpness_metric)
    # return 0.5 * (torch.log(2 * pi * f_var) + torch.square(test_y - f_mean) / (2 * f_var)).mean(dim=combine_dim)
    return log_sharpness_metric
