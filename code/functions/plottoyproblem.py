# -*- coding: utf-8 -*-

import numpy as np
import torch
import gpytorch
from gpytorch.priors import UniformPrior
from matplotlib import pyplot as plt
from matplotlib import rc

def plottoyproblem(X_data, y_data, f_true, X_test, gp_mean, gp_stddev_minus, gp_stddev_plus,
                   beta_minus, beta_plus, delta, color, name=False, chol_jitter=1e-6):

    beta05 = 0.675
    beta09 = 1.645

    deltapercent = delta*100
    X_total = torch.cat([X_test, X_data]).sort()
    X_total = X_total[0]

    if name == 'fullbayes':
        with gpytorch.settings.cholesky_jitter(chol_jitter):
            fbmean = gp_mean(X_total).mean
            # fbmean = gp_mean
            fbstddev = gp_stddev_plus(X_total).stddev
            # fbstddev = gp_stddev
            num_samples = fbmean.size()[0]
            # transpose means and standard deviations of fully Bayesian model for Gaussian mixture model
            fbmeans = torch.transpose(fbmean, 0, 1)
            fbstddevs = torch.transpose(fbstddev, 0, 1)

            # create set of normally distributed variables with means fbmeans and standard deviations fbstddevs
            fbN = torch.distributions.Normal(fbmeans, fbstddevs)

            # generate weights (ones) for Gaussian mixture models
            fbgmmweights = torch.distributions.Categorical(torch.ones(num_samples, ))

            # create Gaussian mixture model using fully Bayesian GP evaluations
            fbgmm = torch.distributions.mixture_same_family.MixtureSameFamily(fbgmmweights, fbN)
            mean = fbgmm.mean.detach()
            stddev_minus = fbgmm.stddev.detach()
            stddev_plus = fbgmm.stddev.detach()
    elif name == 'varfree':
        mean = gp_mean(X_total).mean.detach()
        # mean = gp_mean
        stddev_minus = 1
        stddev_plus = 1
    elif name == 'random':
        mean = gp_mean(X_total).mean.detach()
        # mean = gp_mean
        stddev_minus = gp_stddev_minus(X_total).stddev.detach()*beta_minus
        stddev_plus = gp_stddev_minus(X_total).stddev.detach()*beta_plus
        beta_minus=1
        beta_plus=1
    else:
        mean = gp_mean(X_total).mean.detach()
        # mean = gp_mean
        stddev_minus = gp_stddev_minus(X_total).stddev.detach()
        stddev_plus = gp_stddev_plus(X_total).stddev.detach()
        # stddev = gp_stddev
    # Plot the function, the prediction and the 95% confidence interval
    rc('font', size=50)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    plt.close('all')
    fig = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot()
    ax.set_axisbelow(True)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(X_data, y_data, 'kx', markersize=14, label='Training Data', mew=2.0)
    plt.plot(X_test, f_true, 'ks', markersize=7, label=r'Test Data', mew=0.5)

    plt.fill(torch.cat([X_total, torch.flip(X_total, [0])]),
             torch.cat([mean + beta_minus * stddev_minus,
                        torch.flip(mean + beta_plus * stddev_plus, [0])]).detach(),
             alpha=.2, fc=color, ec='None',
             label='%d percent Confidence Level' % deltapercent)  # r'$\pm \beta^{\frac{1}{2}} \sigma_{\vartheta_0}(x)$')
    plt.plot(X_total, mean, color, label=r'$\mu$(x)', linewidth=4.0)

    plt.xlabel('$x$')
    if name == 'ours' or name=='varfree':
        plt.ylabel('$f(x)$')
    # ylim_UB = 3 # 2**(torch.ceil(torch.log2(torch.max(y_pred+ beta_safe * sig_safepr)))+1)
    # ylim_LB = -1.5 #-2**(torch.ceil(torch.log2(torch.abs(torch.min(y_pred- beta_safe * sig_safepr)))))
    # plt.ylim(ylim_LB, ylim_UB)
    plt.ylim(-12, 17)
    plt.xlim(X_total.min(), X_total.max())
    plt.tick_params(
        axis='both',  # changes apply to both axes
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,
        top=False,  # ticks along the top edge are off
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off
    plt.tight_layout()

    plt.savefig('regressionresults/toy_problem_95interval_' + name + '.pdf' , format='pdf')

    if name == 'vanilla':
        print_name = 'Vanilla (naive)'
    elif name == 'fullbayes':
        print_name = 'Full Bayes (naive)'
    elif name == 'ours':
        print_name = 'Our approach'
    elif name == 'kuleshov':
        print_name = 'Kuleshov et al. (2018)'
    elif name == 'varfree':
        print_name = 'Marx et al. (2022)'
    elif name == 'random':
        print_name = 'Vovk et al. (2020)'
        
    print('Showing results using the following approach: ' + print_name + '. Close figure window to continue.')
    plt.show()
