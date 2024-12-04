# -*- coding: utf-8 -*-

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import pickle
from matplotlib import pyplot as plt
from matplotlib import rc
# automated table. For more details, see https://www.babushk.in/posts/diy-automated-papers.html
import os
import sys
import shelve
from contextlib import closing

def plot_bayesianoptimization(name_saving, datastr):
    # load data

    with open('bayesianoptimizationresults/boresults' + name_saving + '.pkl', 'rb') as f:
        [regret_calib_total, regret_base_total] = pickle.load(f)

    momentary_regret_calib_total = [torch.tensor(reg[1:]) - torch.tensor(reg[:-1]) for reg in regret_calib_total]
    momentary_regret_base_total = [torch.tensor(reg[1:]) - torch.tensor(reg[:-1]) for reg in regret_base_total]

    simple_regret_calib_total = []
    simple_regret_base_total = []
    for reg in momentary_regret_calib_total:
        reg_min = np.inf
        simple_regret_calib = []
        for mom_reg in reg:
            reg_min = min(reg_min, mom_reg.item())
            simple_regret_calib.append(reg_min)
        simple_regret_calib_total.append(simple_regret_calib)

    for reg in momentary_regret_base_total:
        reg_min = np.inf
        simple_regret_base = []
        for mom_reg in reg:
            reg_min = min(reg_min, mom_reg.item())
            simple_regret_base.append(reg_min)
        simple_regret_base_total.append(simple_regret_base)

    # Plot the function, the prediction and the 95% confidence interval
    rc('font', size=26)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    fig = plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot()
    ax.set_axisbelow(True)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    median_calib = torch.tensor(regret_calib_total).quantile(0.5, 0)
    lwdec_calib = torch.tensor(regret_calib_total).quantile(0.1, 0)
    updec_calib = torch.tensor(regret_calib_total).quantile(0.9, 0)

    median_vanilla_naive = torch.tensor(regret_base_total).quantile(0.5, 0)
    lwdec_vanilla_naive = torch.tensor(regret_base_total).quantile(0.1, 0)
    updec_vanilla_naive = torch.tensor(regret_base_total).quantile(0.9, 0)

    iterations = torch.tensor(range(median_calib.size()[0]))

    # plt.plot(deltas_tensor, perc_calib, 'kx', markersize=18, label='Observations', mew=5.0)
    plt.plot(iterations, median_calib, 'b-', linewidth=4.0,
             label='Our approach + UCB')  # 'k-', label=r'$f(x)$', linewidth=4.0)
    # plt.plot(X_test, y_pred, 'r-.', label=r'$\mu_{\vartheta_0}$(x)', linewidth=4.0)
    # plt.plot(X_test, y_predfb, 'r-.', label=r'$\mu_{{FB}}$(x)', linewidth=4.0)
    plt.fill(torch.cat([iterations, torch.flip(iterations, [0])]),
             torch.cat([lwdec_calib,
                        torch.flip(updec_calib, [0])]).detach(),
             alpha=.2, fc='b', ec='None')  # r'$\pm \bar{\beta}^{\frac{1}{2}} {\sigma}_{\vartheta^\prime}(x)$')

    # plot vanilla (naive) approach
    plt.plot(median_vanilla_naive, 'r-', linewidth=4.0,
             label='Vanilla UCB')  # 'k-', label=r'$f(x)$', linewidth=4.0)
    # plt.plot(X_test, y_pred, 'r-.', label=r'$\mu_{\vartheta_0}$(x)', linewidth=4.0)
    # plt.plot(X_test, y_predfb, 'r-.', label=r'$\mu_{{FB}}$(x)', linewidth=4.0)
    plt.fill(torch.cat([iterations, torch.flip(iterations, [0])]),
             torch.cat([lwdec_vanilla_naive,
                        torch.flip(updec_vanilla_naive, [0])]).detach(),
             alpha=.2, fc='r', ec='None')  # r'$\pm \bar{\beta}^{\frac{1}{2}} {\sigma}_{\vartheta^\prime}(x)$')

    plt.legend(loc='upper left', ncol=1, prop={'size': 26})
    # fig = plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    # ax = fig.add_subplot()
    # ax.set_axisbelow(True)
    #
    # # Hide the right and top spines
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # ax.set_xlim(0, 100)
    # handles, labels = ax.get_legend_handles_labels()
    # legend = ax.legend(handles, labels, loc='upper left', ncol=3, prop={'size': 26}, frameon=False)

    ax.set_xlim(-2, iterations[-1])
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative regret')
    plt.tight_layout()
    plt.savefig('bo_regret_' + name_saving + '.pdf', format='pdf')

    fig = plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot()
    ax.set_axisbelow(True)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plot simple regret
    median_calib = torch.tensor(simple_regret_calib_total).quantile(0.5, 0)
    lwdec_calib = torch.tensor(simple_regret_calib_total).quantile(0.1, 0)
    updec_calib = torch.tensor(simple_regret_calib_total).quantile(0.9, 0)

    median_vanilla_naive = torch.tensor(simple_regret_base_total).quantile(0.5, 0)
    lwdec_vanilla_naive = torch.tensor(simple_regret_base_total).quantile(0.1, 0)
    updec_vanilla_naive = torch.tensor(simple_regret_base_total).quantile(0.9, 0)

    iterations = torch.tensor(range(median_calib.size()[0]))

    # plt.plot(deltas_tensor, perc_calib, 'kx', markersize=18, label='Observations', mew=5.0)
    plt.plot(iterations, median_calib, 'b-', linewidth=4.0,
             label='Our approach + UCB')  # 'k-', label=r'$f(x)$', linewidth=4.0)
    # plt.plot(X_test, y_pred, 'r-.', label=r'$\mu_{\vartheta_0}$(x)', linewidth=4.0)
    # plt.plot(X_test, y_predfb, 'r-.', label=r'$\mu_{{FB}}$(x)', linewidth=4.0)
    plt.fill(torch.cat([iterations, torch.flip(iterations, [0])]),
             torch.cat([lwdec_calib,
                        torch.flip(updec_calib, [0])]).detach(),
             alpha=.2, fc='b', ec='None')  # r'$\pm \bar{\beta}^{\frac{1}{2}} {\sigma}_{\vartheta^\prime}(x)$')

    # plot vanilla (naive) approach
    plt.plot(median_vanilla_naive, 'r-', linewidth=4.0,
             label='Vanilla UCB')  # 'k-', label=r'$f(x)$', linewidth=4.0)
    # plt.plot(X_test, y_pred, 'r-.', label=r'$\mu_{\vartheta_0}$(x)', linewidth=4.0)
    # plt.plot(X_test, y_predfb, 'r-.', label=r'$\mu_{{FB}}$(x)', linewidth=4.0)
    plt.fill(torch.cat([iterations, torch.flip(iterations, [0])]),
             torch.cat([lwdec_vanilla_naive,
                        torch.flip(updec_vanilla_naive, [0])]).detach(),
             alpha=.2, fc='r', ec='None')  # r'$\pm \bar{\beta}^{\frac{1}{2}} {\sigma}_{\vartheta^\prime}(x)$')

    plt.legend(loc='upper right', ncol=1, prop={'size': 26})
    # fig = plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    # ax = fig.add_subplot()
    # ax.set_axisbelow(True)
    #
    # # Hide the right and top spines
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # ax.set_xlim(0, 100)
    # handles, labels = ax.get_legend_handles_labels()
    # legend = ax.legend(handles, labels, loc='upper left', ncol=3, prop={'size': 26}, frameon=False)
    ax.set_xlim(-2, iterations[-1])
    plt.ylim(-1, 10)
    plt.xlabel('Iteration')
    plt.ylabel('Simple regret')
    plt.tight_layout()
    plt.savefig('bo_simple_regret_' + name_saving + '.pdf', format='pdf')
