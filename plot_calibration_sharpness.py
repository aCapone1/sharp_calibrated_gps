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

class store:
    def __init__(self, build='.'):
        name = os.path.basename(sys.argv[0])
        name += '.shelve'
        self.name = os.path.join(build, name)

    def __getitem__(self, key):
        with closing(shelve.open(self.name)) as db:
            value = db[key]
        return value

    def __setitem__(self, key, value):
        with closing(shelve.open(self.name)) as db:
            db[key] = value

def print_calibration_results(name_saving, datastr, plot=False):

    # load data

    with open('regressionresults/percanderrs' + name_saving + datastr + '.pkl', 'rb') as f:
        [perc_robust, perc_cal_total, perc_vanilla_naive_total,
         perc_fullbayes_naive_total, perc_robust_kuleshov,
         perc_robust_random, perc_robust_varfree, perc_fullbayes_naive_total,
         NLL_cal_total, NLL_kuleshov_total, NLL_varfree_total,
         NLL_random_total, NLL_vanilla_total, NLL_fullbayes_total,
         CI95_cal_total, CI95_kuleshov_total, CI95_varfree_total,
         CI95_random_total, CI95_vanilla_total, CI95_fullbayes_total,
         sharpness_cal_total, sharpness_kuleshov_total,
         sharpness_random_total, sharpness_varfree_total,
         sharpness_vanilla_naive_total, sharpness_fullbayes_naive_total, deltas] = pickle.load(f)


    deltas_tensor = torch.tensor(np.multiply(deltas,100))


    print('Expected calibration error metric (mean):')
    print('Our approach %f \u00B1 %f:, Kuleshov %f\u00B1 %f:, Vovk: %f\u00B1 %f, '
          'Marx: %f\u00B1 %f, Vanilla (naive): %f\u00B1 %f, Full Bayes (naive): %f\u00B1 %f' %
          (((torch.tensor(perc_cal_total) / 100 - torch.tensor(deltas)) ** 2).mean(),
           ((torch.tensor(perc_cal_total) / 100 - torch.tensor(deltas)) ** 2).std(),
           ((torch.tensor(perc_robust_kuleshov) / 100 - torch.tensor(deltas)) ** 2).mean(),
           ((torch.tensor(perc_robust_kuleshov) / 100 - torch.tensor(deltas)) ** 2).std(),
           ((torch.tensor(perc_robust_random) / 100 - torch.tensor(deltas)) ** 2).mean(),
           ((torch.tensor(perc_robust_random) / 100 - torch.tensor(deltas)) ** 2).std(),
           ((torch.tensor(perc_robust_varfree) / 100 - torch.tensor(deltas) )** 2).mean(),
           ((torch.tensor(perc_robust_varfree) / 100 - torch.tensor(deltas)) ** 2).std(),
           ((torch.tensor(perc_vanilla_naive_total) / 100 - torch.tensor(deltas)) ** 2).mean(),
           ((torch.tensor(perc_vanilla_naive_total) / 100 - torch.tensor(deltas)) ** 2).std(),
           ((torch.tensor(perc_fullbayes_naive_total) / 100 - torch.tensor(deltas)) ** 2).mean(),
           ((torch.tensor(perc_fullbayes_naive_total) / 100 - torch.tensor(deltas)) ** 2).std()))


    print('Sharpness metric (mean):')
    print('Our approach %f \u00B1 %f:, Kuleshov %f\u00B1 %f:, Vovk: %f\u00B1 %f, '
          'Marx: %f\u00B1 %f, Vanilla (naive): %f\u00B1 %f, Full Bayes (naive): %f\u00B1 %f' %
          ((torch.tensor(sharpness_cal_total)).mean(), (torch.tensor(sharpness_cal_total)).std(),
           (torch.tensor(sharpness_kuleshov_total)).mean(), (torch.tensor(sharpness_kuleshov_total)).std(),
           (torch.tensor(sharpness_random_total)).mean(), (torch.tensor(sharpness_random_total)).std(),
           (torch.tensor(sharpness_varfree_total)).mean(), (torch.tensor(sharpness_varfree_total)).std(),
           (torch.tensor(sharpness_vanilla_naive_total)).mean(), (torch.tensor(sharpness_vanilla_naive_total)).std(),
           (torch.tensor(sharpness_fullbayes_naive_total)).mean(), (torch.tensor(sharpness_fullbayes_naive_total)).std()))


    print('95 percent CI metric (mean):')
    print('Our approach %f \u00B1 %f:, Kuleshov %f\u00B1 %f:, Vovk: %f\u00B1 %f, '
          'Marx: %f\u00B1 %f, Vanilla (naive): %f\u00B1 %f, Full Bayes (naive): %f\u00B1 %f' %
          ((torch.tensor(CI95_cal_total)).mean(), (torch.tensor(CI95_cal_total)).std(),
           (torch.tensor(CI95_kuleshov_total)).mean(), (torch.tensor(CI95_kuleshov_total)).std(),
           (torch.tensor(CI95_random_total)).mean(), (torch.tensor(CI95_random_total)).std(),
           (torch.tensor(CI95_varfree_total)).mean(), (torch.tensor(CI95_varfree_total)).std(),
           (torch.tensor(CI95_vanilla_total)).mean(), (torch.tensor(CI95_vanilla_total)).std(),
           (torch.tensor(CI95_fullbayes_total)).mean(), (torch.tensor(CI95_fullbayes_total)).std()))

    print('NLL metric (mean):')
    print('Our approach %f \u00B1 %f:, Kuleshov %f\u00B1 %f:, Vovk: %f\u00B1 %f, '
          'Marx: %f\u00B1 %f, Vanilla (naive): %f\u00B1 %f, Full Bayes (naive): %f\u00B1 %f' %
          ((torch.tensor(NLL_cal_total)).mean(), (torch.tensor(NLL_cal_total)).std(),
           (torch.tensor(NLL_kuleshov_total)).mean(), (torch.tensor(NLL_kuleshov_total)).std(),
           (torch.tensor(NLL_random_total)).mean(), (torch.tensor(NLL_random_total)).std(),
           (torch.tensor(NLL_varfree_total)).mean(), (torch.tensor(NLL_varfree_total)).std(),
           (torch.tensor(NLL_vanilla_total)).mean(), (torch.tensor(NLL_vanilla_total)).std(),
           (torch.tensor(NLL_fullbayes_total)).mean(), (torch.tensor(NLL_fullbayes_total)).std()))

    if plot:
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

        median_calib = 100-torch.tensor(perc_calib).quantile(0.5, 0)
        lwdec_calib = 100-torch.tensor(perc_calib).quantile(0.1, 0)
        updec_calib = 100-torch.tensor(perc_calib).quantile(0.9, 0)

        median_kuleshov = 100-torch.tensor(perc_robust_kuleshov).quantile(0.5, 0)
        lwdec_kuleshov = 100-torch.tensor(perc_robust_kuleshov).quantile(0.1, 0)
        updec_kuleshov = 100-torch.tensor(perc_robust_kuleshov).quantile(0.9, 0)

        median_vanilla_naive = 100-torch.tensor(perc_vanilla_naive).quantile(0.5, 0)
        lwdec_vanilla_naive = 100-torch.tensor(perc_vanilla_naive).quantile(0.1, 0)
        updec_vanilla_naive = 100-torch.tensor(perc_vanilla_naive).quantile(0.9, 0)

        median_fullbayes_naive = 100-torch.tensor(perc_fullbayes_naive).quantile(0.5, 0)
        lwdec_fullbayes_naive = 100-torch.tensor(perc_fullbayes_naive).quantile(0.1, 0)
        updec_fullbayes_naive = 100-torch.tensor(perc_fullbayes_naive).quantile(0.9, 0)

        median_sharpness_calib = torch.tensor(sharpness_cal_total).quantile(0.5, 0)
        lwdec_sharpness_calib = torch.tensor(sharpness_cal_total).quantile(0.1, 0)
        updec_sharpness_calib = torch.tensor(sharpness_cal_total).quantile(0.9, 0)

        median_sharpness_vanilla_naive = torch.tensor(sharpness_vanilla_naive).quantile(0.5, 0)
        lwdec_sharpness_vanilla_naive = torch.tensor(sharpness_vanilla_naive).quantile(0.1, 0)
        updec_sharpness_vanilla_naive = torch.tensor(sharpness_vanilla_naive).quantile(0.9, 0)

        median_sharpness_fullbayes = torch.tensor(sharpness_fullbayes_naive).quantile(0.5, 0)
        lwdec_sharpness_fullbayes = torch.tensor(sharpness_fullbayes_naive).quantile(0.1, 0)
        updec_sharpness_fullbayes = torch.tensor(sharpness_fullbayes_naive).quantile(0.9, 0)

        median_sharpness_kuleshov = torch.tensor(sharpness_kuleshov_total).quantile(0.5, 0)
        lwdec_sharpness_kuleshov = torch.tensor(sharpness_kuleshov_total).quantile(0.1, 0)
        updec_sharpness_kuleshov = torch.tensor(sharpness_kuleshov_total).quantile(0.9, 0)

        plt.plot(100-deltas_tensor,100-deltas_tensor, 'k--', linewidth=4.0,
                 label='Desired')
        plt.legend(loc='upper left', ncol=3, prop={'size': 26})
        # plt.plot(deltas_tensor, perc_calib, 'kx', markersize=18, label='Observations', mew=5.0)
        plt.plot(100-deltas_tensor, median_calib, 'b-', linewidth=4.0,
                 label='Our approach') #'k-', label=r'$f(x)$', linewidth=4.0)
        # plt.plot(X_test, y_pred, 'r-.', label=r'$\mu_{\vartheta_0}$(x)', linewidth=4.0)
        # plt.plot(X_test, y_predfb, 'g-.', label=r'$\mu_{{FB}}$(x)', linewidth=4.0)
        plt.fill(torch.cat([100-deltas_tensor, torch.flip(100-deltas_tensor, [0])]),
                 torch.cat([lwdec_calib,
                            torch.flip(updec_calib, [0])]).detach(),
                 alpha=.2, fc='b', ec='None')  # r'$\pm \bar{\beta}^{\frac{1}{2}} {\sigma}_{\vartheta^\prime}(x)$')

        # plot vanilla (naive) approach
        plt.plot(100 - deltas_tensor, median_vanilla_naive, 'g-', linewidth=4.0,
                 label='Vanilla (naive)')  # 'k-', label=r'$f(x)$', linewidth=4.0)
        # plt.plot(X_test, y_pred, 'r-.', label=r'$\mu_{\vartheta_0}$(x)', linewidth=4.0)
        # plt.plot(X_test, y_predfb, 'g-.', label=r'$\mu_{{FB}}$(x)', linewidth=4.0)
        plt.fill(torch.cat([100 - deltas_tensor, torch.flip(100 - deltas_tensor, [0])]),
                 torch.cat([lwdec_vanilla_naive,
                            torch.flip(updec_vanilla_naive, [0])]).detach(),
                 alpha=.2, fc='g', ec='None')  # r'$\pm \bar{\beta}^{\frac{1}{2}} {\sigma}_{\vartheta^\prime}(x)$')

        # plot fully bayesian (naive) approach
        plt.plot(100 - deltas_tensor, median_fullbayes_naive, 'm-', linewidth=4.0,
                 label='Full Bayes')  # 'k-', label=r'$f(x)$', linewidth=4.0)
        # plt.plot(X_test, y_pred, 'r-.', label=r'$\mu_{\vartheta_0}$(x)', linewidth=4.0)
        # plt.plot(X_test, y_predfb, 'g-.', label=r'$\mu_{{FB}}$(x)', linewidth=4.0)
        plt.fill(torch.cat([100 - deltas_tensor, torch.flip(100 - deltas_tensor, [0])]),
                 torch.cat([lwdec_fullbayes_naive,
                            torch.flip(updec_fullbayes_naive, [0])]).detach(),
                 alpha=.2, fc='m', ec='None')  # r'$\pm \bar{\beta}^{\frac{1}{2}} {\sigma}_{\vartheta^\prime}(x)$')
        #
        # fig = plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
        # ax = fig.add_subplot()
        # ax.set_axisbelow(True)
        #
        # # Hide the right and top spines
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)

        # plt.plot(deltas_tensor, perc_calib, 'kx', markersize=18, label='Observations', mew=5.0)
        plt.plot(100-deltas_tensor, median_kuleshov, 'r-', linewidth=4.0,
                 label='Kuleshov')  # 'k-', label=r'$f(x)$', linewidth=4.0)
        plt.fill(torch.cat([100-deltas_tensor, torch.flip(100-deltas_tensor, [0])]),
                 torch.cat([lwdec_kuleshov,
                            torch.flip(updec_kuleshov, [0])]).detach(),
                 alpha=.2, fc='r', ec='None')  # r'$\pm \bar{\beta}^{\frac{1}{2}} {\sigma}_{\vartheta^\prime}(x)$')
        ax.set_xlim(0, 100)
        # handles, labels = ax.get_legend_handles_labels()
        # legend = ax.legend(handles, labels, loc='upper left', ncol=3, prop={'size': 26}, frameon=False)
        plt.xlabel('Expected confidence level')
        plt.ylabel('Observed confidence level')
        plt.tight_layout()
        plt.savefig('calibration' + name_saving + datastr + '.pdf', format='pdf')

        # plot sharpness

        fig = plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot()
        ax.set_axisbelow(True)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # plt.plot(deltas_tensor, perc_calib, 'kx', markersize=18, label='Observations', mew=5.0)
        plt.plot(100-deltas_tensor, median_sharpness_calib, 'b-', linewidth=4.0,
                 label='Our Approach') #'k-', label=r'$f(x)$', linewidth=4.0)
        # plt.plot(X_test, y_pred, 'r-.', label=r'$\mu_{\vartheta_0}$(x)', linewidth=4.0)
        # plt.plot(X_test, y_predfb, 'g-.', label=r'$\mu_{{FB}}$(x)', linewidth=4.0)
        plt.fill(torch.cat([100-deltas_tensor, torch.flip(100-deltas_tensor, [0])]),
                 torch.cat([lwdec_sharpness_calib,
                            torch.flip(updec_sharpness_calib, [0])]).detach(),
                 alpha=.2, fc='b', ec='None')  # r'$\pm \bar{\beta}^{\frac{1}{2}} {\sigma}_{\vartheta^\prime}(x)$')

        plt.plot(100-deltas_tensor, median_sharpness_vanilla_naive, 'g-', linewidth=4.0,
                 label='Vanilla (naive)')  # 'k-', label=r'$f(x)$', linewidth=4.0)
        # plt.plot(X_test, y_pred, 'r-.', label=r'$\mu_{\vartheta_0}$(x)', linewidth=4.0)
        # plt.plot(X_test, y_predfb, 'g-.', label=r'$\mu_{{FB}}$(x)', linewidth=4.0)
        plt.fill(torch.cat([100-deltas_tensor, torch.flip(100-deltas_tensor, [0])]),
                 torch.cat([lwdec_sharpness_vanilla_naive,
                            torch.flip(updec_sharpness_vanilla_naive, [0])]).detach(),
                 alpha=.2, fc='g', ec='None')  # r'$\pm \bar{\beta}^{\frac{1}{2}} {\sigma}_{\vartheta^\prime}(x)$')

        plt.plot(100-deltas_tensor, median_sharpness_fullbayes, 'm-', linewidth=4.0,
                 label='Full Bayes')  # 'k-', label=r'$f(x)$', linewidth=4.0)
        # plt.plot(X_test, y_pred, 'r-.', label=r'$\mu_{\vartheta_0}$(x)', linewidth=4.0)
        # plt.plot(X_test, y_predfb, 'g-.', label=r'$\mu_{{FB}}$(x)', linewidth=4.0)
        plt.fill(torch.cat([100-deltas_tensor, torch.flip(100-deltas_tensor, [0])]),
                 torch.cat([lwdec_sharpness_fullbayes,
                            torch.flip(updec_sharpness_fullbayes, [0])]).detach(),
                 alpha=.2, fc='m', ec='None')  # r'$\pm \bar{\beta}^{\frac{1}{2}} {\sigma}_{\vartheta^\prime}(x)$')
        #
        # fig = plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
        # ax = fig.add_subplot()
        # ax.set_axisbelow(True)
        #
        # # Hide the right and top spines
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)

        # plt.plot(100-deltas_tensor, perc_calib, 'kx', markersize=18, label='Observations', mew=5.0)
        plt.plot(100-deltas_tensor, median_sharpness_kuleshov, 'r-', linewidth=4.0,
                 label='Vanilla + Kuleshov')  # 'k-', label=r'$f(x)$', linewidth=4.0)
        # plt.plot(X_test, y_pred, 'r-.', label=r'$\mu_{\vartheta_0}$(x)', linewidth=4.0)
        # plt.plot(X_test, y_predfb, 'g-.', label=r'$\mu_{{FB}}$(x)', linewidth=4.0)
        plt.fill(torch.cat([100-deltas_tensor, torch.flip(100-deltas_tensor, [0])]),
                 torch.cat([lwdec_sharpness_kuleshov,
                            torch.flip(updec_sharpness_kuleshov, [0])]).detach(),
                 alpha=.2, fc='r', ec='None')  # r'$\pm \bar{\beta}^{\frac{1}{2}} {\sigma}_{\vartheta^\prime}(x)$')
        ax.set_xlim(0, 100)
        plt.xlabel('Expected confidence level')
        plt.ylabel('Log sharpness')

        # plt.legend(loc='upper right', ncol=3, prop={'size': 26})


        # save legend separately
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -10), ncol=5, prop={'size': 26}, frameon=False)
        leg = legend.figure
        leg.canvas.draw()
        bbox = legend.get_window_extent().transformed(leg.dpi_scale_trans.inverted())
        leg.savefig('legend.pdf', format='pdf', bbox_inches=bbox)
        legend.remove()
        plt.tight_layout()
        ax.set_yscale('log')
        plt.savefig('sharpness' + name_saving + datastr + '.pdf', format='pdf')

        # plt.savefig('safegp_N' + X_data.shape[0].__str__() + '.pdf', format='pdf')

        # plt.show()

