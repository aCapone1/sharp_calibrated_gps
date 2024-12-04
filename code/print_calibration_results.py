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

def print_calibration_results(name_saving, datastr, plot=False, forlatex=False):

    # load data

    with open('regressionresults/percanderrs' + name_saving + datastr + '.pkl', 'rb') as f:
        [perc_robust, perc_cal_total, perc_vanilla_naive_total,
         perc_fullbayes_naive_total, perc_robust_kuleshov,
         perc_robust_random, perc_robust_varfree, perc_fullbayes_naive_total,
         NLL_cal_total, NLL_kuleshov_total, NLL_varfree_total,
         NLL_random_total, NLL_vanilla_total, NLL_fullbayes_total,
         sigma_cal_total, sigma_kuleshov_total, sigma_varfree_total, sigma_random_total,
         CI95_cal_total, CI95_kuleshov_total, CI95_varfree_total,
         CI95_random_total, CI95_vanilla_total, CI95_fullbayes_total, deltas] = pickle.load(f)


    deltas_tensor = torch.tensor(np.multiply(deltas,100))

    if forlatex:
        mathtext = '%.2g \u00B1 %.2g, & %.2g\u00B1 %.2g, & %.2g\u00B1 %.2g, & %.2g\u00B1 %.2g, & %.2g\u00B1 %.2g, & %.2g\u00B1 %.2g'
        mathtext_novanilla = '%.2g \u00B1 %.2g, & %.2g\u00B1 %.2g, & %.2g\u00B1 %.2g, & %.2g\u00B1 %.2g'
    else:
        mathtext = '%.2g \u00B1 %.2g, %.2g\u00B1 %.2g, %.2g\u00B1 %.2g, %.2g\u00B1 %.2g, %.2g\u00B1 %.2g, %.2g\u00B1 %.2g'
        mathtext_novanilla = '%.2g \u00B1 %.2g, %.2g\u00B1 %.2g, %.2g\u00B1 %.2g, %.2g\u00B1 %.2g'

    print('\nExpected calibration error metric (mean):')
    print('Our approach, Kuleshov, Vovk, Marx, Vanilla (naive), Full Bayes (naive):\n' +mathtext %
          ((((torch.tensor(perc_cal_total) / 100 - torch.tensor(deltas)) ** 2).mean(1)).mean(),
           (((torch.tensor(perc_cal_total) / 100 - torch.tensor(deltas)) ** 2).mean(1)).std(),
           (((torch.tensor(perc_robust_kuleshov) / 100 - torch.tensor(deltas)) ** 2).mean(1)).mean(),
           (((torch.tensor(perc_robust_kuleshov) / 100 - torch.tensor(deltas)) ** 2).mean(1)).std(),
           (((torch.tensor(perc_robust_random) / 100 - torch.tensor(deltas)) ** 2).mean(1)).mean(),
           (((torch.tensor(perc_robust_random) / 100 - torch.tensor(deltas)) ** 2).mean(1)).std(),
           (((torch.tensor(perc_robust_varfree) / 100 - torch.tensor(deltas)) ** 2).mean(1)).mean(),
           (((torch.tensor(perc_robust_varfree) / 100 - torch.tensor(deltas)) ** 2).mean(1)).std(),
           (((torch.tensor(perc_vanilla_naive_total) / 100 - torch.tensor(deltas)) ** 2).mean(1)).mean(),
           (((torch.tensor(perc_vanilla_naive_total) / 100 - torch.tensor(deltas)) ** 2).mean(1)).std(),
           (((torch.tensor(perc_fullbayes_naive_total) / 100 - torch.tensor(deltas)) ** 2).mean(1)).mean(),
           (((torch.tensor(perc_fullbayes_naive_total) / 100 - torch.tensor(deltas)) ** 2).mean(1)).std()))

    print('\n95 percent CI metric (mean):')
    print('Our approach, Kuleshov, Vovk, Marx, Vanilla (naive), Full Bayes (naive):\n' +mathtext %
          ((torch.tensor(CI95_cal_total)).mean(), (torch.tensor(CI95_cal_total)).std(),
           (torch.tensor(CI95_kuleshov_total)).mean(), (torch.tensor(CI95_kuleshov_total)).std(),
           (torch.tensor(CI95_random_total)).mean(), (torch.tensor(CI95_random_total)).std(),
           (torch.tensor(CI95_varfree_total)).mean(), (torch.tensor(CI95_varfree_total)).std(),
           (torch.tensor(CI95_vanilla_total)).mean(), (torch.tensor(CI95_vanilla_total)).std(),
           (torch.tensor(CI95_fullbayes_total)).mean(), (torch.tensor(CI95_fullbayes_total)).std()))

    print('\nNLL metric (mean):')
    print('Our approach, Kuleshov, Vovk, Marx, Vanilla (naive), Full Bayes (naive):\n' +mathtext %
          ((torch.tensor(NLL_cal_total)).mean(), (torch.tensor(NLL_cal_total)).std(),
           (torch.tensor(NLL_kuleshov_total)).mean(), (torch.tensor(NLL_kuleshov_total)).std(),
           (torch.tensor(NLL_random_total)).mean(), (torch.tensor(NLL_random_total)).std(),
           (torch.tensor(NLL_varfree_total)).mean(), (torch.tensor(NLL_varfree_total)).std(),
           (torch.tensor(NLL_vanilla_total)).mean(), (torch.tensor(NLL_vanilla_total)).std(),
           (torch.tensor(NLL_fullbayes_total)).mean(), (torch.tensor(NLL_fullbayes_total)).std()))


    print('\nSigma metric (mean):')
    print('Our approach, Kuleshov, Vovk, Marx:\n' + mathtext_novanilla %
          ((torch.tensor(sigma_cal_total)).mean(), (torch.tensor(sigma_cal_total)).std(),
           (torch.tensor(sigma_kuleshov_total)).mean(), (torch.tensor(sigma_kuleshov_total)).std(),
           (torch.tensor(sigma_random_total)).mean(), (torch.tensor(sigma_random_total)).std(),
           (torch.tensor(sigma_varfree_total)).mean(), (torch.tensor(sigma_varfree_total)).std()))
