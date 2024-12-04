import numpy as np
import torch
import scipy.stats as st
from copy import deepcopy

max_test_partition = 1000

def evaleceandsharpness(f_true, mean, stddev, beta):
    # METHODS:
    # -2 = Naive
    # -1 = Vovk et al., 2005
    # 0 = Ours
    # 1 = Kuleshov et al., 2018
    # 2 = Marx et al., 2022
    # 3 = Vovk et al., 2020

    # _, var = gp.predict(X, return_std=True)
    err_pred = beta * stddev
    err = f_true - mean

    # compute prediction error
    pred_err = (err - err_pred).clip(min=0)
    bound_err = pred_err > 0
    percentage = torch.sum(bound_err) / pred_err.shape[0] * 100

    return percentage

def get_CI95(calibrated_gps, kuleshov_calibrated_gps, varfree_calibrated_gps, stddev0, fbstddev0, test_x, test_y):

    deltas = [0.025, 0.975]
    # deltas = [0.005, 0.995]

    beta_bayesian_positive = st.norm.ppf(deltas[1])
    beta_bayesian_negative = st.norm.ppf(deltas[0])

    cal_gp_interp, betas_cal_interp = calibrated_gps.get_calibrated_gaussian_processes(deltas)
    _, betas_kuleshov_interp = kuleshov_calibrated_gps.get_calibrated_gaussian_processes(deltas)
    _, betas_varfree_interp = varfree_calibrated_gps.get_calibrated_gaussian_processes(deltas)
    betas_random_interp = kuleshov_calibrated_gps.get_randomly_interpolated_betas(deltas, test_y.size()[0])

    chunk_size = 200
    n_chunks = np.int64(test_x.size()[0] / chunk_size + 1)
    test_x_chunks = [test_x[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]

    # divide test data into bite-size chunks to avoid memory overload
    CI95_cal = []
    for j in range(n_chunks):
        test_x_chunk = test_x_chunks[j]

        CI95_cal.append(betas_cal_interp[0] * cal_gp_interp[0](test_x_chunk).stddev.detach() - betas_cal_interp[1] * cal_gp_interp[
            1](test_x_chunk).stddev.detach())
    CI95_cal = torch.cat(CI95_cal)
    CI95_kuleshov = betas_kuleshov_interp[0] * stddev0.detach() - betas_kuleshov_interp[1] * stddev0.detach()
    CI95_varfree = betas_varfree_interp[0] * torch.ones(stddev0.detach().size()[0]) - \
                   betas_varfree_interp[1] * torch.ones(stddev0.detach().size()[0])
    CI95_random = betas_random_interp[0] * stddev0.detach() - betas_random_interp[1] * stddev0.detach()
    CI95_vanilla = beta_bayesian_positive * stddev0.detach() - beta_bayesian_negative * stddev0.detach()

    if len(fbstddev0) > 0:
        CI95_fb = beta_bayesian_positive * fbstddev0.detach() - beta_bayesian_negative * fbstddev0.detach()
    else:
        CI95_fb = torch.tensor(np.nan)

    return CI95_cal.mean(), CI95_kuleshov.mean(), CI95_varfree.mean(), CI95_random.mean(), CI95_vanilla.mean(), CI95_fb.mean()


def get_NLL(calibrated_gps, kuleshov_calibrated_gps, varfree_calibrated_gps,
            mean0, stddev0, fbmean0, fbstddev0, test_x, test_y, deltas = False):
    #
    # dimx = test_x.size()[1]
    # if deltas==False:
    n_points = 21
    deltas_nll = torch.linspace(1, 0, n_points).tolist()
    deltas_nll_bayesian = torch.linspace(0.99999, 0.00001, n_points)
    betas_bayesian = st.norm.ppf(1 - deltas_nll_bayesian)
    # else:
    #     deltas_nll = deepcopy(deltas)
    #     if deltas_nll[0]<deltas_nll[1]:
    #         deltas_nll.reverse()
    #     n_points = len(deltas_nll)
    #     deltas_nll_bayesian = torch.tensor(deltas_nll)
    #     deltas_nll_bayesian[-1] = 0.00001
    #     deltas_nll_bayesian[0] = 0.99999
    #     betas_bayesian = st.norm.ppf(1 - deltas_nll_bayesian)
    centered_pred = test_y - mean0
    # divide predictions and test data into bite-size chunks
    chunk_size = 200
    n_chunks = np.int64(test_x.size()[0] / chunk_size + 1)
    test_x_chunks = [test_x[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
    centered_pred_chunks = [centered_pred[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]
    stddev0_chunks = [stddev0[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]

    NLL_cal = []
    NLL_kuleshov = []
    NLL_varfree = []
    NLL_random = []
    NLL_vanilla = []

    sigma_cal = []
    sigma_kuleshov = []
    sigma_varfree = []
    sigma_random = []


    _, betas_kuleshov_interp = kuleshov_calibrated_gps.get_calibrated_gaussian_processes(deltas_nll)
    _, betas_varfree_interp = varfree_calibrated_gps.get_calibrated_gaussian_processes(deltas_nll)
    betas_random_interp = kuleshov_calibrated_gps.get_randomly_interpolated_betas(deltas_nll, 1)

    for j in range(n_chunks):
        test_x_chunk = test_x_chunks[j]
        centered_pred_chunk = centered_pred_chunks[j]
        stddev0_chunk = stddev0_chunks[j]

        q_fun_vanilla = [betas_bayesian[i] * stddev0_chunk for i in range(n_points)]
        NLL_vanilla.append(compute_NLL(q_fun_vanilla, centered_pred_chunk))
        del q_fun_vanilla

        q_fun_cal = []
        for i in range(n_points):
            cal_gp_interp_i, betas_cal_interp_i = calibrated_gps.get_calibrated_gaussian_processes([deltas_nll[i]])
            stddev_cal_chunk = cal_gp_interp_i[0](test_x_chunk).stddev.detach()
            q_fun_cal.append(betas_cal_interp_i[0] * stddev_cal_chunk)
        NLL_cal.append(compute_NLL(q_fun_cal, centered_pred_chunk))
        sigma_cal.append(compute_sigma(q_fun_cal))
        # delete to free up memory
        del q_fun_cal

        q_fun_kuleshov = [betas_kuleshov_interp[i] * stddev0_chunk for i in range(n_points)]
        q_fun_varfree = [betas_varfree_interp[i] * torch.ones(centered_pred_chunk.size()[0]) for i in range(n_points)]
        q_fun_random = [betas_random_interp[i] * stddev0_chunk for i in range(n_points)]


        NLL_kuleshov.append(compute_NLL(q_fun_kuleshov, centered_pred_chunk))
        NLL_varfree.append(compute_NLL(q_fun_varfree, centered_pred_chunk))
        NLL_random.append(compute_NLL(q_fun_random, centered_pred_chunk))

        sigma_kuleshov.append(compute_sigma(q_fun_kuleshov))
        sigma_varfree.append(compute_sigma(q_fun_varfree))
        sigma_random.append(compute_sigma(q_fun_random))

    NLL_cal = torch.tensor(NLL_cal)
    NLL_kuleshov = torch.tensor(NLL_kuleshov)
    NLL_varfree = torch.tensor(NLL_varfree)
    NLL_random = torch.tensor(NLL_random)
    NLL_vanilla = torch.tensor(NLL_vanilla)

    sigma_cal = torch.tensor(sigma_cal)
    sigma_kuleshov = torch.tensor(sigma_kuleshov)
    sigma_varfree = torch.tensor(sigma_varfree)
    sigma_random = torch.tensor(sigma_random)

    if len(fbmean0) > 0:
        centered_pred_fb = test_y - fbmean0
        q_fun_fullbayes = [betas_bayesian[i] * fbstddev0 for i in range(n_points)]
        NLL_fullbayes = compute_NLL(q_fun_fullbayes, centered_pred_fb)
    else:
        NLL_fullbayes = torch.tensor(np.nan)


    return NLL_cal.mean(), NLL_kuleshov.mean(), NLL_varfree.mean(), \
           NLL_random.mean(), NLL_vanilla.mean(), NLL_fullbayes.mean(), \
           sigma_cal.mean(), sigma_kuleshov.mean(), sigma_varfree.mean(), sigma_random.mean()

def compute_NLL(q_fun, y):
    diff_q_y = [q - y for q in q_fun]

    n_deltas = len(q_fun)
    ddelta = 1/n_deltas
    n_data = y.size()[0]
    nll = torch.zeros(n_data)
    for j in range(n_data):
        if diff_q_y[0][j] > 0:
            i_negative = n_deltas-2
            i_positive = n_deltas-1
        elif diff_q_y[0][j] < 0:
            i_negative = 0
            i_positive = 1

            for i in range(n_deltas-1):
                if diff_q_y[i][j] < 0 < diff_q_y[i + 1][j]:
                    i_negative = i
                    i_positive = i + 1
                    break
        if n_data == 1:
            dqddelta_j = (q_fun[i_positive] - q_fun[i_negative]) / ddelta
        else:
            dqddelta_j = (q_fun[i_positive][j] - q_fun[i_negative][j])/ddelta
        nll[j] = -torch.log(1/dqddelta_j)

    nll = nll[~nll.isinf()]
    nll = nll[~nll.isnan()]
    return nll.mean()

def compute_sigma(q_fun):
    # compute mean

    n_deltas = len(q_fun)
    n_data = q_fun[0].size()[0]
    sigma = torch.zeros(n_data)
    q_fun_j = torch.zeros(n_deltas)

    for j in range(n_data):
        for i in range(n_deltas-1):
            q_fun_j[i] = q_fun[i][j]
        mu_j = q_fun_j.mean()
        var_j = ((q_fun_j-mu_j)**2).mean()
        sigma[j] = torch.sqrt(var_j)

    return sigma.mean()




