from plotnine import *
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# --------------------------------------------------------------------------
# make sure you have the algorithms.py file in the current directory!
from algorithms import *
# --------------------------------------------------------------------------
save_dir = os.getcwd() + '/'

def run_RWM(x, y, theta_hat, V, kappa, npost, model, implementation):
    N = x.shape[0]
    d = x.shape[1]
    method = RWM(y, x, V, x0 = theta_hat, model=model, nburn=0, npost=npost, kappa = kappa, implementation=implementation)
    acc_rate = method.get('acc_rate')
    acc_rate_ratio1 = method.get('acc_rate_ratio1')
    meanSJD = method.get('meanSJD')
    cpu_time = method.get('cpu_time')
    expected_B = N
    ESS = method.get('ESS')[d-1]
    save_results = np.array([N, d, kappa, acc_rate, meanSJD, cpu_time, ESS, expected_B, acc_rate_ratio1])

    return save_results


def run_MH_SS(x, y, theta_hat, V, kappa, taylor_order, npost, model, implementation):

    N = x.shape[0]
    d = x.shape[1]

    method = MH_SS(y, x, V, x0 = theta_hat, model=model, control_variates=True, bound = 'new', taylor_order=taylor_order, chi=0, nburn=0, npost=npost, kappa=kappa, implementation=implementation)
    acc_rate = method.get('acc_rate')
    acc_rate_ratio1 = method.get('acc_rate_ratio1')
    meanSJD = method.get('meanSJD')
    cpu_time = method.get('cpu_time')
    expected_B = np.mean(method.get('BoverN'))*method.get('N')
    ESS = np.max(method.get('ESS'))
    save_results = np.array([N, d, kappa, acc_rate, meanSJD, cpu_time, ESS, expected_B, acc_rate_ratio1])

    return save_results


def run_SMH(x, y, theta_hat, V, kappa, bound, taylor_order, npost, model, implementation):
    N = x.shape[0]
    d = x.shape[1]
    method = SMH(y, x, V, x0 = theta_hat, kappa=kappa, model=model, bound=bound, taylor_order=taylor_order, nburn=0, npost=npost, implementation=implementation)
    acc_rate = method.get('acc_rate')
    acc_rate_ratio1 = method.get('acc_rate_ratio1')
    meanSJD = method.get('meanSJD')
    cpu_time = method.get('cpu_time')
    expected_B = np.mean(method.get('BoverN'))*method.get('N')
    ESS = method.get('ESS')[d-1]
    save_results = np.array([N, d, kappa, acc_rate, meanSJD, cpu_time, ESS, expected_B, acc_rate_ratio1])

    return save_results

def run_methods(N, d, set_kappa, rep=2, npost=100000, model='logistic', implementation='vectorised'):

    length_kappa = len(set_kappa)
    data = simulate_data(N, d, model)
    y = data.get('y')
    x = data.get('x')

    theta_hat, V = get_theta_hat_and_var_cov_matrix(model, x, y)

    colnames = np.array(['N','d', 'kappa', 'acc_rate', 'meanSJD', 'cpu_time', 'ESS', 'expected_B', 'acc_rate_ratio1'])
    ncol = len(colnames)
    mhss1_save_results = np.zeros((length_kappa, ncol))
    mhss1_save_results[:] = np.nan
    mhss2_save_results = np.zeros((length_kappa, ncol))
    mhss2_save_results[:] = np.nan
    rwm_save_results = np.zeros((length_kappa, ncol))
    rwm_save_results[:] = np.nan
    smh1_save_results = np.zeros((length_kappa, ncol))
    smh1_save_results[:] = np.nan
    smh2_save_results = np.zeros((length_kappa, ncol))
    smh2_save_results[:] = np.nan
    smh1_chris_save_results = np.zeros((length_kappa, ncol))
    smh1_chris_save_results[:] = np.nan
    smh2_chris_save_results = np.zeros((length_kappa, ncol))
    smh2_chris_save_results[:] = np.nan

    for i in range(rep):
        for j in range(length_kappa):

            print('rep ' + str(i) + ' out of ' + str(rep) + ': kappa = ' + str(j))
            kappa = set_kappa[j]

            mhss1_save_results[j, :] = run_MH_SS(x, y, theta_hat, V, kappa, taylor_order = 1, npost=npost, model=model, implementation=implementation)
            mhss2_save_results[j, :] = run_MH_SS(x, y, theta_hat, V, kappa, taylor_order = 2, npost=npost, model=model, implementation=implementation)
            rwm_save_results[j, :] = run_RWM(x, y, theta_hat, V, kappa, npost, model=model, implementation=implementation)
            smh1_save_results[j, :] = run_SMH(x, y, theta_hat, V, kappa, bound='orig', taylor_order=1, npost=npost, model=model, implementation=implementation)
            smh2_save_results[j, :] = run_SMH(x, y, theta_hat, V, kappa, bound='orig', taylor_order=2, npost=npost, model=model, implementation=implementation)
            smh1_chris_save_results[j, :] = run_SMH(x, y, theta_hat, V, kappa, bound='ChrisS', taylor_order=1, npost=npost, model=model, implementation=implementation)
            smh2_chris_save_results[j, :] = run_SMH(x, y, theta_hat, V, kappa, bound='ChrisS', taylor_order=2, npost=npost, model=model, implementation=implementation)

            mhss1 = pd.DataFrame(mhss1_save_results, columns=colnames)
            mhss2 = pd.DataFrame(mhss2_save_results, columns=colnames)
            rwm = pd.DataFrame(rwm_save_results, columns=colnames)
            smh1 = pd.DataFrame(smh1_save_results, columns=colnames)
            smh2 = pd.DataFrame(smh2_save_results, columns=colnames)
            smh1_chris = pd.DataFrame(smh1_chris_save_results, columns=colnames)
            smh2_chris = pd.DataFrame(smh2_chris_save_results, columns=colnames)

            mhss1_file_name = save_dir + 'ScalingParameterMHSS1' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'
            mhss2_file_name = save_dir + 'ScalingParameterMHSS1' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'
            rwm_file_name = save_dir + 'ScalingParameterRWM' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'
            smh1_file_name = save_dir + 'ScalingParameterSMH1' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'
            smh2_file_name = save_dir + 'ScalingParameterSMH2' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'
            smh1_chris_file_name = save_dir + 'ScalingParameterSMH1NB' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'
            smh2_chris_file_name = save_dir + 'ScalingParameterSMH2NB' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'

            save_file(mhss1, mhss1_file_name)
            save_file(mhss2, mhss2_file_name)
            save_file(rwm, rwm_file_name)
            save_file(smh1, smh1_file_name)
            save_file(smh2, smh2_file_name)
            save_file(smh1_chris, smh1_chris_file_name)
            save_file(smh2_chris, smh2_chris_file_name)

kappa = np.round(np.arange(0.1,4,0.2), 2)
# ------------------------------------------
# Results needed for Figures 3 and 4
# ------------------------------------------
run_methods(N=30000, d=100, set_kappa=kappa) 

# ------------------------------------------
# Results needed for Figure 10
# ------------------------------------------
run_methods(N=100000, d=100, set_kappa=kappa)