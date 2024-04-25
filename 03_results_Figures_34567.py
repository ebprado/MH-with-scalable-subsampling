import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
import os
# --------------------------------------------------------------------------
# make sure you have the algorithms.py file in the current directory!
import algorithms 
# --------------------------------------------------------------------------
save_dir = os.getcwd()

colnames = np.array(['N','d', 'kappa', 'acc_rate', 'meanSJD', 'cpu_time', 'ESS', 'expected_B', 'acc_rate_ratio1'])

def run_RWM(x, y, theta_hat, V, npost, model, kappa=2.4):
    N = x.shape[0]
    d = x.shape[1]
    method = RWM(y, x, V, x0 = theta_hat, model=model, nburn=0, npost=npost, kappa = kappa)
    acc_rate = method.get('acc_rate')
    acc_rate_ratio1 = method.get('acc_rate_ratio1')
    meanSJD = method.get('meanSJD')
    cpu_time = method.get('cpu_time')
    expected_B = N
    ESS = np.max(method.get('ESS'))
    save_results = np.array([N, d, kappa, acc_rate, meanSJD, cpu_time, ESS, expected_B, acc_rate_ratio1])
    save_results = pd.DataFrame(save_results[None, :], columns=colnames)

    return save_results

def run_MH_SS(x, y, theta_hat, V, taylor_order, npost, model, kappa=1.5):
    N = x.shape[0]
    d = x.shape[1]

    if d == 100 and N == 1000:
        kappa = 2.4

    method = tunaMH(y, x, V, x0 = theta_hat, model=model, control_variates=True, bound = 'new', taylor_order=taylor_order, chi=0, nburn=0, npost=npost, kappa=kappa)
    acc_rate = method.get('acc_rate')
    acc_rate_ratio1 = method.get('acc_rate_ratio1')
    meanSJD = method.get('meanSJD')
    cpu_time = method.get('cpu_time')
    expected_B = np.mean(method.get('BoverN'))*method.get('N')
    ESS = np.max(method.get('ESS'))
    save_results = np.array([N, d, kappa, acc_rate, meanSJD, cpu_time, ESS, expected_B, acc_rate_ratio1])
    save_results = pd.DataFrame(save_results[None, :], columns=colnames)

    return save_results

def run_SMH(x, y, theta_hat, V, bound, taylor_order, npost, model):

    N = x.shape[0]
    d = x.shape[1]

    if bound == 'orig':
        if taylor_order == 1:
            kappa = 1

            if N <= 3500 and d >= 30:
                kappa = 2.4

            if N == 1000 and d >= 30:
                kappa = 2.4

            if N == 10000 and d > 30:
                kappa = 2.4

            if N == 31622 and d == 100:
                kappa = 2.4

        elif taylor_order == 2:
            kappa = 2

            if N <= 3500 and d > 30:
                kappa = 2.4

            if N == 1000 and d >= 30:
                kappa = 2.4

            if N == 10000 and d >= 100:
                kappa = 2.4

    if taylor_order == 1 and bound == 'ChrisS':
        kappa = 0.5

    if taylor_order == 2 and bound == 'ChrisS':
        kappa = 1.5
        if N == 1000 and d >= 100:
            kappa = 2.4

    print(kappa)
    method = smh(y, x, V, x0 = theta_hat, model=model, kappa=kappa, bound=bound, taylor_order=taylor_order, nburn=0, npost=npost)
    acc_rate = method.get('acc_rate')
    acc_rate_ratio1 = method.get('acc_rate_ratio1')
    meanSJD = method.get('meanSJD')
    cpu_time = method.get('cpu_time')
    expected_B = np.mean(method.get('BoverN'))*method.get('N')
    ESS = np.max(method.get('ESS'))
    save_results = np.array([N, d, kappa, acc_rate, meanSJD, cpu_time, ESS, expected_B, acc_rate_ratio1])
    save_results = pd.DataFrame(save_results[None, :], columns=colnames)

    return save_results

def run_methods(N, d, model, rep=1, npost=10000):

    data = simulate_data(N, d, model)
    y = data.get('y')
    x = data.get('x')

    theta_hat, V = get_theta_hat_and_var_cov_matrix(model, x, y)
    filename_end = "N" + str(N) + 'd' + str(d) + 'Rep' + str(rep) + '.pickle'

    tuna1_save_results = run_MH_SS(x, y, theta_hat, V, taylor_order = 1, npost=npost, model=model)
    tuna2_save_results = run_MH_SS(x, y, theta_hat, V, taylor_order = 2, npost=npost, model=model)
    rwm_save_results = run_RWM(x, y, theta_hat, V, npost, model=model)
    smh1_save_results = run_SMH(x, y, theta_hat, V, bound='orig', taylor_order=1, npost=npost, model=model)
    smh2_save_results = run_SMH(x, y, theta_hat, V, bound='orig', taylor_order=2, npost=npost, model=model)
    smh1_chris_save_results = run_SMH(x, y, theta_hat, V, bound='ChrisS', taylor_order=1, npost=npost, model=model)
    smh2_chris_save_results = run_SMH(x, y, theta_hat, V, bound='ChrisS', taylor_order=2, npost=npost, model=model)

    tuna1_file_name = save_dir + model + 'EfficiencyMetricsTunaCV1True' + filename_end
    tuna2_file_name = save_dir + model + 'EfficiencyMetricsTunaCV2True' + filename_end
    rwm_file_name = save_dir + model + 'EfficiencyMetricsRWM' + filename_end
    smh1_file_name = save_dir + model + 'EfficiencyMetricsSMH' + filename_end
    smh2_file_name = save_dir + model + 'EfficiencyMetricsSMH2' + filename_end
    smh1_chris_file_name = save_dir + model + 'EfficiencyMetricsSMH1ChrisS' + filename_end
    smh2_chris_file_name = save_dir + model + 'EfficiencyMetricsSMH2ChrisS' + filename_end

    save_file(tuna1_save_results, tuna1_file_name)
    save_file(tuna2_save_results, tuna2_file_name)
    save_file(rwm_save_results, rwm_file_name)
    save_file(smh1_save_results, smh1_file_name)
    save_file(smh2_save_results, smh2_file_name)
    save_file(smh1_chris_save_results, smh1_chris_file_name)
    save_file(smh2_chris_save_results, smh2_chris_file_name)

def run_simulation(d, npost, model = 'logistic'):
    N = np.array([1000, 3162, 10000, 31622, 100000])
    len_N = len(N)
    for j in range(len_N):
        print('N = ' + str(N[j]))
        run_methods(N[j], d=d, model=model, npost=npost)

run_simulation(10, npost=100000)
run_simulation(30, npost=100000)
run_simulation(50, npost=100000)
run_simulation(100, npost=100000)

def get_results(N, model='logistic', rep=1):

    set_d = np.array([10, 30, 50, 100])
    colnames = np.array(['N','d', 'kappa', 'acc_rate', 'meanSJD', 'cpu_time', 'ESS', 'expected_B', 'method', 'acc_rate_ratio1'])
    store_results = np.zeros((len(set_d), len(colnames)))
    store_results[:] = np.nan
    store_results = pd.DataFrame({'N':{},'d':{}, 'kappa':{}, 'acc_rate':{}, 'meanSJD':{}, 'cpu_time':{}, 'ESS':{}, 'expected_B':{}, 'method': {}, 'acc_rate_ratio1':{}})

    for d in set_d:
        filename_end = "N" + str(N) + 'd' + str(d) + 'Rep' + str(rep) + '.pickle'

        tuna1_file_name = save_dir + model + 'EfficiencyMetricsTunaCV1True' + filename_end
        tuna2_file_name = save_dir + model + 'EfficiencyMetricsTunaCV2True' + filename_end
        rwm_file_name = save_dir + model + 'EfficiencyMetricsRWM' + filename_end
        smh1_file_name = save_dir + model + 'EfficiencyMetricsSMH' + filename_end
        smh2_file_name = save_dir + model + 'EfficiencyMetricsSMH2' + filename_end

        smh1_chris_file_name = save_dir + model + 'EfficiencyMetricsSMH1ChrisS' + filename_end
        smh2_chris_file_name = save_dir + model + 'EfficiencyMetricsSMH2ChrisS' + filename_end

        if os.path.exists(tuna1_file_name):
            with open(tuna1_file_name, 'rb') as f:
                tuna1_results = pickle.load(f)
                tuna1_results['method'] = 'MH-SS-1'
                store_results = pd.concat([store_results, tuna1_results])

        if os.path.exists(tuna2_file_name):
            with open(tuna2_file_name, 'rb') as f:
                tuna2_results = pickle.load(f)
                tuna2_results['method'] = 'MH-SS-2'
                store_results = pd.concat([store_results, tuna2_results])

        if os.path.exists(rwm_file_name):
            with open(rwm_file_name, 'rb') as f:
                rwm_results = pickle.load(f)
                rwm_results['method'] = 'RWM'
                store_results = pd.concat([store_results, rwm_results])
        
        if os.path.exists(smh1_file_name):
            with open(smh1_file_name, 'rb') as f:
                smh1_results = pickle.load(f)
                smh1_results['method'] = 'SMH-1'
                store_results = pd.concat([store_results, smh1_results])
        
        if os.path.exists(smh2_file_name):                
            with open(smh2_file_name, 'rb') as f:
                smh2_results = pickle.load(f)
                smh2_results['method'] = 'SMH-2'
                store_results = pd.concat([store_results, smh2_results])

        if os.path.exists(smh1_chris_file_name):
            with open(smh1_chris_file_name, 'rb') as f:
                smh1_chris_results = pickle.load(f)
                smh1_chris_results['method'] = 'SMH-1-NB'
                store_results = pd.concat([store_results, smh1_chris_results])

        if os.path.exists(smh2_chris_file_name):
            with open(smh2_chris_file_name, 'rb') as f:
                smh2_chris_results = pickle.load(f)
                smh2_chris_results['method'] = 'SMH-2-NB'
                store_results = pd.concat([store_results, smh2_chris_results])

    store_results.columns = colnames

    return store_results

part1 = get_results(1000)
part2 = get_results(3162)
part3 = get_results(10000)
part4 = get_results(31622)
part5 = get_results(100000)

result_simulation = pd.concat([part1, part2, part3, part4, part5])

# save_file(test, '00_results_all_vectorised.pickle')
save_file(result_simulation, '00_results_all_loop.pickle')