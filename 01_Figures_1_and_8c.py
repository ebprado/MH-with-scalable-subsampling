from plotnine import *
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
# --------------------------------------------------------------------------
# make sure you have the algorithms.py file in the current directory!
import algorithms 
# --------------------------------------------------------------------------
save_dir = os.getcwd()

colnames = np.array(['N','d', 'kappa', 'acc_rate', 'meanSJD', 'cpu_time', 'ESS', 'expected_B', 'acc_rate_ratio1'])

def run_tuna(x, y, theta_hat, V, taylor_order, npost, model, kappa=1.5):
    N = x.shape[0]
    d = x.shape[1]

    if N == 31622:
        if d == 5:
            chi=1e-3
            kappa=0.03
        elif d == 10:
            chi = 5e-4
            kappa = 0.04
        elif d == 20:
            chi = 1e-4
            kappa = 0.06
        elif d == 30:
            chi = 5e-5
            kappa = 0.065
        elif d == 40:
            chi = 3e-5
            kappa = 0.08
        elif d == 50:
            chi = 2e-5
            kappa = 0.09
        elif d == 60:
            chi = 1e-5
            kappa = 0.1

    method = tunaMH(y, x, V, x0 = theta_hat, model=model, control_variates=False, taylor_order = 0, bound = 'new', chi=chi, nburn=0, npost=npost, kappa=kappa)
    acc_rate = method.get('acc_rate')
    acc_rate_ratio1 = method.get('acc_rate_ratio1')
    meanSJD = method.get('meanSJD')
    cpu_time = method.get('cpu_time')
    expected_B = np.mean(method.get('BoverN'))*method.get('N')
    ESS = np.max(method.get('ESS'))
    save_results = np.array([N, d, kappa, acc_rate, meanSJD, cpu_time, ESS, expected_B, acc_rate_ratio1])
    save_results = pd.DataFrame(save_results[None, :], columns=colnames)

    return save_results

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

def run_SMH(x, y, theta_hat, V, bound, taylor_order, npost, model):
    
    N = x.shape[0]
    d = x.shape[1]

    if d == 5:
        kappa = 2
    elif d == 10:
        kappa = 1.5
    elif d == 20:
        kappa = 1
    elif d == 30:
        kappa = 0.89
    elif d == 40:
        kappa = 0.8
    elif d == 50:
        kappa = 0.69
    elif d == 60:
        kappa = 0.65

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

    filename_beg = save_dir + model + 'Related_works'
    filename_end = "N" + str(N) + 'd' + str(d) + 'Rep' + str(rep) + '.pickle'

    tuna = run_tuna(x, y, theta_hat, V, taylor_order = 1, npost=npost, model=model)
    smh1 = run_SMH(x,  y, theta_hat, V, taylor_order = 1, npost=npost, model=model, bound='orig')
    rwm  = run_RWM(x,  y, theta_hat, V, npost = npost, model=model)

    tuna_file_name = filename_beg + 'Tuna' + filename_end
    smh1_file_name = filename_beg + 'SMH1' + filename_end
    rwm_file_name  = filename_beg + 'RWM' + filename_end

    save_file(tuna,  tuna_file_name)
    save_file(rwm,   rwm_file_name)
    save_file(smh1,  smh1_file_name)

def simulation(d, npost, model = 'logistic'):
    N = np.array([31622])
    len_N = len(N)
    for j in range(len_N):
        print('N = ' + str(N[j]))
        run_methods(N[j], d=d, model=model, npost=npost)

simulation(1, npost=100000)
simulation(2, npost=100000)
simulation(5, npost=100000)
simulation(10, npost=100000)
simulation(20, npost=100000)
simulation(30, npost=100000)
simulation(40, npost=100000)
simulation(50, npost=100000)
simulation(60, npost=100000)

def get_results(N, model='logistic', rep=1):

    set_d = np.array([1, 2, 5, 10, 20, 30, 40, 50, 60])
    colnames = np.array(['N','d', 'kappa', 'acc_rate', 'meanSJD', 'cpu_time', 'ESS', 'expected_B', 'method', 'acc_rate_ratio1'])
    store_results = np.zeros((len(set_d), len(colnames)))
    store_results[:] = np.nan
    store_results = pd.DataFrame({'N':{},'d':{}, 'kappa':{}, 'acc_rate':{}, 'meanSJD':{}, 'cpu_time':{}, 'ESS':{}, 'expected_B':{}, 'method': {}, 'acc_rate_ratio1':{}})

    for d in set_d:
        filename_beg = save_dir + model + 'Related_works'
        filename_end = "N" + str(N) + 'd' + str(d) + 'Rep' + str(rep) + '.pickle'

        tuna_file_name = filename_beg + 'Tuna' + filename_end
        rwm_file_name = filename_beg + 'RWM' + filename_end
        smh1_acc_file_name = filename_beg + 'SMH1' + filename_end

        if os.path.exists(tuna_file_name):
            with open(tuna_file_name, 'rb') as f:
                tuna_results = pickle.load(f)
                tuna_results['method'] = 'Tuna'
                store_results = pd.concat([store_results, tuna_results])

        if os.path.exists(rwm_file_name):
            with open(rwm_file_name, 'rb') as f:
                rwm_results = pickle.load(f)
                rwm_results['method'] = 'RWM'
                store_results = pd.concat([store_results, rwm_results])
        
        if os.path.exists(smh1_acc_file_name):
            with open(smh1_acc_file_name, 'rb') as f:
                smh1_acc_results = pickle.load(f)
                smh1_acc_results['method'] = 'SMH-1'
                store_results = pd.concat([store_results, smh1_acc_results])

    store_results.columns = colnames

    store_results['acc_rate'] = store_results.acc_rate.astype(float)
    store_results['ESS'] = store_results.ESS.astype(float)
    store_results['cpu_time'] = store_results.cpu_time.astype(float)
    store_results['expected_B'] = store_results.expected_B.astype(float)
    store_results['N'] = store_results.N.astype(float)
    store_results['meanSJD'] = store_results.meanSJD.astype(float)

    store_results['MSJD_over_B'] = np.log10(store_results['N'] * store_results['meanSJD'] / store_results['expected_B'])
    store_results['ESS_per_second'] = np.log10(store_results['ESS'] / store_results['cpu_time'])
    store_results['ESS_over_B'] = np.log10(store_results['ESS'] / store_results['expected_B'])
    store_results['log_expected_B'] = np.log10(store_results['expected_B'])
    
    store_results['N'] = np.log10(store_results['N'])
    store_results['d'] = store_results.d.astype(int)

    store_results['method'] = pd.Categorical(store_results.method, categories=['SMH-1', 'Tuna', 'RWM'])

    return store_results

sim_results = get_results(31622)

height_plot = 6
width_plot = 8

# -------------------------------------------------
# Figure 1 (a)
# -------------------------------------------------

plot = (ggplot(sim_results) + 
 aes(x='d', y='ESS_per_second', colour='method') +  
    geom_line(linetype = "dashed", size=3) +
    geom_point(aes(shape = 'method'), size=8) +
 labs(
      y = 'log ESS per second',
    #   x = r'log$_{10} N$'
      x = 'd'
      ) + 
theme_bw(base_size = 30) +
theme(plot_title = element_text(size = 30, hjust = 0.5),
    strip_text_y = element_text(angle = 0),
    legend_position = 'bottom',
    legend_title = element_blank(),
    panel_grid_major = element_blank(),
    panel_grid_minor = element_blank()) + 
    scale_color_manual(values=["#B2DF8A", "#CAB2D6", "#FB9A99"])
    )

plot.save('related_works_ESS_per_second.pdf', height=height_plot, width=width_plot)

# -------------------------------------------------
# Figure 1 (b)
# -------------------------------------------------

plot = (ggplot(sim_results) + 
 aes(x='d', y='log_expected_B', colour='method') +  
    geom_line(linetype = "dashed", size=3) +
    geom_point(aes(shape = 'method'), size=8) +
 labs(
      y = 'log E(B)',
    #   x = r'log$_{10} N$'
      x = 'd'
      ) + 
theme_bw(base_size = 30) +
theme(plot_title = element_text(size = 30, hjust = 0.5),
    strip_text_y = element_text(angle = 0),
    legend_position = 'bottom',
    legend_title = element_blank(),
    panel_grid_major = element_blank(),
    panel_grid_minor = element_blank()) +
    scale_color_manual(values=["#B2DF8A", "#CAB2D6", "#FB9A99"])
    )

plot.save('related_works_E_B.pdf', height=height_plot, width=width_plot)

# -------------------------------------------------
# Figure 1 (c)
# -------------------------------------------------
plot = (ggplot(sim_results) + 
 aes(x='d', y='acc_rate', colour='method') +  
    geom_line(linetype = "dashed", size=3) +
    geom_point(aes(shape = 'method'), size=8) +
    ylim(0, 0.63) +
 labs(
      y = 'Acceptance rate',
    #   x = r'log$_{10} N$'
      x = 'd'
      ) + 
theme_bw(base_size = 30) +
theme(plot_title = element_text(size = 30, hjust = 0.5),
    strip_text_y = element_text(angle = 0),
    legend_position = 'bottom',
    legend_title = element_blank(),
    panel_grid_major = element_blank(),
    panel_grid_minor = element_blank()) +
    scale_color_manual(values=["#B2DF8A", "#CAB2D6", "#FB9A99"])
    )

plot.save('related_works_acc_rate.pdf', height=height_plot, width=width_plot)

# -------------------------------------------------
# Figure 8 (c)
# -------------------------------------------------

plot = (ggplot(sim_results) + 
 aes(x='d', y='acc_rate', colour='method') +  
    geom_line(linetype = "dashed", size=2) +
    geom_point(aes(shape = 'method'), size=6) +
    geom_segment(aes(x = 0, xend = 60, y = 0.234, yend = 0.234), colour='#FB9A99', size=0.4, linetype='dotted') +
    geom_segment(aes(x = 0, xend = 60, y = 0.45, yend = 0.45), colour='#1F78B4', size=0.4, linetype='dotted') +
    ylim(0.2, 0.63) +
 labs(
      y = 'Acceptance rate',
    #   x = r'log$_{10} N$'
      x = 'd'
      ) + 
theme_bw(base_size = 25) +
theme(plot_title = element_text(size = 25, hjust = 0.5),
    strip_text_y = element_text(angle = 0),
    legend_position = 'bottom',
    legend_title = element_blank(),
    panel_grid_major = element_blank(),
    panel_grid_minor = element_blank()) +
    scale_color_manual(values=["#A6CEE3", "#1F78B4", "#FB9A99"])
    )

plot.save('optimal_scaling_acc_rate_d.pdf', height=height_plot, width=width_plot)