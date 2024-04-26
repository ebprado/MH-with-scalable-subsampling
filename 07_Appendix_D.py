from plotnine import *
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

def get_results(N, rep=10):

    set_d = np.array([10, 30, 50, 100])
    colnames = np.array(['N','d', 'kappa', 'acc_rate', 'meanSJD', 'cpu_time', 'ESS', 'expected_B', 'method', 'acc_rate_ratio1'])
    store_results = np.zeros((len(set_d), len(colnames)))
    store_results[:] = np.nan
    store_results = pd.DataFrame({'N':{},'d':{}, 'kappa':{}, 'acc_rate':{}, 'meanSJD':{}, 'cpu_time':{}, 'ESS':{}, 'expected_B':{}, 'method': {}, 'acc_rate_ratio1':{}})

    for i in range(rep):
        for d in set_d:
            tuna1_file_name = save_dir + 'ScalingParameterTunaCV1True' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'
            tuna2_file_name = save_dir + 'ScalingParameterTunaCV2True' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'
            rwm_file_name = save_dir + 'ScalingParameterRWM' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'

            if os.path.exists(tuna1_file_name):
                with open(tuna1_file_name, 'rb') as f:
                    tuna1_results = pickle.load(f)
                    tuna1_results['method'] = 'MH-SS-1'
                    tuna1_results['rep'] = i
                    store_results = pd.concat([store_results, tuna1_results])

            if os.path.exists(tuna2_file_name):
                with open(tuna2_file_name, 'rb') as f:
                    tuna2_results = pickle.load(f)
                    tuna2_results['method'] = 'MH-SS-2'
                    tuna2_results['rep'] = i
                    store_results = pd.concat([store_results, tuna2_results])

            if os.path.exists(rwm_file_name):
                with open(rwm_file_name, 'rb') as f:
                    rwm_results = pickle.load(f)
                    rwm_results['method'] = 'RWM'
                    rwm_results['rep'] = i
                    store_results = pd.concat([store_results, rwm_results])
                
    store_results = store_results.dropna(subset=['N', 'd'])
    
    store_results['MSJD_over_B'] = store_results['meanSJD'] / store_results['expected_B']
    store_results['ESS_per_second'] = store_results['ESS'] / store_results['cpu_time']

    store_results['N'] = store_results.N.astype(int)
    store_results['d'] = store_results.d.astype(int)

    store_results['N'] = 'N = ' + store_results.N.astype(str)
    store_results['d'] = 'd = ' + store_results.d.astype(str)

    store_results['d'] = pd.Categorical(store_results.d, categories=['d = 10', 'd = 30', 'd = 50', 'd = 100', 'd = 200', 'd = 300'])
    store_results['method'] = pd.Categorical(store_results.method, categories= ['MH-SS-1', 'MH-SS-2', 'RWM'])
    
    return store_results

set_kappa = np.round(np.arange(0.1,4,0.2), 2)
test1 = get_results(1000)
test2 = get_results(10000)
test3 = get_results(30000)
test4 = get_results(100000)
test = pd.concat([test3])

height_plot = 6
width_plot = 8

# --------------------------------------------------------------
# Figure 8 (a)
# --------------------------------------------------------------

plot = (ggplot(test[(test['d'] == 'd = 100')]) +
aes(x='kappa', y='MSJD_over_B', colour='method') +  
    geom_line(linetype = "dashed", size=2) +
    geom_point(aes(shape = 'method'), size=6) +
    geom_segment(aes(x = 1.5, xend = 1.5, y = 0, yend = 4e-7), colour='#1F78B4', size=0.4, linetype='dotted') +
 labs(
      y = r'MSJD/$\mathbb{E}(B)$',
      x = r'$\lambda$'
      ) + 
theme_bw(base_size = 25) +
theme(plot_title = element_text(size = 25, hjust = 0.5),
    strip_text_y = element_text(angle = 0),
    legend_position = 'bottom',
    legend_title = element_blank(),
    panel_grid_major = element_blank(),
    panel_grid_minor = element_blank()) +
scale_color_brewer(type = 'qual', palette = 'Paired'))

plot.save('optimal_scaling_lambda_MSJD.pdf', height=height_plot, width=width_plot)

# --------------------------------------------------------------
# Figure 8 (b)
# --------------------------------------------------------------

plot = (ggplot(test[(test['d'] == 'd = 100')]) +
aes(x='acc_rate', y='MSJD_over_B', colour='method') +  
    geom_line(linetype = "dashed", size=2) +
    geom_point(aes(shape = 'method'), size=6) +
    geom_segment(aes(x = 0.45, xend = 0.45, y = 0, yend = 4e-7), colour='#1F78B4', size=0.4, linetype='dotted') +
 labs(
      y = r'MSJD/$\mathbb{E}(B)$',
      x = 'Acceptance rate'
      ) + 
theme_bw(base_size = 25) +
theme(plot_title = element_text(size = 25, hjust = 0.5),
    strip_text_y = element_text(angle = 0),
    legend_position = 'bottom',
    legend_title = element_blank(),
    panel_grid_major = element_blank(),
    panel_grid_minor = element_blank()) +
scale_color_brewer(type = 'qual', palette = 'Paired'))

plot.save('optimal_scaling_acc_rate_MSJD.pdf', height=height_plot, width=width_plot)

# --------------------------------------------------------------
# Figure 8 (d)
# --------------------------------------------------------------

plot = (ggplot(test[(test['d'] == 'd = 50')]) + 
aes(x='kappa', y='acc_rate', colour='method') +  
    geom_line(linetype = "dashed", size=2) +
    geom_point(aes(shape = 'method'), size=6) +
 geom_segment(aes(x = 0.1, xend = 2.38, y = 0.234, yend = 0.234), colour='#FB9A99', size=0.4, linetype='dotted') +
 geom_segment(aes(x = 2.38, xend = 2.38, y = 0, yend = 0.234), colour='#FB9A99', size=0.4, linetype='dotted') +
 geom_segment(aes(x = 0.1, xend = 1.5, y = 0.45, yend = 0.45), colour='#1F78B4', size=0.4, linetype='dotted') +
 geom_segment(aes(x = 1.5, xend = 1.5, y = 0, yend = 0.45), colour='#1F78B4', size=0.4, linetype='dotted') +
 labs(
      y = 'Acceptance rate',
      x = r'$\lambda$'
      ) + 
theme_bw(base_size = 25) +
theme(plot_title = element_text(size = 25, hjust = 0.5),
    strip_text_y = element_text(angle = 0),
    legend_position = 'bottom',
    legend_title = element_blank(),
    panel_grid_major = element_blank(),
    panel_grid_minor = element_blank()) +
scale_color_manual(values=["#A6CEE3", "#1F78B4", "#FB9A99"]))

plot.save('optimal_scaling_lambda_acc.pdf', height=height_plot, width=width_plot)
