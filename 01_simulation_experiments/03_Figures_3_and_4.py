try:
    from plotnine import *
    import pandas as pd
    from mhssteste import *
except ImportError:
    raise ImportError("Please make sure plotnine, pandas and mhssteste are ALL installed!")

save_dir = os.getcwd() + '/'

def get_results(N, which_figure, rep=10):

    set_d = np.array([10, 30, 50, 100])
    colnames = np.array(['N','d', 'kappa', 'acc_rate', 'meanSJD', 'cpu_time', 'ESS', 'expected_B', 'method', 'acc_rate_ratio1'])
    store_results = np.zeros((len(set_d), len(colnames)))
    store_results[:] = np.nan
    store_results = pd.DataFrame({'N':{},'d':{}, 'kappa':{}, 'acc_rate':{}, 'meanSJD':{}, 'cpu_time':{}, 'ESS':{}, 'expected_B':{}, 'method': {}, 'acc_rate_ratio1':{}})

    for i in range(rep):
        for d in set_d:

            mhss1_file_name = save_dir + 'ScalingParameterMHSS1' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'
            mhss2_file_name = save_dir + 'ScalingParameterMHSS1' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'
            rwm_file_name = save_dir + 'ScalingParameterRWM' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'
            smh1_file_name = save_dir + 'ScalingParameterSMH1' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'
            smh2_file_name = save_dir + 'ScalingParameterSMH2' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'
            smh1_chris_file_name = save_dir + 'ScalingParameterSMH1NB' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'
            smh2_chris_file_name = save_dir + 'ScalingParameterSMH2NB' + "N" + str(N) + 'd' + str(d) + 'Rep' + str(i) + '.pickle'

            if os.path.exists(mhss1_file_name):
                with open(mhss1_file_name, 'rb') as f:
                    mhss1_results = pickle.load(f)
                    mhss1_results['method'] = 'MH-SS-1'
                    mhss1_results['rep'] = i
                    store_results = pd.concat([store_results, mhss1_results])

            if os.path.exists(mhss2_file_name):
                with open(mhss2_file_name, 'rb') as f:
                    mhss2_results = pickle.load(f)
                    mhss2_results['method'] = 'MH-SS-2'
                    mhss2_results['rep'] = i
                    store_results = pd.concat([store_results, mhss2_results])

            if os.path.exists(rwm_file_name):
                with open(rwm_file_name, 'rb') as f:
                    rwm_results = pickle.load(f)
                    rwm_results['method'] = 'RWM'
                    rwm_results['rep'] = i
                    store_results = pd.concat([store_results, rwm_results])

            if os.path.exists(smh1_file_name):
                with open(smh1_file_name, 'rb') as f:
                    smh1_results = pickle.load(f)
                    smh1_results['method'] = 'SMH-1'
                    smh1_results['rep'] = i
                    store_results = pd.concat([store_results, smh1_results])
                
            if os.path.exists(smh2_file_name):
                with open(smh2_file_name, 'rb') as f:
                    smh2_results = pickle.load(f)
                    smh2_results['method'] = 'SMH-2'
                    smh2_results['rep'] = i
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
                
    store_results = store_results.dropna(subset=['N', 'd'])
    
    store_results['MSJD_over_B'] = store_results['meanSJD'] / store_results['expected_B']
    store_results['ESS_per_second'] = store_results['ESS'] / store_results['cpu_time']
    store_results['alpha_2'] = store_results['acc_rate'] /  store_results['acc_rate_ratio1']
    store_results['alpha_2'] = np.where(store_results['method'] == 'RWM', store_results['acc_rate'], store_results['alpha_2'])
    # We did this for graphical purposes only, so that MH-SS-2 and RWM alpha_1 curves can be jointly visualised!
    store_results['acc_rate_ratio1'] = np.where(store_results['method'] == 'RWM', store_results['acc_rate'], store_results['acc_rate_ratio1'] + 0.01)

    store_results['N'] = store_results.N.astype(int)
    store_results['d'] = store_results.d.astype(int)

    store_results['N'] = 'N = ' + store_results.N.astype(str)
    store_results['d'] = 'd = ' + store_results.d.astype(str)

    store_results['d'] = pd.Categorical(store_results.d, categories=['d = 10', 'd = 30', 'd = 50', 'd = 100', 'd = 200', 'd = 300'])

    if which_figure == 3:
        store_results['method'] = pd.Categorical(store_results.method, categories= ['MH-SS-1', 'MH-SS-2', 'RWM'])
    elif which_figure == 4:
        store_results['method'] = pd.Categorical(store_results.method, categories= ['MH-SS-1', 'MH-SS-2'])

    # store_results['method'] = pd.Categorical(store_results.method, categories= ['MH-SS-1', 'MH-SS-2', 'SMH-1', 'SMH-2', 'SMH-1-NB', 'SMH-2-NB', 'RWM'])
    # store_results['method'] = pd.Categorical(store_results.method, categories= ['MH-SS-1', 'MH-SS-2', 'SMH-1', 'SMH-2', 'RWM'])
    
    return store_results

set_kappa = np.round(np.arange(0.1,4,0.2), 2)

results_fig3 = get_results(30000, which_figure=3)
results_fig3 = results_fig3[((results_fig3['method'] == 'MH-SS-1') | (results_fig3['method'] == 'MH-SS-2') | (results_fig3['method'] == 'RWM'))]

results_fig4 = get_results(30000, which_figure=4)
results_fig4 = results_fig4[((results_fig4['method'] == 'MH-SS-1') | (results_fig4['method'] == 'MH-SS-2'))]

results_figx = get_results(100000, which_figure=3)

height_plot = 6
width_plot = 8

# --------------------------------------------------------------
# Figure 3 (panel a)
# --------------------------------------------------------------

plot = (ggplot(results_fig3) +
aes(x='kappa', y='acc_rate_ratio1', colour='method') +  
    geom_line(linetype = "dashed", size=2) +
    geom_point(aes(shape = 'method'), size=6) +
 labs(
      y = r'$\alpha_1$',
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

plot.save('optimal_scaling_lambda_acc_alpha_1.pdf', height=height_plot, width=width_plot)

# --------------------------------------------------------------
# Figure 3 (panel b)
# --------------------------------------------------------------

plot = (ggplot(results_fig3) +
aes(x='kappa', y='alpha_2', colour='method') +  
    geom_line(linetype = "dashed", size=2) +
    geom_point(aes(shape = 'method'), size=6) +
 labs(
      y = r'$\alpha_2$',
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

plot.save('optimal_scaling_lambda_acc_alpha_2.pdf', height=height_plot, width=width_plot)

# ----------------------------------------------------
# Figure 4 (panel a)
# ----------------------------------------------------

plot = (ggplot(results_fig4) +
aes(x='kappa', y='MSJD_over_B', colour='method') +  
    geom_line(linetype = "dashed", size=2) +
    geom_point(aes(shape = 'method'), size=6) +
    # geom_segment(aes(x = 1.5, xend = 1.5, y = 0, yend = 4e-7), colour='#1F78B4', size=0.4, linetype='dotted') +
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

# ----------------------------------------------------
# Figure 4 (panel b)
# ----------------------------------------------------

plot = (ggplot(results_fig4) +
# plot = (ggplot(test[(test['d'] == 'd = 100')]) +
aes(x='acc_rate', y='MSJD_over_B', colour='method') +  
    geom_line(linetype = "dashed", size=2) +
    geom_point(aes(shape = 'method'), size=6) +
    # geom_segment(aes(x = 0.45, xend = 0.45, y = 0, yend = 4e-7), colour='#1F78B4', size=0.4, linetype='dotted') +
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
