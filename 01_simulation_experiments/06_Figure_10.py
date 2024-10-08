from PyMHSS import *

save_dir = os.getcwd() + '/'

def get_results(N, rep=10):

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
    store_results['method'] = pd.Categorical(store_results.method, categories= ['MH-SS-1', 'MH-SS-2', 'SMH-1', 'SMH-2', 'SMH-1-NB', 'SMH-2-NB', 'RWM'])
    
    return store_results

results_fig = get_results(100000)

height_plot = 6
width_plot = 8

# ----------------------------------------------------
# Figure 10 (panel e)
# ----------------------------------------------------
plot = (ggplot(results_fig) +
aes(x='kappa', y='acc_rate', colour='method') +  
    geom_line(linetype = "dashed", size=2) +
    geom_point(aes(shape = 'method'), size=6) +
    # geom_segment(aes(x = 1.5, xend = 1.5, y = 0, yend = 4e-7), colour='#1F78B4', size=0.4, linetype='dotted') +
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
scale_color_brewer(type = 'qual', palette = 'Paired'))
plot
plot.save('additional_results_acc_rate_d_lambda.pdf', height=height_plot, width=width_plot)

# ----------------------------------------------------
# Figure 10 (panel a)
# ----------------------------------------------------
results_fig['method'] = results_fig.method.astype(str)

plot = (ggplot(results_fig[results_fig['method'] == 'SMH-1']) +
aes(x='acc_rate', y='MSJD_over_B', colour='method') +  
    geom_line(linetype = "dashed", size=2) +
    geom_point(aes(shape = 'method'), size=6) +
    # geom_segment(aes(x = 1.5, xend = 1.5, y = 0, yend = 4e-7), colour='#1F78B4', size=0.4, linetype='dotted') +
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
scale_color_manual(values=["#B2DF8A"]))

plot
plot.save('additional_results_msjd_acc_rate_smh_1.pdf', height=height_plot, width=width_plot)

# ----------------------------------------------------
# Figure 10 (panel b)
# ----------------------------------------------------
plot = (ggplot(results_fig[results_fig['method'] == 'SMH-1-NB']) +
aes(x='acc_rate', y='MSJD_over_B', colour='method') +  
    geom_line(linetype = "dashed", size=2) +
    geom_point(aes(shape = 'method'), size=6) +
    # geom_segment(aes(x = 1.5, xend = 1.5, y = 0, yend = 4e-7), colour='#1F78B4', size=0.4, linetype='dotted') +
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
scale_color_manual(values=["#FB9A99"]))
plot

plot.save('additional_results_msjd_acc_rate_smh_1_nb.pdf', height=height_plot, width=width_plot)

# ----------------------------------------------------
# Figure 10 (panel c)
# ----------------------------------------------------
plot = (ggplot(results_fig[results_fig['method'] == 'SMH-2']) +
aes(x='acc_rate', y='MSJD_over_B', colour='method') +  
    geom_line(linetype = "dashed", size=2) +
    geom_point(aes(shape = 'method'), size=6) +
    # geom_segment(aes(x = 1.5, xend = 1.5, y = 0, yend = 4e-7), colour='#1F78B4', size=0.4, linetype='dotted') +
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
scale_color_manual(values=["#33A02C"]))

plot
plot.save('additional_results_msjd_acc_rate_smh_2.pdf', height=height_plot, width=width_plot)

# ----------------------------------------------------
# Figure 10 (panel d)
# ----------------------------------------------------
plot = (ggplot(results_fig[results_fig['method'] == 'SMH-2-NB']) +
aes(x='acc_rate', y='MSJD_over_B', colour='method') +  
    geom_line(linetype = "dashed", size=2) +
    geom_point(aes(shape = 'method'), size=6) +
    # geom_segment(aes(x = 1.5, xend = 1.5, y = 0, yend = 4e-7), colour='#1F78B4', size=0.4, linetype='dotted') +
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
scale_color_manual(values=["#E31A1C"]))
plot
plot.save('additional_results_msjd_acc_rate_smh_2_nb.pdf', height=height_plot, width=width_plot)
