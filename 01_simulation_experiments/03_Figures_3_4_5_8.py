from PyMHSS import *
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

os.chdir(RESULTS_DIR)

def organise_outputs(type_plot, implementation, remove_tuna_024 = True):

    # with open('00_results_all_' + implementation + '_10_times.pickle', 'rb') as f:
    #     store_results = pickle.load(f)

    store_results = pd.read_pickle('00_results_all_' + implementation + '_10_times.pickle')

    if type_plot == 1:
        if remove_tuna_024:
            store_results = store_results[(store_results['method'] != 'SMH-1-NB') & (store_results['method'] != 'SMH-2-NB') & (store_results['method'] != 'Tuna_024')]
            categories_method = ['MH-SS-1', 'MH-SS-2', 'SMH-1', 'SMH-2', 'RWM', 'Tuna']           

        else:
            store_results = store_results[(store_results['method'] != 'SMH-1-NB') & (store_results['method'] != 'SMH-2-NB')]
            categories_method = ['MH-SS-1', 'MH-SS-2', 'SMH-1', 'SMH-2', 'RWM', 'Tuna', 'Tuna_024']
    else:
        if remove_tuna_024:
            store_results = store_results[(store_results['method'] != 'Tuna_024') & (store_results['method'] != 'Tuna')]
        store_results = store_results[(store_results['method'] != 'RWM') & (store_results['method'] != 'Tuna')]
        categories_method = ['MH-SS-1', 'MH-SS-2', 'SMH-1', 'SMH-2', 'SMH-1-NB', 'SMH-2-NB']

    store_results['KSD'] = store_results.KSD.astype(float)
    store_results['ESS'] = store_results.ESS.astype(float)
    store_results['cpu_time'] = store_results.cpu_time.astype(float)
    store_results['expected_B'] = store_results.expected_B.astype(float)
    store_results['N'] = store_results.N.astype(float)
    store_results['meanSJD'] = store_results.meanSJD.astype(float)
    store_results['MSJD_over_B'] = np.log10(store_results['N'] * store_results['meanSJD'] / store_results['expected_B'])
    store_results['ESS_per_second'] = np.log10(store_results['ESS'] / store_results['cpu_time'])        

    # This is needed because Tuna and SMH-1 require more MCMC iterations to reach reasonable ESS
    # store_results.loc[(store_results['method'] == 'Tuna_024'), 'ESS'] = store_results.loc[(store_results['method'] == 'Tuna_024'), 'ESS']*10
    store_results.loc[(store_results['method'] == 'Tuna'), 'ESS'] = store_results.loc[(store_results['method'] == 'Tuna'), 'ESS']/100
    # store_results.loc[(store_results['method'] == 'SMH-1'), 'ESS'] = store_results.loc[(store_results['method'] == 'SMH-1'), 'ESS']/10

    store_results['ESS_over_B'] = np.log10(store_results['ESS'] / store_results['expected_B'])
    store_results['log_expected_B'] = np.log10(store_results['expected_B'])
    store_results['log_KSD'] = np.log10(store_results['KSD'])

    store_results.loc[(store_results['kappa'] == 2.4) & (store_results['method'] != 'RWM'), 'MSJD_over_B']  = np.nan
    store_results.loc[(store_results['kappa'] == 2.4) & (store_results['method'] != 'RWM'), 'ESS_per_second']  = np.nan
    store_results.loc[(store_results['kappa'] == 2.4) & (store_results['method'] != 'RWM'), 'ESS_over_B']  = np.nan
    
    store_results['N'] = np.log10(store_results['N'])
    store_results['N'] = pd.Categorical(np.round(store_results['N'], 2))

    store_results['d'] = store_results.d.astype(int)
    store_results['d'] = 'd = ' + store_results.d.astype(str)
    store_results['d'] = pd.Categorical(store_results.d, categories=['d = 3', 'd = 10', 'd = 30', 'd = 50', 'd = 100'])
    store_results['method'] = pd.Categorical(store_results.method, categories=categories_method)
    
    return store_results

# ----------------------------------------
type_plot = 1
implementation = 'loop'
# implementation = 'vectorised'
# ----------------------------------------

data_plot = organise_outputs(type_plot, implementation, remove_tuna_024=True)

height_plot = 6
width_plot = 8

# --------------------------------------------------------------
# Figure 3
# --------------------------------------------------------------
plot = (ggplot(data_plot) + 
 aes(x='N', y='log_expected_B', color='method') +
    geom_boxplot() +
   labs(
     y = 'log$_{10}$ E(B)',
     x = 'log$_{10}$ n'
     ) + 
stat_summary(
    aes(group = 'method', color='method'),
    fun_y = np.mean,
    geom = "line",
    linetype = "dashed"
  ) +
theme_bw(base_size = 12) +
theme(plot_title = element_text(size = 20, hjust = 0.5),
    strip_text_y = element_text(angle = 0),
    legend_position = 'bottom',
    legend_title = element_blank(),
    panel_grid_major = element_blank(),
    panel_grid_minor = element_blank()) +
facet_wrap('d', scales='free_y') + 
scale_color_brewer(type = 'qual', palette = 'Paired'))

plot.save('type_' + str(type_plot) + str(implementation) + '_Expected_B_by_N_10_times.pdf', height=height_plot, width=width_plot)

# --------------------------------------------------------------
# Figure 4
# --------------------------------------------------------------
plot = (ggplot(data_plot) + 
 aes(x='N', y='ESS_per_second', color='method') +
    geom_boxplot() +
 labs(
      y = 'log$_{10}$ ESS per second',
      x = 'log$_{10}$ n'
      ) + 
stat_summary(
    aes(group = 'method', color='method'),
    fun_y = np.mean,
    geom = "line",
    linetype = "dashed"
  ) +      
theme_bw(base_size = 12) +
theme(plot_title = element_text(size = 20, hjust = 0.5),
    strip_text_y = element_text(angle = 0),
    legend_position = 'bottom',
    legend_title = element_blank(),
    panel_grid_major = element_blank(),
    panel_grid_minor = element_blank()) +
facet_wrap('d', scales='free_y')  +
scale_color_brewer(type = 'qual', palette = 'Paired'))

plot.save('type_' + str(type_plot) + str(implementation) + '_ESS_per_second_10_times.pdf', height=height_plot, width=width_plot)

# --------------------------------------------------------------
# Figure 5 (a)
# --------------------------------------------------------------

plot = (ggplot(data_plot[(data_plot['d'] == 'd = 30')]) + 
 aes(x='N', y='log_expected_B', color='method') +
    # geom_line(linetype = "dashed", size=2.5) +
    # geom_point(aes(shape = 'method'), size=7) +
    geom_boxplot(size=1.5) +
 labs(
      y = 'log$_{10}$ E(B)',
      x = 'log$_{10}$ n'
      ) + 
stat_summary(
    aes(group = 'method', color='method'),
    fun_y = np.mean,
    geom = "line",
    linetype = "dashed"
  ) +          
theme_bw(base_size = 27) +
theme(plot_title = element_text(size = 15, hjust = 0.5),
    strip_text_y = element_text(angle = 0),
    legend_position = 'bottom',
    legend_title = element_blank(),
    panel_grid_major = element_blank(),
    panel_grid_minor = element_blank()) +
facet_wrap('d') +
scale_color_brewer(type = 'qual', palette = 'Paired'))

plot.save('type_' + str(type_plot) + str(implementation) + '_Expected_B_by_N_d_30_10_times.pdf', height=8, width=10)

# --------------------------------------------------------------
# Figure 5 (b)
# --------------------------------------------------------------

plot = (ggplot(data_plot[(data_plot['d'] == 'd = 30')]) + 
 aes(x='N', y='ESS_per_second', color='method') +
    # geom_line(linetype = "dashed", size=2.5) +
    # geom_point(aes(shape = 'method'), size=7) +
    geom_boxplot(size=1.5) +
  labs(
      y = 'log$_{10}$ ESS per second',
      x = 'log$_{10}$ n'
      ) + 
stat_summary(
    aes(group = 'method', color='method'),
    fun_y = np.mean,
    geom = "line",
    linetype = "dashed"
  ) +        
theme_bw(base_size = 27) +
theme(plot_title = element_text(size = 15, hjust = 0.5),
    strip_text_y = element_text(angle = 0),
    legend_position = 'bottom',
    legend_title = element_blank(),
    panel_grid_major = element_blank(),
    panel_grid_minor = element_blank()) +
facet_wrap('d', scales='free_y') + 
scale_color_brewer(type = 'qual', palette = 'Paired'))

plot.save('type_' + str(type_plot) + str(implementation) + '_ESS_per_second_d_30_10_times.pdf', height=8, width=10)

# --------------------------------------------------------------
# Figure 9 (panels a and b)
# --------------------------------------------------------------

plot = (ggplot(data_plot[(data_plot['d'] == 'd = 50') | (data_plot['d'] == 'd = 100')]) + 
 aes(x='N', y='ESS_per_second', color='method') +
    # geom_line(linetype = "dashed", size=1) +
    # geom_point(aes(shape = 'method'), size=4) +
    geom_boxplot(size=1.5) +
  labs(
      y = 'log$_{10}$ ESS per second',
      x = 'log$_{10}$ n'
      ) + 
stat_summary(
    aes(group = 'method', color='method'),
    fun_y = np.mean,
    geom = "line",
    linetype = "dashed"
  ) +           
theme_bw(base_size = 15) +
theme(plot_title = element_text(size = 15, hjust = 0.5),
    strip_text_y = element_text(angle = 0),
    legend_position = 'bottom',
    legend_title = element_blank(),
    panel_grid_major = element_blank(),
    panel_grid_minor = element_blank()) +
facet_wrap('d', scales='free_y', ncol=1) + 
scale_color_brewer(type = 'qual', palette = 'Paired'))

plot.save('type_' + str(type_plot) + str(implementation) + '_ESS_per_second_d_50_100_10_times.pdf', height=8, width=5)

# --------------------------------------------------------------
# Figure 13
# --------------------------------------------------------------

plot = (ggplot(data_plot) +
 aes(x='N', y='ESS_over_B', color='method') +  
    # geom_line(linetype = "dashed", size=0.8) +
    # geom_point(aes(shape = 'method'), size=2.5) +
    geom_boxplot() +
 labs(
      y = 'log$_{10}$ ESS/E(B)',
      x = 'log$_{10}$ n'
      ) + 
stat_summary(
    aes(group = 'method', color='method'),
    fun_y = np.mean,
    geom = "line",
    linetype = "dashed"
  ) +         
theme_bw(base_size = 12) +
theme(plot_title = element_text(size = 20, hjust = 0.5),
    strip_text_y = element_text(angle = 0),
    legend_position = 'bottom',
    legend_title = element_blank(),
    panel_grid_major = element_blank(),
    panel_grid_minor = element_blank()) +
facet_wrap('d', scales='free_y') + 
scale_color_brewer(type = 'qual', palette = 'Paired'))

plot.save('type_' + str(type_plot) + str(implementation) + '_ESS_over_B_10_times.pdf', height=height_plot, width=width_plot)
