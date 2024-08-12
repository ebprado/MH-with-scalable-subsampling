try:
    from plotnine import *
    import pandas as pd
    from mhssteste import *
except ImportError:
    raise ImportError("Please make sure plotnine, pandas and mhssteste are ALL installed!")

save_dir = os.getcwd() + '/'

def run_MH_SS(N, d, cv, chi = -1, npost=100000, rep = 10, model = 'logistic', implementation='loop'):
    
    for k in range(rep):
        print(str(k + 1) + ' out of ' + str(rep))

        data = simulate_data(N, d, model)
        y = data.get('y')
        x = data.get('x')

        theta_hat, V = get_theta_hat_and_var_cov_matrix(model, x, y)

        colnames = np.array(['method','N','d', 'chi', 'acc_rate_min', 'acc_rate_max', 'acc_rate_ori'])

        if cv == True:

            mh_ss_min_cv1 = MH_SS(y, x, V, x0 = theta_hat, control_variates=cv, bound = 'new', model=model, phi_function = 'min', taylor_order = 1, chi=0, nburn=0, npost=npost, implementation=implementation)
            mh_ss_max_cv1 = MH_SS(y, x, V, x0 = theta_hat, control_variates=cv, bound = 'new', model=model, phi_function = 'max', taylor_order = 1, chi=0, nburn=0, npost=npost, implementation=implementation)
            mh_ss_ori_cv1 = MH_SS(y, x, V, x0 = theta_hat, control_variates=cv, bound = 'new', model=model, phi_function = 'original', taylor_order = 1, chi=0, nburn=0, npost=npost, implementation=implementation)

            chi = 0

            acc_rate_min_cv1 = mh_ss_min_cv1.get('acc_rate')
            acc_rate_max_cv1 = mh_ss_max_cv1.get('acc_rate')
            acc_rate_ori_cv1 = mh_ss_ori_cv1.get('acc_rate')
            
            save_results_cv1 = np.array(['MH-SS-1', N, d, chi, acc_rate_min_cv1, acc_rate_max_cv1, acc_rate_ori_cv1])

            save_results = pd.DataFrame(save_results_cv1[None], columns=colnames)
            file_name = save_dir + 'PhiComparisonCV' + str(cv) + "N" + str(N) + 'd' + str(d) + 'rep' + str(k) + '.pickle'

        else: 

            # Tuna = MH-SS without control variates and phi_function = 'original'

            tuna_min = MH_SS(y, x, V, x0 = theta_hat, control_variates=False, bound='', model=model,  phi_function = 'min', chi=chi, nburn=0, npost=npost)
            tuna_max = MH_SS(y, x, V, x0 = theta_hat, control_variates=False, bound='', model=model,  phi_function = 'max', chi=chi, nburn=0, npost=npost)
            tuna_ori = MH_SS(y, x, V, x0 = theta_hat, control_variates=False, bound='', model=model,  phi_function = 'original', chi=chi, nburn=0, npost=npost)

            acc_rate_min = tuna_min.get('acc_rate')
            acc_rate_max = tuna_max.get('acc_rate')
            acc_rate_ori = tuna_ori.get('acc_rate')

            save_results = np.array(['Tuna', N, d, chi, acc_rate_min, acc_rate_max, acc_rate_ori])
            save_results = pd.DataFrame(save_results[None], columns=colnames)
            file_name = save_dir + 'PhiComparisonCV' + str(cv) + "N" + str(N) + 'd' + str(d) + 'rep' + str(k) + '.pickle'

        with open(file_name, 'wb') as f:
            pickle.dump(save_results, f, pickle.HIGHEST_PROTOCOL)

run_MH_SS(1000, 5, cv=True)
run_MH_SS(1000, 15, cv=True)
run_MH_SS(1000, 30, cv=True)
run_MH_SS(1000, 50, cv=True)
run_MH_SS(1000, 100, cv=True)

run_MH_SS(10000, 5, cv=True)
run_MH_SS(10000, 15, cv=True)
run_MH_SS(10000, 30, cv=True)
run_MH_SS(10000, 50, cv=True)
run_MH_SS(10000, 100, cv=True)
            
run_MH_SS(100000, 5, cv=True)
run_MH_SS(100000, 15, cv=True)
run_MH_SS(100000, 30, cv=True)
run_MH_SS(100000, 50, cv=True)
run_MH_SS(100000, 100, cv=True)

def get_acc_rates(N, cv=True, rep=10):

    set_d = np.array([5, 15, 30, 50, 100])
    colnames = ['method','N', 'd', 'chi', 'phi_min', 'phi_max', 'phi_ori']
    store_results = np.zeros((len(set_d), len(colnames)))
    store_results[:] = np.nan
    store_results = pd.DataFrame({'method':{}, 'N':{},'d':{}, 'chi':{}, 'acc_rate_min':{}, 'acc_rate_max':{}, 'acc_rate_ori':{}})

    for d in set_d:
        for k in range(rep):
            file_name = save_dir + 'PhiComparisonCV' + str(cv) + "N" + str(N) + 'd' + str(d) + 'rep' + str(k) + '.pickle'

            if os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    results = pickle.load(f)
                store_results = pd.concat([store_results, results])

    store_results.columns = colnames

    store_results['N'] = store_results.N.astype(int)
    store_results['N'] = store_results.N.astype(str)
    store_results['N'] = np.where(store_results['N'] == '1000', r'$10^3$', store_results['N'])
    store_results['N'] = np.where(store_results['N'] == '10000', r'$10^4$', store_results['N'])
    store_results['N'] = np.where(store_results['N'] == '100000', r'$10^5$', store_results['N'])
    store_results['d'] = store_results.d.astype(int)

    store_results['N'] = 'n = ' + store_results.N.astype(str)
    store_results['d'] = store_results.d.astype(str)

    store_results['d'] = pd.Categorical(store_results.d, categories=['5', '15', '30', '50', '100'])

    store_results = pd.melt(store_results, id_vars=['method','N','d','chi'])

    store_results['variable'] = np.where(store_results['variable'] == 'phi_min', r'$\gamma = 0$', store_results['variable'])
    store_results['variable'] = np.where(store_results['variable'] == 'phi_ori', r'$\gamma = 0.5$', store_results['variable'])
    store_results['variable'] = np.where(store_results['variable'] == 'phi_max', r'$\gamma = 1$', store_results['variable'])
    store_results['variable'] = pd.Categorical(store_results.variable, categories=[r'$\gamma = 0$', r'$\gamma = 0.5$', r'$\gamma = 1$'])
    store_results['value'] = store_results['value'].astype(float)

    return store_results

test1_cv = get_acc_rates(1000, cv=True)
test2_cv = get_acc_rates(10000, cv=True)
test3_cv = get_acc_rates(100000, cv=True)

test_cv = pd.concat([test1_cv, test2_cv, test3_cv])

# --------------------------------------------------------------
# Figure 2
# --------------------------------------------------------------

plot = (ggplot(test_cv[test_cv['method'] == 'Tuna+CV-1']) +
aes(x='d', y='value', colour='variable') +
geom_boxplot() +
labs(y = 'Acceptance rate',
     x = 'd',
     ) +
theme(legend_position = 'right') + 
theme_bw(base_size = 12) +
theme(plot_title = element_text(size = 20, hjust = 0.5),
    strip_text_y = element_text(angle = 0),
    legend_position = 'bottom',
    legend_title = element_blank(),
    panel_grid_major = element_blank(),
    panel_grid_minor = element_blank()) +
facet_wrap('~N'))

plot.save('comparison_phi_functions.pdf', height=4.5, width=7)