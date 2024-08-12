try:
    from plotnine import *
    import pandas as pd
    from mhssteste import *
except ImportError:
    raise ImportError("Please make sure plotnine, pandas and mhssteste are ALL installed!")

save_dir = os.getcwd() + '/'

colnames = np.array(['N','d', 'kappa', 'acc_rate', 'meanSJD', 'cpu_time', 'ESS', 'expected_B', 'acc_rate_ratio1'])

def run_tuna(x, y, theta_hat, V, taylor_order, npost, model, implementation, kappa=1.5):
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

    method = MH_SS(y, x, V, x0 = theta_hat, model=model, control_variates=False, taylor_order = taylor_order, bound = 'new', chi=chi, nburn=0, npost=npost, kappa=kappa,implementation=implementation)
    acc_rate = method.get('acc_rate')
    acc_rate_ratio1 = method.get('acc_rate_ratio1')
    meanSJD = method.get('meanSJD')
    cpu_time = method.get('cpu_time')
    expected_B = np.mean(method.get('BoverN'))*method.get('N')
    ESS = np.max(method.get('ESS'))
    save_results = np.array([N, d, kappa, acc_rate, meanSJD, cpu_time, ESS, expected_B, acc_rate_ratio1])
    save_results = pd.DataFrame(save_results[None, :], columns=colnames)

    return save_results

def run_RWM(x, y, theta_hat, V, npost, model, implementation, kappa=2.4):
    N = x.shape[0]
    d = x.shape[1]
    method = RWM(y, x, V, x0 = theta_hat, model=model, nburn=0, npost=npost, kappa = kappa, implementation=implementation)
    acc_rate = method.get('acc_rate')
    acc_rate_ratio1 = method.get('acc_rate_ratio1')
    meanSJD = method.get('meanSJD')
    cpu_time = method.get('cpu_time')
    expected_B = N
    ESS = np.max(method.get('ESS'))
    save_results = np.array([N, d, kappa, acc_rate, meanSJD, cpu_time, ESS, expected_B, acc_rate_ratio1])
    save_results = pd.DataFrame(save_results[None, :], columns=colnames)

    return save_results

def run_SMH(x, y, theta_hat, V, bound, taylor_order, npost, model, implementation):
    
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
    method = SMH(y, x, V, x0 = theta_hat, model=model, kappa=kappa, bound=bound, taylor_order=taylor_order, nburn=0, npost=npost,implementation=implementation)
    acc_rate = method.get('acc_rate')
    acc_rate_ratio1 = method.get('acc_rate_ratio1')
    meanSJD = method.get('meanSJD')
    cpu_time = method.get('cpu_time')
    expected_B = np.mean(method.get('BoverN'))*method.get('N')
    ESS = np.max(method.get('ESS'))
    save_results = np.array([N, d, kappa, acc_rate, meanSJD, cpu_time, ESS, expected_B, acc_rate_ratio1])
    save_results = pd.DataFrame(save_results[None, :], columns=colnames)

    return save_results

def run_methods(N, d, model, implementation='loop', npost=100000):

    npost_tuna = 10000000

    data = simulate_data(N, d, model)
    y = data.get('y')
    x = data.get('x')

    theta_hat, V = get_theta_hat_and_var_cov_matrix(model, x, y)

    filename_beg = save_dir + model + 'Related_works'
    filename_end = "N" + str(N) + 'd' + str(d) + '.pickle'

    tuna = run_tuna(x, y, theta_hat, V, taylor_order = 0, npost=npost_tuna, model=model, implementation=implementation)
    smh1 = run_SMH(x,  y, theta_hat, V, taylor_order = 1, npost=npost, model=model, bound='orig', implementation=implementation)
    rwm  = run_RWM(x,  y, theta_hat, V, npost = npost, model=model, implementation=implementation)

    tuna_file_name = filename_beg + 'Tuna' + filename_end
    smh1_file_name = filename_beg + 'SMH1' + filename_end
    rwm_file_name  = filename_beg + 'RWM' + filename_end

    save_file(tuna,  tuna_file_name)
    save_file(rwm,   rwm_file_name)
    save_file(smh1,  smh1_file_name)

def simulation(d, model = 'logistic'):
    N = np.array([31622])
    len_N = len(N)
    for j in range(len_N):
        print('N = ' + str(N[j]))
        run_methods(N[j], d=d, model=model)

simulation(5)
simulation(10)
simulation(20)
simulation(30)
simulation(40)
simulation(50)
simulation(60)

def get_results(N, model='logistic'):

    set_d = np.array([5, 10, 20, 30, 40, 50, 60])
    colnames = np.array(['N','d', 'kappa', 'acc_rate', 'meanSJD', 'cpu_time', 'ESS', 'expected_B', 'method', 'acc_rate_ratio1'])
    store_results = np.zeros((len(set_d), len(colnames)))
    store_results[:] = np.nan
    store_results = pd.DataFrame({'N':{},'d':{}, 'kappa':{}, 'acc_rate':{}, 'meanSJD':{}, 'cpu_time':{}, 'ESS':{}, 'expected_B':{}, 'method': {}, 'acc_rate_ratio1':{}})

    for d in set_d:
        filename_beg = save_dir + model + 'Related_works'
        filename_end = "N" + str(N) + 'd' + str(d) + '.pickle'

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
