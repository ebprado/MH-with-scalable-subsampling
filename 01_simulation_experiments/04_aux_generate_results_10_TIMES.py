from PyMHSS import *

save_dir = '/home/pradoe/code/01_PyMHSS/01_simulation_experiments/results/'

colnames = np.array(['N','d', 'kappa', 'acc_rate', 'meanSJD', 'cpu_time', 'ESS', 'KSD', 'expected_B', 'acc_rate_ratio1'])

def run_RWM(x, y, theta_hat, V, npost, model, implementation, kappa=2.4):
    N = x.shape[0]
    d = x.shape[1]
    method = RWM(y, x, V, x0 = theta_hat, model=model, nburn=0, npost=npost, kappa = kappa, implementation=implementation, calculate_ksd=False)
    acc_rate = method.get('acc_rate')
    acc_rate_ratio1 = method.get('acc_rate_ratio1')
    meanSJD = method.get('meanSJD')
    cpu_time = method.get('cpu_time')
    expected_B = N
    ESS = np.max(method.get('ESS'))
    KSD = method.get('KSD')
    save_results = np.array([N, d, kappa, acc_rate, meanSJD, cpu_time, ESS, KSD, expected_B, acc_rate_ratio1])
    save_results = pd.DataFrame(save_results[None, :], columns=colnames)

    return save_results

def run_MH_SS(x, y, theta_hat, V, taylor_order, npost, model, implementation, kappa=1.5, tuna_acc_rate = 0.24):
    N = x.shape[0]
    d = x.shape[1]

    if d == 100 and N == 1000:
        kappa = 2.4
    
    if tuna_acc_rate == 0.6:
        if taylor_order == 0:
            if d == 10:
                if N == 1000:
                    kappa = 0.14
                    chi = 1e-3
                elif N == 3162:
                    kappa = 0.08
                    chi = 9e-4
                elif N == 10000:
                    kappa = 0.04
                    chi = 2e-3
                elif N == 31622:
                    kappa = 0.025
                    chi = 1e-3
                elif N == 100000:
                    kappa = 0.015
                    chi = 1e-3
            elif d == 30:
                if N == 1000:
                    kappa = 0.21
                    chi = 2e-4
                elif N == 3162:
                    kappa = 0.11
                    chi = 2e-4
                elif N == 10000:
                    kappa = 0.06
                    chi = 2e-4
                elif N == 31622:
                    kappa = 0.04
                    chi = 1e-4
                elif N == 100000:
                    kappa = 0.02
                    chi = 2e-4

            elif d == 50:
                if N == 1000:
                    kappa = 0.25
                    chi = 6e-5
                elif N == 3162:
                    kappa = 0.15
                    chi = 6e-5
                elif N == 10000:
                    kappa = 0.09
                    chi = 6e-5
                elif N == 31622:
                    kappa = 0.045
                    chi = 6e-5
                elif N == 100000:
                    kappa = 0.03
                    chi = 5e-5

            elif d == 100:
                if N == 1000:
                    kappa = 0.35
                    chi = 1e-5
                elif N == 3162:
                    kappa = 0.22
                    chi = 1.5e-5
                elif N == 10000:
                    kappa = 0.12
                    chi = 1.5e-5
                elif N == 31622:
                    kappa = 0.07
                    chi = 1.5e-5
                elif N == 100000:
                    kappa = 0.04
                    chi = 1.5e-5
    else:
        if taylor_order == 0:
            if d == 10:
                if N == 1000:
                    kappa = 0.65
                    chi = 2e-4
                elif N == 3162:
                    kappa = 0.4
                    chi = 2e-4
                elif N == 10000:
                    kappa = 0.25
                    chi = 2e-4
                elif N == 31622:
                    kappa = 0.11
                    chi = 1e-4
                elif N == 100000:
                    kappa = 0.11
                    chi = 2e-3
            if d == 30:
                if N == 1000:
                    kappa = 0.9
                    chi = 4e-5
                elif N == 3162:
                    kappa = 0.6
                    chi = 2e-5
                elif N == 10000:
                    kappa = 0.35
                    chi = 2e-5
                elif N == 31622:
                    kappa = 0.17
                    chi = 2e-5
                elif N == 100000:
                    kappa = 0.1
                    chi = 2e-5

            if d == 50:
                if N == 1000:
                    kappa = 1.1
                    chi = 2e-5
                elif N == 3162:
                    kappa = 0.75
                    chi = 1e-5
                elif N == 10000:
                    kappa = 0.4
                    chi = 1e-5
                elif N == 31622:
                    kappa = 0.20
                    chi = 1e-5
                elif N == 100000:
                    kappa = 0.12
                    chi = 2e-5

            if d == 100:
                if N == 1000:
                    kappa = 2.4
                    chi = 0
                elif N == 3162:
                    kappa = 1
                    chi = 2e-6
                elif N == 10000:
                    kappa = 0.5
                    chi = 2e-6
                elif N == 31622:
                    kappa = 0.30
                    chi = 2e-6
                elif N == 100000:
                    kappa = 0.17
                    chi = 2e-6
            
    # Tuna = MH-SS without control variates (control_variates = False) and phi_function='original'
    if taylor_order == 0:
        method = MH_SS(y, x, V, x0 = theta_hat, model=model, control_variates=False, phi_function='original', taylor_order=taylor_order, chi=chi, nburn=0, npost=npost, kappa=kappa, nthin=10**2, implementation=implementation, calculate_ksd=False)    
    else:
        method = MH_SS(y, x, V, x0 = theta_hat, model=model, control_variates=True, taylor_order=taylor_order, chi=0, nburn=0, npost=npost, kappa=kappa, implementation=implementation, calculate_ksd=False)

    acc_rate = method.get('acc_rate')
    acc_rate_ratio1 = method.get('acc_rate_ratio1')
    meanSJD = method.get('meanSJD')
    cpu_time = method.get('cpu_time')
    expected_B = np.mean(method.get('BoverN'))*method.get('N')
    ESS = np.max(method.get('ESS'))
    KSD = method.get('KSD')
    save_results = np.array([N, d, kappa, acc_rate, meanSJD, cpu_time, ESS, KSD, expected_B, acc_rate_ratio1])
    save_results = pd.DataFrame(save_results[None, :], columns=colnames)

    return save_results

def run_SMH(x, y, theta_hat, V, bound, taylor_order, npost, model, implementation):

    N = x.shape[0]
    d = x.shape[1]

    if bound == 'orig':
        if taylor_order == 1:
            # kappa = 1

            # if N <= 3500 and d >= 30:
            #     kappa = 2.4

            # if N == 1000 and d >= 30:
            #     kappa = 2.4

            # if N == 10000 and d > 30:
            #     kappa = 2.4

            # if N == 31622 and d == 100:
            #     kappa = 2.4

            if d == 10:
                kappa = 1.4
            elif N <= 3162 and d >= 30:
                kappa = 2.4
            elif N == 10000 and d == 30:
                kappa = 0.8
            elif N == 10000 and d >= 50:
                kappa = 2.4            
            elif N == 31622 and d == 30:
                kappa = 0.8       
            elif N == 31622 and d == 50:
                kappa = 0.65    
            elif N == 31622 and d == 100:
                kappa = 2.4  
            elif N == 100000 and d == 30:
                kappa = 0.8
            elif N == 100000 and d == 50:
                kappa = 0.65
            elif N == 100000 and d == 100:
                kappa = 0.5

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

    if kappa != 2.4:    
        method = SMH(y, x, V, x0 = theta_hat, model=model, kappa=kappa, bound=bound, taylor_order=taylor_order, nburn=0, npost=npost, nthin = 10, implementation=implementation, calculate_ksd=False)
        acc_rate = method.get('acc_rate')
        acc_rate_ratio1 = method.get('acc_rate_ratio1')
        meanSJD = method.get('meanSJD')
        cpu_time = method.get('cpu_time')
        expected_B = np.mean(method.get('BoverN'))*method.get('N')
        ESS = np.max(method.get('ESS'))
        KSD = method.get('KSD')
        save_results = np.array([N, d, kappa, acc_rate, meanSJD, cpu_time, ESS, KSD, expected_B, acc_rate_ratio1])        
        save_results = pd.DataFrame(save_results[None, :], columns=colnames)

        return save_results

def run_methods(N, d, model, implementation='vectorised'):

    npost_tuna = 10000000
    npost_smh1 = 1000000
    npost = 100000

    for i in range(10):

        print('Replicate: ' + str(i))

        data = simulate_data(N, d, model)
        y = data.get('y')
        x = data.get('x')

        theta_hat, V = get_theta_hat_and_var_cov_matrix(model, x, y)
        filename_end = "N" + str(N) + 'd' + str(d) + 'Imp' + str(implementation) + 'rep' + str(i) + '.pickle'

        # tuna_060_results = run_MH_SS(x, y, theta_hat, V, taylor_order = 0, npost=npost_tuna, model=model, implementation = implementation, tuna_acc_rate=0.6)
        # tuna_024_results = run_MH_SS(x, y, theta_hat, V, taylor_order = 0, npost=npost_tuna, model=model, implementation = implementation, tuna_acc_rate=0.24)

        # mhss1_save_results = run_MH_SS(x, y, theta_hat, V, taylor_order = 1, npost=npost, model=model, implementation = implementation)
        # mhss2_save_results = run_MH_SS(x, y, theta_hat, V, taylor_order = 2, npost=npost, model=model, implementation = implementation)
        # rwm_save_results = run_RWM(x, y, theta_hat, V, npost, model=model, implementation = implementation)
        smh1_save_results = run_SMH(x, y, theta_hat, V, bound='orig', taylor_order=1, npost=npost_smh1, model=model, implementation = implementation)
        # smh2_save_results = run_SMH(x, y, theta_hat, V, bound='orig', taylor_order=2, npost=npost, model=model, implementation = implementation)
        # smh1_chris_save_results = run_SMH(x, y, theta_hat, V, bound='ChrisS', taylor_order=1, npost=npost, model=model, implementation = implementation)
        # smh2_chris_save_results = run_SMH(x, y, theta_hat, V, bound='ChrisS', taylor_order=2, npost=npost, model=model, implementation = implementation)

        tuna_060_name = save_dir + model + 'EfficiencyMetricsTuna_060_acc_rate' + filename_end
        tuna_024_name = save_dir + model + 'EfficiencyMetricsTuna_024_acc_rate' + filename_end
        mhss1_file_name = save_dir + model + 'EfficiencyMetricsMHSS1' + filename_end
        mhss2_file_name = save_dir + model + 'EfficiencyMetricsMHSS2' + filename_end
        rwm_file_name = save_dir + model + 'EfficiencyMetricsRWM' + filename_end
        smh1_file_name = save_dir + model + 'EfficiencyMetricsSMH' + filename_end
        smh2_file_name = save_dir + model + 'EfficiencyMetricsSMH2' + filename_end
        smh1_chris_file_name = save_dir + model + 'EfficiencyMetricsSMH1NB' + filename_end
        smh2_chris_file_name = save_dir + model + 'EfficiencyMetricsSMH2NB' + filename_end

        # save_file(tuna_060_results, tuna_060_name)
        # save_file(tuna_024_results, tuna_024_name)        
        # save_file(mhss1_save_results, mhss1_file_name)
        # save_file(mhss2_save_results, mhss2_file_name)
        # save_file(rwm_save_results, rwm_file_name)
        save_file(smh1_save_results, smh1_file_name)
        # save_file(smh2_save_results, smh2_file_name)
        # save_file(smh1_chris_save_results, smh1_chris_file_name)
        # save_file(smh2_chris_save_results, smh2_file_name)

def run_many_times(d, implementation, model = 'logistic'):
    N = np.array([100000, 31622, 10000, 3162, 1000])
    len_N = len(N)
    for j in range(len_N):
        run_methods(N[j], d=d, model=model, implementation=implementation)

# run_many_times(10, implementation='vectorised')
# run_many_times(30, implementation='vectorised')
# run_many_times(50, implementation='vectorised')
# run_many_times(100, implementation='vectorised')

def get_results(N, implementation, model='logistic', rep=1):

    set_d = np.array([10, 30, 50, 100])
    colnames = np.array(['N','d', 'kappa', 'acc_rate', 'meanSJD', 'cpu_time', 'ESS', 'KSD', 'expected_B', 'method', 'acc_rate_ratio1'])
    store_results = np.zeros((len(set_d), len(colnames)))
    store_results[:] = np.nan
    store_results = pd.DataFrame({'N':{},'d':{}, 'kappa':{}, 'acc_rate':{}, 'meanSJD':{}, 'cpu_time':{}, 'ESS':{}, 'KSD':{}, 'expected_B':{}, 'method': {}, 'acc_rate_ratio1':{}})

    for i in range(10):
        for d in set_d:
            filename_end = "N" + str(N) + 'd' + str(d) + 'Imp' + str(implementation) + 'rep' + str(i) + '.pickle'
            tuna_060_name = save_dir + model + 'EfficiencyMetricsTuna_060_acc_rate' + filename_end
            tuna_024_name = save_dir + model + 'EfficiencyMetricsTuna_024_acc_rate' + filename_end
            mhss1_file_name = save_dir + model + 'EfficiencyMetricsMHSS1' + filename_end
            mhss2_file_name = save_dir + model + 'EfficiencyMetricsMHSS2' + filename_end
            rwm_file_name = save_dir + model + 'EfficiencyMetricsRWM' + filename_end
            smh1_file_name = save_dir + model + 'EfficiencyMetricsSMH' + filename_end
            smh2_file_name = save_dir + model + 'EfficiencyMetricsSMH2' + filename_end
            smh1_chris_file_name = save_dir + model + 'EfficiencyMetricsSMH1NB' + filename_end
            smh2_chris_file_name = save_dir + model + 'EfficiencyMetricsSMH2NB' + filename_end

            if os.path.exists(tuna_060_name):
                with open(tuna_060_name, 'rb') as f:
                    tuna_results_orig = pickle.load(f)
                    tuna_results_orig['method'] = 'Tuna'
                    store_results = pd.concat([store_results, tuna_results_orig])

            if os.path.exists(tuna_024_name):
                with open(tuna_024_name, 'rb') as f:
                    tuna_results = pickle.load(f)
                    tuna_results['method'] = 'Tuna_024'
                    store_results = pd.concat([store_results, tuna_results])

            if os.path.exists(mhss1_file_name):
                with open(mhss1_file_name, 'rb') as f:
                    mhss1_results = pickle.load(f)
                    mhss1_results['method'] = 'MH-SS-1'
                    store_results = pd.concat([store_results, mhss1_results])

            if os.path.exists(mhss2_file_name):
                with open(mhss2_file_name, 'rb') as f:
                    mhss2_results = pickle.load(f)
                    mhss2_results['method'] = 'MH-SS-2'
                    store_results = pd.concat([store_results, mhss2_results])

            if os.path.exists(rwm_file_name):
                with open(rwm_file_name, 'rb') as f:
                    rwm_results = pickle.load(f)
                    if rwm_results is not None:
                        rwm_results['method'] = 'RWM'
                        store_results = pd.concat([store_results, rwm_results])
            
            if os.path.exists(smh1_file_name):
                with open(smh1_file_name, 'rb') as f:
                    smh1_results = pickle.load(f)
                    if smh1_results is not None:
                        smh1_results['method'] = 'SMH-1'
                        # smh1_results['method'] = 'SMH-1-acc-20pc'
                        store_results = pd.concat([store_results, smh1_results])
            
            if os.path.exists(smh2_file_name):                
                with open(smh2_file_name, 'rb') as f:
                    smh2_results = pickle.load(f)
                    if smh2_results is not None:
                        smh2_results['method'] = 'SMH-2'
                        store_results = pd.concat([store_results, smh2_results])

            if os.path.exists(smh1_chris_file_name):
                with open(smh1_chris_file_name, 'rb') as f:
                    smh1_chris_results = pickle.load(f)
                    if smh1_chris_results is not None:                
                        smh1_chris_results['method'] = 'SMH-1-NB'
                        store_results = pd.concat([store_results, smh1_chris_results])

            if os.path.exists(smh2_chris_file_name):
                with open(smh2_chris_file_name, 'rb') as f:
                    smh2_chris_results = pickle.load(f)
                    if smh2_chris_results is not None:
                        smh2_chris_results['method'] = 'SMH-2-NB'
                        store_results = pd.concat([store_results, smh2_chris_results])

    store_results.columns = colnames

    return store_results

vec1 = get_results(1000, implementation='vectorised')
vec2 = get_results(3162, implementation='vectorised')
vec3 = get_results(10000, implementation='vectorised')
vec4 = get_results(31622, implementation='vectorised')
vec5 = get_results(100000, implementation='vectorised')

results_vectorised = pd.concat([vec1, vec2, vec3, vec4, vec5])

save_file(results_vectorised, save_dir + '00_results_all_vectorised_10_times.pickle')

results_vectorised = results_vectorised.drop('cpu_time', axis=1)

with open(save_dir + '00_results_all_loop.pickle', 'rb') as f:

    one_time_loop_results = pickle.load(f)
    one_time_loop_results = one_time_loop_results[['N', 'd', 'method', 'cpu_time']]

results_vectorised=results_vectorised.merge(one_time_loop_results, on=['N', 'd', 'method'])

save_file(results_vectorised, save_dir + '00_results_all_loop_10_times.pickle')