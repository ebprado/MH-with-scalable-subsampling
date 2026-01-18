from PyMHSS import *

dataset = 'hepmass'
model = 'probit'

# dir = os.getcwd()
dir = '/home/pradoe/code/01_PyMHSS/02_real_world_applications/03_hepmass/'

train = pd.read_table(dir + 'all_train.csv', sep=',', index_col=False)
train
train.columns = ['label', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
                  'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19','f20', 'f21',
                          'f22', 'f23', 'f24', 'f25', 'f26', 'mass']
n_tot = train.shape[0]
n_sample = 1000000

np.random.seed(101)
idx = np.random.choice(n_tot, n_sample, replace=False)
y = train['label'][idx]
y = np.asarray(y)
x = train.drop(['label'], axis=1)
x = x.iloc[idx, :]
N = len(y)

x_mean = np.mean(x, axis=0)
x_std = np.sqrt(np.var(x, axis=0))
x = (x - x_mean)/x_std
x = np.asarray(x)
x = np.column_stack((np.repeat(1, N), x))
d = x.shape[1]

theta_hat, V = get_theta_hat_and_var_cov_matrix(model, x, y, basic_lr = 0.0000001) # probit
theta_hat, V = get_theta_hat_and_var_cov_matrix(model, x, y, basic_lr = 0.000001) # logistic

theta_hat_file_name = dir + str(dataset) + model + 'theta_hat' + '.pickle'
V_file_name = dir + str(dataset) + model + 'V' + '.pickle'

save_file(theta_hat, theta_hat_file_name)
save_file(V, V_file_name)

theta_hat = open_file(theta_hat_file_name)
V = open_file(V_file_name)

colnames = np.array(['N','d', 'kappa', 'acc_rate', 'meanSJD', 'cpu_time', 'ESS', 'KSD', 'expected_B', 'acc_rate_ratio1'])

def run_RWM(x, y, theta_hat, V, nburn, npost, nthin, model, implementation, kappa=2.4):
    N = x.shape[0]
    d = x.shape[1]
    method = RWM(y, x, V, x0 = theta_hat, model=model, nburn=0, npost=npost, nthin=nthin, kappa = kappa, implementation=implementation, calculate_ksd=True)
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

def run_MH_SS(x, y, theta_hat, V, taylor_order, nburn, npost, nthin, model, implementation, kappa=1.5, tuna_acc_rate = 0.24):
    N = x.shape[0]
    d = x.shape[1]
            
    # Tuna = MH-SS without control variates (control_variates = False) and phi_function='original'
    if taylor_order == 0:
        if tuna_acc_rate == 0.6:
            chi = 1e-5
            kappa = 0.013
        else:
            chi = 1e-6
            kappa = 0.11
            
        method = MH_SS(y, x, V, x0 = theta_hat, model=model, control_variates=False, phi_function='original', taylor_order=taylor_order, chi=chi, nburn=nburn, npost=npost, kappa=kappa, nthin=nthin, implementation=implementation, calculate_ksd=True)    
    else:
        method = MH_SS(y, x, V, x0 = theta_hat, model=model, control_variates=True, taylor_order=taylor_order, chi=0, nburn=nburn, npost=npost, nthin=nthin, kappa=kappa, implementation=implementation, calculate_ksd=True)

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

def run_SMH(x, y, theta_hat, V, bound, taylor_order, nburn, npost, nthin, model, implementation):

    N = x.shape[0]
    d = x.shape[1]

    if taylor_order == 1:
        kappa = 1

    elif taylor_order == 2:
        kappa = 2

    print(kappa)

    method = SMH(y, x, V, x0 = theta_hat, model=model, kappa=kappa, bound=bound, taylor_order=taylor_order, nburn=nburn, npost=npost, nthin = nthin, implementation=implementation, calculate_ksd=True)
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

def run_methods(dataset, model, implementation='loop'):
    
    nburn = 0
    npost = 100000
    npost_tuna = 100000000
    nthin = 10
    nthin_tuna = 10000

    for i in range(10):

        print('Replicate: ' + str(i))

        filename_end = str(dataset) + 'Imp' + str(implementation) + 'rep' + str(i) + '.pickle'

        if model == 'logistic':
                tuna_060_results = run_MH_SS(x, y, theta_hat, V, taylor_order = 0, npost=npost_tuna, nburn=nburn, nthin=nthin_tuna, model=model, implementation = implementation, tuna_acc_rate=0.6)
                tuna_024_results = run_MH_SS(x, y, theta_hat, V, taylor_order = 0, npost=npost_tuna, nburn=nburn, nthin=nthin_tuna, model=model, implementation = implementation, tuna_acc_rate=0.24)

        mhss1_save_results = run_MH_SS(x, y, theta_hat, V, taylor_order = 1, npost=npost, nburn=nburn, nthin=nthin, model=model, implementation = implementation)
        mhss2_save_results = run_MH_SS(x, y, theta_hat, V, taylor_order = 2, npost=npost, nburn=nburn, nthin=nthin, model=model, implementation = implementation)
        rwm_save_results = run_RWM(x, y, theta_hat, V, npost=npost, nburn=nburn, nthin=nthin, model=model, implementation = implementation)
        smh1_save_results = run_SMH(x, y, theta_hat, V, bound='orig', taylor_order=1, npost=npost, nburn=nburn, nthin=nthin, model=model, implementation = implementation)
        smh2_save_results = run_SMH(x, y, theta_hat, V, bound='orig', taylor_order=2, npost=npost, nburn=nburn, nthin=nthin, model=model, implementation = implementation)

        tuna_060_name = dir + model + 'EfficiencyMetricsTuna_060_acc_rate' + filename_end
        tuna_024_name = dir + model + 'EfficiencyMetricsTuna_024_acc_rate' + filename_end
        mhss1_file_name = dir + model + 'EfficiencyMetricsMHSS1' + filename_end
        mhss2_file_name = dir + model + 'EfficiencyMetricsMHSS2' + filename_end
        rwm_file_name = dir + model + 'EfficiencyMetricsRWM' + filename_end
        smh1_file_name = dir + model + 'EfficiencyMetricsSMH' + filename_end
        smh2_file_name = dir + model + 'EfficiencyMetricsSMH2' + filename_end

        if model == 'logistic':
                save_file(tuna_060_results, tuna_060_name)
                save_file(tuna_024_results, tuna_024_name)        
        save_file(mhss1_save_results, mhss1_file_name)
        save_file(mhss2_save_results, mhss2_file_name)
        save_file(rwm_save_results, rwm_file_name)
        save_file(smh1_save_results, smh1_file_name)
        save_file(smh2_save_results, smh2_file_name)

run_methods(dataset, model=model, implementation='loop')

def get_results(dataset, implementation, model, rep=10):

    colnames = np.array(['N','d', 'kappa', 'acc_rate', 'meanSJD', 'cpu_time', 'ESS', 'KSD', 'expected_B', 'method', 'acc_rate_ratio1'])

    store_results = np.zeros((rep, len(colnames)))
    store_results[:] = np.nan
    store_results = pd.DataFrame({'N':{},'d':{}, 'kappa':{}, 'acc_rate':{}, 'meanSJD':{}, 'cpu_time':{}, 'ESS':{}, 'KSD':{}, 'expected_B':{}, 'method': {}, 'acc_rate_ratio1':{}})

    for i in range(rep):

        filename_end = str(dataset) + 'Imp' + str(implementation) + 'rep' + str(i) + '.pickle'
        tuna_060_name = dir + model + 'EfficiencyMetricsTuna_060_acc_rate' + filename_end
        tuna_024_name = dir + model + 'EfficiencyMetricsTuna_024_acc_rate' + filename_end
        mhss1_file_name = dir + model + 'EfficiencyMetricsMHSS1' + filename_end
        mhss2_file_name = dir + model + 'EfficiencyMetricsMHSS2' + filename_end
        rwm_file_name = dir + model + 'EfficiencyMetricsRWM' + filename_end
        smh1_file_name = dir + model + 'EfficiencyMetricsSMH' + filename_end
        smh2_file_name = dir + model + 'EfficiencyMetricsSMH2' + filename_end

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
                                store_results = pd.concat([store_results, smh1_results])
        
        if os.path.exists(smh2_file_name):                
                with open(smh2_file_name, 'rb') as f:
                        smh2_results = pickle.load(f)
                        if smh2_results is not None:
                                smh2_results['method'] = 'SMH-2'
                                store_results = pd.concat([store_results, smh2_results])

    store_results.columns = colnames

    return store_results

results_loop = get_results(dataset, implementation='loop', model=model)

save_file(results_loop, dir +  '00_results_all_loop_10_times_' + str(dataset) + '_' + str(model) + '.pickle')

# -------------------------------------------------------------------------------
# Summary statistics
# -------------------------------------------------------------------------------

db = pd.read_pickle(dir +  '00_results_all_loop_10_times_' + str(dataset) + '_' + str(model) + '.pickle')
table= db

table['KSD'] = table['KSD'].astype(float)
table['ESS'] = table['ESS'].astype(float)
table['cpu_time'] = table['cpu_time'].astype(float)
table['expected_B'] = table['expected_B'].astype(float)
table['ESS_per_second'] = table['ESS']/table['cpu_time']
table['ESS_over_B'] = table['ESS']/table['expected_B']

table.groupby(['method']).agg({'expected_B': ['mean', 'std']}).reset_index()
table.groupby(['method']).agg({'ESS_per_second': ['mean', 'std']}).reset_index()
table.groupby(['method']).agg({'ESS_over_B': ['mean', 'std']}).reset_index()
table.groupby(['method']).agg({'KSD': ['mean', 'std']}).reset_index()
