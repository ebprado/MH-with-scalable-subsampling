import pandas as pd
import os as os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import scipy.special as sc
# --------------------------------------------------------------------------
# make sure you have the algorithms.py file in the current directory!
import algorithms 
# --------------------------------------------------------------------------

# Link to the UCI machine learning repository 
# https://archive.ics.uci.edu/dataset/347/hepmass
# ------------------------------------------------------------------
dataset = 'hepmass'
model = 'logistic'

dir = os.getcwd()

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

theta_hat, V = get_theta_hat_and_var_cov_matrix(model, x, y, basic_lr = 0.000001)

theta_hat_file_name = dir + str(dataset) + model + 'theta_hat' + '.pickle'
V_file_name = dir + str(dataset) + model + 'V' + '.pickle'

save_file(theta_hat, theta_hat_file_name)
save_file(V, V_file_name)

theta_hat = open_file(theta_hat_file_name)
V = open_file(V_file_name)

nburn = 0
npost = 100000

# ------------------------------------------------------------------------------------
# Original Tuna
# ------------------------------------------------------------------------------------
tuna_orig = tunaMH(y, x, V, x0 = theta_hat, control_variates=False, bound = 'new', model=model, phi_function = 'original', chi=1e-5, nburn=nburn, npost=npost, kappa=0.013)
tuna_orig.get('lambda')

# ------------------------------------------------------------------------------------
# Tuna+CV, SMH and RWM
# ------------------------------------------------------------------------------------
mh_ss_1 = tunaMH(y, x, V, x0 = theta_hat, control_variates=True, taylor_order=1, bound = 'new', model=model, chi=0, nburn=nburn, npost=npost, kappa=1.5)
mh_ss_2 = tunaMH(y, x, V, x0 = theta_hat, control_variates=True, taylor_order=2, bound = 'new', model=model, chi=0, nburn=nburn, npost=npost, kappa=1.5)
smh1 = smh(y, x, V, x0=theta_hat, kappa=1, bound='orig', taylor_order = 1, model=model, nburn=nburn, npost=npost)
smh2 = smh(y, x, V, x0=theta_hat, kappa=2, bound='orig', taylor_order = 2, model=model, nburn=nburn, npost=npost)
rwm = RWM(y, x, V, x0=theta_hat, model=model, nburn=nburn, npost=npost, kappa = 2.4)

vector_loop = 'loop'

tuna_file_name = str(dataset) + model + vector_loop + '_Tuna' + '.pickle'
mh_ss_1_file_name = str(dataset) + model + vector_loop + '_mh_ss_1' + '.pickle'
mh_ss_2_file_name = str(dataset) + model + vector_loop + '_mh_ss_2' + '.pickle'
smh1_file_name = str(dataset) + model + vector_loop + '_SMH1' + '.pickle'
smh2_file_name = str(dataset) + model + vector_loop + '_SMH2' + '.pickle'
rwm_file_name = str(dataset) + model + vector_loop + '_RWM' + '.pickle'

save_file(tuna_orig, tuna_file_name)
save_file(mh_ss_1, mh_ss_1_file_name)
save_file(mh_ss_2, mh_ss_2_file_name)
save_file(smh1, smh1_file_name)
save_file(smh2, smh2_file_name)
save_file(rwm, rwm_file_name)

tuna_orig = open_file(tuna_file_name)
mh_ss_1 = open_file(mh_ss_1_file_name)
mh_ss_2 = open_file(mh_ss_2_file_name)
smh1 = open_file(smh1_file_name)
smh2 = open_file(smh2_file_name)
rwm = open_file(rwm_file_name)

# Acceptance rate ------------------
tuna_acc_rate = tuna_orig.get('acc_rate')
mh_ss_1_acc_rate = mh_ss_1.get('acc_rate')
mh_ss_2_acc_rate = mh_ss_2.get('acc_rate')
smh1_acc_rate = smh1.get('acc_rate')
smh2_acc_rate = smh2.get('acc_rate')
rwm_acc_rate = rwm.get('acc_rate')

# E(B)/N --------------------------- 
tuna_EB = np.mean(tuna_orig.get('BoverN'))*N
mh_ss_1_EB = np.mean(mh_ss_1.get('BoverN'))*N
mh_ss_2_EB = np.mean(mh_ss_2.get('BoverN'))*N
smh1_EB = np.mean(smh1.get('BoverN'))*N
smh2_EB = np.mean(smh2.get('BoverN'))*N

# ESS per second -------------------
tuna_ess = np.mean(tuna_orig.get('ESS') / tuna_orig.get('cpu_time'))
mh_ss_1_ess = np.mean(mh_ss_1.get('ESS') / mh_ss_1.get('cpu_time'))
mh_ss_2_ess = np.mean(mh_ss_2.get('ESS') / mh_ss_2.get('cpu_time'))
smh1_ess = np.mean(smh1.get('ESS') / smh1.get('cpu_time'))
smh2_ess = np.mean(smh2.get('ESS') / smh2.get('cpu_time'))
rwm_ess = np.mean(rwm.get('ESS') / rwm.get('cpu_time'))

# ESS / E(B) --------------------
tuna_ess_EB = np.mean(tuna_orig.get('ESS') / tuna_EB)
mh_ss_1_ess_EB = np.mean(mh_ss_1.get('ESS') / mh_ss_1_EB)
mh_ss_2_ess_EB = np.mean(mh_ss_2.get('ESS') / mh_ss_2_EB)
smh1_ess_EB = np.mean(smh1.get('ESS') / smh1_EB)
smh2_ess_EB = np.mean(smh2.get('ESS') / smh2_EB)
rwm_ess_EB = np.mean(rwm.get('ESS') / N)

# Metrics
tuna_save_results = np.array([model,  'Tuna',    tuna_acc_rate,  tuna_EB,  tuna_ess,  tuna_ess_EB])
mh_ss_1_save_results = np.array([model, 'MH-SS-1', mh_ss_1_acc_rate, mh_ss_1_EB, mh_ss_1_ess, mh_ss_1_ess_EB])
mh_ss_2_save_results = np.array([model, 'MH-SS-2', mh_ss_2_acc_rate, mh_ss_2_EB, mh_ss_2_ess, mh_ss_2_ess_EB])
smh1_save_results = np.array([model,  'SMH-1',   smh1_acc_rate,  smh1_EB,  smh1_ess,  smh1_ess_EB])
smh2_save_results = np.array([model,  'SMH-2',   smh2_acc_rate,  smh2_EB,  smh2_ess,  smh2_ess_EB])
rwm_save_results = np.array([model,   'RWM',     rwm_acc_rate,   N,        rwm_ess,   rwm_ess_EB])

colnames = ['Model', 'Algorithm', 'Acc. rate', 'E(B)', 'ESSs', 'ESS/E(B)']
save_results = np.array([tuna_save_results, mh_ss_1_save_results, mh_ss_2_save_results, smh1_save_results, smh2_save_results, rwm_save_results])
save_results = pd.DataFrame(save_results, columns = colnames)