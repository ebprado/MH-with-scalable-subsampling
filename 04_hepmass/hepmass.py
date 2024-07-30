import pandas as pd
import os as os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import scipy.special as sc
# --------------------------------------------------------------------------
# make sure you have the algorithms.py file in the current directory!
from algorithms import *
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
npost_tuna = 100000000
vector_loop = 'loop'

# ------------------------------------------------------------------------------------
# Tuna
# ------------------------------------------------------------------------------------
tuna = tunaMH(y, x, V, x0 = theta_hat, control_variates=False, bound = 'new', model=model, phi_function = 'original', chi=1e-5, nburn=nburn, npost=npost, kappa=0.013)

# ------------------------------------------------------------------------------------
# MH-SS, SMH and RWM
# ------------------------------------------------------------------------------------
mhss1 = MH_SS(y, x, V, x0 = theta_hat, control_variates=True, taylor_order=1, bound = 'new', model=model, chi=0, nburn=nburn, npost=npost, kappa=1.5, implementation=vector_loop)
mhss2 = MH_SS(y, x, V, x0 = theta_hat, control_variates=True, taylor_order=2, bound = 'new', model=model, chi=0, nburn=nburn, npost=npost, kappa=1.5, implementation=vector_loop)
smh1 = SMH(y, x, V, x0=theta_hat, kappa=1, bound='orig', taylor_order = 1, model=model, nburn=nburn, npost=npost, implementation=vector_loop)
smh2 = SMH(y, x, V, x0=theta_hat, kappa=2, bound='orig', taylor_order = 2, model=model, nburn=nburn, npost=npost, implementation=vector_loop)
rwm = RWM(y, x, V, x0=theta_hat, model=model, nburn=nburn, npost=npost, kappa = 2.4, implementation=vector_loop)

tuna_file_name = str(dataset) + model + vector_loop + '_Tuna' + '.pickle'
mhss1_file_name = str(dataset) + model + vector_loop + '_mhss1' + '.pickle'
mhss2_file_name = str(dataset) + model + vector_loop + '_mhss2' + '.pickle'
smh1_file_name = str(dataset) + model + vector_loop + '_SMH1' + '.pickle'
smh2_file_name = str(dataset) + model + vector_loop + '_SMH2' + '.pickle'
rwm_file_name = str(dataset) + model + vector_loop + '_RWM' + '.pickle'

save_file(tuna, tuna_file_name)
save_file(mhss1, mhss1_file_name)
save_file(mhss2, mhss2_file_name)
save_file(smh1, smh1_file_name)
save_file(smh2, smh2_file_name)
save_file(rwm, rwm_file_name)

tuna = open_file(tuna_file_name)
mhss1 = open_file(mhss1_file_name)
mhss2 = open_file(mhss2_file_name)
smh1 = open_file(smh1_file_name)
smh2 = open_file(smh2_file_name)
rwm = open_file(rwm_file_name)

# Acceptance rate ------------------
tuna_acc_rate = tuna.get('acc_rate')
mhss1_acc_rate = mhss1.get('acc_rate')
mhss2_acc_rate = mhss2.get('acc_rate')
smh1_acc_rate = smh1.get('acc_rate')
smh2_acc_rate = smh2.get('acc_rate')
rwm_acc_rate = rwm.get('acc_rate')

# E(B)/N --------------------------- 
tuna_EB = np.mean(tuna.get('BoverN'))*N
mhss1_EB = np.mean(mhss1.get('BoverN'))*N
mhss2_EB = np.mean(mhss2.get('BoverN'))*N
smh1_EB = np.mean(smh1.get('BoverN'))*N
smh2_EB = np.mean(smh2.get('BoverN'))*N

# ESS per second -------------------
tuna_ess = np.mean(tuna.get('ESS') / tuna.get('cpu_time'))
mhss1_ess = np.mean(mhss1.get('ESS') / mhss1.get('cpu_time'))
mhss2_ess = np.mean(mhss2.get('ESS') / mhss2.get('cpu_time'))
smh1_ess = np.mean(smh1.get('ESS') / smh1.get('cpu_time'))
smh2_ess = np.mean(smh2.get('ESS') / smh2.get('cpu_time'))
rwm_ess = np.mean(rwm.get('ESS') / rwm.get('cpu_time'))

# ESS / E(B) --------------------
tuna_ess_EB = np.mean(tuna.get('ESS') / tuna_EB)
mhss1_ess_EB = np.mean(mhss1.get('ESS') / mhss1_EB)
mhss2_ess_EB = np.mean(mhss2.get('ESS') / mhss2_EB)
smh1_ess_EB = np.mean(smh1.get('ESS') / smh1_EB)
smh2_ess_EB = np.mean(smh2.get('ESS') / smh2_EB)
rwm_ess_EB = np.mean(rwm.get('ESS') / N)

# Metrics --------------------
tuna_save_results = np.array([model,  'Tuna',    tuna_acc_rate,  tuna_EB,  tuna_ess,  tuna_ess_EB])
mhss1_save_results = np.array([model, 'MH-SS-1', mhss1_acc_rate, mhss1_EB, mhss1_ess, mhss1_ess_EB])
mhss2_save_results = np.array([model, 'MH-SS-2', mhss2_acc_rate, mhss2_EB, mhss2_ess, mhss2_ess_EB])
smh1_save_results = np.array([model,  'SMH-1',   smh1_acc_rate,  smh1_EB,  smh1_ess,  smh1_ess_EB])
smh2_save_results = np.array([model,  'SMH-2',   smh2_acc_rate,  smh2_EB,  smh2_ess,  smh2_ess_EB])
rwm_save_results = np.array([model,   'RWM',     rwm_acc_rate,   N,        rwm_ess,   rwm_ess_EB])

colnames = ['Model', 'Algorithm', 'Acc. rate', 'E(B)', 'ESSs', 'ESS/E(B)']
save_results = np.array([tuna_save_results, mhss1_save_results, mhss2_save_results, smh1_save_results, smh2_save_results, rwm_save_results])
save_results = pd.DataFrame(save_results, columns = colnames)