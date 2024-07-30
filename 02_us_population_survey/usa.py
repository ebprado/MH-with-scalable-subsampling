import pandas as pd
import os as os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import scipy.special as sc
import patsy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
# --------------------------------------------------------------------------
# make sure you have the algorithms.py file in the current directory!
from algorithms import *
# --------------------------------------------------------------------------

dataset = 'usa'
model = 'logistic' 
dir = os.getcwd()

train = pd.read_table(dir + 'usa_00002.csv', sep=',')

n_tot = train.shape[0]

# Drop two columns ------------------------------------------
train = train.drop(['YEAR', 'SAMPLE', 'SERIAL', 'CBSERIAL','CLUSTER','STRATA','GQ','PERNUM', 'BIRTHQTR', 'RACED', 'EDUCD', 'EMPSTATD', 'VETSTATD'], axis=1)
train['SEX'] = train['SEX'] - 1
train['HCOVANY'] = train['HCOVANY'] - 1

# MARST = marital status (1,2,3,4,5,6,9)
# RACE = race (1,2,3,4,5,6,7,8,9)
# EDUC = Education attainment (0,1,2,3,4,5,6,7,8,9,10,11,99)
# EMPSTAT = Employment status (0,1,2,3,9)
# VETSTAT = Veteran status (0,1,2,9)

train['MARST'] = pd.Categorical(train.MARST)
train['RACE'] = pd.Categorical(train.RACE)
train['EDUC'] = pd.Categorical(train.EDUC)
train['EMPSTAT'] = pd.Categorical(train.EMPSTAT)
train['VETSTAT'] = pd.Categorical(train.VETSTAT)

# Take a sample of the data --------------
n_samples = 500000

# Removing people with unknown (9999998) and NA (9999999) incomes
idx1 = np.asarray(train['INCTOT'] < 9999998)
train = train.loc[idx1, :]

# Set a seed for the subsamples
np.random.seed(102)
idx = np.random.choice(train.shape[0], n_samples, replace=False)

# Generate the design matrix 
y, x = patsy.dmatrices("INCTOT ~ 1 + HHWT + PERWT + FAMSIZE + NCHILD + SEX + AGE + MARST + RACE + HCOVANY + EDUC", train)
y = np.asarray(y[idx] > 25000)[:, 0]*1
x = np.asarray(x[idx, :])

N = len(y)

pd.DataFrame(x).corr()

x_mean = np.mean(x, axis=0)
x_mean[0] = 0
x_std = np.sqrt(np.var(x, axis=0))
x_std[0] = 1
x = (x - x_mean)/x_std
x = np.asarray(x)
d = x.shape[1]

mod = LogisticRegression(solver='liblinear', random_state=0)
mod.fit(x, y)
mod.coef_

theta_hat, V = get_theta_hat_and_var_cov_matrix(model, x, y, x0 = mod.coef_[0])
np.sum(logistic_grad_log_target_i(theta_hat, x, y), axis=0)

theta_hat_file_name = dir + str(dataset) + model + 'theta_hat' + '.pickle'
V_file_name = dir + str(dataset) + model + 'V' + '.pickle'

save_file(theta_hat, theta_hat_file_name)
save_file(V, V_file_name)

theta_hat = open_file(theta_hat_file_name)
V = open_file(V_file_name)

p_hat = (1 / (1 + np.exp(-x @ theta_hat)))
sns.kdeplot(p_hat)
plt.show()

nburn = 0
npost = 100000
npost_tuna = 100000000
vector_loop = 'loop'

# ------------------------------------------------------------------------------------
# Tuna
# ------------------------------------------------------------------------------------
tuna = MH_SS(y, x, V, x0 = theta_hat, control_variates=False, bound = 'new', model=model, phi_function = 'original', chi=1e-5, nburn=nburn, npost=npost_tuna, kappa=0.019, nthin=1000, implementation=vector_loop)

# ------------------------------------------------------------------------------------
# MH-SS, SMH and RWM
# ------------------------------------------------------------------------------------
mhss1 = MH_SS(y, x, V, x0 = theta_hat, control_variates=True, taylor_order=1, bound = 'new', model=model, chi=0, nburn=nburn, npost=npost, kappa=1.5)
mhss2 = MH_SS(y, x, V, x0 = theta_hat, control_variates=True, taylor_order=2, bound = 'new', model=model, chi=0, nburn=nburn, npost=npost, kappa=1.5)
smh1 = SMH(y, x, V, x0=theta_hat, kappa=1, bound='orig', taylor_order = 1, model=model, nburn=nburn, npost=npost)
smh2 = SMH(y, x, V, x0=theta_hat, kappa=2, bound='orig', taylor_order = 2, model=model, nburn=nburn, npost=npost)
rwm = RWM(y, x, V, x0=theta_hat, model=model, nburn=nburn, npost=npost, kappa = 2.4)

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