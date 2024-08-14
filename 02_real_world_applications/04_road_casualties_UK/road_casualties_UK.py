from PyMHSS import *
import patsy

dataset = 'collisions'
model = 'poisson'

# --------------------------------------------------------------------------
# The datasets below can be obtained through the R package "stats19". 
# Open an R script and uncomment the commands below.
# --------------------------------------------------------------------------

# install.packages('stats19')
# library(stats19)

# if(curl::has_internet()) {
  
#   dl_stats19(year = 2020, type = "casualty")
#   dl_stats19(year = 2021, type = "casualty")
#   dl_stats19(year = 2021, type = "casualty")
  
#   casualties_2020 = read_casualties(year = 2020)
#   casualties_2021 = read_casualties(year = 2021)
#   casualties_2022 = read_casualties(year = 2022)
# }

col_2020 = pd.read_csv('dft-road-casualty-statistics-collision-2020.csv', low_memory=False)
col_2021 = pd.read_csv('dft-road-casualty-statistics-collision-2021.csv', low_memory=False)
col_2022 = pd.read_csv('dft-road-casualty-statistics-collision-2022.csv', low_memory=False)

collision_data = pd.concat([col_2020, col_2021, col_2022])

collision_data = collision_data.drop(['location_easting_osgr', 'location_northing_osgr', 'day_of_week', 'local_authority_district', 'local_authority_highway', 'first_road_number', 'second_road_number', 'lsoa_of_accident_location'], axis=1)
collision_data.shape
collision_data.dtypes

collision_data['accident_year'] = pd.Categorical(collision_data.accident_year)
collision_data['accident_severity'] = pd.Categorical(collision_data.accident_severity)
collision_data['first_road_class'] = pd.Categorical(collision_data['first_road_class'])
collision_data['carriageway_hazards'] = pd.Categorical(collision_data['carriageway_hazards'])
collision_data['pedestrian_crossing_physical_facilities'] = pd.Categorical(collision_data['pedestrian_crossing_physical_facilities'])
collision_data['urban_or_rural_area'] = pd.Categorical(collision_data['urban_or_rural_area'])

y, x = patsy.dmatrices("number_of_casualties ~ 1 + accident_year + accident_severity + speed_limit + number_of_vehicles + first_road_class + pedestrian_crossing_physical_facilities + carriageway_hazards + urban_or_rural_area", collision_data) # OK

x_mean = np.mean(x, axis=0)
x_mean[0] = 0
x_std = np.sqrt(np.var(x, axis=0))
x_std[0] = 1
x = (x - x_mean)/x_std
x = np.asarray(x)
d = x.shape[1]

y = np.array(y)[:, 0]
N = len(y)

theta_hat, V = get_theta_hat_and_var_cov_matrix(model, x, y)

theta_hat_file_name = dir + str(dataset) + model + 'theta_hat' + '.pickle'
V_file_name = dir + str(dataset) + model + 'V' + '.pickle'

save_file(theta_hat, theta_hat_file_name)
save_file(V, V_file_name)

theta_hat = open_file(theta_hat_file_name)
V = open_file(V_file_name)

nburn = 0
npost_tuna = 10000000
npost = 100000
vector_loop = 'loop'

# ------------------------------------------------------------------------------------
# Tuna
# ------------------------------------------------------------------------------------
tuna = MH_SS(y, x, V, x0 = theta_hat, control_variates=False, bound = 'new', model=model, phi_function = 'original', chi=1e-7, nburn=nburn, npost=npost_tuna, kappa=0.19, nthin=1000, implementation=vector_loop)

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