import os
os.getcwd()
os.chdir('/Users/estevaoprado/Documents/GitHub/MH-with-flexible-sub-sampling/03_paper_revision')
from algorithms import *

N = 10000
d = 2
npost=100000
nthin = 1
model = 'logistic'
implementation='vectorised'
    
data = simulate_data(N, d, model)
y = data.get('y')
x = data.get('x')

theta_hat, V = get_theta_hat_and_var_cov_matrix(model, x, y)

mh_ss_cv1 = MH_SS(y, x, V, x0 = theta_hat, model=model, phi_function = 'min', taylor_order = 1, chi=0, nburn=0, npost=npost, implementation=implementation, nthin=nthin)
mh_ss_cv1_rnd_sel = MH_SS_random_selection(y, x, V, x0 = theta_hat, model=model, phi_function = 'min', taylor_order = 1, chi=0, nburn=0, npost=npost, implementation=implementation, nthin=nthin)
# mh_ss_cv2 = MH_SS(y, x, V, x0 = theta_hat, model=model, phi_function = 'min', taylor_order = 2, chi=0, nburn=0, npost=npost, implementation=implementation, nthin=nthin)
# mh_ss_cv2_rnd_sel = MH_SS_random_selection(y, x, V, x0 = theta_hat, model=model, phi_function = 'min', taylor_order = 2, chi=0, nburn=0, npost=npost, implementation=implementation, nthin=nthin)

rwm = RWM(y, x, V, x0 = theta_hat, model=model, nburn=0, npost=npost, implementation=implementation, nthin=nthin)

np.mean(mh_ss_cv1.get('BoverN')*N)
# np.mean(mh_ss_cv2.get('BoverN')*N)

np.mean(mh_ss_cv1_rnd_sel.get('BoverN')*N)
# np.mean(mh_ss_cv2_rnd_sel.get('BoverN')*N)

x1 = mh_ss_cv1.get('parameters')[:, 1]
# x2 = mh_ss_cv2.get('parameters')[:, 1]
y1 = mh_ss_cv1_rnd_sel.get('parameters')[:, 1]
# y2 = mh_ss_cv2_rnd_sel.get('parameters')[:, 1]
z = rwm.get('parameters')[:, 1]

pd_x1 = pd.DataFrame(x1)
pd_x1['algorithm'] = 'MH-SS-1'
pd_x1.columns = ['samples', 'algorithm']

# pd_x2 = pd.DataFrame(x2)
# pd_x2['algorithm'] = 'MH-SS-2'
# pd_x2.columns = ['samples', 'algorithm']

pd_y1 = pd.DataFrame(y1)
pd_y1['algorithm'] = 'MH-SS-1 (random selection)'
pd_y1.columns = ['samples', 'algorithm']

# pd_y2 = pd.DataFrame(y2)
# pd_y2['algorithm'] = 'MH-SS-2 (random selection)'
# pd_y2.columns = ['samples', 'algorithm']

pd_z = pd.DataFrame(z)
pd_z['algorithm'] = 'RWM'
pd_z.columns = ['samples', 'algorithm']

df = pd.concat([pd_x1,
                # pd_x2,
                pd_y1,
                # pd_y2,
                pd_z], axis=0)

height_plot = 8
width_plot = 12

plot = (ggplot(df) + 
aes(x='samples', colour='algorithm', linetype='algorithm') +  
    geom_density(adjust=5, size=1.5) +
labs(
      y = 'Density',
    #   x = r'log$_{10} N$'
      x = 'Posterior samples'
      ) + 
theme_bw(base_size = 25) +
theme(plot_title = element_text(size = 25, hjust = 0.5),
    strip_text_y = element_text(angle = 0),
    legend_position = 'bottom',
    legend_title = element_blank(),
    panel_grid_major = element_blank(),
    panel_grid_minor = element_blank()) + 
    scale_color_manual(values=["#B2DF8A", "#CAB2D6", "#FB9A99"])
    +
geom_vline(xintercept = theta_hat[1], linetype='dashed', alpha=0.7)
)

plot
plot.save('reviewer2_comment1.pdf', height=height_plot, width=width_plot)