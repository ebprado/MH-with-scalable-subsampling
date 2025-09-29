try:
    import numpy as np
    import pandas as pd
    from plotnine import *
    from tqdm import tqdm
    import statsmodels.api as sm
    import scipy.stats as scs
    import scipy.special as sc
    import tensorflow as tf
except ImportError:
    raise ImportError("Please make sure numpy, pandas, tqdm, statsmodels, scipy and tensorflow are ALL installed!")

import pickle
import time 
import os
from typing import Union
from jax import jit, vmap, lax
import jax.numpy as jnp

def save_file(file, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)

def open_file(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def simulate_data(N, d, model, std_theta = 1):

    """
    Simulate synthetic data either from a logistic, probit or poisson regression model.

    Parameters
    ----------

    N : number of observations
    d : number of parameters in the linear predictor
    model : either 'logistic', 'probit' or 'poisson'
    std_theta : standard deviation of the normal distribution used to simulate the covariates
    """

    if model == 'logistic':
        data = simulate_logistic_regression(N, d, std_theta)    
    elif model == 'probit':
        data = simulate_probit_regression(N, d, std_theta)
    elif model == 'poisson':
        data = simulate_poisson_regression(N, d, std_theta)
    elif model == 'mixture':
        data = simulate_mixture(N, d)
    return data

def multivariate_norm(mean, cholesky_dec, d):

    """ Sample from a multivariate normal using Choleski decomposition """

    z = np.random.normal(0, 1, d)
    return mean + cholesky_dec @ z

def simulate_mixture(N, d):

    """
    Simulate synthetic data from a logistic model.
    
    Parameters
    ----------

    N : number of observations
    d : number of parameters in the linear predictor
    """
    delta = 1000/N
    x = np.random.normal(0, 1, N * d) # simulate covariates  
    x = np.array(x).reshape(N, d)
    x[:, 0] = np.repeat(1, N)
    theta = np.array([N**(-0.2)])  
    lin_pred = np.asarray(x @theta)
    w = np.random.uniform(0, 1, n) <= 0.5 + delta
    y = w * lin_pred + (1 - w)*(-lin_pred) + np.random.normal(0, 1, n)
    return {
        'y': y,
        'x': x,
        'theta': theta,
    }

def simulate_logistic_regression(N, d, std_theta):

    """
    Simulate synthetic data from a logistic model.
    
    Parameters
    ----------

    N : number of observations
    d : number of parameters in the linear predictor
    std_theta : standard deviation of the normal distribution used to simulate the covariates
    """

    x = np.random.normal(0, 1, N * d) # simulate covariates
    x = np.array(x).reshape(N, d)
    x[:, 0] = np.repeat(1, N)
    theta_0 = np.zeros(d)
    Sigma_0 = 100*np.identity(d)
    Sigma_0_inv = np.linalg.inv(Sigma_0)
    theta = np.random.normal(0, std_theta, d)/np.sqrt(d) # simulate parameters
    linear_predictor = np.asarray(x @theta)
    p = (1 / (1 + np.exp(-linear_predictor)))
    y = np.random.binomial(1, p, N) # generate the response variable

    return {
        'y': y,
        'x': x,
        'theta_0': theta_0,
        'Sigma_0_inv': Sigma_0_inv,
        'theta': theta,
        'p': p
    }

def simulate_probit_regression(N, d, std_theta):

    """
    Simulate synthetic data from a probit model.

    Parameters
    ----------

    N : number of observations
    d : number of parameters in the linear predictor
    std_theta : standard deviation of the normal distribution used to simulate the covariates
    """

    x = np.random.normal(0, 1, N * d) # simulate covariates
    x = np.array(x).reshape(N, d)
    x[:, 0] = np.repeat(1, N)
    theta_0 = np.zeros(d)
    Sigma_0 = 100*np.identity(d)
    Sigma_0_inv = np.linalg.inv(Sigma_0)
    theta = np.random.normal(0, std_theta, d)/np.sqrt(d) # simulate parameters
    linear_predictor = np.asarray(x @ theta)
    p = scs.norm.cdf(linear_predictor)
    y = np.random.binomial(1, p, N) # generate the response variable
    
    return {
        'y': y,
        'x': x,
        'theta_0': theta_0,
        'Sigma_0_inv': Sigma_0_inv,
        'theta': theta,
        'p': p
    }

def simulate_poisson_regression(N, d, std_theta):

    """
    Simulate synthetic data from a poisson model.

    Parameters
    ----------

    N : number of observations
    d : number of parameters in the linear predictor
    std_theta : standard deviation of the normal distribution used to simulate the covariates
    """

    x = np.random.normal(0, 1, N * d) # simulate covariates
    x = np.array(x).reshape(N, d)
    x[:, 0] = np.repeat(1, N)
    theta_0 = np.zeros(d)
    Sigma_0 = 100*np.identity(d)
    Sigma_0_inv = np.linalg.inv(Sigma_0)
    theta = np.random.normal(0, std_theta, d)/np.sqrt(d) # simulate parameters
    eta = x @ theta
    poisson_rate = np.asarray(np.log(1 + np.exp(eta)))
    y = np.random.poisson(poisson_rate, N) # generate the response variable

    return {
        'y': y,
        'x': x,
        'theta_0': theta_0,
        'Sigma_0_inv': Sigma_0_inv,
        'theta': theta
    }

def L1_norm(x):
    """ Calculate the L1-norm """
    return np.linalg.norm(x,1)

def L2_norm_vector(x):
    """ Calculate the L1-norm of a vector """
    return np.linalg.norm(x,2)

def L2_norm_matrix(x):
    """ Calculate the L1-norm of a vector """
    return np.linalg.norm(x,2,1)

def mixture_model_log_target_i(beta, y, x, delta=10**(-3)):
    """ Log-likelihood of a mixture model """
    log_lik = np.log((0.5 + delta) * scs.norm.pdf(y + x @ beta) + (0.5 - delta) * scs.norm.pdf(y - x @ beta))
    return log_lik

def mixture_model_grad_log_target_i(beta, x, y, delta=10**(-3)):
    """ Gradient of the log-likelihood of a mixture model """
    a = 0.5 + delta
    b = 0.5 - delta
    t = x @ beta              
    u = y + t
    v = y - t
    phi_u = scs.norm.pdf(u)
    phi_v = scs.norm.pdf(v)
    
    D = a * phi_u + b * phi_v
    num = b * v * phi_v - a * u * phi_u
    weights = num / D         
    
    # grad = x.T @ weights      
    grad = x * weights[:, None]      
    return grad

def mixture_model_hessian_log_target_i(beta, x, y, delta=10**(-3)):
    """
    Hessian of the log-likelihood of a mixture model: 
        log_lik = sum log((0.5+δ) φ(y+xβ) + (0.5-δ) φ(y-xβ))
    """
    nrow = len(y)
    a = 0.5 + delta
    b = 0.5 - delta
    t = x @ beta
    u = y + t
    v = y - t
    
    phi_u = scs.norm.pdf(u)
    phi_v = scs.norm.pdf(v)
    D = a * phi_u + b * phi_v
    
    # numerator for gradient
    num = b * v * phi_v - a * u * phi_u
    
    # derivative terms
    num_prime = b * phi_v * (v**2 - 1) + a * phi_u * (u**2 - 1)
    D_prime = -a * u * phi_u + b * v * phi_v
    
    # scalar coefficients for Hessian
    coeffs = (num_prime * D - num * D_prime) / (D**2)   # shape (n,)
    
    hessian_log_lik = -np.asarray([ coeffs[i] * x[i, None].T @ x[i, None] for i in range(nrow) ])
    return hessian_log_lik, -coeffs

def logistic_log_target_i(beta, y, x):
    """ Log-likelihood of a logistic regression model """
    betaTx = x @ beta
    log_lik = -np.log(1+ np.exp(betaTx)) + betaTx * y
    return log_lik

def logistic_grad_log_target_i(beta, x, y):
    """ Gradient of the log-likelihood of a logistic regression model """
    gradient_log_lik = x * (y - 1 / (1 + np.exp(-x @ beta)))[:, None]
    return gradient_log_lik

def logistic_hessian_log_target_i(beta, x, y):
    """ Hessian of the log-likelihood of a logistic regression model """
    nrow = len(y)
    prob = 1/(1 + np.exp(-x @ beta))
    aux = prob*(1-prob)
    hessian_log_lik = -np.asarray([ aux[i] * x[i, None].T @ x[i, None] for i in range(nrow) ])
    return hessian_log_lik, -aux

def probit_log_target_i(beta, y, x):
    """ Log-likelihood of a probit regression model """
    eta = x @ beta
    PHI = scs.norm.cdf(eta)   
    log_lik = y * np.log(PHI) + (1 - y) * np.log(1 - PHI)
    return log_lik

def probit_grad_log_target_i(beta, x, y):
    """ Gradient of the log-likelihood of a probit regression model """
    eta = x @ beta
    phi = scs.norm.pdf(eta)
    PHI = scs.norm.cdf(eta)
    gradient_log_lik = x * (y * (phi / PHI) - (1 - y) * phi/(1-PHI))[:, None]
    return gradient_log_lik

def probit_hessian_log_target_i(beta, x, y):
    """ Hessian of the log-likelihood of a probit regression model """
    nrow = len(y)
    eta = x @ beta
    phi = scs.norm.pdf(eta)
    PHI = scs.norm.cdf(eta)
    aux = -phi * y * ((eta * PHI + phi) / PHI**2) + phi * (1-y) * ((eta * (1 - PHI) - phi) / (1 - PHI)**2)
    hessian_log_lik = np.asarray([ aux[i] * x[i, None].T @ x[i, None] for i in range(nrow) ])
    return hessian_log_lik, aux

def poisson_log_target_i(beta, y, x):
    """ Log-likelihood of a poisson regression model with expectation log (1 + exp (x * beta))"""
    eta = x @ beta
    lambda_poisson = np.log(1 + np.exp(eta))
    log_lik = y * np.log(lambda_poisson) - lambda_poisson - sc.gammaln(y + 1)
    return log_lik

def poisson_grad_log_target_i(beta, x, y):
    """ Gradient of the log-likelihood of a poisson regression model with expectation log (1 + exp (x * beta))"""
    eta = x @ beta
    lambda_poisson = np.log(1 + np.exp(eta))
    gradient_log_lik = x * ((y / lambda_poisson - 1) * (1 / (1 + np.exp(-eta))))[:, None]

    return gradient_log_lik

def poisson_hessian_log_target_i(beta, x, y):
    """ Hessian of the log-likelihood of a poisson regression model with expectation log (1 + exp (x * beta))"""
    nrow = len(y)
    eta = x @ beta
    exp_eta = np.exp(eta)
    aux = -(exp_eta / (1 + exp_eta)**2) * (y * (exp_eta / (np.log(1 + exp_eta))**2 - 1/np.log(1 + exp_eta)) + 1)
    hessian_log_lik = np.asarray([ aux[i] * x[i, None].T @ x[i, None] for i in range(nrow) ])
    return hessian_log_lik, aux

def sgd(grad_log_target, x, y, k, x0 = None, tol=1e-10):
    """ Our own implementation of the stochastic gradient descent algorithm """

    n = len(y)
    d = x.shape[1]
    h = 1/k
    if x0 == None:
        par_theta = np.zeros(d) # initial values
    else:
        par_theta = x0
    i = 0
    j = 0
    diff = 1

    aux_idx = np.random.choice(range(n), n)
    # while i <= k and diff > tol:
    while i <= 1000000 and diff > tol:
        aux_previous_par = par_theta
        if j == n:
            j = 0
        par_theta = par_theta + h*grad_log_target(par_theta, x[aux_idx[j], :], y[aux_idx[j]])
        # if d == 1: par_theta = par_theta[0, :]
        diff = L1_norm(par_theta - aux_previous_par)
        i = i+1
        j = j+1
        # print(par_theta)
    return par_theta

def check_bound(x, V, kappa, model, nsamples=1000):
    N = x.shape[0]
    d = x.shape[1]
    cov_mat = (kappa / np.sqrt(d))**2 * V
    cholesky_dec = np.linalg.cholesky(cov_mat)
    cholesky_dec_V = np.linalg.cholesky(V)
    theta_prime_minus_theta = np.zeros(nsamples*d).reshape(nsamples,d)
    theta_minus_theta_hat = np.zeros(nsamples*d).reshape(nsamples,d)
    zero = np.zeros(d)
    
    if model == 'logistic':
        norm_x = np.sum(0.25*L2_norm_matrix(x)**2)
    if model == 'probit':
        norm_x = np.sum(L2_norm_matrix(x)**2)

    for i in range(nsamples):
        theta_prime_minus_theta[i, :] = multivariate_norm(zero, cholesky_dec, d)
        theta_minus_theta_hat[i, :] = multivariate_norm(zero, cholesky_dec_V, d)

    norm_theta_prime_minus_theta = np.mean(L2_norm_vector(theta_prime_minus_theta))
    norm_theta_minus_theta_hat = np.mean(L2_norm_vector(theta_minus_theta_hat))
    first_order_bound = 0.5*norm_x * norm_theta_prime_minus_theta * norm_theta_minus_theta_hat

    return {'bound': first_order_bound,
            'C': norm_x,
            'expected_norm_theta_theta_hat': norm_theta_minus_theta_hat,
            'expected_norm_theta_prime_theta': norm_theta_prime_minus_theta}

def sgd_tf(grad_log_target_i, x, y, x0 = None, basic_lr = None, tol=1e-7, iter_max = 100000):

    """ Implementation of the stochastic gradient descent algorithm using TensorFlow """

    d = x.shape[1]
    N = x.shape[0]

    # Set up initial values
    if x0 is None:
        initial_par = np.zeros(d) # initial values
    else:
        initial_par = x0

    parameters = tf.convert_to_tensor(initial_par, dtype = tf.float32)
    parameters = tf.Variable(parameters)
    norm_grad = 0

    # Define loss function (norm of the gradient) and the gradient
    def compute_gradient(par, x, y):
        initial_grad = -grad_log_target_i(par.numpy(),x,y) # evaluate the gradient at theta
        initial_grad = tf.convert_to_tensor(initial_grad, dtype=tf.float32) # convert it into tensor
        grad = [tf.reduce_sum(initial_grad, axis=0)] # sum by row
        return grad

    if basic_lr is None:
        basic_lr = 1/N

    optimizer = tf.keras.optimizers.SGD(learning_rate = basic_lr)
    # optimizer = tf.keras.optimizers.legacy.SGD(learning_rate = basic_lr)
    # optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = basic_lr)
    j = 0

    for k in range(iter_max):
        
        curr_par = 1*parameters # save previous parameters
        curr_norm_grad = 1*norm_grad
        grad = compute_gradient(parameters, x, y)
        optimizer.apply_gradients(zip(grad, [parameters])) # Update parameters
        norm_grad = tf.norm(grad, ord=1) # calculate the norm_1 of the gradient at theta
        diff_norm_grad = np.abs(norm_grad - curr_norm_grad)
        norm_diff_par = tf.norm(curr_par - parameters, ord=1) # calculate the norm_1(theta - theta_previous)
        j = j + 1
        if k%100 == 0:
            print(k)
            print(norm_grad)
            print(norm_diff_par)
        # if diff_norm_grad < tol or norm_diff_par < tol:
        if norm_diff_par < tol:
            break
    
    return parameters

def effective_sample_size(x):

    """ Implementation of the positive and monotone sequence to calculate effective sample size; see Section 3.3 of Geyer (1992) """
    
    nrow = len(x)
    ncol = x.shape[1]
    # Organise the autocovariances in a matrix with 2 columns where the autocovs 2m and 2m + 1 are in the same row.
    sum_adjacent_pairs_cov = [np.sum(sm.tsa.acovf(x[:, col]).reshape(-1,2), axis=1) for col in range(ncol)]
    largest_cov_n = sum_adjacent_pairs_cov[0].shape[0]

    L1 = [] # save largest m for the POSITIVE sequence
    L2 = [] # save largest m for the MONOTONE sequence
    
    for col in range(ncol):
        # Find the largest m such that the autocovs for a given column are all positive
        aux_positive_seq = np.min(np.where(sum_adjacent_pairs_cov[col] < 0), axis=1)
        if np.size(aux_positive_seq) != 0:
            L1.append(aux_positive_seq[0])
        else:
            L1.append(largest_cov_n)
        
        # Conditioned on the autocovs being all positive, find the largest m such that they are monotone
        aux_L1 = L1[col]
        aux_monotone_seq = np.where(np.diff(sum_adjacent_pairs_cov[col][0:aux_L1]) > 0) # find where monotonicity breaks

        if np.size(aux_monotone_seq) != 0:
            L2.append(np.min(aux_monotone_seq))
        else:
            L2.append(aux_L1)
        
    # Combine the POSITIVE and MONOTONE sequences
    L1 = np.asarray(L1)
    L2 = np.asarray(L2)
    L = np.minimum(L1, L2)

    sum_autocorr = np.asarray([np.sum(sm.tsa.acf(x[:, col], nlags=L[col])) for col in range(ncol)])
    ess = nrow/(1 + 2*sum_autocorr)
    
    return ess

def define_target_and_bounds(x, y, theta_hat, model, control_variates, taylor_order):

    """ Define all the terms in | l(theta) - l(theta') - r(theta, theta') <= c_i M(theta, theta')
        based on the corresponding model, except for M(theta, theta').
    """

    if model == 'logistic':
        log_target_i = logistic_log_target_i
        grad_log_target_i = logistic_grad_log_target_i
        hessian_log_target_i = logistic_hessian_log_target_i

    elif model == 'probit':
        log_target_i = probit_log_target_i
        grad_log_target_i = probit_grad_log_target_i
        hessian_log_target_i = probit_hessian_log_target_i

    elif model == 'poisson':
        log_target_i = poisson_log_target_i
        grad_log_target_i = poisson_grad_log_target_i
        hessian_log_target_i = poisson_hessian_log_target_i
    elif model == 'mixture':
        log_target_i = mixture_model_log_target_i
        grad_log_target_i = mixture_model_grad_log_target_i
        hessian_log_target_i = mixture_model_hessian_log_target_i

    if control_variates == True:
        grad_at_theta_hat = grad_log_target_i(theta_hat, x, y)
        sum_grad_at_theta_hat = np.sum(grad_at_theta_hat, axis=0)

        if taylor_order == 1:
            sum_hess_at_theta_hat = 0
            hess_at_theta_hat = 0

            U = lambda theta, idx: -(log_target_i(theta, y[idx, True], x[idx, :]) - grad_at_theta_hat[idx, :] @ theta)
            
            norm_x = L2_norm_matrix(x)**2

            if model == 'logistic':
                Ly = 0.25
                c_i = Ly * norm_x
            if model == 'probit':
                Ly = 1
                c_i = Ly * norm_x
            if model == 'poisson':
                Ly = 0.25 + 0.168*y
                c_i = Ly * norm_x
            if model == 'mixture':
                Ly = np.maximum(1, y**2 - 1)
                c_i = Ly * norm_x

        if taylor_order == 2:
            hess_at_theta_hat, aux_hess_at_theta_hat = hessian_log_target_i(theta_hat, x, y)
            sum_hess_at_theta_hat = np.sum(hess_at_theta_hat, axis=0)

            U = lambda theta, idx: -(log_target_i(theta, y[idx, True], x[idx, :]) - grad_at_theta_hat[idx, :] @ theta - 0.5 * aux_hess_at_theta_hat[idx] * (x[idx, :] @ (theta - theta_hat))**2)

            norm_x = L2_norm_matrix(x)**3

            if model == 'logistic':
                Ly = np.sqrt(3)/18
                c_i = Ly * norm_x
            if model == 'probit':
                Ly = 0.3
                c_i = Ly * norm_x # largest eigen value of the Hessian; see ChrisS notes
            if model == 'poisson':
                Ly = np.sqrt(3)/18 + 0.061*y
                c_i = Ly * norm_x

    else:
        sum_grad_at_theta_hat = 0
        sum_hess_at_theta_hat = 0
        
        U = lambda theta, idx: -log_target_i(theta, y[idx, True], x[idx, :])
        norm_x = L2_norm_matrix(np.abs(x))

        if model == 'logistic': 
            Ly = 1
            c_i = Ly * norm_x
        elif model == 'probit':
            # c_i = something * norm_x
            return print("It is not possible to bound the gradient of the log-target for Tuna without control variates")
        elif model == 'poisson':
            Ly = np.maximum(1, y)
            c_i = Ly * norm_x
    
    return U, c_i, log_target_i, sum_grad_at_theta_hat, sum_hess_at_theta_hat

def M_theta_theta_prime(theta, theta_prime, theta_hat, control_variates, taylor_order):
    
    """ Define M(theta, theta') """

    if control_variates == True:

        p1_omega = L2_norm_vector(theta - theta_hat)
        p2_omega = L2_norm_vector(theta_prime - theta)
        p1_omega_prime = L2_norm_vector(theta_prime - theta_hat)
        
        omega_numerator = (theta - theta_hat) @ (theta_prime - theta)
        omega_denominator = p1_omega * p2_omega
        omega = omega_numerator / omega_denominator

        omega_prime_numerator = (theta_prime - theta_hat) @ (theta_prime - theta)
        omega_prime_denominator = p1_omega_prime * p2_omega
        omega_prime = omega_prime_numerator / omega_prime_denominator

        if taylor_order == 1:

            m1 = D_1(omega)
            m2 = D_1(omega_prime)
            M = p2_omega * np.maximum(p1_omega*m1, p1_omega_prime*m2)

        if taylor_order == 2:

            m1 = D_2(omega)
            m2 = D_2(omega_prime)
            M = 0.5 * p2_omega * (1/6 * p2_omega**2 + p1_omega**2 * m1 + p1_omega_prime**2 * m2)

    else:
        M = L2_norm_vector(theta - theta_prime)

    return M

def D_1(omega):
    """ See Section 'Regression models: Further improvement on bounds' """
    return (1 + np.abs(omega))/2

def D_2(omega, aux1 = 3**1.5):
    """ See Section 'Regression models: Further improvement on bounds' """
    abs_omega = np.abs(omega)
    c_2 = np.sqrt(2 + 0.25*(omega)**2) - 0.5 * abs_omega
    return 1/(c_2 * aux1) * (2 + abs_omega * c_2)**1.5

def get_grad_log_target_i(model):
    if model =='logistic':
        grad_log_target = logistic_grad_log_target_i
    elif model == 'probit':
        grad_log_target = probit_grad_log_target_i
    elif model == 'poisson':
        grad_log_target = poisson_grad_log_target_i
    elif model == 'mixture':
        grad_log_target = mixture_model_grad_log_target_i

    return grad_log_target

def get_theta_hat_and_var_cov_matrix(model, x, y, x0=None, basic_lr=None, iter_max=100000):

    """ Find an estimate (theta hat) of the posterior mode via SGD and evaluate the Hessian at theta hat """

    if model =='logistic':
        grad_log_target = logistic_grad_log_target_i
    elif model == 'probit':
        grad_log_target = probit_grad_log_target_i
    elif model == 'poisson':
        grad_log_target = poisson_grad_log_target_i
    elif model == 'mixture':
        grad_log_target = mixture_model_grad_log_target_i
                
    # theta_hat = gd(grad_log_target, x=x, y=y, d=d, k=N)
    theta_hat = sgd_tf(grad_log_target, x=x, y=y,x0=x0, basic_lr=basic_lr, iter_max=iter_max).numpy()

    if model == 'logistic':    
        p_hat = 1/(1+ np.exp(-x @ theta_hat))
        p1_p = (p_hat*(1-p_hat))[:, None]
        V = np.linalg.inv(x.T @ (x * p1_p))

    elif model == 'probit':
        eta = x @ theta_hat
        phi = scs.norm.pdf(eta)
        PHI = scs.norm.cdf(eta)
        aux = (y * phi * (phi + eta*PHI) / PHI**2 + (1 - y) * phi * (phi - eta * PHI * (1 - PHI))/ (1 - PHI)**2)[:, None]
        V = np.linalg.inv(x.T @ (x * aux))

    elif model == 'poisson':
        eta = x @ theta_hat
        exp_eta = np.exp(eta)
        aux = ((exp_eta / (1 + exp_eta)**2) * (y * (exp_eta / np.log(1 + exp_eta)**2 - 1/np.log(1 + exp_eta)) + 1))[:, None]
        V = np.linalg.inv(x.T @ (x * aux))

    elif model == 'mixture':
        hess_at_theta_hat, aux = mixture_model_hessian_log_target_i(theta_hat, x, y)
        aux = -aux[:, None]
        V = np.linalg.inv(x.T @ (x * aux))

    return theta_hat, V

def RWM(y, x, V, x0, model, nburn, npost, implementation, nthin=1, kappa = 2.4, calculate_ksd=False):

    """ 
    General description: random-walk Metropolis algorithm

    Parameters
    ----------
    y : dependent/response univariate variable
    x : an n x d design matrix 
    V : covariance matrix of the proposal distribution for the random-walk proposal
    x0: initial parameter values
    model : it can be 'logistic', 'probit' and 'poisson'
    nburn : number of MCMC iterations for the burn-in period
    npost : number of MCMC iterations for the post-burn-in period
    implementation: either 'loop' or 'vectorised'
    kappa : scaling parameter of the random-walk proposal distribution
    
    Returns
    -------
    parameters : a matrix with posterior samples
    acc_rate : overall acceptance probability of the algorithm (i.e., alpha1 * alpha2)
    cpu_time : how long it took to run (in seconds)
    meanSJD : mean squared jump distance
    ESS : effective sample size
    N : number of observations
    d : number of parameters
    """

    n = len(y)
    d = len(x0)
    theta = x0

    nmcmc = nburn + npost
    store_size = int(npost/nthin)
    save_parameters = np.zeros((store_size, d))
    acceptance_rate = 0
    sum_SJD = 0

    # Help faster sampling from MVN
    cov_mat = (kappa / np.sqrt(d))**2 * V
    cholesky_dec = np.linalg.cholesky(cov_mat)

    if model =='logistic':
        log_target_i = logistic_log_target_i

    elif model == 'probit':
        log_target_i = probit_log_target_i
    elif model == 'poisson':
        log_target_i = poisson_log_target_i
    elif model == 'mixture':
        log_target_i = mixture_model_log_target_i

    U = lambda theta: -log_target_i(theta, y, x)
    U_j = lambda theta, idx: -log_target_i(theta, y[idx, True], x[idx, :])

    start_time = time.time()

    U_theta = np.sum(U(theta))

    for i in tqdm(range(nmcmc), desc='Running', ncols=75):

        # Propose new candidate values for theta
        theta_prime = multivariate_norm(theta, cholesky_dec, d)

        # Metropolis-Hastings ratio
        if implementation == 'vectorised':
            U_theta_prime = np.sum(U(theta_prime))
            r = U_theta - U_theta_prime
        elif implementation == 'loop':
            U_theta_prime = 0
            for j in range(n):
                U_theta_prime = U_theta_prime + U_j(theta_prime, j)
            r = U_theta - U_theta_prime

        if np.random.exponential() > -r:
            U_theta = U_theta_prime*1
            aux_theta_SJD = theta
            theta = theta_prime
            if i >= nburn:
                acceptance_rate = acceptance_rate + 1
                sum_SJD = sum_SJD + L2_norm_vector(aux_theta_SJD - theta_prime)**2

        if i >= nburn and ((i-nburn)%nthin == 0):
            curr_idx = int((i-nburn)/nthin)
            save_parameters[curr_idx, :] = theta

    cpu_time = time.time() - start_time

    EffectiveSampleSize = effective_sample_size(save_parameters)

    if calculate_ksd:
        grad_log_target_i = get_grad_log_target_i(model)
        ksd = compute_ksd(save_parameters, grad_log_target_i, x, y)
    else:
        ksd = None

    return {'parameters': save_parameters,
            'acc_rate': acceptance_rate/npost,
            'cpu_time': cpu_time,
            'meanSJD': sum_SJD/npost,
            'ESS': EffectiveSampleSize,
            'KSD': ksd,
            'N': n,
            'd': d}

def get_psi(model, x, y, bound, taylor_order):

    """ See Equation 13 in Cornish et al (ICML, 2019) """

    if taylor_order == 1:

        if bound == 'ChrisS':
            norm_x = L2_norm_matrix(x)**2
        else:
            norm_x = np.max(np.abs(x)**2, axis=1)

        if model == 'logistic':
            psi = 1/4 * norm_x
        elif model == 'probit':
            psi = norm_x
        elif model == 'poisson':
            psi = np.maximum(0.168*y, 0.25) * norm_x
    
    elif taylor_order == 2:

        if bound == 'ChrisS':
            norm_x = L2_norm_matrix(x)**3
        else:
            norm_x = np.max(np.abs(x)**3, axis=1)

        if model == 'logistic':
            psi = np.sqrt(3)/18 * norm_x
        elif model == 'probit':
            psi = 0.3 * norm_x
        elif model == 'poisson':
            psi = np.maximum(0.061*y, np.sqrt(3)/18) * norm_x

    return psi

def get_phi_theta_theta_prime(theta, theta_prime, theta_hat, bound, taylor_order=1):
    
    """ See Equation 13 in Cornish et al (ICML, 2019) """

    if bound == 'orig' and taylor_order == 1:
        phi_theta_theta_prime = 1/2 * (L1_norm(theta - theta_hat)**2 + L1_norm(theta_prime - theta_hat)**2)
    elif bound == 'orig' and taylor_order == 2:
        phi_theta_theta_prime = 1/6 * (L1_norm(theta - theta_hat)**3 + L1_norm(theta_prime - theta_hat)**3)
    elif bound == 'ChrisS':
        phi_theta_theta_prime = M_theta_theta_prime(theta, theta_prime, theta_hat, control_variates=True, taylor_order=taylor_order)

    return phi_theta_theta_prime

def SMH(y, x, V, x0, nburn, npost, kappa, model, implementation, bound = 'orig', taylor_order = 1, control_variates = True, nthin = 1, calculate_ksd = False):

    """ 
    General description: Scalable Metropolish-Hastings algorithm (Cornish et al, ICML 2019)

    Parameters
    ----------
    y : dependent/response univariate variable
    x : an n x d design matrix 
    V : covariance matrix of the proposal distribution for the random-walk proposal
    x0: initial parameter values
    nburn : number of MCMC iterations for the burn-in period
    npost : number of MCMC iterations for the post-burn-in period
    kappa : scaling parameter of the random-walk proposal distribution
    model : it can be 'logistic', 'probit' and 'poisson'
    implementation: either 'loop' or 'vectorised'
    bound : if bounds = 'orig', then the original bounds presented in the SMH paper are used. If bound = 'ChrisS', then the resulting algorithm is the SMH-NB.

    taylor_order: order of the control-variates. It's either 1 or 2.
    nthin : Every nthin draw is kept to be returned to the user. 

    Returns
    -------
    parameters : a matrix with posterior samples
    acc_rate : overall acceptance probability of the algorithm (i.e., alpha1 * alpha2)
    acc_rate_ratio1 : Stage 1 acceptance probability  (i.e., alpha1 only)
    BoverN : Average batch size over the total number of observations
    cpu_time : how long it took to run (in seconds)
    meanSJD : mean squared jump distance
    ESS : effective sample size
    N : number of observations
    d : number of parameters
    """

    # Pre-processing (control variates)
    n = len(y)
    d = len(x0)
    
    theta_hat = x0
    aux_range = range(n)

    # U_bar_2_i; see Section G.1. Logistic regression
    psi = get_psi(model, x, y, bound, taylor_order)
    # MCMC details
    nmcmc = nburn + npost
    store_size = int(npost/nthin)
    store_samples = np.zeros((store_size, d))
    save_B = np.zeros((store_size, 1))
    store_samples[:] = np.nan
    save_B[:] = np.nan
    aux_idx = 0
    acceptance_rate = 0
    count_acc_rate_ratio1 = 0
    sum_SJD = 0
    subsample_idx_initial = range(n)

    sum_psi = np.sum(psi)
    weights = psi/sum_psi
    E_M = np.sqrt(np.trace(V)) # rough number based on E(||theta - theta'||_1) ~= 0.8 * trace(V), assuming theta - theta' ~ N(0, FisherInformationMatrix^-1)
    n_samples = np.minimum(10000000, int(sum_psi* E_M**2 *nmcmc*1.5))
    sample_idx = np.random.choice(aux_range, n_samples, p=weights)

    U, nothing, log_target_i, sum_grad_at_theta_hat, sum_hess_at_theta_hat  = define_target_and_bounds(x, y, theta_hat, model, control_variates, taylor_order)
    
    # Help speed up sampling from MVN
    cov_mat = (kappa / np.sqrt(d))**2 * V
    cholesky_dec = np.linalg.cholesky(cov_mat)
    
    theta = multivariate_norm(theta_hat, cholesky_dec, d)

    start_time = time.time()
    
    for i in tqdm(range(nmcmc), ncols=75, desc='Running: '):

        # Propose new candidate values for theta
        theta_prime = multivariate_norm(theta, cholesky_dec, d)
        
        # phi(theta, theta'); see the left-hand side of equation 
        phi_theta_theta_prime = get_phi_theta_theta_prime(theta, theta_prime, theta_hat, bound, taylor_order=taylor_order)
        poisson_rate = phi_theta_theta_prime * sum_psi
        
        # Sample B ~ Poisson(phi(theta, theta') * Psi)
        B = np.random.poisson(poisson_rate)

        if B == 0:
            r = sum_grad_at_theta_hat @ (theta_prime - theta)
            if taylor_order == 2:
                r = r + 0.5 * (theta_prime - theta_hat) @ sum_hess_at_theta_hat @ (theta_prime - theta_hat) - 0.5 * (theta - theta_hat) @ sum_hess_at_theta_hat @ (theta - theta_hat)
            
            if np.random.exponential() > -r:
                aux_theta_SJD = theta
                theta = theta_prime
                if i>= nburn:
                    acceptance_rate = acceptance_rate + 1
                    sum_SJD = sum_SJD + L2_norm_vector(aux_theta_SJD - theta_prime)**2

        # If phi_theta_theta_prime * sum_psi >= n, then perform a Metropolis-Hastings step
        elif poisson_rate >= n:
            B = n
            subsample_idx = subsample_idx_initial
            U_theta = -log_target_i(theta, y, x)
            U_theta_prime = -log_target_i(theta_prime, y, x)
            mh_ratio = np.sum(U_theta - U_theta_prime)

            if np.random.exponential() > -mh_ratio:
                aux_theta_SJD = theta
                theta = theta_prime
                if i>= nburn:
                    acceptance_rate = acceptance_rate + 1
                    sum_SJD = sum_SJD + L2_norm_vector(aux_theta_SJD - theta_prime)**2
        
        # If B < n, then perform a Factorised Metropolis-Hastings step
        else:
            # Acceptance probability (PART 1): left-hand side of equation 10 (i.e., 1 ^ pi^hat_k(theta')/pi^hat_k(theta))
            log_ratio1 = sum_grad_at_theta_hat @ (theta_prime - theta)
            if taylor_order == 2:
                log_ratio1 = log_ratio1 + 0.5 * (theta_prime - theta_hat) @ sum_hess_at_theta_hat @ (theta_prime - theta_hat) - 0.5 * (theta - theta_hat) @ sum_hess_at_theta_hat @ (theta - theta_hat)
            
            # Auxiliary variable to check whether the proposed theta should be rejected
            reject_theta_prime = False
            
            # If the acceptance probability PART 1 is all right, then work away
            if np.random.exponential() > -log_ratio1:
                if i >= nburn:
                    count_acc_rate_ratio1 = count_acc_rate_ratio1 + 1

                # Reinitialise the vector of indices if we got to the end of it
                if aux_idx + B > n_samples:
                    aux_idx = 0

                # Loop through the vector of indices
                subsample_idx = sample_idx[aux_idx:aux_idx+B]
                aux_idx = aux_idx + B + 1
                # subsample_idx = np.random.choice(aux_range, size=B, p=psi/sum_psi, replace=False) # This is what they call "Alias table"

                # Acceptance probability (PART 2): see right-hand side of equation 10 (i.e., prod_i (1 ^ tilde(pi)_i(theta') * hat(pi)_k,i(theta)...))
                lambda_i = U(theta_prime, subsample_idx) - U(theta, subsample_idx)

                # See Algorithm 1 (Bj ~ Bernoulli(lambda/lambda_bar))
                lambda_bar_i = phi_theta_theta_prime * psi[subsample_idx]
                
                if implementation == 'vectorised':
                    if any(np.random.uniform(0, 1, B) < lambda_i/lambda_bar_i):
                        reject_theta_prime = True
                elif implementation == 'loop':
                    for k in subsample_idx:
                        # Acceptance probability (PART 2): see right-hand side of equation 10 (i.e., prod_i (1 ^ tilde(pi)_i(theta') * hat(pi)_k,i(theta)...))
                        lambda_i = U(theta_prime, k) - U(theta, k)
                        # See Algorithm 1 (Bj ~ Bernoulli(lambda/lambda_bar))
                        lambda_bar_i = phi_theta_theta_prime * psi[k]
                            
                        if np.random.uniform(0, 1, 1) < lambda_i/lambda_bar_i:
                            reject_theta_prime = True
                            break

                if reject_theta_prime == False:
                    aux_theta_SJD = theta
                    theta = theta_prime
                    if i>= nburn:
                        acceptance_rate = acceptance_rate + 1
                        sum_SJD = sum_SJD + L2_norm_vector(aux_theta_SJD - theta_prime)**2

        if i >= nburn and ((i-nburn)%nthin == 0):
            curr_idx = int((i-nburn)/nthin)
            store_samples[curr_idx, :] = theta
            save_B[curr_idx, :] = B

    cpu_time = time.time() - start_time

    EffectiveSampleSize = effective_sample_size(store_samples)
    
    if calculate_ksd:
        grad_log_target_i = get_grad_log_target_i(model)
        ksd = compute_ksd(store_samples, grad_log_target_i, x, y)
    else:
        ksd = None

    return {'parameters': store_samples,
            'acc_rate': acceptance_rate/npost,
            'acc_rate_ratio1': count_acc_rate_ratio1/npost,
            'BoverN': save_B/n,
            'cpu_time': cpu_time,
            'meanSJD': sum_SJD/npost,
            'ESS': EffectiveSampleSize,
            'KSD': ksd,
            'N': n,
            'd': d}

def MH_SS(y, x, V, x0, nburn, npost, model, implementation, control_variates = True, chi = 0, taylor_order=1, phi_function = 'min', kappa = 1.5, nthin = 1, calculate_ksd=False):

    """ 
    General description: Metropolis-Hastings with Scalable Subsampling algorithm. This implementation can also
    be used to run the Tuna algorithm (Zhang et al, NeurIPS 2020) if control_variates = False and chi > 0.

    Parameters
    ----------
    y : dependent/response univariate variable
    x : an n x d design matrix 
    V : covariance matrix of the proposal distribution for the random-walk proposal
    x0: initial parameter values
    nburn : number of MCMC iterations for the burn-in period
    npost : number of MCMC iterations for the post-burn-in period
    model : it can be 'logistic', 'probit' and 'poisson'
    implementation: either 'loop' or 'vectorised'

    control_variates: control_variates == False results in the Tuna algorithm. If control_variates == True, then the MH_SS algorithm is run
    chi : Tuna additional hyperparameter. In MH-SS, chi = 0. In the Tuna algorithm, chi > 0
    taylor_order: order of the control-variates. It's either 1 or 2 for MH-SS, zero otherwise (i.e., Tuna algorithm)
    phi_function: If phi_function == 'min', then gamma = 0 and the expectation of the Poisson auxiliary variable is optimally designed. On the other hand, phi_function == 'max' denotes gamma = 1
    kappa : scaling parameter of the random-walk proposal distribution
    nthin : Every nthin draw is kept to be returned to the user
    
    Returns
    -------
    parameters : a matrix with posterior samples
    acc_rate : overall acceptance probability of the algorithm (i.e., alpha1 * alpha2)
    acc_rate_ratio1 : Stage 1 acceptance probability  (i.e., alpha1 only)
    BoverN : Average batch size over the total number of observations
    cpu_time : how long it took to run (in seconds)
    meanSJD : mean squared jump distance
    ESS : effective sample size
    chi : Tuna additional hyperparameter
    N : number of observations
    d : number of parameters
    lambda : chi should be set so that lambda < 1 following the Tuna paper.
    """

    n = len(y)
    d = len(x0)
    nmcmc = nburn + npost
    store_size = int(npost/nthin)
    save_parameters = np.zeros((store_size, d))
    save_B = np.zeros((store_size, 1))
    save_lambda = 0

    acceptance_rate = 0
    count_acc_rate_ratio1 = 0
    sum_SJD = 0
    subsample_idx_initial = range(n)
    aux_idx = 0
    
    theta_hat = x0

    control_variates = control_variates
    taylor_order = taylor_order
    model = model

    U, c_i, log_target_i, sum_grad_at_theta_hat, sum_hess_at_theta_hat = define_target_and_bounds(x, y, theta_hat, model, control_variates, taylor_order)
    
    # Help faster sampling from MVN
    cov_mat = (kappa / np.sqrt(d))**2 * V
    cholesky_dec = np.linalg.cholesky(cov_mat)

    C = np.sum(c_i)
    weights = c_i/C
    E_M = (kappa / np.sqrt(d)) * np.sqrt(np.trace(V)) # rough number based on E(||theta - theta'||_2) from TunaMH without CV.
    n_samples = np.minimum(10000000, int(C* E_M**2 *nmcmc*1.5))
    sample_idx = np.random.choice(range(n), n_samples, p=weights)

    theta = multivariate_norm(theta_hat, cholesky_dec, d)

    start_time = time.time()

    for i in tqdm(range(nmcmc), desc='Running', ncols=75):

        # Propose new candidate values for theta
        theta_prime = multivariate_norm(theta, cholesky_dec, d)

        # Calculate the bound, which is a function M and C
        M = M_theta_theta_prime(theta, theta_prime, theta_hat, control_variates, taylor_order)

        _lambda = chi * (C**2) * (M**2) # see the bottom of page 20
        poisson_rate = _lambda + C*M
        B = np.random.poisson(poisson_rate)

        if B == 0:
            if control_variates == True:
                r = sum_grad_at_theta_hat @ (theta_prime - theta)
                if taylor_order == 2:
                    r = r + 0.5 * (theta_prime - theta_hat) @ sum_hess_at_theta_hat @ (theta_prime - theta_hat) - 0.5 * (theta - theta_hat) @ sum_hess_at_theta_hat @ (theta - theta_hat)
            else:
                r = 0 # accept theta_prime

        # Perform a RWM step
        elif poisson_rate >= n:
            B = n
            subsample_idx = subsample_idx_initial
            U_theta = -log_target_i(theta, y, x)
            U_theta_prime = -log_target_i(theta_prime, y, x)
            r = np.sum(U_theta - U_theta_prime)
        
        else:
            if control_variates == True:
                log_ratio1 = sum_grad_at_theta_hat @ (theta_prime - theta)
                if taylor_order == 2:
                    log_ratio1 = log_ratio1 + 0.5 * (theta_prime - theta_hat) @ sum_hess_at_theta_hat @ (theta_prime - theta_hat) - 0.5 * (theta - theta_hat) @ sum_hess_at_theta_hat @ (theta - theta_hat)
            else:
                log_ratio1 = 0 # move on
            # If the acceptance probability PART 1 is all right, then work away
            if np.random.exponential() > -log_ratio1:
                if i >= nburn:
                    count_acc_rate_ratio1 = count_acc_rate_ratio1 + 1

                # Reinitialise the vector of indices if we've got to the end of it
                if aux_idx + B > n_samples:
                    aux_idx = 0

                # Loop through the vector of indices
                subsample_idx = sample_idx[aux_idx:aux_idx+B]
                c_i_subsample = c_i[subsample_idx]                
                aux_idx = aux_idx + B + 1

                U_theta = U(theta, subsample_idx)
                U_theta_prime = U(theta_prime, subsample_idx)

                # Calculate phi and phi_prime; see Equation 6 (page 20)
                if phi_function == 'min':
                    diff_U = U_theta_prime - U_theta
                    phi = np.minimum(0, diff_U) + c_i_subsample * M
                    phi_prime = phi - diff_U
                    # phi_prime = np.minimum(0, -diff_U) + c_i_subsample * M

                elif phi_function == 'max':
                    diff_U = U_theta_prime - U_theta
                    phi = np.maximum(0, diff_U)
                    # phi_prime = phi - diff_U
                    phi_prime = np.maximum(0, -diff_U)

                else:
                    phi = 0.5 * (U_theta + U_theta_prime) - U_theta + 0.5 * c_i_subsample * M
                    phi_prime = 0.5 * (U_theta + U_theta_prime) - U_theta_prime + 0.5 * c_i_subsample * M

                # Form minibatch; see the bottom of page 20
                prob_add_to_I = (_lambda * c_i_subsample + C * phi) / (_lambda * c_i_subsample + C * c_i_subsample * M)
                n_obs_bundled = len(prob_add_to_I)

                I = np.where(np.random.uniform(0, 1, n_obs_bundled) < prob_add_to_I)[0]

                # Metropolis-Hastings ratio; see Algorithm 4 on page 21
                if implementation == 'vectorised':
                    r = np.sum(np.log((_lambda*c_i_subsample[I] + C * phi_prime[I])/(_lambda*c_i_subsample[I] + C * phi[I])))
                elif implementation == 'loop':
                    r = 0
                    for j in I:
                        r = r + np.sum(np.log((_lambda*c_i_subsample[j] + C * phi_prime[j])/(_lambda*c_i_subsample[j] + C * phi[j])))

            else:
                r = -np.inf # i.e., reject theta_prime

        if np.random.exponential() > -r:
            aux_theta_SJD = theta
            theta = theta_prime
            if i>= nburn:
                acceptance_rate = acceptance_rate + 1
                sum_SJD = sum_SJD + L2_norm_vector(aux_theta_SJD - theta_prime)**2
                
        if i >= nburn and ((i-nburn)%nthin == 0):
            curr_idx = int((i-nburn)/nthin)
            save_parameters[curr_idx, :] = theta
            save_B[curr_idx, :] = B
            save_lambda = save_lambda + _lambda
    
    cpu_time = time.time() - start_time
        
    EffectiveSampleSize = effective_sample_size(save_parameters)
    
    if calculate_ksd:
        grad_log_target_i = get_grad_log_target_i(model)
        ksd = compute_ksd(save_parameters, grad_log_target_i, x, y)
    else:
        ksd = None

    return {'parameters': save_parameters,
            'acc_rate': acceptance_rate/npost,
            'acc_rate_ratio1': count_acc_rate_ratio1/npost,
            'BoverN': save_B/n,
            'cpu_time': cpu_time,
            'meanSJD': sum_SJD/npost,
            'ESS': EffectiveSampleSize,
            'KSD': ksd,
            'chi': chi,
            'N': n,
            'd': d,
            'lambda': save_lambda/npost}

def MH_SS_random_selection(y, x, V, x0, nburn, npost, model, implementation, control_variates = True, chi = 0, taylor_order=1, phi_function = 'min', kappa = 1.5, nthin = 1, calculate_ksd=False):

    """ 
    General description: Metropolis-Hastings with Scalable Subsampling algorithm. This implementation can also
    be used to run the Tuna algorithm (Zhang et al, NeurIPS 2020) if control_variates = False and chi > 0.

    Parameters
    ----------
    y : dependent/response univariate variable
    x : an n x d design matrix 
    V : covariance matrix of the proposal distribution for the random-walk proposal
    x0: initial parameter values
    nburn : number of MCMC iterations for the burn-in period
    npost : number of MCMC iterations for the post-burn-in period
    model : it can be 'logistic', 'probit' and 'poisson'
    implementation: either 'loop' or 'vectorised'

    control_variates: control_variates == False results in the Tuna algorithm. If control_variates == True, then the MH_SS algorithm is run
    chi : Tuna additional hyperparameter. In MH-SS, chi = 0. In the Tuna algorithm, chi > 0
    taylor_order: order of the control-variates. It's either 1 or 2 for MH-SS, zero otherwise (i.e., Tuna algorithm)
    phi_function: If phi_function == 'min', then gamma = 0 and the expectation of the Poisson auxiliary variable is optimally designed. On the other hand, phi_function == 'max' denotes gamma = 1
    kappa : scaling parameter of the random-walk proposal distribution
    nthin : Every nthin draw is kept to be returned to the user
    
    Returns
    -------
    parameters : a matrix with posterior samples
    acc_rate : overall acceptance probability of the algorithm (i.e., alpha1 * alpha2)
    acc_rate_ratio1 : Stage 1 acceptance probability  (i.e., alpha1 only)
    BoverN : Average batch size over the total number of observations
    cpu_time : how long it took to run (in seconds)
    meanSJD : mean squared jump distance
    ESS : effective sample size
    chi : Tuna additional hyperparameter
    N : number of observations
    d : number of parameters
    lambda : chi should be set so that lambda < 1 following the Tuna paper.
    """

    n = len(y)
    d = len(x0)
    nmcmc = nburn + npost
    store_size = int(npost/nthin)
    save_parameters = np.zeros((store_size, d))
    save_B = np.zeros((store_size, 1))
    save_lambda = 0

    acceptance_rate = 0
    count_acc_rate_ratio1 = 0
    sum_SJD = 0
    subsample_idx_initial = range(n)
    aux_idx = 0
    
    theta_hat = x0

    control_variates = control_variates
    taylor_order = taylor_order
    model = model

    U, c_i, log_target_i, sum_grad_at_theta_hat, sum_hess_at_theta_hat = define_target_and_bounds(x, y, theta_hat, model, control_variates, taylor_order)
    
    # Help faster sampling from MVN
    cov_mat = (kappa / np.sqrt(d))**2 * V
    cholesky_dec = np.linalg.cholesky(cov_mat)

    C = np.sum(c_i)
    weights = c_i/C
    E_M = (kappa / np.sqrt(d)) * np.sqrt(np.trace(V)) # rough number based on E(||theta - theta'||_2) from TunaMH without CV.
    n_samples = np.minimum(10000000, int(C* E_M**2 *nmcmc*1.5))
    # sample_idx = np.random.choice(range(n), n_samples, p=weights)
    sample_idx = np.random.choice(range(n), n_samples)

    theta = multivariate_norm(theta_hat, cholesky_dec, d)

    start_time = time.time()

    for i in tqdm(range(nmcmc), desc='Running', ncols=75):

        # Propose new candidate values for theta
        theta_prime = multivariate_norm(theta, cholesky_dec, d)

        # Calculate the bound, which is a function M and C
        M = M_theta_theta_prime(theta, theta_prime, theta_hat, control_variates, taylor_order)

        _lambda = chi * (C**2) * (M**2) # see the bottom of page 20
        poisson_rate = _lambda + C*M
        B = np.random.poisson(poisson_rate)

        if B == 0:
            if control_variates == True:
                r = sum_grad_at_theta_hat @ (theta_prime - theta)
                if taylor_order == 2:
                    r = r + 0.5 * (theta_prime - theta_hat) @ sum_hess_at_theta_hat @ (theta_prime - theta_hat) - 0.5 * (theta - theta_hat) @ sum_hess_at_theta_hat @ (theta - theta_hat)
            else:
                r = 0 # accept theta_prime

        # Perform a RWM step
        elif poisson_rate >= n:
            B = n
            subsample_idx = subsample_idx_initial
            U_theta = -log_target_i(theta, y, x)
            U_theta_prime = -log_target_i(theta_prime, y, x)
            r = np.sum(U_theta - U_theta_prime)
        
        else:
            if control_variates == True:
                log_ratio1 = sum_grad_at_theta_hat @ (theta_prime - theta)
                if taylor_order == 2:
                    log_ratio1 = log_ratio1 + 0.5 * (theta_prime - theta_hat) @ sum_hess_at_theta_hat @ (theta_prime - theta_hat) - 0.5 * (theta - theta_hat) @ sum_hess_at_theta_hat @ (theta - theta_hat)
            else:
                log_ratio1 = 0 # move on
            # If the acceptance probability PART 1 is all right, then work away
            if np.random.exponential() > -log_ratio1:
                if i >= nburn:
                    count_acc_rate_ratio1 = count_acc_rate_ratio1 + 1

                # Reinitialise the vector of indices if we've got to the end of it
                if aux_idx + B > n_samples:
                    aux_idx = 0

                # Loop through the vector of indices
                subsample_idx = sample_idx[aux_idx:aux_idx+B]
                c_i_subsample = c_i[subsample_idx]                
                aux_idx = aux_idx + B + 1

                U_theta = U(theta, subsample_idx)
                U_theta_prime = U(theta_prime, subsample_idx)

                # Calculate phi and phi_prime; see Equation 6 (page 20)
                if phi_function == 'min':
                    diff_U = U_theta_prime - U_theta
                    phi = np.minimum(0, diff_U) + c_i_subsample * M
                    phi_prime = phi - diff_U
                    # phi_prime = np.minimum(0, -diff_U) + c_i_subsample * M

                elif phi_function == 'max':
                    diff_U = U_theta_prime - U_theta
                    phi = np.maximum(0, diff_U)
                    # phi_prime = phi - diff_U
                    phi_prime = np.maximum(0, -diff_U)

                else:
                    phi = 0.5 * (U_theta + U_theta_prime) - U_theta + 0.5 * c_i_subsample * M
                    phi_prime = 0.5 * (U_theta + U_theta_prime) - U_theta_prime + 0.5 * c_i_subsample * M

                # Form minibatch; see the bottom of page 20
                prob_add_to_I = (_lambda * c_i_subsample + C * phi) / (_lambda * c_i_subsample + C * c_i_subsample * M)
                n_obs_bundled = len(prob_add_to_I)

                I = np.where(np.random.uniform(0, 1, n_obs_bundled) < prob_add_to_I)[0]

                # Metropolis-Hastings ratio; see Algorithm 4 on page 21
                if implementation == 'vectorised':
                    r = np.sum(np.log((_lambda*c_i_subsample[I] + C * phi_prime[I])/(_lambda*c_i_subsample[I] + C * phi[I])))
                elif implementation == 'loop':
                    r = 0
                    for j in I:
                        r = r + np.sum(np.log((_lambda*c_i_subsample[j] + C * phi_prime[j])/(_lambda*c_i_subsample[j] + C * phi[j])))

            else:
                r = -np.inf # i.e., reject theta_prime

        if np.random.exponential() > -r:
            aux_theta_SJD = theta
            theta = theta_prime
            if i>= nburn:
                acceptance_rate = acceptance_rate + 1
                sum_SJD = sum_SJD + L2_norm_vector(aux_theta_SJD - theta_prime)**2
                
        if i >= nburn and ((i-nburn)%nthin == 0):
            curr_idx = int((i-nburn)/nthin)
            save_parameters[curr_idx, :] = theta
            save_B[curr_idx, :] = B
            save_lambda = save_lambda + _lambda
    
    cpu_time = time.time() - start_time
        
    EffectiveSampleSize = effective_sample_size(save_parameters)

    if calculate_ksd:
        grad_log_target_i = get_grad_log_target_i(model)
        ksd = compute_ksd(save_parameters, grad_log_target_i, x, y)
    else:
        ksd = None

    return {'parameters': save_parameters,
            'acc_rate': acceptance_rate/npost,
            'acc_rate_ratio1': count_acc_rate_ratio1/npost,
            'BoverN': save_B/n,
            'cpu_time': cpu_time,
            'meanSJD': sum_SJD/npost,
            'ESS': EffectiveSampleSize,
            'KSD': ksd,
            'chi': chi,
            'N': n,
            'd': d,
            'lambda': save_lambda/npost}

"""
The k_0_fun and imq_KSD functions are from the SGMCMCJax package (see https://joss.theoj.org/papers/10.21105/joss.04113)
"""
Array = Union[np.ndarray, jnp.ndarray]

@jit
def k_0_fun(
    parm1: Array,
    parm2: Array,
    gradlogp1: Array,
    gradlogp2: Array,
    c: float = 1.0,
    beta: float = -0.5,
) -> float:
    """KSD kernel with the IMQ kernel and the 2 norm: http://proceedings.mlr.press/v70/gorham17a/gorham17a.pdf

    Args:
        parm1 (Array): sampled parameter 1
        parm2 (Array): sampled parameter 2
        gradlogp1 (Array): gradient of sampled parameter 1
        gradlogp2 (Array): gradient of sampled parameter 2
        c (float, optional): intercept parameter in the IMQ kernel. Defaults to 1.
        beta (float, optional): exponent parameter in the IMQ kernel. Defaults to -0.5.

    Returns:
        float: value of kernel for the pair of samples
    """
    diff = parm1 - parm2
    dim = parm1.shape[0]
    base = c**2 + jnp.dot(diff, diff)
    term1 = jnp.dot(gradlogp1, gradlogp2) * base**beta
    term2 = -2 * beta * jnp.dot(gradlogp1, diff) * base ** (beta - 1)
    term3 = 2 * beta * jnp.dot(gradlogp2, diff) * base ** (beta - 1)
    term4 = -2 * dim * beta * (base ** (beta - 1))
    term5 = -4 * beta * (beta - 1) * base ** (beta - 2) * jnp.sum(jnp.square(diff))
    return term1 + term2 + term3 + term4 + term5


_batch_k_0_fun_rows = jit(vmap(k_0_fun, in_axes=(None, 0, None, 0, None, None)))

@jit
def imq_KSD(samples: Array, grads: Array) -> Array:
    """Kernel Stein Discrepancy with IMQ kernel

    Args:
        samples (Array): MCMC samples
        grads (Array): gradients of the MCMC samples

    Returns:
        float: estimate of the KSD
    """
    c, beta = 1.0, -0.5
    N = samples.shape[0]

    # we use lax.scan rather than a nested vmap as the latter becomes very slow for high dimensional problems with lots of samples.
    def body_ksd(le_sum, x):
        my_sample, my_grad = x
        le_sum += jnp.sum(
            _batch_k_0_fun_rows(my_sample, samples, my_grad, grads, c, beta)
        )
        return le_sum, None

    le_sum, _ = lax.scan(body_ksd, 0.0, (samples, grads))
    return jnp.sqrt(le_sum) / N

def compute_ksd(post_samples, grad_log_target_i, x_train, y_train):
    n_post_samples = post_samples.shape[0]
    grad_mcmc_samples = np.array([np.sum(grad_log_target_i(post_samples[j, :], x_train, y_train), axis=0) for j in range(n_post_samples)])
    return float(imq_KSD(post_samples, grad_mcmc_samples))

def MH_SS_without_DA(y, x, V, x0, nburn, npost, model, implementation, control_variates = True, chi = 0, taylor_order=1, phi_function = 'min', kappa = 1.5, nthin = 1, calculate_ksd=False):

    """ 
    General description: Metropolis-Hastings with Scalable Subsampling algorithm. This implementation can also
    be used to run the Tuna algorithm (Zhang et al, NeurIPS 2020) if control_variates = False and chi > 0.

    Parameters
    ----------
    y : dependent/response univariate variable
    x : an n x d design matrix 
    V : covariance matrix of the proposal distribution for the random-walk proposal
    x0: initial parameter values
    nburn : number of MCMC iterations for the burn-in period
    npost : number of MCMC iterations for the post-burn-in period
    model : it can be 'logistic', 'probit' and 'poisson'
    implementation: either 'loop' or 'vectorised'

    control_variates: control_variates == False results in the Tuna algorithm. If control_variates == True, then the MH_SS algorithm is run
    chi : Tuna additional hyperparameter. In MH-SS, chi = 0. In the Tuna algorithm, chi > 0
    taylor_order: order of the control-variates. It's either 1 or 2 for MH-SS, zero otherwise (i.e., Tuna algorithm)
    phi_function: If phi_function == 'min', then gamma = 0 and the expectation of the Poisson auxiliary variable is optimally designed. On the other hand, phi_function == 'max' denotes gamma = 1
    kappa : scaling parameter of the random-walk proposal distribution
    nthin : Every nthin draw is kept to be returned to the user
    
    Returns
    -------
    parameters : a matrix with posterior samples
    acc_rate : overall acceptance probability of the algorithm (i.e., alpha1 * alpha2)
    acc_rate_ratio1 : Stage 1 acceptance probability  (i.e., alpha1 only)
    BoverN : Average batch size over the total number of observations
    cpu_time : how long it took to run (in seconds)
    meanSJD : mean squared jump distance
    ESS : effective sample size
    chi : Tuna additional hyperparameter
    N : number of observations
    d : number of parameters
    lambda : chi should be set so that lambda < 1 following the Tuna paper.
    """

    n = len(y)
    d = len(x0)
    nmcmc = nburn + npost
    store_size = int(npost/nthin)
    save_parameters = np.zeros((store_size, d))
    save_B = np.zeros((store_size, 1))
    save_lambda = 0

    acceptance_rate = 0
    count_acc_rate_ratio1 = 0
    sum_SJD = 0
    subsample_idx_initial = range(n)
    aux_idx = 0
    
    theta_hat = x0

    control_variates = control_variates
    taylor_order = taylor_order
    model = model

    U, c_i, log_target_i, sum_grad_at_theta_hat, sum_hess_at_theta_hat = define_target_and_bounds(x, y, theta_hat, model, control_variates, taylor_order)
    
    # Help faster sampling from MVN
    cov_mat = (kappa / np.sqrt(d))**2 * V
    cholesky_dec = np.linalg.cholesky(cov_mat)

    C = np.sum(c_i)
    weights = c_i/C
    E_M = (kappa / np.sqrt(d)) * np.sqrt(np.trace(V)) # rough number based on E(||theta - theta'||_2) from TunaMH without CV.
    n_samples = np.minimum(10000000, int(C* E_M**2 *nmcmc*1.5))
    sample_idx = np.random.choice(range(n), n_samples, p=weights)

    theta = multivariate_norm(theta_hat, cholesky_dec, d)

    start_time = time.time()

    for i in tqdm(range(nmcmc), desc='Running', ncols=75):

        # Propose new candidate values for theta
        theta_prime = multivariate_norm(theta, cholesky_dec, d)

        # Calculate the bound, which is a function M and C
        M = M_theta_theta_prime(theta, theta_prime, theta_hat, control_variates, taylor_order)

        _lambda = chi * (C**2) * (M**2) # see the bottom of page 20
        poisson_rate = _lambda + C*M
        B = np.random.poisson(poisson_rate)

        if B == 0:
            if control_variates == True:
                r = sum_grad_at_theta_hat @ (theta_prime - theta)
                if taylor_order == 2:
                    r = r + 0.5 * (theta_prime - theta_hat) @ sum_hess_at_theta_hat @ (theta_prime - theta_hat) - 0.5 * (theta - theta_hat) @ sum_hess_at_theta_hat @ (theta - theta_hat)
            else:
                r = 0 # accept theta_prime
                log_ratio1 = 0

        # Perform a RWM step
        elif poisson_rate >= n:
            B = n
            subsample_idx = subsample_idx_initial
            U_theta = -log_target_i(theta, y, x)
            U_theta_prime = -log_target_i(theta_prime, y, x)
            r = np.sum(U_theta - U_theta_prime)
            log_ratio1 = 0
        
        else:
            if control_variates == True:
                log_ratio1 = sum_grad_at_theta_hat @ (theta_prime - theta)
                if taylor_order == 2:
                    log_ratio1 = log_ratio1 + 0.5 * (theta_prime - theta_hat) @ sum_hess_at_theta_hat @ (theta_prime - theta_hat) - 0.5 * (theta - theta_hat) @ sum_hess_at_theta_hat @ (theta - theta_hat)
            else:
                log_ratio1 = 0 # move on
            # If the acceptance probability PART 1 is all right, then work away
            if log_ratio1 is not None:
            # if np.random.exponential() > -log_ratio1:
                if i >= nburn:
                    count_acc_rate_ratio1 = count_acc_rate_ratio1 + 1

                # Reinitialise the vector of indices if we've got to the end of it
                if aux_idx + B > n_samples:
                    aux_idx = 0

                # Loop through the vector of indices
                subsample_idx = sample_idx[aux_idx:aux_idx+B]
                c_i_subsample = c_i[subsample_idx]                
                aux_idx = aux_idx + B + 1

                U_theta = U(theta, subsample_idx)
                U_theta_prime = U(theta_prime, subsample_idx)

                # Calculate phi and phi_prime; see Equation 6 (page 20)
                if phi_function == 'min':
                    diff_U = U_theta_prime - U_theta
                    phi = np.minimum(0, diff_U) + c_i_subsample * M
                    phi_prime = phi - diff_U
                    # phi_prime = np.minimum(0, -diff_U) + c_i_subsample * M

                elif phi_function == 'max':
                    diff_U = U_theta_prime - U_theta
                    phi = np.maximum(0, diff_U)
                    # phi_prime = phi - diff_U
                    phi_prime = np.maximum(0, -diff_U)

                else:
                    phi = 0.5 * (U_theta + U_theta_prime) - U_theta + 0.5 * c_i_subsample * M
                    phi_prime = 0.5 * (U_theta + U_theta_prime) - U_theta_prime + 0.5 * c_i_subsample * M

                # Form minibatch; see the bottom of page 20
                prob_add_to_I = (_lambda * c_i_subsample + C * phi) / (_lambda * c_i_subsample + C * c_i_subsample * M)
                n_obs_bundled = len(prob_add_to_I)

                I = np.where(np.random.uniform(0, 1, n_obs_bundled) < prob_add_to_I)[0]

                # Metropolis-Hastings ratio; see Algorithm 4 on page 21
                if implementation == 'vectorised':
                    r = np.sum(np.log((_lambda*c_i_subsample[I] + C * phi_prime[I])/(_lambda*c_i_subsample[I] + C * phi[I])))
                elif implementation == 'loop':
                    r = 0
                    for j in I:
                        r = r + np.sum(np.log((_lambda*c_i_subsample[j] + C * phi_prime[j])/(_lambda*c_i_subsample[j] + C * phi[j])))

            else:
                r = -np.inf # i.e., reject theta_prime

        if np.random.exponential() > -(r + log_ratio1):
            aux_theta_SJD = theta
            theta = theta_prime
            if i>= nburn:
                acceptance_rate = acceptance_rate + 1
                sum_SJD = sum_SJD + L2_norm_vector(aux_theta_SJD - theta_prime)**2
                
        if i >= nburn and ((i-nburn)%nthin == 0):
            curr_idx = int((i-nburn)/nthin)
            save_parameters[curr_idx, :] = theta
            save_B[curr_idx, :] = B
            save_lambda = save_lambda + _lambda
    
    cpu_time = time.time() - start_time
        
    EffectiveSampleSize = effective_sample_size(save_parameters)
    
    if calculate_ksd:
        grad_log_target_i = get_grad_log_target_i(model)
        ksd = compute_ksd(save_parameters, grad_log_target_i, x, y)
    else:
        ksd = None

    return {'parameters': save_parameters,
            'acc_rate': acceptance_rate/npost,
            'acc_rate_ratio1': count_acc_rate_ratio1/npost,
            'BoverN': save_B/n,
            'cpu_time': cpu_time,
            'meanSJD': sum_SJD/npost,
            'ESS': EffectiveSampleSize,
            'KSD': ksd,
            'chi': chi,
            'N': n,
            'd': d,
            'lambda': save_lambda/npost}

# def MH_SS_multimodal(y, x, V, x0, nburn, npost, model, implementation, control_variates = True, chi = 0, taylor_order=1, phi_function = 'min', kappa = 1.5, nthin = 1, calculate_ksd=False):

#     """ 
#     General description: Metropolis-Hastings with Scalable Subsampling algorithm. This implementation can also
#     be used to run the Tuna algorithm (Zhang et al, NeurIPS 2020) if control_variates = False and chi > 0.

#     Parameters
#     ----------
#     y : dependent/response univariate variable
#     x : an n x d design matrix 
#     V : covariance matrix of the proposal distribution for the random-walk proposal
#     x0: initial parameter values
#     nburn : number of MCMC iterations for the burn-in period
#     npost : number of MCMC iterations for the post-burn-in period
#     model : it can be 'logistic', 'probit' and 'poisson'
#     implementation: either 'loop' or 'vectorised'

#     control_variates: control_variates == False results in the Tuna algorithm. If control_variates == True, then the MH_SS algorithm is run
#     chi : Tuna additional hyperparameter. In MH-SS, chi = 0. In the Tuna algorithm, chi > 0
#     taylor_order: order of the control-variates. It's either 1 or 2 for MH-SS, zero otherwise (i.e., Tuna algorithm)
#     phi_function: If phi_function == 'min', then gamma = 0 and the expectation of the Poisson auxiliary variable is optimally designed. On the other hand, phi_function == 'max' denotes gamma = 1
#     kappa : scaling parameter of the random-walk proposal distribution
#     nthin : Every nthin draw is kept to be returned to the user
    
#     Returns
#     -------
#     parameters : a matrix with posterior samples
#     acc_rate : overall acceptance probability of the algorithm (i.e., alpha1 * alpha2)
#     acc_rate_ratio1 : Stage 1 acceptance probability  (i.e., alpha1 only)
#     BoverN : Average batch size over the total number of observations
#     cpu_time : how long it took to run (in seconds)
#     meanSJD : mean squared jump distance
#     ESS : effective sample size
#     chi : Tuna additional hyperparameter
#     N : number of observations
#     d : number of parameters
#     lambda : chi should be set so that lambda < 1 following the Tuna paper.
#     """
#     aux_theta_hat = list(x0.values())
#     n = len(y)
#     d = len(aux_theta_hat[0])
#     nmcmc = nburn + npost
#     store_size = int(npost/nthin)
#     save_parameters = np.zeros((store_size, d))
#     save_B = np.zeros((store_size, 1))
#     save_lambda = 0

#     acceptance_rate = 0
#     count_acc_rate_ratio1 = 0
#     sum_SJD = 0
#     subsample_idx_initial = range(n)
#     aux_idx = 0
    
#     theta_hat = aux_theta_hat

#     control_variates = control_variates
#     taylor_order = taylor_order
#     model = model

#     # U, c_i, log_target_i, sum_grad_at_theta_hat, sum_hess_at_theta_hat = define_target_and_bounds(x, y, theta_hat, model, control_variates, taylor_order)
    
#     # Call define_target_and_bounds for each element of theta_hat (list of numpy arrays)
#     results = [define_target_and_bounds(x, y, th, model, control_variates, taylor_order) for th in theta_hat]

#     # Unpack results for each mode (assuming two modes for multimodal case)
#     U_list, c_i_list, log_target_i_list, sum_grad_at_theta_hat_list, sum_hess_at_theta_hat_list = zip(*results)

#     # Use the first mode's outputs for the first iteration
#     U, c_i, log_target_i, sum_grad_at_theta_hat, sum_hess_at_theta_hat = U_list[0], c_i_list[0], log_target_i_list[0], sum_grad_at_theta_hat_list[0], sum_hess_at_theta_hat_list[0]

#     # Help faster sampling from MVN
#     cov_mat = (kappa / np.sqrt(d))**2 * V
#     cholesky_dec = np.linalg.cholesky(cov_mat)

#     C = np.sum(c_i)
#     weights = c_i/C
#     E_M = (kappa / np.sqrt(d)) * np.sqrt(np.trace(V)) # rough number based on E(||theta - theta'||_2) from TunaMH without CV.
#     n_samples = np.minimum(10000000, int(C* E_M**2 *nmcmc*1.5))
#     sample_idx = np.random.choice(range(n), n_samples, p=weights)

#     theta = multivariate_norm(aux_theta_hat[0], cholesky_dec, d)

#     start_time = time.time()

#     for i in tqdm(range(nmcmc), desc='Running', ncols=75):

#         # Propose new candidate values for theta
#         theta_prime = multivariate_norm(theta, cholesky_dec, d)

#         # Choose the closest theta_hat to (theta + theta')/2
#         aux_k_star = [L2_norm_vector((theta + theta_prime)/2 - th) for th in aux_theta_hat]
#         k_star = np.argmin(aux_k_star)
#         theta_hat = aux_theta_hat[k_star]

#         # Use the corresponding precomputed values based on theta_hat_k_star
#         c_i = c_i_list[k_star]
#         sum_grad_at_theta_hat = sum_grad_at_theta_hat_list[k_star]
#         sum_hess_at_theta_hat = sum_hess_at_theta_hat_list[k_star]
#         U = U_list[k_star]
#         log_target_i = log_target_i_list[k_star]

#         # Calculate the bound, which is a function M and C
#         M = M_theta_theta_prime(theta, theta_prime, theta_hat, control_variates, taylor_order)

#         _lambda = chi * (C**2) * (M**2) # see the bottom of page 20
#         poisson_rate = _lambda + C*M
#         B = np.random.poisson(poisson_rate)

#         if B == 0:
#             if control_variates == True:
#                 r = sum_grad_at_theta_hat @ (theta_prime - theta)
#                 if taylor_order == 2:
#                     r = r + 0.5 * (theta_prime - theta_hat) @ sum_hess_at_theta_hat @ (theta_prime - theta_hat) - 0.5 * (theta - theta_hat) @ sum_hess_at_theta_hat @ (theta - theta_hat)
#             else:
#                 r = 0 # accept theta_prime

#         # Perform a RWM step
#         elif poisson_rate >= n:
#             B = n
#             subsample_idx = subsample_idx_initial
#             U_theta = -log_target_i(theta, y, x)
#             U_theta_prime = -log_target_i(theta_prime, y, x)
#             r = np.sum(U_theta - U_theta_prime)
        
#         else:
#             if control_variates == True:
#                 log_ratio1 = sum_grad_at_theta_hat @ (theta_prime - theta)
#                 if taylor_order == 2:
#                     log_ratio1 = log_ratio1 + 0.5 * (theta_prime - theta_hat) @ sum_hess_at_theta_hat @ (theta_prime - theta_hat) - 0.5 * (theta - theta_hat) @ sum_hess_at_theta_hat @ (theta - theta_hat)
#             else:
#                 log_ratio1 = 0 # move on
#             # If the acceptance probability PART 1 is all right, then work away
#             if np.random.exponential() > -log_ratio1:
#                 if i >= nburn:
#                     count_acc_rate_ratio1 = count_acc_rate_ratio1 + 1

#                 # Reinitialise the vector of indices if we've got to the end of it
#                 if aux_idx + B > n_samples:
#                     aux_idx = 0

#                 # Loop through the vector of indices
#                 subsample_idx = sample_idx[aux_idx:aux_idx+B]
#                 c_i_subsample = c_i[subsample_idx]                
#                 aux_idx = aux_idx + B + 1

#                 U_theta = U(theta, subsample_idx)
#                 U_theta_prime = U(theta_prime, subsample_idx)

#                 # Calculate phi and phi_prime; see Equation 6 (page 20)
#                 if phi_function == 'min':
#                     diff_U = U_theta_prime - U_theta
#                     phi = np.minimum(0, diff_U) + c_i_subsample * M
#                     phi_prime = phi - diff_U
#                     # phi_prime = np.minimum(0, -diff_U) + c_i_subsample * M

#                 elif phi_function == 'max':
#                     diff_U = U_theta_prime - U_theta
#                     phi = np.maximum(0, diff_U)
#                     # phi_prime = phi - diff_U
#                     phi_prime = np.maximum(0, -diff_U)

#                 else:
#                     phi = 0.5 * (U_theta + U_theta_prime) - U_theta + 0.5 * c_i_subsample * M
#                     phi_prime = 0.5 * (U_theta + U_theta_prime) - U_theta_prime + 0.5 * c_i_subsample * M

#                 # Form minibatch; see the bottom of page 20
#                 prob_add_to_I = (_lambda * c_i_subsample + C * phi) / (_lambda * c_i_subsample + C * c_i_subsample * M)
#                 n_obs_bundled = len(prob_add_to_I)

#                 I = np.where(np.random.uniform(0, 1, n_obs_bundled) < prob_add_to_I)[0]

#                 # Metropolis-Hastings ratio; see Algorithm 4 on page 21
#                 if implementation == 'vectorised':
#                     r = np.sum(np.log((_lambda*c_i_subsample[I] + C * phi_prime[I])/(_lambda*c_i_subsample[I] + C * phi[I])))
#                 elif implementation == 'loop':
#                     r = 0
#                     for j in I:
#                         r = r + np.sum(np.log((_lambda*c_i_subsample[j] + C * phi_prime[j])/(_lambda*c_i_subsample[j] + C * phi[j])))

#             else:
#                 r = -np.inf # i.e., reject theta_prime

#         if np.random.exponential() > -r:
#             aux_theta_SJD = theta
#             theta = theta_prime
#             if i>= nburn:
#                 acceptance_rate = acceptance_rate + 1
#                 sum_SJD = sum_SJD + L2_norm_vector(aux_theta_SJD - theta_prime)**2
                
#         if i >= nburn and ((i-nburn)%nthin == 0):
#             curr_idx = int((i-nburn)/nthin)
#             save_parameters[curr_idx, :] = theta
#             save_B[curr_idx, :] = B
#             save_lambda = save_lambda + _lambda
    
#     cpu_time = time.time() - start_time
        
#     EffectiveSampleSize = effective_sample_size(save_parameters)
    
#     if calculate_ksd:
#         grad_log_target_i = get_grad_log_target_i(model)
#         ksd = compute_ksd(save_parameters, grad_log_target_i, x, y)
#     else:
#         ksd = None

#     return {'parameters': save_parameters,
#             'acc_rate': acceptance_rate/npost,
#             'acc_rate_ratio1': count_acc_rate_ratio1/npost,
#             'BoverN': save_B/n,
#             'cpu_time': cpu_time,
#             'meanSJD': sum_SJD/npost,
#             'ESS': EffectiveSampleSize,
#             'KSD': ksd,
#             'chi': chi,
#             'N': n,
#             'd': d,
#             'lambda': save_lambda/npost}

# # beta = np.array([0.063])
# n = 10**6
# delta=1000/n
# data = simulate_mixture(n, d=1)
# y = data['y']
# x = data['x']
# beta = data['theta']
# d = len(beta)

# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters

# # Simulate ys (assuming ys is y from above)
# ys = y
# # Generate beta1s and compute log-likelihoods
# beta1s = np.linspace(-1.5 * beta, 1.5 * beta, 200)
# lls = np.zeros_like(beta1s)

# # Vectorized computation for speed
# for i in range(len(beta1s)):
#     beta1 = beta1s[i]
#     lls[i] = np.sum(np.log((0.5 + delta) * scs.norm.pdf(ys + beta1, scale=1) +
#                            (0.5 - delta) * scs.norm.pdf(ys - beta1, scale=1)))

# # Plot
# plt.plot(beta1s, lls, label="log pi(beta1,0,...,0)")
# plt.xlabel("beta1")
# plt.ylabel("log pi(beta1,0,...,0)")
# plt.axvline(x=beta, color="red", label="mu")
# plt.legend()
# plt.show()

# theta_hat = {'mode1': beta, 'mode2': -beta}

# hess_at_theta_hat, aux = mixture_model_hessian_log_target_i(beta, x, y)

# V = np.linalg.inv(x.T @ (x * aux[:, None]))

# teste = MH_SS_multimodal(y, x, V=V, x0=theta_hat, kappa=1.5, nburn=0, nthin=1, npost=100000, model='mixture', implementation='vectorised', control_variates = True, chi = 0, taylor_order=1)
# import matplotlib.pyplot as plt    

# plt.plot(teste['parameters'])
# teste['acc_rate']
# teste['BoverN'].mean()
# teste_rwm = RWM(y, x, V=V, x0=theta_hat['mode1'], nburn=0, npost=1000, model='mixture', implementation='vectorised')
# plt.plot(teste_rwm['parameters'])