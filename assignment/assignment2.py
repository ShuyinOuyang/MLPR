#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 06/11/2021 14:12
# @Author  : Shuyin Ouyang
# @File    : assignment2.py

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
import time

data = np.load('ct_data.npz')
X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']
y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']

# 1.a
print('Q1')
def get_standard_error(X):
    return np.std(X)/np.sqrt(np.size(X))

mean_y_train = np.mean(y_train)
mean_y_val = np.mean(y_val)
print('The mean of y_train: %s, is it zero? %s (with numerical rounding errors)' % (mean_y_train, np.isclose(mean_y_train, 0)))
standard_error_y_val = get_standard_error(y_val)
print('The mean of y_val: %s, with "standard error" %s' % (mean_y_val, standard_error_y_val))
mean_y_train_5785 = np.mean(y_train[:5785])
standard_error_y_train_5785 = get_standard_error(y_train[:5785])
print('The mean of first 5,785 entries in the y_train: %s, with "standard error" %s' % (mean_y_train_5785, standard_error_y_train_5785))

# 1.b
constant_column = []
duplicated_column = []
for i in range(X_train.T.shape[0]):
    if np.all(X_train.T[i] == X_train.T[i][0]):
        constant_column.append(i)
        # print('Column: ', i)
# print('--------------')
for i in range(X_train.T.shape[0]):
    temp = X_train.T[i + 1:] == X_train.T[i]
    for j in range(temp.shape[0]):
        if np.all(temp[j]):
            duplicated_column.append(j + i + 1)
            # print('Column: %s %s' % (i, j + i + 1))

delete_column = sorted(list(set(constant_column + duplicated_column)), reverse=True)
# print(delete_column)

for i in delete_column:
    X_train = np.delete(X_train, i, 1)
    X_val = np.delete(X_val, i, 1)
    X_test = np.delete(X_test, i, 1)


# 2
print('Q2')
def fit_linreg(X, yy, alpha=30):
    n_row = X.shape[0]
    X = np.append(X, np.ones([n_row, 1]), axis=1)
    n_col = X.shape[1]
    A = np.identity(n_col) * np.sqrt(alpha)
    A[-1, -1] = 0
    X = np.append(X, A, axis=0)
    res = np.linalg.lstsq(X, np.append(yy, np.zeros(X.shape[1])), rcond=None)[0]
    return res[:-1], res[-1]


def params_unwrap(param_vec, shapes, sizes):
    """Helper routine for minimize_list"""
    args = []
    pos = 0
    for i in range(len(shapes)):
        sz = sizes[i]
        args.append(param_vec[pos:pos + sz].reshape(shapes[i]))
        pos += sz
    return args


def params_wrap(param_list):
    """Helper routine for minimize_list"""
    param_list = [np.array(x) for x in param_list]
    shapes = [x.shape for x in param_list]
    sizes = [x.size for x in param_list]
    param_vec = np.zeros(sum(sizes))
    pos = 0
    for param in param_list:
        sz = param.size
        param_vec[pos:pos + sz] = param.ravel()
        pos += sz
    unwrap = lambda pvec: params_unwrap(pvec, shapes, sizes)
    return param_vec, unwrap


def minimize_list(cost, init_list, args):
    """Optimize a list of arrays (wrapper of scipy.optimize.minimize)

    The input function "cost" should take a list of parameters,
    followed by any extra arguments:
        cost(init_list, *args)
    should return the cost of the initial condition, and a list in the same
    format as init_list giving gradients of the cost wrt the parameters.

    The options to the optimizer have been hard-coded. You may wish
    to change disp to True to get more diagnostics. You may want to
    decrease maxiter while debugging. Although please report all results
    in Q2-5 using maxiter=500.

    The Matlab code comes with a different optimizer, so won't give the same
    results.
    """
    opt = {'maxiter': 500, 'disp': False}
    init, unwrap = params_wrap(init_list)

    def wrap_cost(vec, *args):
        E, params_bar = cost(unwrap(vec), *args)
        vec_bar, _ = params_wrap(params_bar)
        return E, vec_bar

    res = minimize(wrap_cost, init, args, 'L-BFGS-B', jac=True, options=opt)
    return unwrap(res.x)


def linreg_cost(params, X, yy, alpha):
    """Regularized least squares cost function and gradients

    Can be optimized with minimize_list -- see fit_linreg_gradopt for a
    demonstration.

    Inputs:
    params: tuple (ww, bb): weights ww (D,), bias bb scalar
         X: N,D design matrix of input features
        yy: N,  real-valued targets
     alpha: regularization constant

    Outputs: (E, [ww_bar, bb_bar]), cost and gradients
    """
    # Unpack parameters from list
    ww, bb = params

    # forward computation of error
    ff = np.dot(X, ww) + bb
    res = ff - yy
    E = np.dot(res, res) + alpha * np.dot(ww, ww)

    # reverse computation of gradients
    ff_bar = 2 * res
    bb_bar = np.sum(ff_bar)
    ww_bar = np.dot(X.T, ff_bar) + 2 * alpha * ww

    return E, [ww_bar, bb_bar]


def fit_linreg_gradopt(X, yy, alpha):
    """
    fit a regularized linear regression model with gradient opt

         ww, bb = fit_linreg_gradopt(X, yy, alpha)

     Find weights and bias by using a gradient-based optimizer
     (minimize_list) to improve the regularized least squares cost:

       np.sum(((np.dot(X,ww) + bb) - yy)**2) + alpha*np.dot(ww,ww)

     Inputs:
             X N,D design matrix of input features
            yy N,  real-valued targets
         alpha     scalar regularization constant

     Outputs:
            ww D,  fitted weights
            bb     scalar fitted bias
    """
    D = X.shape[1]
    args = (X, yy, alpha)
    init = (np.zeros(D), np.array(0))
    ww, bb = minimize_list(linreg_cost, init, args)
    return ww, bb


def logreg_cost(params, X, yy, alpha):
    """Regularized logistic regression cost function and gradients

    Can be optimized with minimize_list -- see fit_linreg_gradopt for a
    demonstration of fitting a similar function.

    Inputs:
    params: tuple (ww, bb): weights ww (D,), bias bb scalar
         X: N,D design matrix of input features
        yy: N,  real-valued targets
     alpha: regularization constant

    Outputs: (E, [ww_bar, bb_bar]), cost and gradients
    """
    # Unpack parameters from list
    ww, bb = params

    # Force targets to be +/- 1
    yy = 2 * (yy == 1) - 1

    # forward computation of error
    aa = yy * (np.dot(X, ww) + bb)
    sigma = 1 / (1 + np.exp(-aa))
    E = -np.sum(np.log(sigma)) + alpha * np.dot(ww, ww)

    # reverse computation of gradients
    aa_bar = sigma - 1
    bb_bar = np.dot(aa_bar, yy)
    ww_bar = np.dot(X.T, yy * aa_bar) + 2 * alpha * ww

    return E, (ww_bar, bb_bar)


def fit_logreg_gradopt(X, yy, alpha):
    D = X.shape[1]
    args = (X, yy, alpha)
    init = (np.zeros(D), np.array(0))
    ww, bb = minimize_list(logreg_cost, init, args)
    return ww, bb

def rmse_3(params, X, y):
    ww, bb = params
    ff = np.dot(X, ww) + bb
    res = ff - y
    return np.sqrt(np.mean(res ** 2))

# w_1, b_1 = fit_linreg_gradopt(X_train, y_train, 30)
# w_2, b_2 = fit_linreg(X_train, y_train)
# rmse_1_train = rmse_3((w_1, b_1), X_train, y_train)
# rmse_1_val = rmse_3((w_1, b_1), X_val, y_val)
# rmse_2_train = rmse_3((w_2, b_2), X_train, y_train)
# rmse_2_val = rmse_3((w_2, b_2), X_val, y_val)
# print('RMSE on the training set using fit_linreg_gradopt: %s' % (rmse_1_train))
# print('RMSE on the validation set using fit_linreg_gradopt: %s' % (rmse_1_val))
# print('RMSE on the training set using fit_linreg: %s' % (rmse_2_train))
# print('RMSE on the validation set using fit_linreg: %s' % (rmse_2_val))

# 3
print('Q3')
alpha = 30

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def X_transform(X, y, alpha):
    K = 20  # number of thresholded classification problems to fit
    mx = np.max(y)
    mn = np.min(y)
    hh = (mx - mn) / (K + 1)
    thresholds = np.linspace(mn + hh, mx - hh, num=K, endpoint=True)
    # V = np.array([])
    # bk = np.array([])
    res_matrix = np.array([])
    for kk in range(K):
        labels = y > thresholds[kk]
        # ... fit logistic regression to these labels
        ww, bb = fit_logreg_gradopt(X, labels, alpha)
        feature = np.array([sigmoid(X @ ww + bb)])
        if res_matrix.size == 0:
            res_matrix = feature
        else:
            res_matrix = np.append(res_matrix, feature, axis=0)
    return res_matrix.T


# X_train_20 = X_transform(X_train, y_train, alpha)
# train_param = fit_linreg_gradopt(X_train_20, y_train, alpha)
# X_val_20 = X_transform(X_val, y_val, alpha)
# RMSE_train = rmse_3(train_param, X_train_20, y_train)
# RMSE_val = rmse_3(train_param, X_val_20, y_val)
# print('Training root mean square errors: %s' % (RMSE_train))
# print('Validation root mean square errors: %s' %(RMSE_val))


def nn_cost(params, X, yy=None, alpha=None):
    """NN_COST simple neural network cost function and gradients, or predictions

           E, params_bar = nn_cost([ww, bb, V, bk], X, yy, alpha)
                    pred = nn_cost([ww, bb, V, bk], X)

     Cost function E can be minimized with minimize_list

     Inputs:
             params (ww, bb, V, bk), where:
                    --------------------------------
                        ww K,  hidden-output weights
                        bb     scalar output bias
                         V K,D hidden-input weights
                        bk K,  hidden biases
                    --------------------------------
                  X N,D input design matrix
                 yy N,  regression targets
              alpha     scalar regularization for weights

     Outputs:
                     E  sum of squares error
            params_bar  gradients wrt params, same format as params
     OR
               pred N,  predictions if only params and X are given as inputs
    """
    # Unpack parameters from list
    ww, bb, V, bk = params

    # Forwards computation of cost
    A = np.dot(X, V.T) + bk[None, :]  # N,K
    P = 1 / (1 + np.exp(-A))  # N,K
    F = np.dot(P, ww) + bb  # N,
    if yy is None:
        # user wants prediction rather than training signal:
        return F
    res = F - yy  # N,
    E = np.dot(res, res) + alpha * (np.sum(V * V) + np.dot(ww, ww))  # 1x1

    # Reverse computation of gradients
    F_bar = 2 * res  # N,
    ww_bar = np.dot(P.T, F_bar) + 2 * alpha * ww  # K,
    bb_bar = np.sum(F_bar)  # scalar
    P_bar = np.dot(F_bar[:, None], ww[None, :])  # N,
    A_bar = P_bar * P * (1 - P)  # N,
    V_bar = np.dot(A_bar.T, X) + 2 * alpha * V  # K,
    bk_bar = np.sum(A_bar, 0)

    return E, (ww_bar, bb_bar, V_bar, bk_bar)


# 4
print('Q4')
# random initialized parameters
def fit_nn(X, yy, alpha, param=None, K=20):
    D = X.shape[1]
    args = (X, yy, alpha)
    if param:
        init = param
    else:
        # init = (np.random.rand(K), np.random.rand(), np.random.rand(K, D), np.random.rand(K))
        init = (0.1 * np.random.rand(K)/np.sqrt(K), np.random.rand(), 0.1 * np.random.rand(K, D)/np.sqrt(D), np.random.rand(K))
    params = minimize_list(nn_cost, init, args)
    return params

def rmse(prediction, y):
    res = prediction - y
    return np.sqrt(np.mean(res**2))

# Q3 initialized parameters
def hidden_parameter(X, y, alpha):
    K = 20  # number of thresholded classification problems to fit
    mx = np.max(y)
    mn = np.min(y)
    hh = (mx - mn) / (K + 1)
    thresholds = np.linspace(mn + hh, mx - hh, num=K, endpoint=True)
    V = np.array([])
    bk = np.array([])
    for kk in range(K):
        labels = y > thresholds[kk]
        # ... fit logistic regression to these labels
        ww, bb = fit_logreg_gradopt(X, labels, alpha)
        if V.size == 0:
            V = np.array([ww])
        else:
            V = np.append(V, [ww], axis=0)
        if bk.size == 0:
            bk = np.array([bb])
        else:
            bk = np.append(bk, [bb], axis=0)
    return V, bk

# ww K,  hidden-output weights
# bb     scalar output bias
#  V K,D hidden-input weights
# bk K,  hidden biases

# V, bk = hidden_parameter(X_train, y_train, alpha)
# ww, bb = train_param[0], train_param[1]
#
# params_q3 = fit_nn(X_train, y_train, alpha, (ww, bb, V, bk))
# params_rand = fit_nn(X_train, y_train, alpha)
#
# prediction_rand = nn_cost(params_rand, X_val)
# prediction_q3 = nn_cost(params_q3, X_val)
#
# error_rand = rmse(prediction_rand, y_val)
# error_q3 = rmse(prediction_q3, y_val)
#
# print('The root mean square error of random initialization: %s' % (error_rand))
# print('The root mean square error of using Q3 initialization: %s' % (error_q3))


def rbf_fn(X1, X2):
    """Helper routine for gp_post_par"""
    return np.exp((np.dot(X1, (2 * X2.T)) - np.sum(X1 * X1, 1)[:, None]) - np.sum(X2 * X2, 1)[None, :])


def gauss_kernel_fn(X1, X2, ell, sigma_f):
    """Helper routine for gp_post_par"""
    return sigma_f ** 2 * rbf_fn(X1 / (np.sqrt(2) * ell), X2 / (np.sqrt(2) * ell))


def gp_post_par(X_rest, X_obs, yy, sigma_y=0.01, ell=5.0, sigma_f=0.01):
    """GP_POST_PAR means and covariances of a posterior Gaussian process

         rest_cond_mu, rest_cond_cov = gp_post_par(X_rest, X_obs, yy)
         rest_cond_mu, rest_cond_cov = gp_post_par(X_rest, X_obs, yy, sigma_y, ell, sigma_f)

     Calculate the means and covariances at all test locations of the posterior Gaussian
     process conditioned qqqqqqon the observations yy at observed locations X_obs.

     Inputs:
                 X_rest GP test locations
                  X_obs locations of observations
                     yy observed values
                sigma_y observation noise standard deviation
                    ell kernel function length scale
                sigma_f kernel function standard deviation

     Outputs:
           rest_cond_mu mean at each location in X_rest
          rest_cond_cov covariance matrix between function values at all test locations
    """
    X_rest = X_rest[:, None]
    X_obs = X_obs[:, None]
    K_rest = gauss_kernel_fn(X_rest, X_rest, ell, sigma_f)
    K_rest_obs = gauss_kernel_fn(X_rest, X_obs, ell, sigma_f)
    K_obs = gauss_kernel_fn(X_obs, X_obs, ell, sigma_f)
    M = K_obs + sigma_y ** 2 * np.eye(yy.size)
    M_cho, M_low = cho_factor(M)
    rest_cond_mu = np.dot(K_rest_obs, cho_solve((M_cho, M_low), yy))
    rest_cond_cov = K_rest - np.dot(K_rest_obs, cho_solve((M_cho, M_low), K_rest_obs.T))

    return rest_cond_mu, rest_cond_cov


# 5
print('Q5')
def PI(yy, mu, cov):
    return norm.cdf((mu - np.max(yy)) / np.sqrt(cov))


def train_nn_reg(alpha, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val):
    # trains the neural network from Q4 for a given alpha parameters and returns the validation RMSE.
    params = fit_nn(X_train, y_train, alpha)
    prediction = nn_cost(params, X_val)
    rmse_val = np.sqrt(np.mean((prediction - y_val) ** 2))

    return params, rmse_val
#
#
alpha = np.arange(0, 50, 0.02)
index = list(np.random.choice(len(alpha), size=3, replace=False))
X_obs = alpha[index]
X_rest = np.delete(alpha, index)

# yy = []
# params_list = []
# for a in X_obs:
#     params, rmse_a = train_nn_reg(a)
#     y = np.log(error_rand) - np.log(rmse_a)
#     params_list.append(params)
#     yy.append(y)
# yy = np.array(yy)
# for i in range(5):
#     rest_cond_mu, rest_cond_cov = gp_post_par(X_rest, X_obs, yy)
#     diag_cov = np.diag(rest_cond_cov, k=0)
#     f = PI(yy, rest_cond_mu, diag_cov)
#     max_alpha = np.argmax(f)
#     params, rmse_val = train_nn_reg(X_rest[max_alpha])
#     params_list.append(params)
#     yy = np.append(yy, np.log(error_rand) - np.log(rmse_val))
#     X_obs = np.append(X_obs, X_rest[max_alpha])
#     X_rest = np.delete(X_rest, max_alpha)
#
# best_alpha = X_obs[np.argmax(yy)]
# best_params = params_list[np.argmax(yy)]
# prediction_val = nn_cost(best_params, X_val)
# prediction_test = nn_cost(best_params, X_test)
# rmse_val = rmse(prediction_val, y_val)
# rmse_test = rmse(prediction_test, y_test)
# print("The best alpha: %s" % (best_alpha))
# print("Validation RMSE under best alpha: %s" % (rmse_val))
# print("Test RMSE under best alpha: %s" % (rmse_test))

# yy = [0.00982922 0.06427645 0.02195344 0.09355739 0.06321719 0.06634082, 0.04792657 0.0475543 ]
# X_obs  = [25.14  7.26 22.66  8.58 11.06  6.32 10.64  4.96]
# best_alpha = 8.58

# 6
print('Q6')

# PCA
def PCA(X,  K):
    x_bar = np.mean(X, 0)
    [U, vecS, VT] = np.linalg.svd(X - x_bar, 0)  # Apply SVD to centred data
    U = U[:, :K]  # NxK "datapoints" transformed into K-dims
    vecS = vecS[:K]  # The diagonal elements of diagonal matrix S, in a vector
    V = VT[:K, :].T  # DxK "features" transformed into K-dims
    X_kdim = U * vecS  # = np.dot(U, np.diag(vecS))
    X_proj = np.dot(X_kdim, V.T) + x_bar  # SVD approx USV' + mean

    return X_proj

def new_method(X_train, y_train, X_val, y_val, alpha=30):
    K = 20
    X_proj = PCA(X_train, K)
    X_val_proj = PCA(X_val, K)
    params, rmse_a = train_nn_reg(alpha, X_proj, y_train, X_val_proj, y_val)
    return params, rmse_a


params, rmse_new = new_method(X_train, y_train, X_val, y_val, 9.2)
print('RMSE of validation with PCA: %s' % (rmse_new))
prediction = nn_cost(params, X_test, None, 9.2)
rmse_test_new = rmse(prediction, y_test)
print('RMSE of test with PCA: %s' % (rmse_test_new))


params_rand = fit_nn(X_train, y_train, 9.2)
prediction = nn_cost(params_rand, X_val, None, 9.2)
rmse_old = rmse(prediction, y_val)

print('RMSE of validation with Q4: %s' % (rmse_old))
prediction = nn_cost(params_rand, X_test, None, 9.2)
rmse_test_old = rmse(prediction, y_test)
print('RMSE of test with Q4: %s' % (rmse_test_old))

# bagging
prediction_res = []
for _ in range(5):
    X_train_sub = X_train[np.random.choice(X_train.shape[0], size=X_train.shape[0], replace=True)]
    # train_data_list.append(X_train_sub)
    params, rmse_new = new_method(X_train, y_train, X_val, y_val, 9.2)
    prediction = nn_cost(params, X_test, None, 9.2)
    prediction_res.append(prediction)
prediction = np.mean(np.array(prediction_res), axis=0)
rmse_bagging = rmse(prediction, y_test)
print('RMSE of test with bagging: %s' % (rmse_bagging))