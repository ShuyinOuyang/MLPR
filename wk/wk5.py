import numpy as np
from numpy.random import randn, uniform
import matplotlib.pyplot as plt

D = 2  # Dimension of the weight space
N_Data_1 = 15  # Number of samples in dataset 1
N_Data_2 = 30  # Number of samples in dataset 2
sigma_w = 2.0
prior_mean = [-5, 0]
prior_precision = np.eye(D) / sigma_w ** 2
# We summarize distributions using their parameters
prior_par = {'mean': prior_mean, 'precision': prior_precision}
# Here we draw the true underlying w. We do this only once
w_tilde = sigma_w * randn(2) + prior_mean
# Draw the inputs for datasets 1 and 2
X_Data_1 = 0.5 * randn(N_Data_1, D)
X_Data_2 = 0.1 * randn(N_Data_2, D) + 0.5
# Draw the outputs for the datasets
sigma_y = 1.0
y_Data_1 = np.dot(X_Data_1, w_tilde) + sigma_y * randn(N_Data_1)
y_Data_2 = np.dot(X_Data_2, w_tilde) + sigma_y * randn(N_Data_2)
# The complete datasets
Data_1 = {'X': X_Data_1,
          'y': y_Data_1}
Data_2 = {'X': X_Data_2,
          'y': y_Data_2}


def posterior_par(prior_par, Data, sigma_y):
    """Calculate posterior parameters.

    Calculate posterior mean and covariance for given prior mean and
    covariance in the par dictionary, given data and given noise
    standard deviation.
    """
    X = Data['X']
    y = Data['y']
    var_y = sigma_y ** 2
    w_0 = prior_par['mean']
    K_0 = prior_par['precision']
    K_N = K_0 + (1 / var_y) * (X.T @ X)
    K_N_invert = np.linalg.solve(K_N, np.eye(len(K_N)))
    w_N = K_N_invert @ K_0 @ w_0 + (1 / var_y) * (K_N_invert @ X.T @ y)
    return {'mean': w_N, 'precision': K_N}

def multi_gaussion(mean, cov):
    x, y = np.random.multivariate_normal(mean, cov, 400).T
    plt.plot(x, y, 'x')
    plt.plot(sum(x)/len(x), sum(y)/len(y), 'o', 'r')
    plt.axis('equal')
    plt.show()