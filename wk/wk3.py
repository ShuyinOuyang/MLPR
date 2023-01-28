import numpy as np
import matplotlib.pyplot as plt


def demo():
    # Train model on synthetic dataset
    N = 200
    X = np.random.rand(N, 1) * 10
    yy = (X > 1) & (X < 3)

    def phi_fn(X):
        return np.concatenate([np.ones((X.shape[0], 1)), X, X ** 2], axis=1)

    ww = np.linalg.lstsq(phi_fn(X), yy, rcond=None)[0]

    # Predictions
    x_grid = np.arange(0, 10, 0.05)[:, None]
    f_grid = np.dot(phi_fn(x_grid), ww)

    # Predictions with alternative weights:
    w2 = [-1, 2, -0.5]  # Values set by hand
    f2_grid = np.dot(phi_fn(x_grid), w2)

    # Show demo
    plt.clf()
    plt.plot(X[yy == 1], yy[yy == 1], 'r+')
    plt.plot(X[yy == 0], yy[yy == 0], 'bo')
    plt.plot(x_grid, f_grid, 'm-')
    plt.plot(x_grid, f2_grid, 'k-')
    plt.ylim([-0.1, 1.1])
    plt.show()


# finite difference
def finite_diff(a):
    sigma_func = lambda x: 1 / (1 + np.exp(-x))
    varepsilon = 1e-5
    derivative = sigma_func(a) * (1 - sigma_func(a))
    finite_difference = (sigma_func(a + varepsilon / 2) - sigma_func(a - varepsilon / 2)) / varepsilon
    return np.isclose(derivative, finite_difference)

# Estimate the mean and covariance from the samples
# def estimate():
m = 2
sigma = 5
alpha = 2
n = 4
x_1 = m + sigma * np.random.randn(int(1e6))
v = n * np.random.randn(int(1e6))
x_2 = alpha * x_1 + v
X = np.array([x_1, x_2])
X_mean = np.mean(X, axis=1)
X_covariance = np.cov(X)
X_mean_formula = np.array([m, alpha*m])
X_covariance_formula = np.array([[sigma**2, alpha*sigma**2], [alpha*sigma**2, alpha**2*sigma**2+n**2]])
print("Example output: ")
print("Real mean of X:%s, Calculated mean of X:%s" % (X_mean, X_mean_formula))
print("Are they closed?: \n%s \nAre they closed with other tolerance?\n%s" % \
      (np.isclose(X_mean_formula, X_mean), np.isclose(X_mean_formula, X_mean, rtol=0.01)))
print("Real covariance of X:\n%s, \nCalculated covariance of X:\n%s" % (X_covariance, X_covariance_formula))
print("Are they closed?: \n%s \nAre they closed with other tolerance?\n%s" % \
      (np.isclose(X_covariance, X_covariance_formula), np.isclose(X_covariance, X_covariance_formula, rtol=0.01)))