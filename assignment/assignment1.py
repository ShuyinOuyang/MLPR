import numpy as np
import traceback
from matplotlib import pyplot as plt

# 1. Getting started
# import data
amp_data = np.load('../amp_data.npz')['amp_data']

# # core question a
def plot_histogram():
    plt.figure()
    plt.plot(amp_data)
    plt.show()

    plt.hist(amp_data, bins=10000)
    print(np.std(amp_data))
    plt.show()

new_amp_data = np.reshape(amp_data[:-(len(amp_data) % 21)], (-1, 21))
train_set, val_set, test_set = np.split(new_amp_data, [int(len(new_amp_data) * 0.7), int(len(new_amp_data) * 0.85)])
np.random.seed(42)
train_set = np.random.permutation(train_set)
val_set = np.random.permutation(val_set)
test_set = np.random.permutation(test_set)
X_shuf_train, y_shuf_train = train_set[:, :-1], train_set[:, -1]
X_shuf_val, y_shuf_val = val_set[:, :-1], val_set[:, -1]
X_shuf_test, y_shuf_test = test_set[:, :-1], test_set[:, -1]


# 2. Curve fitting on a snippet of audio
# Plot the points in one row of your X_shuf_train data against the numbers t

# phi function
def phi_linear(x):
    return np.hstack([np.ones((x.shape[0], 1)), x])


def phi_quadratic(x):
    return np.hstack([np.ones((x.shape[0], 1)), x, x ** 2, x ** 3, x ** 4])


row_n = 142
t = np.array([[i / 20] for i in range(21)])
w_fit_linear = np.linalg.lstsq(phi_linear(t[:-1]), X_shuf_train[row_n], rcond=None)[0]
w_fit_quadratic = np.linalg.lstsq(phi_quadratic(t[:-1]), X_shuf_train[row_n], rcond=None)[0]


# plt.subplot(111)
#
# plt.plot([i / 20 for i in range(20)], X_shuf_train[row_n], 'x')
# plt.plot(1, y_shuf_train[row_n], 'x')
# plt.plot([i / 20 for i in range(21)], phi_linear(t) @ w_fit_linear)
# plt.plot([i / 20 for i in range(21)], phi_quadratic(t) @ w_fit_quadratic)
# plt.show()


# 3. Choosing a polynomial predictor based on performance

def phi(C, K):
    t = np.array([[(20 - C + c) / 20] for c in range(C)])
    return np.hstack([t ** k for k in range(K)])


def make_vv(C, K):
    phi_t_1 = np.array([1 for _ in range(K)])
    pp = phi(C, K)
    try:
        vv = phi_t_1.T @ np.linalg.inv(pp.T @ pp) @ pp.T
    except Exception:
        vv = np.array([])
    return vv


# # linear
# C = 20
# K = 2
# vv = make_vv(C, K)
# y_predict = (vv.T @ X_shuf_train[row_n])
# print(y_predict)
# print((phi_linear(t) @ w_fit_linear)[-1])
# print(np.isclose(y_predict, (phi_linear(t) @ w_fit_linear)[-1]))
#
# # quartic
# K = 5
# vv = make_vv(C, K)
# y_predict = (vv.T @ X_shuf_train[row_n])
# print(y_predict)
# print((phi_quadratic(t) @ w_fit_quadratic)[-1])
# print(np.isclose(y_predict, (phi_quadratic(t) @ w_fit_quadratic)[-1]))


# Find the best K & C to min square error
def find_K_C(K = 10, C = 20):
    res = []
    min_error = -1
    for k in range(2, K + 1):
        for c in range(1, C + 1):
            vv = make_vv(c, k)
            # singular matrix
            if not vv.any():
                continue
            y_predict = (X_shuf_train[:, len(X_shuf_train[0])-c:] @ vv)
            error = sum((y_shuf_train - y_predict) ** 2)
            if min_error == -1:
                min_error = error
                res.append(k)
                res.append(c)
            else:
                if error < min_error:
                    min_error = min(error, min_error)
                    res[0], res[1] = k, c
    return res[0], res[1]


def mean_square_error(X, y, k, c):
    vv = make_vv(c, k)
    if not vv.any():
        return -1
    y_predict = (X[:, len(X[0])-c:] @ vv)
    error = sum((y - y_predict) ** 2) / len(X)
    return error


def report_3c():
    best_k, best_c = find_K_C()
    train_error = mean_square_error(X_shuf_train, y_shuf_train, best_k, best_c)
    val_error = mean_square_error(X_shuf_val, y_shuf_val, best_k, best_c)
    test_error = mean_square_error(X_shuf_test, y_shuf_test, best_k, best_c)
    print("Mean square error on the training set: %s" % (train_error))
    print("Mean square error on the validation set: %s" % (val_error))
    print("Mean square error on the test set: %s" % (test_error))


def find_C(X, y):
    min_mse = -1
    best_c = 0
    best_v = np.array([])
    for c in range(1, 21):
        v_fit = np.linalg.lstsq(X[:, 20-c:], y, rcond=None)[0]
        y_predict = (X[:, 20-c:] @ v_fit)
        mse = sum((y - y_predict) ** 2) / len(X)
        if min_mse == -1:
            min_mse = mse
            best_c = c
            best_v = v_fit
        else:
            if mse < min_mse:
                min_mse = mse
                best_c = c
                best_v = v_fit
    return best_c, best_v
# print(find_C(X_shuf_train, y_shuf_train)[0])
# print(find_C(X_shuf_val, y_shuf_val)[0])
def compare_report():
    k1, c1 = find_K_C()
    c_2, v_2 = find_C(X_shuf_val, y_shuf_val)
    vv = make_vv(c1, k1)
    y_predict1 = (X_shuf_test[:, len(X_shuf_test[0])-c1:] @ vv)
    y_predict2 = (X_shuf_test[:, 20-c_2:] @ v_2)
    mse1 = sum((y_shuf_test - y_predict1) ** 2) / len(X_shuf_test)
    mse2 = sum((y_shuf_test - y_predict2) ** 2) / len(X_shuf_test)
    print(mse1, mse2)

def plot_best_model():
    c_2, v_2 = find_C(X_shuf_val, y_shuf_val)
    y_predict2 = (X_shuf_val[:, 20 - c_2:] @ v_2)
    residuals = y_shuf_val - y_predict2
    print(np.std(residuals))
    plt.figure()
    plt.hist(residuals, bins=10000)
    plt.show()