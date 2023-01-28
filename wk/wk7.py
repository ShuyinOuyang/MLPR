#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 01/11/2021 10:32
# @Author  : Shuyin Ouyang
# @File    : wk7.py

import numpy as np
from matplotlib import pyplot as plt


# def neural_net(X):
#     ff = np.random.randn() * X**2
#     return ff

def sigmoid(a):
    return 1. / (1. + np.exp(-a))


def relu(x):
    return np.maximum(x, 0)


def linear(a):
    return a


a1 = [100, 1]
a2 = [100, 50, 50, 50, 50, 1]
a3 = [100] + [50 for _ in range(100)] + [1]


def neural_net(X, layer_sizes=(100, 50, 1), gg=sigmoid, sigma_w=1):
    for out_size in layer_sizes:
        Wt = sigma_w * np.random.randn(X.shape[1], out_size)
        X = gg(X @ Wt)
        print(X)
    return X


N = 100
X = np.linspace(-2, 2, num=N)[:, None]  # N,1
plt.clf()

def layer_test(X):
    for i in range(12):
        ff_sig = neural_net(X, gg=relu, layer_sizes=a1)
        plt.plot(X, ff_sig)

    plt.show()
    for i in range(12):
        ff_sig = neural_net(X,gg=relu,  layer_sizes=a2)
        plt.plot(X, ff_sig)

    plt.show()
    for i in range(12):
        ff_sig = neural_net(X,gg=relu,  layer_sizes=a3)
        plt.plot(X, ff_sig)

    plt.show()
    for i in range(1):
        ff_sig = neural_net(X,gg=relu, layer_sizes=a3)
        plt.plot(X, ff_sig)

    plt.show()


def numberOfLayers_test(X):
    for i in range(12):
        ff_sig = neural_net(X, layer_sizes=a1)
        plt.plot(X, ff_sig)

    plt.show()
    for i in range(12):
        ff_sig = neural_net(X, layer_sizes=a2)
        plt.plot(X, ff_sig)

    plt.show()
    for i in range(12):
        ff_sig = neural_net(X, layer_sizes=a3)
        plt.plot(X, ff_sig)

    plt.show()



# def neural_net(X, layer_sizes=(100, 50, 1), gg=sigmoid, sigma_w=1):
#     for out_size in layer_sizes:
#         Wt = sigma_w * np.random.rand(X.shape[1], out_size)
#         X = gg(X @ Wt)
#         print(X)
#     return X
#
# history = []
# for i in range(12):
#
#     ff_sig = neural_net(X, gg=sigmoid)
#     history.append(ff_sig)
#     plt.plot(X, ff_sig)
#
# plt.show()