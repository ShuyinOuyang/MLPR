#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 09/11/2021 18:18
# @Author  : Shuyin Ouyang
# @File    : wk8.py

import numpy as np

X = np.array([[1,2,3], [3,6,1], [6,4,1], [8,7,0]])
K = 2
# Find top K principal directions:
E, V = np.linalg.eig(np.cov(X.T))
idx = np.argsort(E)[::-1]
V = V[:, idx[:K]]  # (D,K)
x_bar = np.mean(X, 0)

# Transform down to K-dims:
X_kdim = np.dot(X - x_bar, V)  # (N,K)

# Transform back to D-dims:
X_proj1 = np.dot(X_kdim, V.T) + x_bar  # (N,D)

# # PCA via SVD, for NxD matrix X
# x_bar = np.mean(X, 0)
# [U, vecS, VT] = np.linalg.svd(X - x_bar, 0) # Apply SVD to centred data
# U = U[:, :K] # NxK "datapoints" transformed into K-dims
# vecS = vecS[:K] # The diagonal elements of diagonal matrix S, in a vector
# V = VT[:K, :].T # DxK "features" transformed into K-dims
# X_kdim = U * vecS # = np.dot(U, np.diag(vecS))
# X_proj = np.dot(X_kdim, V.T) + x_bar # SVD approx USV' + mean