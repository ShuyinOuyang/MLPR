#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 12/10/2021 21:22
# @Author  : Shuyin Ouyang
# @File    : wk4.py

from random import random
from math import floor

# cards = [[1, 1],
#          [0, 0],
#          [1, 0]]
# num_cards = len(cards)
#
# N = 0  # Number of times first side is black
# kk = 0  # Out of those, how many times the other side is white
#
# for _ in range(int(1e6)):
#     card = floor(num_cards * random())
#     side = (random() < 0.5)
#     other_side = int(not side)
#     x1 = cards[card][side]
#     x2 = cards[card][other_side]
#     if x1 == 0:
#         N += 1  # just seen another black face
#         kk += (x2 == 1)  # count if other side was white
#
# approx_probability = float(kk) / N
# print(approx_probability)

# import numpy as np
# from matplotlib import pyplot as plt
#
# x1 = [0.5, 0.1, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.35, 0.25]
# x2 = [0.9, 0.8, 0.75, 1.0]
# p_y1 = len(x1) / (len(x1) + len(x2))
# p_y2 = 1 - p_y1
# x1_mean, x1_std = np.mean(x1), np.std(x1)
# x2_mean, x2_std = np.mean(x2), np.std(x2)
#
#
# def pdf(x, mu, sigma):
#     p = (1 / ((2 * np.pi) ** 0.5 * sigma)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
#     return p
#
#
# plt.figure()
# X1 = np.linspace(x1_mean - x1_std * 5, x1_mean + x1_std * 5, num=1000)
# X2 = np.linspace(x2_mean - x2_std * 5, x2_mean + x2_std * 5, num=1000)
# plt.plot(X1, p_y1 * pdf(X1, x1_mean, x1_std), c='blue')
# plt.plot(X2, p_y2 * pdf(X2, x2_mean, x2_std), c='red')
# plt.legend(['p(x,y1)=p(y1)*p(x|y1)', 'p(x,y2)=p(y2)*p(x|y2)'])
# plt.show()
#
# p = p_y1 * pdf(0.6, x1_mean, x1_std) / (p_y1 * pdf(0.6, x1_mean, x1_std) + p_y2 * pdf(0.6, x2_mean, x2_std))
#
# print('p(y=1|x=0.6) is %s' % (p))


# # pseudo code
# # Step1 : classify the hold data set into different groups based on its label
# # perplexity O(N)
# X0, p(y=0) <- filter the data with label = 0
# X1, p(y=1) <- filter the data with label = 1
# # Step2: get the distribution of two groups
# # perplexity mean:O(N) cov：O(DND)
# mean_0 <- E(X0)
# cov_0 <-  E(X0.T@X0) - E(X0).T@E(X0)
# mean_1 <- E(X1)
# cov_1 <-  E(X1@X1.T) - E(X1).T@E(X1)
# distribution0 <- N(X0; mean_0, cov_0)
# distribution1 <- N(X1; mean_1, cov_1)
# # step3: get probability density function
# # perplexity O(D^3)
# pdf0 <- 1/(det|cov_0|)**(1/2)*(2pi)**(D/2)*exp(-1/2*(x-mean_0).T@(cov_0)**(-1)@(x-mean_0))
# pdf1 <- 1/(det|cov_1|)**(1/2)*(2pi)**(D/2)*exp(-1/2*(x-mean_1).T@(cov_1)**(-1)@(x-mean_1))