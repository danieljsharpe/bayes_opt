#!/usr/bin/python

# A simple program to sample functions from a Gaussian process and plot them

from math import exp
import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(x1, x2, variance = 1):
    return exp(-1 * ((x1-x2) ** 2) / (2*variance))

def gram_matrix(xs):
    return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]

xs = np.arange(-1, 1, 0.01)
mean = [0 for x in xs]
gram = gram_matrix(xs)

plt_vals = []
nsamples = 5
for i in range(nsamples):
    ys = np.random.multivariate_normal(mean, gram)
    plt_vals.extend([xs, ys, "k"])
plt.plot(*plt_vals)
plt.show()
