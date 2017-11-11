#!/usr/bin/env python3
import numpy as np
from utils import lowrank

class Model():
    def generate(n = 6, m = 5, p = 4, r = 3, nn = 1, dense = True):
        X = np.random.randn(n, p)
        W = lowrank(p, m, r, nn, dense)
        Y = np.dot(X, W)
        return X, Y, W

    def loss(X, Y, W = None):
        if W is None:
            W = np.zeros((X.shape[1], Y.shape[1]))
        return np.sum((np.dot(X, W) - Y)**2) / 2

    def gradient(X, Y, W):
        return X.T.dot(X.dot(W)-Y)

    def linesearch(X, Y, W, D, grad = None):
        if grad is None:
            grad = gradient(X, Y, W)

        Z = D - W
        a = np.sum(-grad * Z)
        b = np.sum(X.T.dot(X).dot(Z) * Z)
        return np.array([a,b])

