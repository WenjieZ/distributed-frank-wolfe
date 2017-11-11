#!/usr/bin/env python3
import numpy as np
from utils import lowrank

def generate(n = 6, m = 5, p = 4, r = 3, nn = 1):
    X = np.random.randn(n, p)
    W = lowrank(p, m, r, nn)
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

class Model():
    def __init__(self, Z):
        self.X = Z[0]
        self.Y = Z[1]
        self.n = self.Y.shape[0]
        self.m = self.Y.shape[1]
        self.p = self.X.shape[1]

    def loss(self, W = None):
        return loss(self.X, self.Y, W)

    def gradient(self, W):
        return gradient(self.X, self.Y, W)

    def linesearch(self, W, D, grad = None):
        return linesearch(self.X, self.Y, W, D, grad)

class Param():
    def __init__(self, shape):
        self.t = 0
        self.W = np.zeros(shape)
