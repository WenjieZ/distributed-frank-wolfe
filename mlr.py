#!/usr/bin/env python3
import numpy as np
from utils import lowrank

class Model():
    def class2score(y, k):
        Y = np.zeros((len(y), k), dtype = np.int)
        for i, s in enumerate(y):
            Y[i, s] = 1
        return Y
	
    def score2class(Y):
        y = Y.argmax(axis=1)
        return y
	
    def predict(X, W):
        return Model.score2class(np.dot(X, W))

    def generate(n = 8, m = 5, p = 4, r = 3, nn = 1, dense = True):
        X = np.random.randn(n, p)
        W = lowrank(p, m, r, nn, dense)
        y = Model.predict(X, W)
        Y = Model.class2score(y, m)
        return X, Y, W
    
    def score(X, W):
        return np.exp(np.dot(X,W))
	
    def proba(X, W):
        Z = Model.score(X, W)
        return Z / np.reshape(np.sum(Z, axis=1), (-1, 1))

    def loss(X, y, W = None):
        if W is None:
            W = np.zeros((X.shape[1], y.shape[1]))
        if y.ndim > 1:
            y = Model.score2class(y)
        return sum(np.log(np.sum(Model.score(X,W), axis=1))) - np.sum(W.T[y] * X)

    def miss(X, y, W = None):
        if W is None:
            W = np.zeros((X.shape[1], y.shape[1]))
        if y.ndim > 1:
            y = Model.score2class(y)
        y2 = Model.predict(X, W)
        return sum(y2 != y)

    def gradient(X, Y, W):
        return np.dot(X.T, Model.proba(X, W) - Y)


