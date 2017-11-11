#!/usr/bin/env python3
import numpy as np
from utils import lowrank

def class2score(y, k):
	Y = np.zeros((len(y), k), dtype = np.int)
	for i, s in enumerate(y):
		Y[i, s] = 1
	return Y
	
def score2class(Y):
	y = Y.argmax(axis=1)
	return y
	
def predict(X, W):
	return score2class(np.dot(X, W))

def generate(n = 8, m = 5, p = 4, r = 3, nn = 1):
    X = np.random.randn(n, p)
    W = lowrank(p, m, r, nn)
    Y = predict(X, W)
    return X, Y, W
    
def score(X, W):
	return np.exp(np.dot(X,W))
	
def proba(X, W):
	Z = score(X, W)
	return Z / np.reshape(np.sum(Z, axis=1), (-1, 1))

def loss(X, y, W):
    return sum(np.log(np.sum(score(X,W), axis=1))) - np.sum(W.T[y] * X)

def miss(X, y, W):
	y2 = predict(X, W)
	return sum(y2 != y)

def gradient(X, Y, W):
    return np.dot(X.T, proba(X, W) - Y)

class Model():
    def __init__(self, Z):
        self.X = Z[0]
        self.y = Z[1]
        self.m = Z[2]
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]

    def loss(self, W):
        return loss(self.X, self.y, W)
        
    def miss(self, W):
        return miss(self.X, self.y, W)

    def gradient(self, W):
        return gradient(self.X, class2score(self.y, self.m), W)

class Param():
    def __init__(self, shape):
        self.t = 0
        self.W = np.zeros(shape)
