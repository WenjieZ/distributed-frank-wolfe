#!/usr/bin/env python3
import numpy as np
from utils import *

def class2score(y, k = None):
    if k is None:
        k = max(y) + 1
    Y = np.zeros((len(y), k), dtype = np.int)
    for i, s in enumerate(y):
        Y[i, s] = 1
    return Y

def score2class(Y):
    y = Y.argmax(axis=1)
    return y

def predict(X, W, hit = None):
    XW = todense(dot(X, W))
    if hit is None:
        return score2class(XW)
    else:
        return np.argpartition(XW, -hit)[:, -hit:]

def generate(n = 16, m = 5, p = 4, r = 3, nn = 1, seed = 0, **kwargs):
    np.random.seed(seed)
    X = np.random.randn(n, p)
    W = lowrank(p, m - 1, r, nn, addzero = True)
    y = predict(X, W)
    return (X, y), W

#def score(X, W):
#    s = todense(dot(X, W))
#    return np.exp(s - np.max(s))

#def proba(X, W):
#    Z = score(X, W)
#    return Z / np.reshape(np.sum(Z, axis=1), (-1, 1))

def loss(X, y, W = None, k = None):
    if W is None:
        return X.shape[0] * np.log(k)
    y = y.squeeze()
    if y.ndim > 1:
        y = score2class(y)
    s = todense(dot(X, W)).T
    s = np.exp(s - s[y, range(s.shape[1])])
    return sum(np.log(np.sum(s, axis=0)))
    #return sum(np.log(np.sum(score(X,W), axis=1))) - np.sum(W.T[y] * X)

def miss(X, y, W = None, k = None, hit = None):
    if W is None:
        return X.shape[0] * (1 - float(1 if hit is None else hit) / k)
    y = y.squeeze()
    if y.ndim > 1:
        y = score2class(y)
    y2 = predict(X, W, hit)
    if hit is None:
        return sum(y2 != y)
    else:
        return sum([i not in j for i, j in zip(y, y2)])

#def gradient(X, Y, W):
#    return np.dot(X.T, proba(X, W) - Y)

def score2proba(S):
    S = np.exp(S.T - np.max(S, axis=1))
    S = S / np.sum(S, axis=0)   
    return S.T
    
def stats(data, W = LRmatrix([], [], []), k = None, **kwargs):
    X, y = data
    y = y.squeeze()
    if isinstance(W, LRmatrix) and len(W) > 0:
        k = W.n
    elif not isinstance(W, LRmatrix):
        k = W.shape[1]
    XW = dot(X, W)
    if isinstance(XW, LRmatrix):
        XW = XW.todense((X.shape[0], k))
    S = score2proba(XW)
    Y = class2score(y, k)
    XY = np.dot(X.T, Y)
    grad = np.dot(X.T, S) - XY
    grad[:, 0] = 0
    return {'X':X, 'XW':XW, 'XY':XY, 'grad':grad}

def linesearch():
    error

def update(s, D, a):
    s['XW'] = dot(s['X'], a * D) + (1 - a) * s['XW']
    s['grad'] = np.dot(s['X'].T, score2proba(s['XW'])) - s['XY']    # costly
    s['grad'][:, 0] = 0
    return s

if __name__ == '__main__':
    # class2score
    assert np.sum(class2score([1, 0, 3, 2], 5)) == 4
    assert class2score([1, 0, 3, 2], 5).shape == (4, 5)
    assert class2score([2, 3, 4]).shape == (3, 5)
    assert np.sum(class2score([2, 3, 4])) == 3
    
    # score2class
    np.random.seed(0)
    assert sum(score2class(np.random.rand(9, 3))) == 12
    
    # predict
    X = np.array([[1, 2], [3, 4]], np.float_)
    W = LRmatrix([2], [np.array([1, 2])], [np.array([1, 2])])
    assert sum(predict(X, W)) == 2
    assert predict(X, W, 2).shape == (2, 2)

    # generate
    param = {'n':16, 'm':5, 'p':4, 'r':3, 'nn':1, 'seed':1}
    data, W = generate(**param)
    assert np.isclose(prod(data[0]), 1.576354e-17)
    
    # loss
    assert np.isclose(loss(*data, k = 5), 25.751)
    assert np.isclose(loss(*data, W), 22.18814937)
    
    # miss
    assert np.isclose(miss(*data, k = 5), 12.8)
    assert np.isclose(miss(*data, k = 5, hit = 2), 48./5)
    assert miss(*data, W) == 0
    np.random.seed(0)
    assert miss(*data, np.random.randn(4, 5), hit = 3) == 3
    
    # score2proba
    A = np.array([[0, np.log(3)], [np.log(4), 0]])
    assert np.isclose(prod(score2proba(A)), 0.03)
    
    # stats
    s = stats(data, W)
    s = stats(data, k = 5)
    
    # update
    assert np.isclose(prod(update(s, W, 0.5)['grad'][:, 1:]), 2.89565862)

    print('Test: mlr.py...OK')    
