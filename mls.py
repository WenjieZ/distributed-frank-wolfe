#!/usr/bin/env python3
import numpy as np
from utils import *

def generate(n = 16, m = 5, p = 4, r = 3, nn = 1, seed = 0, sigma = 0):
    np.random.seed(seed)
    X = np.random.randn(n, p)
    W = lowrank(p, m, r, nn)
    Y = lrrmul(X, W).todense()
    Y += sigma * np.linalg.norm(Y) / np.sqrt(n*m) * np.random.randn(n, m)
    return (X, Y), W

def loss(X, Y, W = None, **kvargv):
    if W is None:
        W = np.zeros((X.shape[1], Y.shape[1]))
    Y2 = todense(dot(X, W))
    return np.sum((Y2 - Y)**2) / 2

def XX_XY(X, Y):
    return np.dot(X.T, X), np.dot(X.T, Y)

def stats(data, W = LRmatrix([], [], []), **kwargs):
    XX, XY = XX_XY(*data)
    XXW = dot(XX, W)
    grad = XXW + -XY
    if len(W) == 0:
        XXW = np.zeros_like(XY)
        W = np.zeros_like(grad)
    XXW = todense(XXW)
    W = todense(W)
    return {'XX':XX, 'XXW':XXW, 'grad':grad, 'W':W}

def linesearch(XX, XXW, grad, W, D):
    Z = todense(D) - W
    num = np.sum(-grad * Z)
    XXD = dot(XX, D)
    den = np.sum(todense(XXD + -XXW) * Z) 
    return np.array([num, den])

def update(s, D, a):
    s['grad'] -= todense(s['XXW'])
    s['XXW'] = dot(s['XX'], a * D) + (1 - a) * s['XXW']
    s['grad'] += todense(s['XXW'])
    s['W'] = a * D + (1 - a) * s['W']
    return s

if __name__ == '__main__':    
    # generate data
    param = {'n':16, 'm':5, 'p':4, 'r':3, 'nn':1, 'seed':0}
    data, W = generate(**param)
    assert np.isclose(prod(data), 1.29557e-84)

    # loss
    X = np.array([[1, 2], [3, 4]], np.float_)
    Y = np.array([[10, 20], [30, 40]], np.float_)
    assert np.isclose(loss(X, Y), 1500.)
    W = LRmatrix([2], [np.array([1, 2])], [np.array([1, 2])])
    assert np.isclose(loss(X, Y, W), 40.)

    # XX_XY
    assert np.isclose(prod(XX_XY(X, Y)), 15366400000000)

    # stats, linesearch
    D = LRmatrix([4], [np.array([1, 1])], [np.array([1, 0])])
    s = stats((X, Y), W)
    a = linesearch(**s, D = D)
    b = linesearch(**stats((X, Y), W.todense()), D = D.todense())
    assert np.allclose(a, b)
    linesearch(**stats((X, Y)), D = D)

    # update
    assert prod(update(s, D, 0.5)['XXW']) == 86118336

    print('Test: mls.py...OK')
