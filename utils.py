#!/usr/bin/env python3
import numpy as np
from copy import deepcopy
#from scipy.sparse import diags

class LRmatrix:
    def __init__(self, s, u, v):
        self.s = np.array(s)
        self.u = deepcopy(u)
        self.v = deepcopy(v)
        if len(s) > 0:
            self.m = len(u[0])
            self.n = len(v[0])
    
    def __len__(self):
        return len(self.s)

    def __str__(self):
        return str(self.s) + str(self.u) + str(self.v)
    
    def __add__(self, A):
        if isinstance(A, LRmatrix):
            return LRmatrix(np.append(self.s, A.s), self.u + A.u, self.v + A.v)
        elif len(self) == 0:
            return A
        else:
            return self.todense() + A
    
    def __neg__(self):
        return LRmatrix(-self.s, self.u, self.v)

    def __mul__(self, a):
        return LRmatrix(a * self.s, self.u, self.v)

    __rmul__ = __mul__

    def todense(self, shape = None):
        if len(self) > 0:
            shape = (self.m, self.n)
        X = np.zeros(shape)
        for s, u, v in zip(self.s, self.u, self.v):
            X += np.outer(s * u, v)
        return X

def lrrmul(A, W):
    if len(W) == 0:
        return W
    u_new = []
    s_new = []
    for s, u in zip(W.s, W.u):
        result = np.dot(A, u)
        nm = np.linalg.norm(result)
        result /= nm
        s_new.append(s * nm)
        u_new.append(result)
    return LRmatrix(s_new, u_new, W.v)

def dot(X, W):
    return lrrmul(X, W) if isinstance(W, LRmatrix) else np.dot(X, W)

def todense(W):
    return W.todense() if isinstance(W, LRmatrix) else W

class FWpath:
    def __init__(self, U, V, A, timer):
        self.U = U
        self.V = V
        self.A = A
        self.timer = timer

class Performance:
    def __init__(self, risk, good, time):
        self.risk = risk
        self.good = good
        self.time = time


def lowrank(m, n, r, nn = 1, seed = 0, addzero = False):
    np.random.seed(seed)
    u = np.random.randn(m, r)
    u = np.linalg.qr(u)[0].T
    v = np.random.randn(n, r)
    v = np.linalg.qr(v)[0].T
    if addzero:
        v = np.hstack((np.zeros((r, 1)), v))
    s = np.random.rand(r)
    s *= nn / sum(s)
    return LRmatrix(sorted(s, reverse=True), list(u), list(v))

# def lowrank(m, n, r, nn = 1, dense = True):
#     if dense:
#         W = np.random.randn(m,n)
#         U, s, V = np.linalg.svd(W, full_matrices = False)
#         s[r:] = 0
#         s = s / sum(s) * nn
#         S = np.diag(s)
#         W = U.dot(S).dot(V)
#     else:
#         s = np.zeros(min(m, n))
#         s[0:r] = 1/r
#         W = diags(s, shape = (m,n)).toarray()
#     return W

def mat2point(X, *Y):
    return [(x, *y) for x, *y in zip(X, *Y)]

def parseLine(string):
    arr = np.fromstring(string,dtype=float,sep=',')
    return arr[1:]/np.linalg.norm(arr[1:]), int(arr[0])

def point2mat(points): 
    points = list(points)
    mat = [np.array(x) for x in points[0]]
    for p in points[1:]:
        for i in range(len(p)):
            mat[i] = np.vstack((mat[i], p[i]))
    return [[x.squeeze() for x in mat]]

def test(x):
    return hash(str(x))

def apogee(x):
    a, b = min(x), max(x)
    return a if abs(a) > b else b

def firstSVD(A, seed = 0):
    np.random.seed(seed)
    m,n = np.shape(A)
    v = np.random.randn(n)
    for _ in range(20):
        u = np.dot(A,v)
        u /= apogee(u)
        v = np.dot(u,A)
        v /= np.linalg.norm(v, np.inf)
    return u, v

def viaSVD(A):
    u, s, v = np.linalg.svd(A)
    u = u[:,0]
    v = v[0,:]
    return u, v

def prod(a):
    if isinstance(a, np.ndarray):
        return np.prod(a)
    s = 1
    for x in a:
        s *= np.prod(x)
    return s

def allclose(a, b):
    if isinstance(a, np.ndarray):
        return np.allclose(a, b)
    for x, y in zip(a, b):
        if not np.allclose(x, y):
            return False
    return True

if __name__ == '__main__':
    # prod
    a = np.array([1, 2, 3, 4])
    assert np.isclose(prod(a), np.prod(a))
    b = (a, a)
    assert np.isclose(prod(b), 24 * 24)
    
    # LRmatrix
    s = [1., 2.]
    u = [np.array([1., 1.]), np.array([2., 1.])]
    v = [np.array([1., 2.]), np.array([1., 1.])]
    W = LRmatrix(s, u, v)
    assert np.isclose(prod(W.todense()), 360)
    assert len(W) == 2
    A = np.array([[1., 2.], [3., 4.]])
    assert np.allclose(W + A, W.todense() + A)
    assert np.allclose((W + -W).todense(), 0.)
    assert np.allclose((2 * W).todense(), (W * 2).todense())
    # lrrmul
    assert np.isclose(prod(lrrmul(A, W).todense()), 141372)
    assert len(lrrmul(A, LRmatrix([], [], []))) == 0
    # dot
    assert np.isclose(prod(todense(dot(A, W))), 141372)
    assert len(dot(A, LRmatrix([], [], []))) == 0

    # lowrank
    a = lowrank(3, 4, 2, 2)
    u = np.array(a.u)
    v = np.array(a.v)
    assert np.allclose(np.dot(u, u.T), np.array([[1., 0.], [0., 1.]]))
    assert np.allclose(np.dot(v, v.T), np.array([[1., 0.], [0., 1.]]))
#    assert test(lowrank(3, 4, 2, 1, False)) == -4542029373938666413

    # mat2point
    s = 2, 2
    a = np.ones(s)
    b = 2 * np.ones(s)
    c = 3 * np.ones(s)
    d = mat2point(a, b, c)
    assert np.isclose(prod(d), 1296)

    # point2mat
    assert np.isclose(prod(point2mat(d)), 1296)

    # apogee
    a = np.array([1, -2, -4, 5])
    assert apogee(a) == 5
    assert apogee(-a) == -5

    # firstSVD
    np.random.seed(3)
    a = np.random.randn(3,4)
    u, v = viaSVD(a)
    u = u / np.linalg.norm(u, np.inf)
    v = v / np.linalg.norm(v, np.inf)
    u0, v0 = firstSVD(a)
    assert np.allclose(-u, u0) or np.allclose(u, u0)
    assert np.allclose(-v, v0) or np.allclose(v, v0)

    print('Test: utils.py...OK')
