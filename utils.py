#!/usr/bin/env python3
import numpy as np

def lowrank(m, n, r, nn = 1):
    W = np.random.randn(m,n)
    U, s, V = np.linalg.svd(W, full_matrices = False)
    s[r:] = 0
    s = s / sum(s) * nn
    S = np.diag(s)
    W = U.dot(S).dot(V)
    return W

def mat2point(X, *Y):
    return [(x, *y) for x, *y in zip(X, *Y)]

def point2mat(points):
    points = list(points)
    mat = [np.array(x) for x in points[0]]
    for p in points[1:]:
        for i in range(len(p)):
            mat[i] = np.vstack((mat[i], p[i]))
    return [mat]

def hashkey(x):
    return (hash(str(x)), x)

def apogee(x):
    a, b = min(x), max(x)
    return a if abs(a) > b else b

def firstSVD(A):
    m,n = np.shape(A)
    v = np.random.randn(n)
    for _ in range(20):
        u = np.dot(A,v)
        u /= apogee(u)
        v = np.dot(u,A)
        v /= np.linalg.norm(v, np.inf)
    return u, v
