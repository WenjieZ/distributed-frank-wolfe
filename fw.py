#!/usr/bin/env python3
import numpy as np
from utils import *

# lmo
def add(x, y):
    return x + y

def loground(t, c = 0.5):
    return 1 + np.int(c * np.log10(.99 + t))

def centralize(rdd, svd = firstSVD, **kwargs):
    grad = rdd.map(lambda z: z['grad']).setName("grad").reduce(add)
    return svd(grad)

def warmstart(rdd, svd = firstSVD, **kwargs):
    return rdd.map(lambda z: svd(z['grad'])).setName("u,v")

def avgmix(rdd, **kwargs):
    return warmstart(rdd).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))

def poweriter(rdd, max_iter, v = None, t = 0, **kwargs):
    if v is None:
        v = warmstart(rdd).map(lambda z: z[1]).reduce(add)
    elif v == "random":
        v = np.random.randn(kwargs['m'])
    #print("poweriter: ", v[0])

    k = max_iter(t)
    for _ in range(int(k)):
        u = rdd.map(lambda z: np.dot(z['grad'], v)).setName("v").reduce(add)
        u /= np.linalg.norm(u)
        v = rdd.map(lambda z: np.dot(u, z['grad'])).setName("u").reduce(add)
        v /= np.linalg.norm(v)
    if k > int(k):
        u = rdd.map(lambda z: np.dot(z['grad'], v)).reduce(add)
        u /= np.linalg.norm(u)
    return u, v

def regularize(u, v, nn):
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    u *= -nn
    return u, v

# ls
def naivestep(*args, t, **kwargs):
    return 2./ (t + 2)

def linesearch(*args, rdd, u = None, v = None, ls = None, D = None, **kwargs):
    if D is None:
        D = LRmatrix([1], [u], [v])
    a = rdd.map(lambda z: ls(**z, D = D)).reduce(add)
    return min(a[0] / a[1], 1)

def fixedstep(*args, const = 0.01, **kwargs):
    return const

# update
def update(rdd, u, v, a, f):
    return rdd.map(lambda z: f(z, LRmatrix([1], [u], [v]), a))

if __name__ == '__main__':
    from pyspark import SparkContext
    sc = SparkContext("local", "Test fw.py")
    import mls as md
    # X: n*p  W: p*m  Y: n*m
    # =======================

    # default parameters
    param = {'n':16, 'm':5, 'p':4, 'r':3, 'nn':1, 'seed':0}

    # generate data
    data, W = md.generate(**param)

    # prepare data
    points = mat2point(*data)
    dataRDD = sc.parallelize(points, 8).mapPartitions(point2mat)
    assert dataRDD.count() == 8
    statRDD = dataRDD.map(md.stats).cache()
    assert statRDD.count() == 8

    # lmo
    assert np.isclose(prod(centralize(statRDD)), 7.186559722e-06)
    assert np.isclose(prod(avgmix(statRDD)), 275.7011134)
    assert np.isclose(prod(poweriter(statRDD, lambda t: 3)), 2.863723520e-06)
    u, v = regularize(*centralize(statRDD), 1)
    assert np.isclose(prod((u, v)), 2.852219061e-06)

    # ls
    assert np.isclose(naivestep(t = 2), 0.5)
    a = linesearch(rdd = statRDD, u = u, v = v, ls = md.linesearch)
    assert np.isclose(a, 0.3897265)

    # update
    statRDD = update(statRDD, u, v, a, md.update)
    assert np.isclose(prod(centralize(statRDD)), -0.00195662557)

    print('Test: fw.py...OK')
