#!/usr/bin/env python3
from utils import firstSVD
import numpy as np

def add(x, y):
	return x + y
	
def loground(t, c = 1):
    return 1 + np.int(c * np.log10(.99 + t))
    
def gradient(paramRDD, modelRDD):
	rdd = modelRDD.join(paramRDD)
	return rdd.mapValues(lambda x: x[0].gradient(x[1].W))

def centralize(gradRDD, t = None, svd = firstSVD):
    grad = gradRDD.values().reduce(add)
    return svd(grad)

def warmstart(gradRDD, svd = firstSVD):
    return gradRDD.mapValues(lambda x: svd(x))

def avgmix(gradRDD, t = None):
    rdd = warmstart(gradRDD).values()
    return rdd.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))

def poweriter(gradRDD, t, max_iter, v = None):
    if v is None:
        rdd = warmstart(gradRDD).map(lambda x: x[1][1])
        v = rdd.reduce(add)

    for _ in range(max_iter(t)):
        u = gradRDD.map(lambda A: np.dot(A[1], v)).reduce(add)
        u /= np.linalg.norm(u)
        v = gradRDD.map(lambda A: np.dot(u, A[1])).reduce(add)
        v /= np.linalg.norm(v)
    return u, v
    
def regularize(u, v, nn):
	u /= np.linalg.norm(u)
	v /= np.linalg.norm(v)
	u *= -nn
	return u, v	
    
def broadcast(paramRDD, u, v):
    def _f(x, u, v):
        x.D = np.outer(u, v)
        return x
    return paramRDD.mapValues(lambda x: _f(x, u, v))
    
def naivestep(paramRDD, modelRDD = None, gradRDD = None):
    def _f(x):
        x.a = 2. / (x.t + 2)
        return x
    return paramRDD.mapValues(_f), None
    
def linearsearch(paramRDD, modelRDD, gradRDD):
	rdd = modelRDD.join(paramRDD).join(gradRDD).values()
	rdd = rdd.map(lambda c: c[0][0].linesearch(c[0][1].W, c[0][1].D, c[1]))
	a = rdd.reduce(add)
	a = a[0] / a[1]
	
	def _f(x):
		x.a = a
		return x

	return paramRDD.mapValues(_f), a

def descent(paramRDD):
    def _f(x):
        x.W, x.D = (1 - x.a) * x.W + x.a * x.D, x.W
        x.t += 1
        return x
    return paramRDD.mapValues(_f)

def ascent(paramRDD):
	def _f(x):
		x.W, x.D = x.D, (x.W - (1 - x.a) * x.D) / x.a
		x.t -= 1
		return x
	return paramRDD.mapValues(_f)
	
def loss(paramRDD, modelRDD):
	rdd = modelRDD.join(paramRDD).values()
	return rdd.map(lambda x: x[0].loss(x[1].W)).reduce(add)
	
def miss(paramRDD, modelRDD):
	rdd = modelRDD.join(paramRDD).values()
	return rdd.map(lambda x: x[0].miss(x[1].W)).reduce(add)

def iterate(model, param, lmo, stepsize, nn, t, cl = True, cm = True):
	grad = gradient(param, model)
	u, v = lmo(grad, t)
	u, v = regularize(u, v, nn)
	param = broadcast(param, u, v)
	param, ss = stepsize(param, model, grad)
	param = descent(param)
	ls = loss(param, model) if cl else None
	ms = miss(param, model) if cm else None
	return param, (u, v), ss, ls, ms

