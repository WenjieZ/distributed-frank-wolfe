#!/usr/bin/env python3
import numpy as np
from utils import firstSVD

def row_data(z):
    return dict([('X',z[0]), ('Y',z[1]), ('n',z[1].shape[0]),
                 ('m',z[1].shape[1]), ('p',z[0].shape[1])])

def row_param(z):
    z['t'] = 0
    z['W'] = np.zeros((z['p'], z['m']))
    return z

def add(x, y):
    return x + y
	
def loground(t, c = 1):
    return 1 + np.int(c * np.log10(.99 + t))

def gradient(rdd, model):
    def row_gradient(z):
        z['grad'] = model.gradient(z['X'], z['Y'], z['W'])
        return z
    return rdd.map(row_gradient).setName("withGradient").cache()

def centralize(rdd, svd = firstSVD, **kwargs):
    grad = rdd.map(lambda z: z['grad']).reduce(add)
    return svd(grad)

def warmstart(rdd, svd = firstSVD, **kwargs):
    def row_svd(z):
        z['u'], z['v'] = svd(z['grad'])
        return z
    return rdd.map(row_svd)

def avgmix(rdd, **kwargs):
    rdd = warmstart(rdd).map(lambda z: (z['u'], z['v']))
    return rdd.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))

def poweriter(rdd, max_iter, v = None, t = 0):
    if v is None:
        v = warmstart(rdd).map(lambda z: z['v']).reduce(add)
        
    for _ in range(max_iter(t)):
        u = rdd.map(lambda z: np.dot(z['grad'], v)).reduce(add)
        u /= np.linalg.norm(u)
        v = rdd.map(lambda z: np.dot(u, z['grad'])).reduce(add)
        v /= np.linalg.norm(v)
    return u, v
    
def regularize(u, v, nn):
	u /= np.linalg.norm(u)
	v /= np.linalg.norm(v)
	u *= -nn
	return u, v	
    
def broadcast(rdd, u, v):
    def row_dest(z):
        z['D'] = np.outer(u, v)
        return z
    return rdd.map(row_dest).setName("withDest").cache()
    
def naivestep(rdd, *args, **kwargs):
    def row_step(z):
        z['a'] = 2. / (z['t'] + 2)
        return z
    return rdd.map(row_step), None
    
def linearsearch(rdd, model):
    a = rdd.map(lambda z: model.linesearch(z['X'], z['Y'], z['W'], z['D'], z['grad'])).reduce(add)
    a = a[0] / a[1]

    def row_step(z):
        z['a'] = a
        return z

    return rdd.map(row_step), a

def descent(rdd):
    def row_descent(z):
        z['W'], z['D'] = (1 - z['a']) * z['W'] + z['a'] * z['D'], z['W']
        z['t'] += 1
        return z
    return rdd.map(row_descent)

def ascent(rdd):
    def row_ascent(z):
        z['W'], z['D'] = z['D'], (z['W'] - (1 - z['a']) * z['D']) / z['a']
        z['t'] -= 1
        return z
    return rdd.map(row_ascent)
	
def loss(rdd, model):
    return rdd.map(lambda z: model.loss(z['X'], z['Y'], z['W'])).reduce(add)

def miss(rdd, model):
    return rdd.map(lambda z: model.miss(z['X'], z['Y'], z['W'])).reduce(add)

def iterate(rdd, model, lmo, stepsize, nn, t, cl = True, cm = True):
	rdd = gradient(rdd, model)
	u, v = lmo(rdd, t=t)
	u, v = regularize(u, v, nn)
	rdd = broadcast(rdd, u, v)
	rdd, ss = stepsize(rdd, model)
	rdd = descent(rdd)
	ls = loss(rdd, model) if cl else None
	ms = miss(rdd, model) if cm else None
	return rdd, (u, v), ss, ls, ms

