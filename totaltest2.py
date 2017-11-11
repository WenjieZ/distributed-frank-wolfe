import numpy as np
from time import time
from utils import *
import mlr as md
import fw

from pyspark import SparkContext
sc = SparkContext("local", "Total test 1")
# # Warehouse

# X: n*p  W: p*m  Y: n*m
# =======================

# default parameters
param = {'n':200, 'm':10, 'p':100, 'r':5, 'nn':1, 'seed':0}

# generate data
data, W = md.generate(**param)
X1 = data[0][0:100]
X2 = data[0][100:200]
y1 = data[1][0:100]
y2 = data[1][100:200]


# prepare data
points = mat2point(X1, y1)
dataRDD = sc.parallelize(points, 2).mapPartitions(point2mat)


# # Optimization

# parameter
T = 2
lmo = fw.centralize
step = fw.naivestep
nn = 10
# init
U = []
V = []
A = np.array([])
timer = np.zeros(T)
statRDD = dataRDD.map(lambda z: md.stats(z, k = param['m'])).setName("initial").persist()
# iter
for t in range(T):
    u, v = fw.regularize(*lmo(statRDD), nn)
    a = step(rdd = statRDD, u = u, v = v, t = t, ls = md.linesearch)
    timer[t] = time()
    statRDD = fw.update(statRDD, u, v, a, f = md.update)
    statRDD.setName("turn" + str(t)).persist()
    U.append(u)
    V.append(v)
    A = np.append(A, a)


# # Evaluation

emprisk = [md.loss(X1, y1, k = param['m'])]
eva = [md.miss(X2, y2, k = param['m'])]


W0 = LRmatrix([], [], [])
for u, v, a in zip(U, V, A):
    W0 = LRmatrix([a], [u], [v]) + (1 - a) * W0
    emprisk.append(md.loss(X1, y1, W0))
    eva.append(md.miss(X2, y2, W0))
emprisk = [x / emprisk[0] for x in emprisk]

assert np.isclose(prod(emprisk + eva), 581822.96)
print("Total test 2...OK")
