import numpy as np
from time import time
from utils import *
import fw

def add(x, y):
    return x + y
    
def solve(oDataRDD, metadata, md, nn, T, lmo, step):
    # Extract metadata -------------
    n = metadata['n']
    m = metadata['m']
    p = metadata['p']
    
    # Warehouse --------------------
    dataRDD = oDataRDD.mapPartitions(point2mat)
    
    # Optimization  ----------------
    # init
    U = []
    V = []
    A = np.array([])
    statRDD = dataRDD.map(lambda z: md.stats(z, k = m)).setName("initial").persist()
    statRDD.count()         
    timer = np.zeros(T + 1)  
    timer[0] = time()
    
    # iter
    np.random.seed(1)
    for t in range(T):
        u, v = fw.regularize(*lmo(statRDD, m = m, t = t), nn)
        a = step(rdd = statRDD, u = u, v = v, t = t, ls = md.linesearch)
        U.append(u)
        V.append(v)
        A = np.append(A, a)
        tempRDD = fw.update(statRDD, u, v, a, f = md.update)
        tempRDD.setName("turn" + str(t + 1)).persist().count()
        statRDD.unpersist()
        statRDD = tempRDD
        timer[t + 1] = time()  
    
    # Output -----------------------
    return FWpath(U, V, A, timer - timer[0])
    
    
def evaluate(dataRDD, path, metadata, md, W=None):
    n = metadata['n']
    m = metadata['m']
    p = metadata['p']

    testRDD = dataRDD.mapPartitions(point2mat).cache()

    loss = [testRDD.map(lambda z: md.loss(*z, k = m)).reduce(add)]
    eerr = None
    miss = None
    if W is not None:
        eerr = [np.linalg.norm(todense(W))]
    if md.__name__ == 'mlr':
        miss = [testRDD.map(lambda z: md.miss(*z, k = m, hit = 5)).reduce(add)]

    W0 = LRmatrix([], [], [])
    for u, v, a in zip(path.U, path.V, path.A):
        W0 = LRmatrix([a], [u], [v]) + (1 - a) * W0
        loss.append(testRDD.map(lambda z: md.loss(*z, W0)).reduce(add))
        if W is not None:
            eerr.append(np.linalg.norm(todense(W0 + -W)))
        if md.__name__ == "mlr":
            miss.append(testRDD.map(lambda z: md.miss(*z, W0, hit = 5)).reduce(add))

    loss = [y / loss[0] for y in loss]
    if W is not None:
        eerr = [y / eerr[0] for y in eerr]
    if md.__name__ == 'mlr':
        miss = [y / n for y in miss]
    
    return loss, eerr, miss
