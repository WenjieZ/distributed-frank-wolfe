# Distributed Frank-Wolfe Framework for Trace Norm Minimization

This package solves *trace norm minimization* in a distributed way. It is written in Python and works on clusters deployed with Apache SPARK.

## Dependency
Use Python3.5 for the widest compatibility with Apache SPARK. Other versions of Python may not be supported.

## Structure
The package is composed of the front end and the back end. The front end implements high-level functions, which allows the users to run the distributed Frank-Wolfe algorithms without caring about the detail. The back end implements various low-level functions about every aspect of Frank-Wolfe.

### Frontend
The front end (`frontend.py`) has two functions: `solve` and `evaluate`. The former solves a minimization problem, as the name indicates. It returns an `FWpath` object, which stores the solution path. The latter takes a dataset and an `FWpath` object to evaluate the fitness of the result.

### Backend
The back end is the base of the framework. It is composed of two parts: the Frank-Wolfe strategy part, which is generic, and the model part, which is not generic.

We currently have implemented two models: **multitask least square** (`mls`) and **multinomial logistic regression** (`mlr`). Each model is implemented in a separated file (e.g., `mls.py`, `mlr.py`) and follows a certain API, among which the most important functions are `stats` and `update`. The model part is written in pure Python. Users can implement their own models without any prior distributed computing knowledge.

The strategy part (`fw.py`) implements the 4 distributed strategies solving the linear subproblem as well as various step size defining devices. This part is written in PySpark, SPARK's Python API.

## Basic usage
The typical workflow is:

1. Open a PySpark terminal or a PySpark notebook
2. Prepare your own RDD, and `from frontend import solve, evaluate`
3. Feed your RDD to the `solve` function, as well as provide necessary parameters
4. Evaluate your result with `evaluate` function.

The `solve` function takes as input 
- data RDD, 
- metadata dictionary containing the dimension of the input matrices (i.e., n, m, p)
- model (either mls or mlr, should be imported)
- hyperparameter characterizing the intended trace norm
- number of epochs
- linear minimization oracle
- step size oracle

Linear minimization oracle candidates:
```python
fw.centralize                                                             # centralize
fw.avgmix                                                                 # singular vectors mixture
lambda rdd, **kwargs: fw.poweriter(rdd, lambda t: 1, 'random', **kwargs)  # power1
lambda rdd, **kwargs: fw.poweriter(rdd, lambda t: 2, 'random', **kwargs)  # power2
lambda rdd, **kwargs: fw.poweriter(rdd, lambda t: 1, **kwargs)            # power1 with warm start
lambda rdd, **kwargs: fw.poweriter(rdd, lambda t: 2, **kwargs)            # power2 with warm start
lambda rdd, **kwargs: fw.poweriter(rdd, lambda t: fw.loground(t, c=1), 'random', **kwargs)  # powlog
```

Step size oracle candidates:
```python
fw.linesearch
fw.naivestep                                     # 2 / (t+2)
lambda **kwargs: fw.fixedstep(const=0.01)        # fixed step size
```
The `evaluate` function takes as input
- test data RDD,
- solution path yield from `solve`
- metadata dictionary containing the dimension of the input matrices (i.e., n, m, p)
- model (either mls or mlr, should be imported)
- (optional) ground truth matrix

For the mls model, it outputs the objective function value and, if provided with the ground truth, estimation error. For the mlr model, it outputs the objective function value and the top-5 misclassification rate.

This workflow works either on a cluster or a PC. For users with difficulties to install SPARK, I recommend my [Park](https://github.com/WenjieZ/Park) package. It allows to develop and/or test basic Spark code on a single PC without SPARK installed.

## Demo
Two demos are provided, in both notebook format and HTML format.

## Reference
This repository accompanies my working paper 

- A Distributed Frank-Wolfe Framework for Learning Low-Rank Matrices with the Trace Norm

as well as my Ph.D. dissertation

- A Distributed Frank-Wolfe Framework for Trace Norm Minimization

**Difference between these two works**
- My working paper focuses on matrices in the *general space*, whereas my dissertation studies *general matrices*, *symmetric matrices* and *positive semidefinite matrices*.
- My dissertation includes only the slow convergence rate, whereas my working paper quantifies also the fast convergence rate when hypotheses could be made on the distribution of the singular values.
- My working paper presents only the convergence in expectation, whereas my dissertation presents also the convergence in probability.
- My working paper describes only the dense representation, whereas yy dissertation discusses also the low-rank representation.
- My working paper provides a coarse-grained analysis of the elapsed time, whereas my dissertation provides a fine-grained one.

