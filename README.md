# Distributed Frank-Wolfe Framework for Trace Norm Minimization
This repository accompanies my working paper 

- A Distributed Frank-Wolfe Framework for Learning Low-Rank Matrices with the Trace Norm

as well as my PhD dissertation

- A Distributed Frank-Wolfe Framework for Trace Norm Minimization

**Difference between these two works**
- My working paper focuses on matrices in the *general space*. My dissertation studies *general matrices*, *symmetric matrices* and *positive semidefinite matrices*.

- My disseration includes only the slow convergence rate. My working paper quantifies also the fast convergence rate when hypotheses cound be made on the distribution of the singular values.

- My working paper describes only the dense representation. My disseration discusses also the low-rank representation.

- My working paper provides a coarse-grained analysis of the elapsed time. My dissertation provides a fine-grained one.


## Structure
The package is composed of the front end and the back end. The front end implements high level functions, which allows the users to run the distributed Frank-Wolfe algorithms without caring about the detail. The back end implements various low level functions about every aspect of Frank-Wolfe.

### Front end
The front end (`frontend.py`) has two functions: `solve` and `evaluate`. The former solves a minimization problem, as the name indicates. It returns a `FWpath` object, which stores the solution path. The latter takes a dataset and a `FWpath` object to evaluate the fitness of the result.

### Back end
The back end is the base of the framework. It is composed of two parts: the Frank-Wolfe strategy part, which is generic, and the model part, which is not generic.

We currently have implemented two models: **multitask least square** (`mls`) and **multinomial logistic regression** (`mlr`). Each model is implemented in a separated file (e.g., `mls.py`, `mlr.py`) and follows a certain API, among which the most important functions are `stats` and `update`. The model part is written in pure Python. Users can implement their own models without any prior distributed computing knowledge.

The strategy part (`fw.py`) implements the 4 distributed strategies solving the linear subproblem as well as various step size defining devices. This part is written in PySpark, SPARK's Python API.

## Usage
1. Open a PySpark terminal or a PySpark notebook

2. Prepare your own RDD

3. Feed your RDD to the `solve` function, as well as provide necessary parameters

4. Evaluate your result with `evaluate` function.

The workflow works either on a cluster or a PC. For users with difficulties to install SPARK, I recommend my [Park](https://github.com/WenjieZ/Park) package. It allows to develop and/or test basic Spark code on a single PC without SPARK installed.

## Demo
Two demos are provided, in both notebook format and HTML format.

