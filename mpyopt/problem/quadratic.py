import numpy as np
from optimization_problem import OptimizationProblem

class Quadratic(OptimizationProblem):
    """Class for uconstrained quadratic programs:
       min (x-x0) @ A @ (x-x0) + c
    """

    def __init__(self,A,x0,c=0.0):
        self.dim = len(x0)
        self.A  = A
        self.x0 = x0
        self.c = c
        self.minimum = x0

    def eval(self,x): 
        return (x-self.x0) @ self.A @ (x-self.x0)  + self.c

    def grad(self,x): 
        return (self.A + self.A.T) @ (x-self.x0)

    def hess(self,x):
        return (self.A + self.A.T)
