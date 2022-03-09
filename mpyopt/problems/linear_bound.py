import numpy as np
from optimization_problem import OptimizationProblem

class LinearBound(OptimizationProblem):
    """Class for bound constrained linear programs:
       min c @ x
       s.t x in [lb,ub]
       """

    def __init__(self,lb,ub,c):
        self.dim = len(c)
        self.lb = lb
        self.ub = ub
        self.c  = c
        self.minimum = ub*(c<0) + lb*(c>=0)

    def eval(self,x): 
        return self.c @ x

    def grad(self,x): 
        return self.c
