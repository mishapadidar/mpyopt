import numpy as np
from optimization_problem import OptimizationProblem
import cvxpy as cp

class ConvexQuadratic(OptimizationProblem):
    """Class for bound constrained convex quadratic optimization problems:
       min (x-x0) @ A @ (x-x0)
       s.t x in [lb,ub]
       The bounds may be infinite.
       """

    def __init__(self,lb,ub,A,x0):
        assert np.all(np.linalg.eigvals(A)>= 0.0), "Quadratic must be convex"
        self.dim = len(x0)
        self.lb = lb
        self.ub = ub
        self.A  = A
        self.x0 = x0

        # compute the minima
        x = cp.Variable(self.dim)
        objective = cp.Minimize(cp.quad_form(x,A) - 2*x0 @ A @ x )
        constraints = [x >= lb, x<=ub]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        self.minimum = x.value 

    def eval(self,x): 
        return (x-self.x0) @ self.A @ (x-self.x0) 

    def grad(self,x): 
        return (self.A + self.A.T) @ (x-self.x0)

    def hess(self,x):
        return (self.A + self.A.T)