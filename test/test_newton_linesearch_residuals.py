import numpy as np
import sys
sys.path.append("../mpyopt/nonlinear_equations")
from newton_linesearch_residuals import NewtonLinesearchResiduals


def test_1():
  """
  Linear Equation Ax = b
  """
  dim_x = 182
  A = np.random.randn(dim_x,dim_x)
  b = np.random.randn(dim_x)
  F = lambda x: A @ x - b
  H = lambda x: A
  c = np.linalg.solve(A,b)

  x0    = np.zeros(dim_x)
  max_iter = 1
  xopt = NewtonLinesearchResiduals(F,H,x0,max_iter,verbose=True)
  print('Test 1: distance to opt: ',np.linalg.norm(xopt - c))
  return xopt

def test_2():
  """
  A Equality constrained convex QP
  min x^TQx
  s.t Ax = b
  """
  dim_x = 500
  # objective
  Q = np.random.randn(dim_x,dim_x)
  Q = Q @ Q.T + 1e-8*np.eye(dim_x)
  # constraints
  dim_c = 100
  A = np.random.randn(dim_c,dim_x)
  b = np.random.randn(dim_c)
  def F(y):
    x = y[:dim_x]
    lam = y[dim_x:]
    grad = np.zeros(dim_x + dim_c)
    grad[:dim_x] = np.copy(Q @ x + A.T @ lam)
    grad[dim_x:] = np.copy(A @ x - b)
    return grad
  def H(y):
    x = y[:dim_x]
    lam = y[dim_x:]
    hess = np.zeros((dim_x + dim_c,dim_x + dim_c))
    hess[:dim_x,:dim_x] = np.copy(Q)
    hess[:dim_x,dim_x:] = np.copy(A.T)
    hess[dim_x:,:dim_x] = np.copy(A)
    return hess

  # solution by lagrange multipliers
  lam = np.linalg.solve(A @ np.linalg.solve(Q,A.T),b)
  c = np.linalg.solve(Q,A.T @ lam)
  # solution by newton
  x0    = np.zeros(dim_x+dim_c)
  max_iter = 100
  ftarget = 1e-12
  yopt = NewtonLinesearchResiduals(F,H,x0,max_iter=max_iter,ftarget=ftarget,verbose=True)
  xopt = yopt[:dim_x]
  print('Test 2: distance to opt: ',np.linalg.norm(xopt - c))
  return xopt

test_1()
test_2()
