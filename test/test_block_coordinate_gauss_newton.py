import numpy as np
import sys
sys.path.append("../mpyopt/least_squares")
from block_coordinate_gauss_newton import BlockCoordinateGaussNewton


def test_1():
  """
  A linear least squares problem
  """
  dim_x = 2
  
  # generate a linear least squares problem
  n_points = 1000
  X     = np.random.uniform(0,1,(n_points,dim_x))
  c     = np.random.randn(dim_x)
  Y     = X @ c
  resid = lambda m: (X @ m - Y)
  jac   = lambda m,idx: X[:,idx] # define for block

  x0    = 10*np.ones(dim_x)
  max_iter = 100
  gtol     = 1e-5
  block_size = dim_x-1
  xopt = BlockCoordinateGaussNewton(resid,jac,x0,block_size,max_iter,gtol,verbose=False)
  print('Test 1: distance to opt: ',np.linalg.norm(xopt - c))
  return xopt

def test_2():
  """
  A higher dimensional linear least squares problem
  """
  dim_x = 30
  
  # generate a linear least squares problem
  n_points = 5000
  X     = np.random.uniform(0,1,(n_points,dim_x))
  c     = 10*np.random.randn(dim_x)
  Y     = X @ c
  resid = lambda m: (X @ m  - Y)
  jac   = lambda m,idx: X[:,idx] # define for block

  x0    = c + 10*np.random.randn(dim_x) 
  max_iter = 500
  gtol     = 1e-5
  block_size = 10
  xopt = BlockCoordinateGaussNewton(resid,jac,x0,block_size,max_iter,gtol,verbose=False)
  print('Test 2: distance to opt: ',np.linalg.norm(xopt - c))
  return xopt

def test_3():
  """
  Sigmoid fit non-linear least squares
  """
  def finite_difference(f,x0,h=1e-6):
    """Compute the jacobian of f with
    central difference
    """
    h2   = h/2.0
    dim  = len(x0)
    Ep   = x0 + h2*np.eye(dim)
    Fp   = np.array([f(e) for e in Ep])
    Em   = x0 - h2*np.eye(dim)
    Fm   = np.array([f(e) for e in Em])
    jac = (Fp - Fm)/(h)
    return jac.T
  
  # generate data
  n_points = 5000
  dim_x = 8
  X     = np.random.uniform(0,1,(n_points,dim_x))
  c     = 10*np.random.randn(dim_x)
  Y     = 1/(1+np.exp(-X@c))
  resid = lambda m: (1/(1+np.exp(-X@m))- Y)
  def jac(m,idx):
    return finite_difference(resid,m,1e-8)[:,idx]

  x0    = c + 10*np.random.randn(dim_x) 
  max_iter = 1000
  gtol     = 1e-5
  block_size= dim_x
  xopt = BlockCoordinateGaussNewton(resid,jac,x0,block_size,max_iter,gtol,verbose=False)
  print('Test 3: distance to opt: ',np.linalg.norm(xopt - c))
  return xopt

test_1()
test_2()
test_3()
