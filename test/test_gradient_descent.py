import numpy as np
import sys
sys.path.append("../mpyopt/problem")
from quadratic import Quadratic
from rosenbrock import Rosenbrock
sys.path.append("../mpyopt/first_order")
from gradient_descent import GradientDescent


def test_1():
  dim_x = 2
  xopt  = np.random.randn(dim_x)
  A     = np.eye(dim_x)
  prob  = Quadratic(A,xopt)
  x0    = 10*np.ones(dim_x)
  x_res = GradientDescent(prob.eval,prob.grad,x0,gtol=1e-5,ls_type='backtracking')
  f_res = prob.eval(x_res)
  print('Test 1: distance to opt: ',np.linalg.norm(x_res-xopt))
  print('Test 1: best value found: ',f_res)
  return x_res

def test_2():
  """
  A 1 dimensional quadratic test.
  """
  dim_x = 1
  xopt  = np.random.randn(dim_x)
  obj = lambda x: np.sum((x-xopt)**2)
  grad = lambda x: np.array([np.sum(2*(x-xopt))])
  x0    = 10*np.ones(dim_x)
  x_res = GradientDescent(obj,grad,x0,gtol=1e-5,ls_type='quadratic')
  f_res = obj(x_res)
  print('Test 2: distance to opt: ',np.linalg.norm(x_res-xopt))
  print('Test 2: best value found: ',f_res)
  return x_res

def test_3():
  """
  High dimensional quadratic test
  with random quadratic.
  """
  dim_x = 35
  xopt  = np.random.randn(dim_x)
  Q = np.random.randn(dim_x,dim_x)
  A = Q@Q.T + 0.5*np.eye(dim_x)
  c = np.random.randn()
  prob  = Quadratic(A,xopt,c)
  x0    = 10*np.ones(dim_x)

  x_res = GradientDescent(prob.eval,prob.grad,x0,gtol=1e-5,ls_type='cubic')
  f_res = prob.eval(x_res)

  print('Test 3: distance to opt: ',np.linalg.norm(x_res-xopt))
  print('Test 3: best value found: ',f_res)
  #import matplotlib.pyplot as plt
  #plt.plot(fX)
  #plt.show()
  return x_res
  
def test_4():
  """
  Medium dimensional chained Rosenbrock 
  """
  dim_x = 15
  prob = Rosenbrock(dim_x)
  xopt = prob.minimum
  x0   = np.random.uniform(prob.lb,prob.ub)

  x_res = GradientDescent(prob.eval,prob.grad,x0,gtol=1e-5,ls_type='cubic')
  f_res = prob.eval(x_res)

  print('Test 4: distance to opt: ',np.linalg.norm(x_res-xopt))
  print('Test 4: best value found: ',f_res)
  #import matplotlib.pyplot as plt
  #plt.plot(fX)
  #plt.show()
  return x_res

test_1()
test_2()
test_3()
test_4()
