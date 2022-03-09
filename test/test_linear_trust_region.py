import numpy as np
import sys
sys.path.append("../mpyopt/problem")
from quadratic import Quadratic
from rosenbrock import Rosenbrock
sys.path.append("../mpyopt/dfo")
from linear_trust_region import LinearTrustRegion


def test_1():
  dim_x = 2
  xopt  = np.random.randn(dim_x)
  A     = np.eye(dim_x)
  prob  = Quadratic(A,xopt)
  x0    = 10*np.ones(dim_x)
  max_eval = 500
  delta_min = 1e-8
  ltr = LinearTrustRegion(prob.eval,x0,max_eval=max_eval,delta_min=delta_min)
  res = ltr.solve()
  x_res = res['x'] 
  f_res = res['f']
  print('Test 1: distance to opt: ',np.linalg.norm(x_res-xopt))
  print('Test 1: best value found: ',f_res)
  fX = res['fX']
  return res['x']

def test_2():
  """
  A 1 dimensional quadratic test.
  """
  dim_x = 1
  xopt  = np.random.randn(dim_x)
  obj = lambda x: np.sum((x-xopt)**2)
  x0    = 10*np.ones(dim_x)
  max_eval = 1000
  delta_min = 1e-8
  ltr = LinearTrustRegion(obj,x0,max_eval=max_eval,delta_min=delta_min)
  res = ltr.solve()
  x_res = res['x'] 
  f_res = res['f']
  print('Test 2: distance to opt: ',np.linalg.norm(x_res-xopt))
  print('Test 2: best value found: ',f_res)
  fX = res['fX']
  return res['x']

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

  max_eval = 20000
  delta_min = 1e-8
  delta0 = 5.0
  ltr = LinearTrustRegion(prob.eval,x0,max_eval=max_eval,delta = delta0,delta_min=delta_min)

  res = ltr.solve()
  x_res = res['x'] 
  f_res = res['f']
  print('Test 3: distance to opt: ',np.linalg.norm(x_res-xopt))
  print('Test 3: best value found: ',f_res)
  fX = res['fX']
  import matplotlib.pyplot as plt
  plt.plot(fX)
  plt.show()
  return res['x']
  
def test_4():
  """
  Medium dimensional chained Rosenbrock 
  """
  dim_x = 15
  prob = Rosenbrock(dim_x)
  xopt = prob.minimum
  x0   = np.random.uniform(prob.lb,prob.ub)

  max_eval = 20000
  delta_min = 1e-8
  delta0 = 1.0
  ltr = LinearTrustRegion(prob.eval,x0,max_eval=max_eval,delta = delta0,delta_min=delta_min)

  res = ltr.solve()
  x_res = res['x'] 
  f_res = res['f']
  print('Test 4: distance to opt: ',np.linalg.norm(x_res-xopt))
  print('Test 4: best value found: ',f_res)
  fX = res['fX']
  import matplotlib.pyplot as plt
  plt.plot(fX)
  plt.show()
  return res['x']

test_1()
test_2()
test_3()
test_4()
