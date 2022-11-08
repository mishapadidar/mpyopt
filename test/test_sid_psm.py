import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize as sp_minimize
import sys
sys.path.append("../mpyopt/problem")
from quadratic import Quadratic
from rosenbrock import Rosenbrock
sys.path.append("../mpyopt/dfo")
#sys.path.append("../mpyopt/first_order")
from sid_psm import SIDPSM
from linear_trust_region import LinearTrustRegion
from minimum_norm_hessian import MinimumNormHessian


def test_1():
  dim_x = 2
  xopt  = np.random.randn(dim_x)
  A     = np.eye(dim_x)
  prob  = Quadratic(A,xopt)
  x0    = 10*np.ones(dim_x)
  max_eval = 500
  delta_min = 1e-8
  ftarget = 1e-12
  sid = SIDPSM(prob.eval,x0,max_eval=max_eval,delta_min=delta_min)
  res = sid.solve()
  #mnh = MinimumNormHessian(prob.eval,x0,max_eval=max_eval,delta_min=delta_min,ftarget=ftarget)
  #res = mnh.solve()
  x_res = res['x'] 
  f_res = res['f']
  print('Test 1: distance to opt: ',np.linalg.norm(x_res-xopt))
  print('Test 1: best value found: ',f_res)
  fX = res['fX']
  plt.plot(fX,label='SID-PSM')

  fX = []
  def evw(x):
    f = prob.eval(x)
    fX.append(f)
    return f
  res = sp_minimize(evw,x0,method='Nelder-Mead')
  plt.plot(fX,label='nelder-mead')

  ltr = LinearTrustRegion(prob.eval,x0,max_eval=max_eval,delta_min=delta_min)
  res = ltr.solve()
  fX = res['fX']
  plt.plot(fX,label='LTR')
  plt.legend()
  plt.show()
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
  ftarget = 1e-12
  sid = SIDPSM(obj,x0,max_eval=max_eval,delta_min=delta_min)
  res = sid.solve()
  #mnh = MinimumNormHessian(obj,x0,max_eval=max_eval,delta_min=delta_min,ftarget=ftarget)
  #res = mnh.solve()
  x_res = res['x'] 
  f_res = res['f']
  print('Test 2: distance to opt: ',np.linalg.norm(x_res-xopt))
  print('Test 2: best value found: ',f_res)
  fX = res['fX']
  plt.plot(fX,label='SID-PSM')

  fX = []
  def evw(x):
    f = obj(x)
    fX.append(f)
    return f
  res = sp_minimize(evw,x0,method='Nelder-Mead')
  plt.plot(fX,label='nelder-mead')

  ltr = LinearTrustRegion(obj,x0,max_eval=max_eval,delta_min=delta_min)
  res = ltr.solve()
  fX = res['fX']
  plt.plot(fX,label='LTR')
  plt.legend()
  plt.show()
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

  max_eval = 1000
  delta_min = 1e-8
  delta_max = 20.0
  delta0 = 1.0
  #mnh = MinimumNormHessian(prob.eval,x0,max_eval=max_eval,delta = delta0,delta_min=delta_min)
  #res = mnh.solve()
  sid = SIDPSM(prob.eval,x0,max_eval=max_eval,delta=delta0,delta_min=delta_min,delta_max=delta_max)
  res = sid.solve()
  x_res = res['x'] 
  f_res = res['f']
  print('Test 3: distance to opt: ',np.linalg.norm(x_res-xopt))
  print('Test 3: best value found: ',f_res)
  fX = res['fX']
  plt.plot(fX,label='SID-PSM')

  fX = []
  def evw(x):
    f = prob.eval(x)
    fX.append(f)
    return f
  res = sp_minimize(evw,x0,method='Nelder-Mead')
  plt.plot(fX,label='nelder-mead')

  ltr = LinearTrustRegion(prob.eval,x0,max_eval=max_eval,delta = delta0,delta_min=delta_min,delta_max=delta_max)
  res = ltr.solve()
  fX = res['fX']
  plt.plot(fX,label='LTR')
  plt.legend()
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

  max_eval = 500
  delta_min = 1e-8
  delta0 = 0.01
  delta_max = 1.0
  #mnh = MinimumNormHessian(prob.eval,np.copy(x0),max_eval=max_eval,delta = delta0,delta_min=delta_min)
  #res = mnh.solve()
  sid = SIDPSM(prob.eval,x0,max_eval=max_eval,delta=delta0,delta_min=delta_min,delta_max=delta_max)
  res = sid.solve()
  x_res = res['x'] 
  f_res = res['f']
  print('Test 4: distance to opt: ',np.linalg.norm(x_res-xopt))
  print('Test 4: best value found: ',f_res)
  fX = res['fX']
  plt.plot(fX,label='SID-PSM')

  fX = []
  def evw(x):
    f = prob.eval(x)
    fX.append(f)
    return f
  res = sp_minimize(evw,x0,method='Nelder-Mead')
  plt.plot(fX,label='nelder-mead')

  ltr = LinearTrustRegion(prob.eval,x0,max_eval=max_eval,delta = delta0,delta_min=delta_min,delta_max=delta_max)
  res = ltr.solve()
  fX = res['fX']
  plt.plot(fX,label='LTR')
  plt.legend()
  plt.show()
  return res['x']

def test_5():
  """
  A 2 dimensional quadratic with "simulation failures"
  """
  dim_x = 2
  xopt  = np.random.randn(dim_x)
  A     = np.eye(dim_x)
  prob  = Quadratic(A,xopt)
  x0    = 10*np.ones(dim_x)
  # define the sim fails
  x_barrier = (x0 + xopt)/2
  bounds = np.vstack((-np.inf*np.ones(dim_x),np.inf*np.ones(dim_x))).T
  bounds[0,0] = x_barrier[0]
  def obj(x):
    if x[0] < x_barrier[0]:
      return np.inf
    else:
      return prob.eval(x)

  max_eval = 500
  delta_min = 1e-8
  delta_max=10.0
  ftarget = 1e-12
  delta0 = 0.1
  sid = SIDPSM(obj,x0,max_eval=max_eval,delta=delta0,delta_min=delta_min,delta_max=delta_max)
  res = sid.solve()
  #mnh = MinimumNormHessian(prob.eval,x0,max_eval=max_eval,delta_min=delta_min,ftarget=ftarget)
  #res = mnh.solve()
  x_res = res['x'] 
  f_res = res['f']
  print('Test 5: distance to opt: ',np.linalg.norm(x_res-xopt))
  print('Test 5: best value found: ',f_res)
  fX = res['fX']
  #plt.plot(fX,label='SID-PSM')
  plt.plot(np.minimum.accumulate(fX),label='SID-PSM')
  
  fX = []
  def evw(x):
    f = obj(x)
    fX.append(f)
    return f
  res = sp_minimize(evw,x0,method='Nelder-Mead')
  #plt.plot(fX,label='nelder-mead')
  plt.plot(np.minimum.accumulate(fX),label='nelder-mead')

  ltr = LinearTrustRegion(obj,x0,max_eval=max_eval,delta=delta0,delta_min=delta_min,delta_max=delta_max)
  res = ltr.solve()
  fX = res['fX']
  #plt.plot(fX,label='LTR')
  plt.plot(np.minimum.accumulate(fX),label='LTR')

  res = sp_minimize(prob.eval,x0=x0,method='L-BFGS-B',bounds=bounds)
  fopt = obj(res.x)
  plt.axhline(y=fopt, color='r', linestyle='-',label='optimum')
  plt.legend()
  plt.show()
  return res['x']

test_1()
test_2()
test_3()
test_4()
test_5()
