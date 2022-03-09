import numpy as np
from scipy.linalg import qr as rrqr

#def linear_trust_region(obj,x0,delta=0.01,eta=1e-5,max_eval=100,delta_max=1.0,delta_min=1e-6,theta=1e-3):
class linear_trust_region():
   """
   Linear trust region method to minimize obj(x)

   At each iteration we build a linear interpolation model of 
   our objective and minimize within the 2-norm trust region.
   """

  def __init__(self,obj,x0,delta=0.01,eta=1e-5,max_eval=100,delta_max=1.0,delta_min=1e-6,theta=1e-3):
    """
    input:
    obj: function handle for minimization
    x0: starting point, 1d array
    delta: initial trust region size
    eta: sufficient decrease condition parameter in (0,1)
    max_eval: maximum number of evaluations, a stopping criteria
    delta_max: maximum trust region size, typically a few orders of magnitude larger
               than the initial size
    delta_min: smallest trust region size, a stopping criteria
    theta: threshold to determine validity of interpolation set, in (0,1)
    """

    self.obj = obj
    self.x0 = x0
    self.dim_x = len(x0)
    self.delta0 = delta
    self.eta = eta
    self.max_eval = max_eval
    self.delta_min = delta_min
    self.delta_max = delta_max
    self.theta = theta
    self.gamma_inc = 2.0

    # storage
    self.X = np.zeros((0,dim_x))
    self.fX = np.zeros(0)

    assert (max_eval > dim_x + 1), "Need more evals"
    assert delta > 0, "Need larger TR radius"
    assert (delta_min > 0 and delta_min < delta), "Invalid delta_min"
    assert (0 < eta and eta < 1), "Invalid eta"

  def solve():

    x0 = np.copy(self.x0)
    f0 = obj(self.x0)
    delta_k = self.delta0

    # initial sampling
    _X  = x0 + delta_k*np.eye(self.dim_x)
    _fX = np.array([self.obj(xx) for xx in _X])

    # storage
    self.X  = np.copy(_X)
    self.X  = np.vstack((X,x0))
    self.fX = np.copy(_fX)
    self.fX = np.copy(np.append(fX,f0))

    # select the best point to be the center
    idx_best = np.argmin(self.fX)
    idx_other = [ii for ii in range(len(self.fX)) if ii != idx_best]
    x_k = np.copy(self.X[idx_best])
    f_k = np.copy(self.fX[idx_best])
    _X  = np.copy(self.X[idx_other])
    _fX = np.copy(self.fX[idx_other])

    n_evals = len(fX)
    while n_evals < self.max_eval and delta_k > self.delta_min:
      # form the linear model 
      m_k = self.make_model(x_k,f_k,_X,_fX)
      # solve the TR subproblem
      y_plus = self.solve_subproblem(m_k,delta_k)
      # shift back to original domain
      x_plus = np.copy(x_k + y_plus)
      # evaluate model and objective
      m_plus = f_k + m_k @ y_plus
      f_plus = obj(x_plus)
      
      # save eval
      self.X  = np.copy(np.vstack((self.X,x_plus)))
      self.fX = np.copy(np.append(self.fX,f_plus))

      # do ratio rest
      rho = (f_k - f_plus)/(f_k - m_plus)
      # choose next iterate
      if rho >= self.eta:
        x_kp1 = np.copy(x_plus)
        f_kp1 = f_plus
      else:
        x_kp1 = np.copy(x_k)
        f_kp1 = f_k
      # shrink/expand TR
      if rho >= self.eta and np.linalg.norm(x_k - x_plus) >= 0.75*delta_k:
        delta_kp1 = min(self.gamma_inc*delta_k,self.delta_max)
      elif rho >= self.eta:
        delta_kp1 = delta_k
      else:
        delta_kp1 = delta_k/self.gamma_inc

      # get a new model
      _X,_fX = self.get_model_points(x_kp1,f_kp1,delta_kp1):
      
      # prepare for next iteration
      x_k = np.copy(x_kp1)
      f_k = f_kp1
      delta_k = delta_kp1

    result = {}
    result['x']   = x_k
    result['f']   = f_k
    result['X']   = np.copy(self.X)
    result['fX']  = np.copy(self.fX)
    return result

  def make_model(self,x0,f0,_X,_fX):
    """
    Build the linear model from interpolation 
    points.

    input:
    x0: 1d array, (dim_x,), Trust region center
    f0: float, function value at x0
    _X: 2d array, (dim_x,dim_x), interpolation points
        points are rows.
    _fX: 2d array, (dim_x,), interpolation values
 
    output:
    m: 1d array, linear model such that
     f(x) ~ f0 + m @ (x-x0)
    """
    # shift the points
    _Y = np.copy(_X - x0)
    # shift the function values
    _fY = _fX - f0
    # interpolate
    m = np.linalg.solve(_Y,_fY)
    return np.copy(m)

  def solve_subproblem(self,m,delta):
    """
    Solve the Trust region subproblem
      min_x  m @ x
      s.t. ||x|| <= delta
    Problem is solved around the origin.
    So solution should be shifted back around
    trust region center.
    
    m: 1d array, (dim_x,), linear model
    delta: float, trust region radius
    """
    return np.copy(-delta*m/np.linalg.norm(m))

  def get_model_points(self,x0,f0,delta):
    """
    A model selection and improvement routine. 

    Find a set of affinely independent points, within the eval history.
    If no such set exists, propose new evaluation points
    to complete the set.
  
    This method is essentially the AffPoints method from Stefan Wild's
    ORBIT algorithm.
  
    This algorithm ensures, by
    the analysis in Stefan's 2013 paper, Global convergence of radial 
    basis function trust-region algorithms for derivative-free optimization,
    that our model is fully linear.
  
    First we seek a set of sufficiently affinely independent points within a radius
    delta of x0. sufficiently affinely independent is determined by a tolerance 
    theta. If such a set of points exists, then our model will be fully linear on 
    a radius delta. 
    If a full set of n points does not exist, then we look
    to complete the set by adding sufficiently affinely independent points within a 
    radius of delta_max. If we complete the set then our model is fully linear on 
    delta_max.
    Finally, if we still do not have a complete set we 
    evaluate model improvement points.
    """
    # storage for new displacements
    _Y = np.zeros((0,self.dim_x))
    _fY = np.zeros(0)
  
    # rows are basis for null(_Y)
    _Z = np.eye(self.dim_x)     
  
    # find points within distance delta
    idx = np.linalg.norm(self.X-x0) <= delta
    # use shifted points
    Yt   = self.X[idx] - x0
    fYt  = self.fX[idx] - f0
    for jj,dj in enumerate(Yt):
      fj = fYt[jj]
      # check |proj_Z(dj/delta)| >= theta
      if np.linalg.norm(_Z.T @ (_Z @ dj/delta)) >= theta:
        _Y  = np.copy(np.vstack((_Y,dj)))
        _fY = np.copy(np.append(_fY,fj))
        # find new null space
        _Z = null_space(_Y).T
      if len(_Y) = self.dim_x:
        linear = True
        break
      else:
        linear = False
  
    if linear == False:
      # find points within distance delta_max but > delta
      idx = np.logical_and(np.linalg.norm(self.X-x0) <= self.delta_max,\
            np.linalg.norm(self.X-x0) > delta)
      # use shifted points
      Yt   = self.X[idx] - x0
      fYt  = self.fX[idx] - f0
      for jj,dj in enumerate(Yt):
        fj = fYt[jj]
        # check |proj_Z(dj/delta)| >= theta
        if np.linalg.norm(_Z.T @ (_Z @ dj/delta)) >= theta:
          _Y  = np.copy(np.vstack((_Y,dj)))
          _fY = np.copy(np.append(_fY,fj))
          # find new null space
          _Z = null_space(_Y).T
        if len(_Y) = self.dim_x:
          linear = True
          break
        else:
          linear = False
  
    # now propose new points
    if linear == False:
      # evaluate f(Z)
      _fZ = np.array([obj(x0 + delta*zz) for zz in _Z]) - f0
      _Y = np.vstack((_Y,_Z))
      _fY = np.append(_fY,_fZ)

    # return interpolation points
    _X = np.copy(x0 + _Y)
    _fX = np.copy(f0 + _fY)
  
    return _X,_fX
  
