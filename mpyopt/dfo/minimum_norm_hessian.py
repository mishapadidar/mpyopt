import numpy as np
from scipy.linalg import null_space
import cvxpy as cp
from scipy.optimize import minimize as sp_minimize, Bounds as sp_bounds
import sys
#sys.path.append("../first_order")
#from gradient_descent import GradientDescent

class MinimumNormHessian():
  """
  Minimum Norm Hessian trust region method to minimize obj(x)

  At each iteration we build a quadratic interpolation model of 
  our objective and minimize within the infinity-norm trust region.
  """

  def __init__(self,
    obj,
    x0,
    delta=0.01,
    eta=1e-5,
    gamma_inc=2.0,
    max_eval=100,
    delta_max=1.0,
    delta_min=1e-6,
    theta1=1e-2,
    theta2=1e-1,
    theta_exp = 1.,
    max_points=300,
    ftarget=-np.inf):
    """
    input:
    obj: function handle for minimization
    x0: starting point, 1d array
    delta: initial trust region size
    eta: sufficient decrease condition parameter in (0,1)
    gamma_inc: trust region growth coefficient
    max_eval: maximum number of evaluations, a stopping criteria
    delta_max: maximum trust region size, typically a few orders of magnitude larger
               than the initial size
    delta_min: smallest trust region size, a stopping criteria
    theta1: threshold to determine validity of interpolation set, in (0,1)
    theta_exp: search for points for the quadratic model within theta_exp*delta, >= 1
    max_points: max number of points used in the quadratic model
    ftarget: target function value, will stop if this is reached
    """

    self.obj = obj
    self.x0 = x0
    self.dim_x = len(x0)
    self.delta0 = delta
    self.eta = eta
    self.gamma_inc = gamma_inc;
    self.max_eval = max_eval
    self.delta_min = delta_min
    self.delta_max = delta_max
    self.theta1 = theta1
    self.theta2 = theta2
    self.theta_exp = theta_exp
    # cant use more than d(d+1)/2 + d points
    self.max_points = min(max_points,int(self.dim_x*(self.dim_x+1)/2 + self.dim_x))
    self.ftarget=ftarget

    # storage
    self.X = np.zeros((0,self.dim_x))
    self.fX = np.zeros(0)

    # for ill-conditioning
    self.n_fails = 0
    self.max_fails = 3

    assert (max_eval > self.dim_x + 1), "Need more evals"
    assert delta > 0, "Need larger TR radius"
    assert (delta_min > 0 and delta_min < delta), "Invalid delta_min"
    assert (0 < eta and eta < 1), "Invalid eta"

  def solve(self):

    x0 = np.copy(self.x0)
    f0 = self.obj(self.x0)
    delta_k = self.delta0

    # initial sampling
    _X  = x0 + delta_k*np.eye(self.dim_x)
    _fX = np.array([self.obj(xx) for xx in _X])

    # storage
    self.X  = np.copy(_X)
    self.X  = np.vstack((self.X,x0))
    self.fX = np.copy(_fX)
    self.fX = np.copy(np.append(self.fX,f0))

    # select the best point to be the center
    idx_best = np.argmin(self.fX)
    idx_other = [ii for ii in range(len(self.fX)) if ii != idx_best]
    x_k = np.copy(self.X[idx_best])
    f_k = np.copy(self.fX[idx_best])
    _X  = np.copy(self.X[idx_other])
    _fX = np.copy(self.fX[idx_other])

    self.n_evals = len(self.fX)
    while self.n_evals < self.max_eval and delta_k > self.delta_min:
      # form the model 
      alpha_k,beta_k = self.make_model(x_k,f_k,_X,_fX)
      # solve the TR subproblem
      y_plus = self.solve_subproblem(alpha_k,beta_k,delta_k)
      # shift back to original domain
      x_plus = np.copy(x_k + y_plus)
      # evaluate model and objective
      m_plus = f_k + alpha_k @ y_plus + beta_k @ self.quadratic_features([y_plus]).flatten()
      f_plus = self.obj(x_plus)
      self.n_evals += 1
      
      # save eval
      self.X  = np.copy(np.vstack((self.X,x_plus)))
      self.fX = np.copy(np.append(self.fX,f_plus))

      # check termination
      if f_plus <= self.ftarget:
        x_k = np.copy(x_plus)
        f_k = f_plus
        break

      # do ratio rest
      #print(f_k,m_plus,f_plus)
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

      # get a linear model
      _X,_fX = self.affPoints(x_kp1,f_kp1,delta_kp1)
      # TODO: add points
      _X,_fX = self.addPoints(x_kp1,f_kp1,delta_kp1,_X,_fX)

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

  def quadratic_features(self,X):
    """
    Form the set of dim_x*(dim_x+1)/2 quadratic features
    of a symmetric quadratic model.
      feat = [x1^2, ..., xn^2, x1*x2,...,x_{n-1}x_n]
    A quadratic model m(x) = beta @ feat is equivalent to 
    m(x) = x @ B @ x where
      B = np.zeros((dim_x,dim_x)
      B[np.triu_indices(dim_x)] = beta
      B = (B+B.T)/2 

    X: (N,dim_x) array of points
    return
    feat: (N,dim_x*(dim_x+1)/2) array of quadratic features. 
    """
    N,dim_x = np.shape(X)
    n_feat = int(dim_x*(dim_x+1)/2)
    feat = np.zeros((N,n_feat))
    for ii,x in enumerate(X):
      outer = np.outer(x,x)
      x_feat = outer[np.triu_indices(dim_x)]
      feat[ii,:] = np.copy(x_feat)
    return np.copy(feat)

  def make_model(self,x0,f0,_X,_fX):
    """
    Given an interpolation set, compute the coefficients
    of the minimum norm hessian quadratic interpolation model
      m(x) = alpha @ (x-x0) + beta @ quadratic_features(x-x0)
    
    Let Y be the row vectors for the interpolation
    set, centered around zero. Let Z be an orthogonal basis for 
    the null space of Y.T, and Q@R = Y. Let NY.T be the quadratic
    feature vectors, as rows.
    
    alpha and beta satisfy equtions (3.5) and (3.6) of Stefan
    Wild's MNH paper:
    Then there exist a unique alpha, beta, w which satisfy
      Z.T @ NY.T @ NY @ Z @ w = Z.T @ fY
      R @ alpha = Q.T @ (fY - NY.T @ NY @ Z@ w)
      beta = NY @ Z @ w
    where fY are the function values. lambda = Z @ w are the lagrange
    multipliers, and the array shapes are
      w: |len(Y)| - dim_x - 1 
      alpha: (dim_x,)
      beta: (dim_x*(dim_x+1)/2,)
    

    input:
    x0: 1d array, (dim_x,), Trust region center
    f0: float, function value at x0
    _X: 2d array, (dim_x,dim_x), interpolation points
        points are rows.
    _fX: 2d array, (dim_x,), interpolation values
 
    output:
    alpha: 1d array, (dim_x,), linear coefficients
    beta: 1d array, (dim_x*(dim_x+1)/2,) quadratic coefficients
    """
    # shift the points
    _Y = np.copy(_X - x0)
    # shift the function values
    _fY = _fX - f0
    # form quadratic features
    _NYT = self.quadratic_features(_Y) # row vectors
    # compute Null(_Y.T)
    _Z = null_space(_Y.T) # col vectors
    # QR factor _NYT @ Z
    Q,R = np.linalg.qr(_NYT.T @ _Z)
    # solve for lagrange multipliers
    w = np.linalg.solve(R.T @ R, _Z.T @ _fY)
    # solve for beta
    beta = _NYT.T @ _Z @ w
    # QR factor _Y
    Q,R = np.linalg.qr(_Y)
    # solve for alpha
    alpha = np.linalg.solve(R,Q.T @ (_fY - _NYT @ beta))

    # TODO: add jitter to the solve.
    #try:
    #  m = np.linalg.solve(_Y,_fY)
    #except:
    #  # jitter if unstable
    #  _Y = self.jitter(_Y)
    #  self.n_fails +=1 
    #  # increase theta1 to prevent more fails
    #  self.theta1 = 10*self.theta1
    #  if self.n_fails >= self.max_fails:
    #    print("Exiting: Too many failed solves")
    #    print("Try increasing theta1")
    #    result = {}
    #    result['x']   = x0
    #    result['f']   = f0
    #    result['X']   = np.copy(self.X)
    #    result['fX']  = np.copy(self.fX)
    #    return x0
    #  m = np.linalg.solve(_Y,_fY)
    return np.copy(alpha),np.copy(beta)

  def solve_subproblem(self,alpha,beta,delta):
    """
    Solve the Trust region subproblem
      min_x  alpha @ x + beta @ quadratic_features(x)
      s.t. ||x||_inf <= delta
    Problem is solved around the origin.
    So solution should be shifted back around
    trust region center.
    
    alpha: 1d array, linear model parameters
    beta: 1d array, quadratic model parameters
    delta: float, trust region radius
    """
    # form the quadratic features as a matrix
    B = np.zeros((self.dim_x,self.dim_x))
    B[np.triu_indices(self.dim_x)] = beta
    B = (B+B.T)/2 # ensures correct weighting in objective

    # 2-norm TR subproblem
    #def obj(x):
    #  if x @ x < delta**2:
    #    obj2 = x @ alpha + beta @ self.quadratic_features([x]).flatten()
    #    obj1 = x @ alpha + x @ B @ x
    #    if np.abs(obj1-obj2) > 1e-15:
    #      print(obj1-obj2)
    #    return x @ alpha + x @ B @ x 
    #  else: 
    #    return np.inf
    #def grad(x):
    #  return alpha + 2*B @ x 
    #x0 = np.zeros(self.dim_x)
    #x = GradientDescent(obj,grad,x0,alpha0 = 1.0,gamma=0.8,max_iter=100,gtol=1e-7)
    #true = np.copy(-delta*alpha/np.linalg.norm(alpha))
    #return np.copy(x)

    # inf-norm TR subproblem
    def obj(x):
      return x @ alpha + x @ B @ x
    def grad(x):
      return alpha + 2*B @ x 
    bounds = sp_bounds(lb=-delta*np.ones(self.dim_x),ub=delta*np.ones(self.dim_x))
    res = sp_minimize(obj,x0=np.zeros(self.dim_x),jac=grad,method='L-BFGS-B',bounds=bounds,
          options={'gtol':1e-14,'ftol':1e-13})
    return res.x
    #return np.copy(-delta*alpha/np.linalg.norm(alpha))

  def affPoints(self,x0,f0,delta):
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
    theta1. If such a set of points exists, then our model will be fully linear on 
    a radius delta. 
    If we still do not have a complete set we 
    evaluate model improvement points.
    """
    # storage for new displacements
    _Y = np.zeros((0,self.dim_x))
    _fY = np.zeros(0)
  
    # rows are basis for null(_Y)
    _Z = np.eye(self.dim_x)     
  
    # default
    linear = False

    # find points within distance delta
    #idx = np.linalg.norm(self.X-x0,axis=1) <= delta
    idx = np.max(np.abs(self.X-x0),axis=1) <= delta
    # use shifted points
    Yt   = self.X[idx] - x0
    fYt  = self.fX[idx] - f0
    for jj,dj in enumerate(Yt):
      fj = fYt[jj]
      # check |proj_Z(dj/delta)| >= theta1
      if np.linalg.norm(_Z.T @ (_Z @ dj/delta)) >= self.theta1:
        _Y  = np.copy(np.vstack((_Y,dj)))
        _fY = np.copy(np.append(_fY,fj))
        # find new null space
        _Z = null_space(_Y).T
      if len(_Y) == self.dim_x:
        linear = True
        break
      else:
        linear = False
  
    # now propose new points
    if linear == False:
      # evaluate f(Z)
      _fZ = np.array([self.obj(x0 + delta*zz) for zz in _Z]) - f0
      _Y = np.vstack((_Y,delta*_Z))
      _fY = np.append(_fY,_fZ)
      # save the new evals
      self.n_evals += len(_fZ)
      self.X  = np.copy(np.vstack((self.X, x0 + delta*_Z)))
      self.fX = np.copy(np.append(self.fX, f0 + _fZ))

    # return interpolation points
    _X = np.copy(x0 + _Y)
    _fX = np.copy(f0 + _fY)
  
    return _X,_fX

  def addPoints(self,x0,f0,delta,_X,_fX):
    """
    Add points to build the quadratic model. This is essentially
    Algorithm 4.2 from Stefan Wild's MNH paper.

    """
    # shift the points
    _Y = np.copy(_X - x0)
    _fY = np.copy(_fX - f0)
  
    # find points within infinity distance delta
    #idx = np.linalg.norm(self.X-x0,axis=1) <= delta*self.theta_exp
    idx = np.max(np.abs(self.X-x0),axis=1) <= delta*self.theta_exp
    # use shifted points
    Yt   = self.X[idx] - x0
    fYt  = self.fX[idx] - f0
    # form quadratic features
    _NYT = self.quadratic_features(_Y) # row vectors
    for jj,dj in enumerate(Yt):
      # make tilde_Y 
      tilde_Y  = np.copy(np.vstack((_Y,dj)))
      # get tilde_Z = null(tilde_Y.T)
      tilde_Z = null_space(tilde_Y.T) # col vectors
      # make tilde_NYT
      qf = self.quadratic_features([dj]).flatten()
      tilde_NYT  = np.copy(np.vstack((_NYT,qf)))
      # compute sigma_min(tilde_NYT @ tilde_Z)
      sigma_min = np.min(np.linalg.svd(tilde_NYT.T @ tilde_Z,compute_uv=False))
      if sigma_min >= self.theta2:
        # add the points to the interpolation set
        fj = fYt[jj]
        _Y  = np.copy(tilde_Y)
        _fY = np.copy(np.append(_fY,fj))
        # prepare for next iteration
        _Z = np.copy(tilde_Z)
        _NYT = np.copy(tilde_NYT)

      if len(_Y) >= self.max_points:
        # reached maximum allowable model
        break

    # return interpolation points
    _X = np.copy(x0 + _Y)
    _fX = np.copy(f0 + _fY)
  
    return _X,_fX


  def jitter(self,A,jit=1e-10):
    """
    Add a "jitter" to the matrix
  
    input
    A: (n,n) matrix

    return 
    (n,n) matrix, A + jit*np.eye(n)
    """
    return np.copy(A + jit*np.eye(len(A)))
    
  
