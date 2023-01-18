import numpy as np


def GradientDescent(func,grad,x0,alpha0 = 1.0,
  gamma=0.5,max_iter=10000,gtol=1e-3,c_1=1e-4,
  verbose=False,ls_type='quadratic'):
  """
  Gradient descent with sim failure safe armijo linesearch.
  Optimization will stop if any of the stopping criteria are met.

  func: objective function handle, for minimization
  grad: gradient handle
  x0: feasible starting point
  gamma: linesearch decrease parameter
  max_iter: maximimum number of iterations
  gtol: projected gradient tolerance
  c_1: Armijo parameters for linesearch.
           must satisfy 0 < c_1 < c_2 < 1
  ls_type: linesearch type, 'quadratic' or 'backtracking'
  """
  assert (0 < c_1 and c_1< 1), "unsuitable linesearch parameters"
  assert ls_type in ['cubic','quadratic','backtracking']

  # inital guess
  x_k = np.copy(x0)
  dim = len(x_k)

  # minimum step size
  alpha_min = 1e-18
  # initial step size
  alpha_k = alpha0

  # compute gradient
  g_k    = np.copy(grad(x_k))
  # compute function value
  f_k    = np.copy(func(x_k))

  # stop when gradient is flat (within tolerance)
  nn = 0
  stop = False
  while stop==False:
    if verbose:
      print(f_k,np.linalg.norm(g_k))

    # compute search direction
    p_k = - g_k

    # compute step
    x_kp1 = x_k + alpha_k*p_k
    f_kp1 = func(x_kp1);

    # set values for the cubic linesearch
    if ls_type == "cubic":
        alpha_old = None
        f_old = None

    # linsearch with Armijo condition
    armijo = f_kp1 <= f_k + c_1*g_k @ (x_kp1 - x_k)
    while armijo==False:
      # determine the next step size
      if ls_type == "cubic":
        alpha_new = cubic_linesearch(f_k,g_k,p_k,alpha_k,f_kp1,alpha_old,f_old)
        alpha_old = alpha_k
        f_old = f_kp1
        alpha_k = alpha_new
      elif ls_type == "quadratic":
        alpha_k = quadratic_linesearch(f_k,g_k,p_k,alpha_k,f_kp1)
      elif ls_type == "backtracking":
        alpha_k = backtracking_linesearch(alpha_k,gamma)
      # take step
      x_kp1 = np.copy(x_k + alpha_k*p_k)
      # f_kp1
      f_kp1 = func(x_kp1);
      if np.isfinite(f_kp1) == False:
        armijo = False
      else:
        # compute the armijo condition
        armijo = f_kp1 <= f_k + c_1*g_k @ (x_kp1 - x_k)

      # break if alpha is too small
      if alpha_k <= alpha_min:
        if verbose:
          print('Exiting: alpha too small.')
        return x_k

    # gradient
    g_kp1 = np.copy(grad(x_kp1))

    # increase alpha to counter backtracking
    #alpha_k = alpha_k/gamma
    # set alpha to give the same linear decrease as prev step
    alpha_k = alpha_k*(g_k @ -g_k)/(g_kp1 @ -g_kp1) # // N & W Ch. 3 (Initial Step Length)

    # reset for next iteration
    x_k  = np.copy(x_kp1)
    f_k  = f_kp1;
    g_k  = np.copy(g_kp1);

    # update iteration counter
    nn += 1


    # stopping criteria
    if np.linalg.norm(g_k) <= gtol:
      if verbose:
        print('Exiting: gtol reached')
      stop = True
    elif nn >= max_iter:
      if verbose:
        print('Exiting: max_iter reached')
      stop = True


  return x_k

def backtracking_linesearch(alpha,gamma=0.5):
    """
    Return the new value of alpha according to a backtracking linesearch.
    In general, this means dividing alpha by 2.
    """
    return alpha*gamma

def quadratic_linesearch(f0,g0,p,alpha0,f_alpha0):
    """
    Perform a quadratic interpolation linesearch. Method returns the next learning
    rate.

    The quadratic linesearch fits a quadratic interpolation model to the one dimensional function
      phi(alpha) = f(x + alpha*p),
    and minimizes the model to determine the learning rate alpha.
    
    After testing our first linesearch parameter alpha0, we know phi(0) = f(x), 
    phi'(0) = gradf(x) @ p, and phi(alpha0) = f(x+alpha0*p).
    We build a quadratic interpolation model on phi using the three values,
      Q(alpha) = a*alpha^2 + b*alpha + c
    where
      a = (f(x + alpha0*p) - c - b*alpha0)/(alpha0^2)
      b = Q'(0) = phi'(0) = gradf(x) @ p
      c = Q(0) = phi(0) = f(x)
    See Nocedal & Wrigth equation (3.57).
    We then minimize the Quadratic to find the next value of alpha. 
    The minimizer lies in the interval [0,alpha0]


    f0: f(x_k), current function value
    g0: g_k, current gradient value
    p: p_k, current step direction
    alpha0: float, last learning rate tested
    f_alpha0: f(x + alpha0*p), function value using last alpha

    return alpha, new learning rate
    """
    c = f0
    b = g0 @ p
    a = (f_alpha0 - c - b*alpha0)/(alpha0**2)

    alph = -b/(2*a)

    # check the safegaurds; alpha cannot be too close or too far
    if (alph < alpha0/16) or (alph > 15*alpha0/16):
      alph = alpha0/2

    return alph


def cubic_linesearch(f0,g0,p,alpha0,f_alpha0,alpha1=None,f_alpha1=None):
    """
    Perform a cubic interpolation linesearch.

    Invokes a quadratic linesearch if alpha1 is None.

    See section 3 (interpolation subsection) of Nocedal and Wright).

    f0: f(x_k), current function value
    g0: g_k, current gradient value
    p: p_k, current step direction
    alpha0: float, second to last learning rate tested
    f_alpha0: f(x + alpha0*p), function value using alpha0
    alpha1: float, last learning rate tested
    f_alpha1: f(x + alpha1*p), function value using alpha1

    return alpha, new learning rate
    """
    if alpha1 is None:
      return quadratic_linesearch(f0,g0,p,alpha0,f_alpha0)
    # check the safegaurds; alpha1 cannot be too close or too far
    if (alpha1 < alpha0/16) or (alpha1 > 15*alpha0/16):
      return alpha0/2
    
    det = 1.0/alpha0/alpha0/alpha1/alpha1/(alpha1-alpha0)
    r = f_alpha1 - f0 - g0 @ p * alpha1
    t = f_alpha0 - f0 - g0 @ p * alpha0
    a = (1.0/det) * ( (alpha0**2)*r - (alpha1**2)*t)
    b = (1.0/det) * ( -(alpha0**3)*r + (alpha1**3)*t)
    alph = (-b + np.sqrt(b**2 - 3*a*(g0 @ p)))/(3*a)

    # check the safegaurds; alpha cannot be too close or too far
    if (alph < alpha0/16) or (alph > 15*alpha0/16):
      alph = alpha0/2

    return alph
